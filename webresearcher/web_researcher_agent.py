# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: WebResearcher Agent implementing the IterResearch paradigm.

Supports two tool calling modes:
1. OpenAI Function Calling (default): Uses OpenAI-style tools parameter, works with OpenAI/DeepSeek/etc.
2. XML Protocol: Uses <tool_call> tags, compatible with all LLMs including local models
"""
import json
import json5
import re
import datetime
import asyncio
import random
import time

from typing import Any, Callable, Dict, List, Optional
from openai import (
    AsyncOpenAI,
    APIError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    AuthenticationError,
)
import inspect
import sys
sys.path.append('..')
from webresearcher.base import Message, today_date, build_text_completion_prompt, count_tokens as count_tokens_base
from webresearcher.log import logger
from webresearcher.prompt import get_iterresearch_system_prompt, get_iterresearch_system_prompt_fc, TOOL_DESCRIPTIONS
from webresearcher.tool_file import FileParser
from webresearcher.tool_scholar import Scholar
from webresearcher.tool_python import PythonInterpreter
from webresearcher.tool_search import Search
from webresearcher.tool_visit import Visit
from webresearcher.config import (
    LLM_API_KEY, 
    LLM_BASE_URL, 
    OBS_START, 
    OBS_END, 
    MAX_LLM_CALL_PER_RUN,
    FILE_DIR,
    LLM_MODEL_NAME
)


TOOL_CLASS = [
    FileParser(),
    Scholar(),
    Visit(),
    Search(),
    PythonInterpreter(),
]
TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}


class ResearchRound:
    """
    实现了 IterResearch 范式的核心状态管理器。
    
    状态只包含用于构建下一个提示的核心信息：
    - question (Q): 原始问题
    - current_report (R_{i-1}): 上一轮生成的报告
    - last_observation (O_{i-1}): 上一轮工具调用的结果
    """

    def __init__(self, question: str):
        self.question = question

        # R_{i-1}: 上一轮生成的报告，初始为空
        self.current_report = "This is the first round. The report is empty."

        # O_{i-1}: 上一轮工具调用的结果，初始为空
        self.last_observation = "This is the first round. No tool has been called yet."

    def get_context(self, system_prompt: str) -> List[Dict]:
        """
        构建当前轮次的精简上下文。
        这代表了论文中的 Workspace，即状态 s_t = (Q, R_{i-1}, O_{i-1})
        """
        user_content = (
            f"**Question:** {self.question}\n\n"
            f"**Current Report (R_{{i-1}}):**\n{self.current_report}\n\n"
            f"**Last Observation (O_{{i-1}}):**\n{self.last_observation}"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]


class WebResearcherAgent:
    """
    Web Research Agent implementing the IterResearch paradigm.
    
    The IterResearch paradigm uses a cyclical process of:
    1. Workspace (question + previous report + last tool result)
    2. Plan-Action (LLM generates plan and action)
    3. Tool Execution (execute the tool and get result)
    4. Synthesis (update report by integrating new information)
    
    Supports two tool calling modes:
    - use_xml_protocol=True (default): XML-based <tool_call> tags, works better for IterResearch paradigm
    - use_xml_protocol=False: OpenAI-style function calling, works with OpenAI/DeepSeek/etc.
    """

    def __init__(
            self,
            llm_config: Optional[Dict] = None,
            function_list: Optional[List[str]] = None,
            instruction: str = "",
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            model: Optional[str] = None,
            use_xml_protocol: bool = True,  # XML protocol works better for IterResearch paradigm
    ):
        llm_config = dict(llm_config or {})
        if api_key:
            llm_config["api_key"] = api_key
        if base_url:
            llm_config["base_url"] = base_url

        self.llm_config = llm_config
        self.llm_generate_cfg = self.llm_config.get("generate_cfg", {})
        self.model = model or self.llm_config.get("model", LLM_MODEL_NAME)
        self.api_key = self.llm_config.get("api_key", LLM_API_KEY)
        self.base_url = self.llm_config.get("base_url", LLM_BASE_URL)
        self.max_input_tokens = self.llm_config.get("max_input_tokens", 32000)
        self.llm_timeout = self.llm_config.get("llm_timeout", 300.0)
        self.agent_timeout = self.llm_config.get("agent_timeout", 1800.0)
        self.function_list = function_list or list(TOOL_MAP.keys())
        self.instruction = instruction
        self.use_xml_protocol = use_xml_protocol

    def _get_tool_definitions(self) -> List[Dict]:
        """Get tool definitions in OpenAI function calling format."""
        tools = []
        for tool_name in self.function_list:
            if tool_name in TOOL_DESCRIPTIONS:
                tools.append(TOOL_DESCRIPTIONS[tool_name])
            else:
                # Fallback for custom tools
                tools.append({
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "description": f"Custom tool '{tool_name}'",
                        "parameters": {"type": "object", "properties": {}, "required": []}
                    }
                })
        return tools

    def parse_output(self, text: str) -> Dict[str, str]:
        """
        解析 LLM 的单次输出，严格提取 <plan>, <report>, 和 (<tool_call> 或 <answer> 或 <terminate>)。
        这是 IterResearch 范式的核心：LLM 在单次调用中生成三段式输出。
        """
        output = {
            "plan": "",
            "report": "",
            "tool_call": "",
            "answer": "",
            "terminate": False,
            "terminate_reason": "",
        }

        def _extract_last_block(pattern: str) -> str:
            matches = re.findall(pattern, text, flags=re.DOTALL | re.MULTILINE)
            for m in reversed(matches):
                if m and m.strip():
                    return m.strip()
            return ""

        # 1. 提取 <plan>
        plan = _extract_last_block(r"^\s*<plan>(.*?)</plan>")
        if plan:
            output["plan"] = plan

        # 2. 提取 <report>
        output["report"] = _extract_last_block(r"^\s*<report>(.*?)</report>")

        # 3. 提取 <tool_call>、<answer>、<terminate>
        output["tool_call"] = _extract_last_block(r"^\s*<tool_call>(.*?)</tool_call>")
        output["answer"] = _extract_last_block(r"^\s*<answer>(.*?)</answer>")
        term_body = _extract_last_block(r"^\s*<terminate>(.*?)</terminate>")
        if term_body != "":
            output["terminate"] = True
            output["terminate_reason"] = term_body

        if not output["tool_call"] and not output["answer"] and not output["terminate"]:
            logger.warning("LLM output did not contain <tool_call>, <answer>, or <terminate> tag.")

        return output

    async def call_server(
            self, 
            msgs: List[Dict], 
            stop_sequences: Optional[List[str]] = None,
            max_tries: int = 3,
            tools: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Call LLM server with support for both XML-based and native function calling.
        Uses AsyncOpenAI for true async concurrency.
        
        Returns:
            Dict with keys:
            - content: str, the text content
            - reasoning_content: Optional[str], thinking process (for models like DeepSeek R1)
            - tool_calls: Optional[List], native tool calls (when use_xml_protocol=False)
            - raw_message: the original message object
        """
        client = AsyncOpenAI(
            api_key=self.api_key or "EMPTY",
            base_url=self.base_url,
            timeout=self.llm_timeout,
        )

        base_sleep_time = 1
        stop_sequences = stop_sequences or (["<tool_response>"] if self.use_xml_protocol else None)

        for attempt in range(max_tries):
            try:
                request_params = {
                    "model": self.model,
                    "messages": msgs,
                    "temperature": self.llm_generate_cfg.get('temperature', 0.6),
                    "top_p": self.llm_generate_cfg.get('top_p', 0.95),
                    "presence_penalty": self.llm_generate_cfg.get('presence_penalty', 1.1)
                }
                
                # Add stop sequences only for XML mode
                if stop_sequences and self.use_xml_protocol:
                    request_params["stop"] = stop_sequences
                
                # Add tools for function calling mode (non-XML)
                if tools and not self.use_xml_protocol:
                    request_params["tools"] = tools
                
                # Add extra_body for thinking mode (DeepSeek R1 etc.)
                model_thinking_type = self.llm_generate_cfg.get("model_thinking_type", "")
                if model_thinking_type:
                    request_params["extra_body"] = {
                        "thinking": {"type": model_thinking_type}
                    }
                
                # Use native async call
                chat_response = await client.chat.completions.create(**request_params)

                message = chat_response.choices[0].message
                content = message.content or ""
                
                # Extract reasoning_content if available (DeepSeek R1, etc.)
                reasoning_content = getattr(message, 'reasoning_content', None)
                
                # Extract native tool_calls if available
                tool_calls = getattr(message, 'tool_calls', None)
                
                logger.debug(
                    f"Input messages: {msgs}, \n"
                    f"Reasoning_content: {reasoning_content}, \n"
                    f"Tool_calls: {tool_calls}, \n"
                    f"LLM Response: {content}"
                )
                
                return {
                    "content": content.strip() if content else "",
                    "reasoning_content": reasoning_content,
                    "tool_calls": tool_calls,
                    "raw_message": message,
                }

            except RateLimitError as e:
                logger.warning(f"Attempt {attempt + 1} rate limit error: {e}")
            except AuthenticationError as e:
                logger.error(f"Authentication error: {e}")
                break  # Don't retry auth errors
            except (APIError, APIConnectionError, APITimeoutError) as e:
                logger.warning(
                    f"Attempt {attempt + 1} API error: {e}, base_url: {self.base_url}, model: {self.model}")
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} unexpected error: {e}")

            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                sleep_time = min(sleep_time, 30)
                logger.warning(f"Retrying in {sleep_time:.2f}s...")
                await asyncio.sleep(sleep_time)
            else:
                logger.error("All retry attempts exhausted. The LLM call failed.")
        
        return {
            "content": "LLM server error.",
            "reasoning_content": None,
            "tool_calls": None,
            "raw_message": None,
        }

    def count_tokens(self, messages, model=None):
        """Count tokens in messages"""
        if model is None:
            model = self.model or "gpt-4o"  # 使用实例的 model 或默认值
        try:
            # Convert dict messages to Message objects if needed
            full_message = []
            for x in messages:
                if isinstance(x, dict):
                    full_message.append(Message(**x))
                elif isinstance(x, Message):
                    full_message.append(x)
                else:
                    full_message.append(x)

            full_prompt = build_text_completion_prompt(full_message, allow_special=True)
            return count_tokens_base(full_prompt, model)
        except Exception as e:
            logger.warning(f"Failed to count tokens: {e}. Using simple split.")
            return sum(len(str(x).split()) for x in messages)

    async def _execute_function_call(self, tool_call) -> str:
        """Execute an OpenAI-style function call."""
        func_name = tool_call.function.name
        args_str = tool_call.function.arguments
        
        logger.debug(f"Function call: {func_name}({args_str})")
        
        if func_name not in TOOL_MAP:
            return f"Error: Tool {func_name} not found"
        
        try:
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            return f"Error: Failed to decode arguments: {args_str}"
        
        tool = TOOL_MAP[func_name]
        loop = asyncio.get_event_loop()
        
        try:
            # Special handling for python tool: extract code from arguments
            if func_name == "python":
                code = args.get("code", "")
                if not code:
                    return "[Python Interpreter Error]: Empty code. Please provide code in arguments.code"
                result = await loop.run_in_executor(None, tool.call, code)
                return result if isinstance(result, str) else str(result)
            
            if asyncio.iscoroutinefunction(tool.call):
                if func_name == "parse_file":
                    params = {"files": args.get("files")}
                    result = await tool.call(params, file_root_path=FILE_DIR)
                else:
                    result = await tool.call(args)
            else:
                result = await loop.run_in_executor(None, tool.call, args)
            return result if isinstance(result, str) else str(result)
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return f"Error: Tool execution failed. {e}"

    async def _execute_xml_tool(self, tool_call_str: str) -> str:
        """Execute a tool call from XML <tool_call> block."""
        loop = asyncio.get_event_loop()

        try:
            # Check for <code> tag (Python code in XML mode)
            if "<code>" in tool_call_str and "</code>" in tool_call_str:
                code_raw = tool_call_str.split("<code>", 1)[1].rsplit("</code>", 1)[0].strip()
                result = await loop.run_in_executor(None, TOOL_MAP['python'].call, code_raw)
                return result

            # Parse JSON tool call
            tool_call = json5.loads(tool_call_str)
            tool_name = tool_call.get('name', '')
            tool_args = tool_call.get('arguments', {})

            if tool_name not in TOOL_MAP:
                return f"Error: Tool {tool_name} not found"

            tool = TOOL_MAP[tool_name]

            # Special handling for python tool: extract code from arguments
            if tool_name == "python":
                code = tool_args.get("code", "")
                if not code:
                    return "[Python Interpreter Error]: Empty code. Please provide code in arguments.code"
                result = await loop.run_in_executor(None, tool.call, code)
                return str(result) if not isinstance(result, str) else result

            # Handle async/sync tools
            if asyncio.iscoroutinefunction(tool.call):
                if tool_name == "parse_file":
                    params = {"files": tool_args.get("files")}
                    result = await tool.call(params, file_root_path=FILE_DIR)
                else:
                    result = await tool.call(tool_args)
            else:
                result = await loop.run_in_executor(None, tool.call, tool_args)

            return str(result) if not isinstance(result, str) else result

        except Exception as e:
            logger.error(f"Tool call parsing or execution failed: {e}")
            return f"Error: Tool call failed. Input: {tool_call_str}. Error: {e}"

    async def run(self, question, progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None):
        """
        Execute research following the IterResearch paradigm.
        
        Supports two modes:
        - Function Calling (default): Uses OpenAI-style tools parameter
        - XML Protocol: Uses <tool_call> tags in prompts
        
        Loop:
        s_t = (Q, R_{i-1}, O_{i-1})
        LLM(s_t) -> (Plan_i, Report_i, Action_i)
        
        if Action_i == <answer>:
            STOP
        else:
            O_i = Tool(Action_i)
            s_{t+1} = (Q, R_i, O_i)
            LOOP
        """

        async def emit(event: Dict[str, Any]):
            if not callable(progress_callback):
                return
            event.setdefault("timestamp", datetime.datetime.utcnow().isoformat())
            try:
                maybe = progress_callback(event)
                if inspect.isawaitable(maybe):
                    await maybe
            except Exception as callback_err:
                logger.warning(f"progress_callback raised error: {callback_err}")

        start_time = time.time()

        # Initialize research round
        research_round = ResearchRound(question=question)
        
        # Build system prompt based on mode (XML mode uses detailed prompt, function calling uses simpler one)
        if self.use_xml_protocol:
            system_prompt = get_iterresearch_system_prompt(
                today_date(), self.function_list, self.instruction, question=question
            )
        else:
            system_prompt = get_iterresearch_system_prompt_fc(
                today_date(), self.instruction, question=question
            )
        
        # Get tool definitions for function calling mode
        tool_definitions = self._get_tool_definitions() if not self.use_xml_protocol else None

        # Trajectory log for debugging
        full_trajectory_log = []
        prediction = ''
        termination = ''

        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        round_num = 0

        while num_llm_calls_available > 0:
            if time.time() - start_time > self.agent_timeout:
                logger.warning("Agent timeout reached.")
                termination = "timeout"
                prediction = "No answer found (timeout)."
                break

            round_num += 1
            num_llm_calls_available -= 1

            # Build prompt (s_t = Q, R_{i-1}, O_{i-1})
            current_context = research_round.get_context(system_prompt)

            if round_num == 1:
                full_trajectory_log.extend(current_context)

            # Single LLM call (generate P_i, R_i, A_i)
            request_msgs = current_context
            is_last_call = (num_llm_calls_available == 0)
            
            try:
                logger.debug(f"Round {round_num}: Calling LLM. Remaining calls: {num_llm_calls_available}")

                # Force final answer on last call
                if is_last_call:
                    finalize_instruction = (
                        "You have reached the maximum allowed LLM calls for this run. "
                        "Do not call tools anymore. Based on your current report and the information gathered so far, "
                        "provide the final answer now in the three-part format: "
                        "<plan>...</plan> <report>...</report> <answer>...</answer>"
                    )
                    request_msgs = current_context + [{"role": "user", "content": finalize_instruction}]

                response = await self.call_server(request_msgs, tools=tool_definitions)
                content = response["content"]
                reasoning_content = response["reasoning_content"]
                tool_calls = response["tool_calls"]
                raw_message = response["raw_message"]

                # Log reasoning content if available
                if reasoning_content:
                    logger.info(f"[Round {round_num}] Thinking: {reasoning_content[:500]}...")
                    await emit({"type": "thinking", "round": round_num, "content": reasoning_content})

                # Build assistant message for trajectory (preserving reasoning_content)
                if raw_message:
                    msg_dict = raw_message.model_dump(exclude_none=True)
                    if reasoning_content:
                        msg_dict['reasoning_content'] = reasoning_content
                    full_trajectory_log.append(msg_dict)
                else:
                    full_trajectory_log.append({"role": "assistant", "content": content})
                
                logger.debug(f'Round {round_num} LLM response received.')
            except Exception as e:
                logger.error(f"Unknown Error: {e}")
                prediction = f"Error: Unknown {e}"
                termination = 'unknown error'
                break

            # === Function Calling Mode: Handle native tool calls ===
            if not self.use_xml_protocol and tool_calls:
                # Execute each tool call
                tool_results = []
                for tool_call in tool_calls:
                    tool_result = await self._execute_function_call(tool_call)
                    logger.debug(f"Tool {tool_call.function.name} result: {tool_result[:200]}...")
                    
                    await emit({
                        "type": "tool",
                        "round": round_num,
                        "tool_name": tool_call.function.name,
                        "tool_args": tool_call.function.arguments,
                        "observation": tool_result[:1000],
                    })
                    tool_results.append(tool_result)
                
                # Update observation with combined tool results
                research_round.last_observation = "\n\n".join(tool_results)
                
                # Parse any report/answer from content (LLM may still output structured response)
                if content:
                    parsed = self.parse_output(content)
                    if parsed["report"]:
                        research_round.current_report = parsed["report"]
                    if parsed["answer"]:
                        prediction = parsed["answer"]
                        termination = 'answer found'
                        await emit({
                            "type": "final",
                            "round": round_num,
                            "answer": prediction,
                            "report": research_round.current_report,
                            "termination": termination,
                        })
                        break
                continue

            # === XML Protocol Mode: Parse structured output ===
            parsed = self.parse_output(content)
            plan_content = parsed["plan"]
            report_content = parsed["report"]
            action_content = parsed["tool_call"]
            answer_content = parsed["answer"]
            terminate_flag = parsed.get("terminate", False)
            terminate_reason = parsed.get("terminate_reason", "").strip()

            logger.debug(f"Round {round_num} - Plan: {plan_content}")
            logger.debug(f"Round {round_num} - Report: {report_content}")
            logger.debug(f"Round {round_num} - Action: {action_content}")
            logger.debug(f"Round {round_num} - Answer: {answer_content}")
            if terminate_flag:
                logger.debug(f"Round {round_num} - Terminate signaled. Reason: {terminate_reason}")

            await emit({
                "type": "round",
                "round": round_num,
                "plan": plan_content,
                "report": report_content,
                "action": action_content,
                "answer": answer_content,
                "terminate": terminate_flag,
                "reasoning_content": reasoning_content,
            })

            # Update report (R_i)
            if report_content:
                research_round.current_report = report_content
            else:
                logger.warning("No <report> found. Report will not be updated for the next round.")

            # Check for termination
            if answer_content:
                prediction = answer_content
                termination = 'answer found'
                if terminate_flag:
                    termination = 'terminate with answer'
                await emit({
                    "type": "final",
                    "round": round_num,
                    "answer": prediction,
                    "report": research_round.current_report,
                    "termination": termination,
                })
                logger.debug(f"Round {round_num}: LLM provided <answer>. Terminating loop.")
                break
            
            if terminate_flag:
                prediction = terminate_reason if terminate_reason else research_round.current_report.strip()
                termination = 'terminated by llm'
                await emit({
                    "type": "final",
                    "round": round_num,
                    "answer": prediction,
                    "report": research_round.current_report,
                    "termination": termination,
                })
                logger.debug(f"Round {round_num}: LLM signaled <terminate>. Final response prepared.")
                break
            
            if is_last_call:
                fallback_report = research_round.current_report.strip()
                fallback_source = fallback_report or terminate_reason
                prediction = fallback_source if fallback_source else research_round.last_observation
                termination = 'finalized without answer tag'
                await emit({
                    "type": "final",
                    "round": round_num,
                    "answer": prediction,
                    "report": research_round.current_report,
                    "termination": termination,
                })
                logger.warning("Last LLM call did not return <answer> or <terminate>. Promoting accumulated content as final answer.")
                break

            # Execute Action (A_i)
            if action_content:
                try:
                    logger.debug(f"Round {round_num}: Executing tool...")
                    tool_response_str = await self._execute_xml_tool(action_content)

                    # Store tool response O_i for next round s_{t+1}
                    research_round.last_observation = tool_response_str
                    await emit({
                        "type": "tool",
                        "round": round_num,
                        "tool_call": action_content,
                        "observation": tool_response_str,
                    })

                    # Log tool response
                    tool_obs_msg = f"{OBS_START}\n{tool_response_str}\n{OBS_END}"
                    full_trajectory_log.append({"role": "user", "content": tool_obs_msg})

                    logger.debug(f"Round {round_num}: Tool execution completed.")

                except Exception as e:
                    logger.error(f"Error calling tool: {e}")
                    error_str = f"Error executing tool: {e}"
                    research_round.last_observation = error_str
                    await emit({
                        "type": "tool_error",
                        "round": round_num,
                        "tool_call": action_content,
                        "observation": error_str,
                    })
                    full_trajectory_log.append({"role": "user", "content": f"{OBS_START}\n{error_str}\n{OBS_END}"})
            else:
                # LLM produced neither answer nor tool call
                logger.warning("LLM did not produce <answer> or <tool_call>. Forcing answer generation...")

                force_answer_msgs = current_context + [
                    {"role": "user", "content": (
                        "You did not provide a valid response format. "
                        "Based on your current report and the information gathered so far, "
                        "please provide the final answer to the original question. "
                        "Use the three-part format: <plan>...</plan> <report>...</report> <answer>...</answer>"
                    )}
                ]

                try:
                    forced_response = await self.call_server(force_answer_msgs, tools=None)
                    forced_content = forced_response["content"]
                    forced_parsed = self.parse_output(forced_content)

                    if forced_parsed["answer"]:
                        prediction = forced_parsed["answer"]
                        termination = "answer (forced)"
                        await emit({
                            "type": "final",
                            "round": round_num,
                            "answer": prediction,
                            "report": research_round.current_report,
                            "termination": termination,
                        })
                        full_trajectory_log.append({"role": "assistant", "content": forced_content})
                        break
                    else:
                        logger.error("Failed to force answer generation")
                        prediction = "No answer found (format error after retry)."
                        termination = "format error"
                        break
                except Exception as e:
                    logger.error(f"Error during forced answer generation: {e}")
                    prediction = "No answer found (format error)."
                    termination = "format error"
                    break

            # Token limit check
            token_count = self.count_tokens(request_msgs)
            logger.debug(f"Round {round_num} context token count: {token_count}")
            if token_count > self.max_input_tokens:
                logger.warning(f"Token quantity exceeds the limit: {token_count}")
                force_answer_msgs = current_context + [
                    {"role": "user", "content": "You have now reached the maximum context length. "
                                                "Stop making tool calls. Based on your research report, "
                                                "provide the final answer in the three-part format: "
                                                "<plan>...</plan> <report>...</report> <answer>...</answer>"}
                ]
                forced_response = await self.call_server(force_answer_msgs, tools=None)
                content = forced_response["content"]
                parsed = self.parse_output(content)
                prediction = parsed["answer"] if parsed["answer"] else "No answer found (token limit)."
                termination = 'token limit reached'
                await emit({
                    "type": "final",
                    "round": round_num,
                    "answer": prediction,
                    "report": research_round.current_report,
                    "termination": termination,
                })
                full_trajectory_log.append({"role": "assistant", "content": content})
                break

        # Finalize
        if not prediction:
            fallback_report = research_round.current_report.strip()
            if fallback_report:
                prediction = fallback_report
                if not termination:
                    termination = 'report fallback'
            elif num_llm_calls_available == 0:
                prediction = 'No answer found (exceeded available LLM calls).'
                termination = 'exceed available llm calls'
            else:
                prediction = 'No answer found.'
                termination = 'answer not found'

        # Return final result
        result = {
            "question": question,
            "prediction": prediction,
            "report": research_round.current_report,
            "termination": termination,
            "trajectory": full_trajectory_log,
        }

        await emit({
            "type": "status",
            "status": termination or 'completed',
            "answer": prediction,
            "report": research_round.current_report,
        })

        return result


async def main():
    # 1. 定义你的 LLM 配置
    llm_config = {
        "model": LLM_MODEL_NAME,
        "generate_cfg": {
            'max_input_tokens': 32000,
            "temperature": 0.6,
            "top_p": 0.95,
            "presence_penalty": 1.1
        }
    }

    # 2. 实例化 agent
    agent = WebResearcherAgent(
        llm_config=llm_config,
        function_list=['search', 'python']
    )

    question = '刘翔破纪录时候是多少岁?'
    gt = "23岁差2天"

    # 4. 运行完整的并行研究与总结
    final_result = await agent.run(question)

    # 5. 打印结果
    print(f"final_result: {final_result}")


if __name__ == "__main__":
    asyncio.run(main())
