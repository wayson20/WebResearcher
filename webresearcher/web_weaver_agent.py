# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: WebWeaver Agent - Dual-Agent Research Framework with Dynamic Outline

Supports two tool calling modes:
1. OpenAI Function Calling (default): Uses OpenAI-style tools parameter, works with OpenAI/DeepSeek/etc.
2. XML Protocol: Uses <tool_call> tags, compatible with all LLMs including local models
"""
import json5
import re
import datetime
import asyncio
import random
import time
import json
import inspect

from typing import Any, Callable, Dict, List, Optional, Set
from openai import (
    AsyncOpenAI,
    APIError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    AuthenticationError,
)

from webresearcher.base import BaseTool, today_date
from webresearcher.log import logger
from webresearcher.prompt import get_webweaver_planner_prompt, get_webweaver_writer_prompt, TOOL_DESCRIPTIONS
from webresearcher.tool_memory import MemoryBank, RetrieveTool
from webresearcher.tool_planner_search import PlannerSearchTool
from webresearcher.tool_planner_scholar import PlannerScholarTool
from webresearcher.tool_planner_visit import PlannerVisitTool
from webresearcher.tool_planner_python import PlannerPythonTool
from webresearcher.tool_planner_file import PlannerFileTool
from webresearcher.config import (
    LLM_API_KEY, 
    LLM_BASE_URL, 
    OBS_START, 
    MAX_LLM_CALL_PER_RUN, 
    AGENT_TIMEOUT,
    LLM_MODEL_NAME
)


class BaseWebWeaverAgent:
    """
    Base class for WebWeaver agents (Planner and Writer).
    Handles common LLM calling and tool execution logic.
    
    Supports two tool calling modes:
    - use_xml_protocol=True (default): XML-based <tool_call> tags, works better for structured output
    - use_xml_protocol=False: OpenAI-style function calling
    """

    def __init__(self, llm_config: Dict, tool_map: Dict[str, BaseTool], use_xml_protocol: bool = True):
        """
        Initialize base agent with LLM config and tools.
        
        Args:
            llm_config: LLM configuration dict
            tool_map: Dictionary mapping tool names to BaseTool instances
            use_xml_protocol: If True (default), use XML tags; if False, use OpenAI function calling
        """
        self.llm_config = llm_config
        self.llm_generate_cfg = self.llm_config.get("generate_cfg", {})
        self.model = self.llm_config.get("model", LLM_MODEL_NAME)
        self.llm_timeout = self.llm_config.get("llm_timeout", 300.0)
        self.tool_map = tool_map
        self.function_list = list(tool_map.keys())
        self.api_key = self.llm_config.get("api_key", LLM_API_KEY)
        self.base_url = self.llm_config.get("base_url", LLM_BASE_URL)
        self.use_xml_protocol = use_xml_protocol
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.llm_timeout,
        )
        # Cache for idempotent tool calls to avoid redundant executions
        self.cacheable_tools = set(llm_config.get("cacheable_tools", ["retrieve"]))
        self.tool_call_cache: Dict[str, str] = {}

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

    async def call_server(
            self, 
            msgs: List[Dict], 
            stop_sequences: Optional[List[str]] = None,
            max_tries: int = 3,
            tools: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Async LLM API call with retry logic using AsyncOpenAI.
        Unified return structure matching ReactAgent and WebResearcherAgent.
        
        Args:
            msgs: List of message dicts
            stop_sequences: Optional stop sequences
            max_tries: Maximum retry attempts
            tools: Optional tool definitions for function calling mode
            
        Returns:
            Dict with keys:
            - content: str, the text content
            - reasoning_content: Optional[str], thinking process (for models like DeepSeek R1)
            - tool_calls: Optional[List], native tool calls (when use_xml_protocol=False)
            - raw_message: the original message object
        """
        base_sleep_time = 1
        stop_sequences = stop_sequences or ([OBS_START] if self.use_xml_protocol else None)

        for attempt in range(max_tries):
            try:
                request_params = {
                    "model": self.model,
                    "messages": msgs,
                    "temperature": self.llm_generate_cfg.get('temperature', 0.1),
                    "top_p": self.llm_generate_cfg.get('top_p', 0.95),
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
                chat_response = await self.client.chat.completions.create(**request_params)
                
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
                logger.error("All retry attempts exhausted.")

        return {
            "content": "Error: LLM server failed after all retries.",
            "reasoning_content": None,
            "tool_calls": None,
            "raw_message": None,
        }

    async def _execute_function_call(self, tool_call) -> str:
        """Execute an OpenAI-style function call."""
        func_name = tool_call.function.name
        args_str = tool_call.function.arguments
        
        logger.debug(f"Function call: {func_name}({args_str})")
        
        if func_name not in self.tool_map:
            return f"Error: Tool {func_name} not found"
        
        try:
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            return f"Error: Failed to decode arguments: {args_str}"
        
        # Auto-fix common LLM mistakes
        args = self._fix_tool_args(func_name, args)
        
        # Check cache
        cache_key = None
        if func_name in self.cacheable_tools:
            try:
                cache_key = f"{func_name}::" + json.dumps(args, sort_keys=True, ensure_ascii=False)
                if cache_key in self.tool_call_cache:
                    logger.debug(f"Cache hit for tool '{func_name}'")
                    return self.tool_call_cache[cache_key]
            except Exception:
                cache_key = None
        
        tool = self.tool_map[func_name]
        loop = asyncio.get_event_loop()
        
        try:
            if asyncio.iscoroutinefunction(tool.call):
                result = await tool.call(args)
            else:
                result = await loop.run_in_executor(None, tool.call, args)
            
            result_str = str(result) if not isinstance(result, str) else result
            
            # Store in cache
            if cache_key is not None:
                self.tool_call_cache[cache_key] = result_str
            
            return result_str
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return f"Error: Tool execution failed. {e}"

    async def _execute_xml_tool(self, tool_call_str: str) -> str:
        """Execute a tool call from XML <tool_call> block."""
        loop = asyncio.get_event_loop()
        
        try:
            tool_call = json5.loads(tool_call_str)
            tool_name = tool_call.get('name')
            tool_args = tool_call.get('arguments', {})

            if tool_name not in self.tool_map:
                return f"Error: Tool '{tool_name}' not found in agent's tool map."

            # Auto-fix common LLM mistakes
            tool_args = self._fix_tool_args(tool_name, tool_args)

            # Cache check for idempotent tools
            cache_key = None
            if tool_name in self.cacheable_tools:
                try:
                    cache_key = f"{tool_name}::" + json.dumps(tool_args, sort_keys=True, ensure_ascii=False)
                    if cache_key in self.tool_call_cache:
                        logger.debug(f"Cache hit for tool '{tool_name}' with identical arguments.")
                        return self.tool_call_cache[cache_key]
                except Exception:
                    cache_key = None

            tool = self.tool_map[tool_name]

            # Handle async vs sync tools
            if asyncio.iscoroutinefunction(tool.call):
                result = await tool.call(tool_args)
            else:
                result = await loop.run_in_executor(None, tool.call, tool_args)

            result_str = str(result) if not isinstance(result, str) else result

            # Store in cache if applicable
            if cache_key is not None:
                self.tool_call_cache[cache_key] = result_str

            return result_str

        except Exception as e:
            logger.error(f"Tool call failed: {e}")
            return f"Error: Tool call failed. Input: {tool_call_str}. Error: {e}"

    def _fix_tool_args(self, tool_name: str, tool_args: Dict) -> Dict:
        """Auto-fix common LLM mistakes: convert string to array for certain parameters."""
        if tool_name in ['search', 'google_scholar'] and 'query' in tool_args:
            if isinstance(tool_args['query'], str):
                tool_args['query'] = [tool_args['query']]
        elif tool_name == 'visit' and 'url' in tool_args:
            if isinstance(tool_args['url'], str):
                tool_args['url'] = [tool_args['url']]
        elif tool_name == 'parse_file' and 'files' in tool_args:
            if isinstance(tool_args['files'], str):
                tool_args['files'] = [tool_args['files']]
        return tool_args

    # Legacy method for backward compatibility
    async def call_tool(self, tool_call_str: str) -> str:
        """Legacy method - delegates to _execute_xml_tool."""
        return await self._execute_xml_tool(tool_call_str)


class WebWeaverPlanner(BaseWebWeaverAgent):
    """
    Planner Agent for WebWeaver.
    
    Goal: Iteratively search and optimize a research outline.
    Actions: search, write_outline, terminate.
    
    Supports two tool calling modes:
    - use_xml_protocol=True (default): XML-based <tool_call> tags, works better for structured output
    - use_xml_protocol=False: OpenAI-style function calling
    
    Based on WebWeaver paper Section 3.2 and Appendix B.2.
    """

    def __init__(
            self, 
            llm_config: Dict, 
            memory_bank: MemoryBank, 
            function_list: Optional[List[str]] = None, 
            instruction: str = "",
            use_xml_protocol: bool = True,  # XML protocol works better for Planner's structured output
    ):
        """
        Initialize Planner agent.
        
        Args:
            llm_config: LLM configuration
            memory_bank: Shared MemoryBank instance
            function_list: Optional list of tool names to use
            instruction: Optional custom instruction
            use_xml_protocol: If True, use XML tags; if False (default), use OpenAI function calling
        """
        # Planner's tool set - all 5 common tools like WebResearcher
        full_tool_map = {
            "search": PlannerSearchTool(memory_bank),
            "google_scholar": PlannerScholarTool(memory_bank),
            "visit": PlannerVisitTool(memory_bank),
            "python": PlannerPythonTool(memory_bank),
            "parse_file": PlannerFileTool(memory_bank),
        }
        tool_map = {k: v for k, v in full_tool_map.items() if not function_list or k in function_list} or full_tool_map

        super().__init__(llm_config, tool_map, use_xml_protocol=use_xml_protocol)
        self.memory_bank = memory_bank
        self.instruction = instruction
        # system_prompt will be set dynamically in run() based on question language
        self.system_prompt = get_webweaver_planner_prompt(today_date(), self.function_list, instruction)

    def parse_output(self, text: str) -> Dict[str, str]:
        """
        Parse Planner's output: <plan> and (<tool_call> | <write_outline> | <terminate>).
        
        Args:
            text: Raw LLM output
            
        Returns:
            Dict with 'plan', 'action_type', and 'action_content'
        """
        plan_match = re.search(r'<plan>(.*?)</plan>', text, re.DOTALL)
        plan = plan_match.group(1).strip() if plan_match else ""

        tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
        write_outline_match = re.search(r'<write_outline>(.*?)</write_outline>', text, re.DOTALL)
        terminate_match = re.search(r'<terminate>', text, re.DOTALL)

        action_type = None
        action_content = ""

        if terminate_match:
            action_type = "terminate"
        elif write_outline_match:
            action_type = "write_outline"
            action_content = write_outline_match.group(1).strip()
        elif tool_call_match:
            action_type = "tool_call"
            action_content = tool_call_match.group(1).strip()
        else:
            action_type = "error"
            action_content = "No valid action tag found. Must use <tool_call>, <write_outline>, or <terminate>."
            logger.warning(f"Planner output parsing error: {action_content}")

        return {
            "plan": plan,
            "action_type": action_type,
            "action_content": action_content
        }

    async def run(
            self, 
            question: str,
            progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> str:
        """
        Execute Planner's research loop.
        
        Args:
            question: Research question
            progress_callback: Optional callback for progress updates
            
        Returns:
            Final outline string
        """
        async def emit(event: Dict[str, Any]):
            if not callable(progress_callback):
                return
            event.setdefault("timestamp", datetime.datetime.utcnow().isoformat())
            event.setdefault("agent", "planner")
            try:
                maybe = progress_callback(event)
                if inspect.isawaitable(maybe):
                    await maybe
            except Exception as callback_err:
                logger.warning(f"progress_callback raised error: {callback_err}")

        logger.debug("--- [WebWeaver] Planner Agent activated ---")
        
        # Update system prompt based on question language
        self.system_prompt = get_webweaver_planner_prompt(
            today_date(), self.function_list, self.instruction, question=question
        )

        current_outline = "Outline is empty. Start by searching for information."
        last_observation = "No observation yet."
        
        # Get tool definitions for function calling mode
        tool_definitions = self._get_tool_definitions() if not self.use_xml_protocol else None

        for i in range(MAX_LLM_CALL_PER_RUN):
            # Build Planner context
            context_str = (
                f"[Question]\n{question}\n\n"
                f"[Current Outline]\n{current_outline}\n\n"
                f"[Last Observation]\n{last_observation}\n\n"
                f"**IMPORTANT: When you write the outline using <write_outline>, "
                f"you MUST use the SAME LANGUAGE as the [Question] above. Do NOT translate.**"
            )
            # Final iteration: force LLM to output <write_outline>
            is_last_iteration = (i == MAX_LLM_CALL_PER_RUN - 1)
            if is_last_iteration:
                context_str += (
                    "\n[Final Instruction]\n"
                    "This is your last allowed step. You MUST output <write_outline> with the complete final outline. "
                    "Do NOT output <tool_call> or <terminate>."
                )
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": context_str}
            ]

            # Call LLM
            response = await self.call_server(messages, tools=tool_definitions)
            content = response["content"]
            reasoning_content = response["reasoning_content"]
            tool_calls = response["tool_calls"]
            
            # Log and emit reasoning content if available
            if reasoning_content:
                logger.info(f"[Planner Step {i + 1}] Thinking: {reasoning_content[:500]}...")
                await emit({"type": "thinking", "step": i + 1, "content": reasoning_content})

            # === Function Calling Mode: Handle native tool calls ===
            if not self.use_xml_protocol and tool_calls:
                for tool_call in tool_calls:
                    tool_result = await self._execute_function_call(tool_call)
                    logger.debug(f"Planner Step {i + 1}: Tool {tool_call.function.name} executed.")
                    
                    await emit({
                        "type": "tool",
                        "step": i + 1,
                        "tool_name": tool_call.function.name,
                        "tool_args": tool_call.function.arguments,
                        "observation": tool_result[:1000],
                    })
                    last_observation = tool_result
                
                # Parse any outline from content
                if content:
                    parsed = self.parse_output(content)
                    if parsed['action_type'] == "write_outline":
                        current_outline = parsed['action_content']
                        await emit({"type": "outline_updated", "step": i + 1, "outline": current_outline})
                    elif parsed['action_type'] == "terminate":
                        logger.debug("Planner finished. Terminating.")
                        return current_outline
                continue

            # === XML Protocol Mode: Parse structured output ===
            parsed = self.parse_output(content)
            logger.debug(f"Planner Step {i + 1} | Action: {parsed}")
            
            await emit({
                "type": "step",
                "step": i + 1,
                "plan": parsed["plan"],
                "action_type": parsed["action_type"],
                "reasoning_content": reasoning_content,
            })

            # Execute action
            if parsed['action_type'] == "terminate":
                logger.debug("Planner finished. Terminating.")
                return current_outline

            elif parsed['action_type'] == "write_outline":
                current_outline = parsed['action_content']
                last_observation = "Outline successfully updated."
                logger.debug(f"Planner Step {i + 1}: Outline updated.")
                await emit({"type": "outline_updated", "step": i + 1, "outline": current_outline})

            elif parsed['action_type'] == "tool_call":
                tool_call_str = parsed['action_content']
                last_observation = await self._execute_xml_tool(tool_call_str)
                logger.debug(f"Planner Step {i + 1}: Tool executed.")
                await emit({
                    "type": "tool",
                    "step": i + 1,
                    "tool_call": tool_call_str,
                    "observation": last_observation[:1000],
                })

            elif parsed['action_type'] == "error":
                last_observation = parsed['action_content']
                logger.warning(f"Planner Step {i + 1}: Action parse error.")

        logger.warning("Planner reached max iterations.")
        return current_outline


class WebWeaverWriter(BaseWebWeaverAgent):
    """
    Writer Agent for WebWeaver.
    
    Goal: Write report section-by-section based on Planner's outline and memory bank.
    Actions: retrieve, write, terminate.
    
    Supports two tool calling modes:
    - use_xml_protocol=True (default): XML-based <tool_call> tags, works better for structured output
    - use_xml_protocol=False: OpenAI-style function calling
    
    Based on WebWeaver paper Section 3.3 and Appendix B.3.
    """

    def __init__(
            self, 
            llm_config: Dict, 
            memory_bank: MemoryBank, 
            instruction: str = "",
            use_xml_protocol: bool = True,  # XML protocol works better for Writer's structured output
    ):
        """
        Initialize Writer agent.
        
        Args:
            llm_config: LLM configuration
            memory_bank: Shared MemoryBank instance
            instruction: Optional custom instruction
            use_xml_protocol: If True, use XML tags; if False (default), use OpenAI function calling
        """
        # Writer only needs retrieve tool
        tool_map = {
            "retrieve": RetrieveTool(memory_bank),
        }

        super().__init__(llm_config, tool_map, use_xml_protocol=use_xml_protocol)
        self.memory_bank = memory_bank
        self.instruction = instruction
        # system_prompt will be set dynamically in run() based on question language
        self.system_prompt = get_webweaver_writer_prompt(today_date(), instruction)

    def parse_output(self, text: str) -> Dict[str, str]:
        """
        Parse Writer's output: <plan> and (<tool_call> | <write> | <terminate>).
        
        Args:
            text: Raw LLM output
            
        Returns:
            Dict with 'plan', 'action_type', and 'action_content'
        """
        plan_match = re.search(r'<plan>(.*?)</plan>', text, re.DOTALL)
        plan = plan_match.group(1).strip() if plan_match else ""

        tool_call_match = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
        write_match = re.search(r'<write>(.*?)</write>', text, re.DOTALL)
        terminate_match = re.search(r'<terminate>', text, re.DOTALL)

        action_type = None
        action_content = ""

        if terminate_match:
            action_type = "terminate"
        elif write_match:
            action_type = "write"
            action_content = write_match.group(1).strip()
        elif tool_call_match:
            action_type = "tool_call"
            action_content = tool_call_match.group(1).strip()
        else:
            action_type = "error"
            action_content = "No valid action tag found. Must use <tool_call> (retrieve), <write>, or <terminate>."
            logger.warning(f"Writer output parsing error: {action_content}")

        return {
            "plan": plan,
            "action_type": action_type,
            "action_content": action_content
        }

    async def run(
            self, 
            question: str, 
            final_outline: str,
            progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> str:
        """
        Execute Writer's writing loop.
        
        Args:
            question: Research question
            final_outline: Final outline from Planner
            progress_callback: Optional callback for progress updates
            
        Returns:
            Final report string
        """
        async def emit(event: Dict[str, Any]):
            if not callable(progress_callback):
                return
            event.setdefault("timestamp", datetime.datetime.utcnow().isoformat())
            event.setdefault("agent", "writer")
            try:
                maybe = progress_callback(event)
                if inspect.isawaitable(maybe):
                    await maybe
            except Exception as callback_err:
                logger.warning(f"progress_callback raised error: {callback_err}")

        logger.debug("--- [WebWeaver] Writer Agent activated ---")
        
        # Update system prompt based on question language
        self.system_prompt = get_webweaver_writer_prompt(today_date(), self.instruction, question=question)

        report_written_so_far = ""
        last_observation = "No observation yet. Start by retrieving evidence for the first section."
        # Track retrieve calls to avoid redundant tool executions
        seen_retrieve_keys: Set[str] = set()
        retrieve_repeat_counts: Dict[str, int] = {}
        retrieve_results_cache: Dict[str, str] = {}
        steps_since_last_write = 0
        MAX_IDLE_BEFORE_FORCE_WRITE_HINT = 6
        
        # Get tool definitions for function calling mode
        tool_definitions = self._get_tool_definitions() if not self.use_xml_protocol else None

        for i in range(MAX_LLM_CALL_PER_RUN):
            # Build Writer context
            context_str = (
                f"[Question]\n{question}\n\n"
                f"[Final Outline]\n{final_outline}\n\n"
                f"[Report Written So Far]\n{report_written_so_far}\n\n"
                f"[Last Observation]\n{last_observation}\n\n"
                f"**CRITICAL LANGUAGE REQUIREMENT: The report you write using <write> MUST be "
                f"in the SAME LANGUAGE as the [Question] and [Final Outline] above. "
                f"Check the language carefully and DO NOT translate or switch languages.**"
            )
            # Final iteration: force LLM to output <write>
            is_last_iteration = (i == MAX_LLM_CALL_PER_RUN - 1)
            if is_last_iteration:
                context_str += (
                    "\n[Final Instruction]\n"
                    "This is your last allowed step. You MUST output <write> with a well-structured final section. "
                    "Do NOT output <tool_call> or <terminate>."
                )
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": context_str}
            ]

            # Call LLM
            response = await self.call_server(messages, tools=tool_definitions)
            content = response["content"]
            reasoning_content = response["reasoning_content"]
            tool_calls = response["tool_calls"]
            
            # Log and emit reasoning content if available
            if reasoning_content:
                logger.info(f"[Writer Step {i + 1}] Thinking: {reasoning_content[:500]}...")
                await emit({"type": "thinking", "step": i + 1, "content": reasoning_content})

            # === Function Calling Mode: Handle native tool calls ===
            if not self.use_xml_protocol and tool_calls:
                for tool_call in tool_calls:
                    func_name = tool_call.function.name
                    args_str = tool_call.function.arguments
                    
                    # Check for duplicate retrieve calls
                    if func_name == "retrieve":
                        try:
                            args = json.loads(args_str) if args_str else {}
                            key = json.dumps(args, sort_keys=True, ensure_ascii=False)
                        except Exception:
                            key = None
                        
                        if key and key in seen_retrieve_keys:
                            retrieve_repeat_counts[key] = retrieve_repeat_counts.get(key, 1) + 1
                            cached_evidence = retrieve_results_cache.get(key, "")
                            last_observation = (
                                f"Evidence already retrieved. Here it is again:\n\n{cached_evidence}\n\n"
                                "You MUST now proceed to <write> the section."
                            )
                            steps_since_last_write += 1
                            continue
                        elif key:
                            seen_retrieve_keys.add(key)
                    
                    tool_result = await self._execute_function_call(tool_call)
                    logger.debug(f"Writer Step {i + 1}: Tool {func_name} executed.")
                    
                    # Cache retrieve results
                    if func_name == "retrieve" and key:
                        retrieve_results_cache[key] = tool_result
                    
                    await emit({
                        "type": "tool",
                        "step": i + 1,
                        "tool_name": func_name,
                        "tool_args": args_str,
                        "observation": tool_result[:1000],
                    })
                    last_observation = tool_result
                    steps_since_last_write += 1
                
                # Parse any write/terminate from content
                if content:
                    parsed = self.parse_output(content)
                    if parsed['action_type'] == "write":
                        section_prose = parsed['action_content']
                        report_written_so_far += "\n\n" + section_prose
                        last_observation = f"Section written successfully:\n{section_prose}\n"
                        steps_since_last_write = 0
                        await emit({"type": "section_written", "step": i + 1, "content": section_prose})
                    elif parsed['action_type'] == "terminate":
                        logger.debug("Writer finished. Terminating.")
                        return report_written_so_far
                
                # Force write hint if idling too long
                if steps_since_last_write >= MAX_IDLE_BEFORE_FORCE_WRITE_HINT:
                    last_observation += "\nInstruction: You MUST output <write> now."
                continue

            # === XML Protocol Mode: Parse structured output ===
            parsed = self.parse_output(content)
            logger.debug(f"Writer Step {i + 1} | Action: {parsed}")
            
            await emit({
                "type": "step",
                "step": i + 1,
                "plan": parsed["plan"],
                "action_type": parsed["action_type"],
                "reasoning_content": reasoning_content,
            })

            # Execute action
            if parsed['action_type'] == "terminate":
                logger.debug("Writer finished. Terminating.")
                return report_written_so_far

            elif parsed['action_type'] == "write":
                section_prose = parsed['action_content']
                report_written_so_far += "\n\n" + section_prose
                last_observation = f"Section written successfully:\n{section_prose}\n"
                logger.debug(f"Writer Step {i + 1}: Section written.")
                steps_since_last_write = 0
                await emit({"type": "section_written", "step": i + 1, "content": section_prose})

            elif parsed['action_type'] == "tool_call":
                tool_call_str = parsed['action_content']
                # Try to parse tool call for caching
                try:
                    tool_call_parsed = json5.loads(tool_call_str)
                    tool_name = tool_call_parsed.get('name')
                    tool_args = tool_call_parsed.get('arguments', {})
                except Exception:
                    tool_name, tool_args = None, None

                # Check for duplicate retrieve calls
                if tool_name == "retrieve":
                    try:
                        key = json.dumps(tool_args, sort_keys=True, ensure_ascii=False)
                    except Exception:
                        key = None

                    if key and key in seen_retrieve_keys:
                        retrieve_repeat_counts[key] = retrieve_repeat_counts.get(key, 1) + 1
                        cached_evidence = retrieve_results_cache.get(key, "")
                        last_observation = (
                            f"Evidence already retrieved:\n\n{cached_evidence}\n\n"
                            "You MUST now proceed to <write> the section."
                        )
                        steps_since_last_write += 1
                        continue
                    elif key:
                        seen_retrieve_keys.add(key)

                last_observation = await self._execute_xml_tool(tool_call_str)
                
                # Cache retrieve results
                if tool_name == "retrieve" and key:
                    retrieve_results_cache[key] = last_observation
                
                logger.debug(f"Writer Step {i + 1}: Evidence retrieved.")
                steps_since_last_write += 1
                await emit({
                    "type": "tool",
                    "step": i + 1,
                    "tool_call": tool_call_str,
                    "observation": last_observation[:1000],
                })

            elif parsed['action_type'] == "error":
                last_observation = parsed['action_content']
                logger.warning(f"Writer Step {i + 1}: Action parse error.")
                steps_since_last_write += 1

            # Force write hint if idling too long
            if steps_since_last_write >= MAX_IDLE_BEFORE_FORCE_WRITE_HINT:
                last_observation += (
                    "\nInstruction: You have gathered sufficient evidence. "
                    "You MUST output <write> with a well-structured section now."
                )

        logger.warning("Writer reached max iterations.")
        return report_written_so_far


class WebWeaverAgent:
    """
    WebWeaver main orchestrator.
    
    Coordinates Planner and Writer agents in sequence.
    Based on WebWeaver paper dual-agent framework.
    
    Supports two tool calling modes:
    - use_xml_protocol=True (default): XML-based <tool_call> tags, works better for structured output
    - use_xml_protocol=False: OpenAI-style function calling
    """

    def __init__(
            self,
            llm_config: Optional[Dict] = None,
            function_list: Optional[List[str]] = None,
            instruction: str = "",
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            model: Optional[str] = None,
            use_xml_protocol: bool = True,  # XML protocol works better for WebWeaver's structured output
    ):
        """
        Initialize WebWeaver agent.
        
        Args:
            llm_config: LLM configuration dict
            function_list: Optional list of tool names for Planner
            instruction: Optional custom instruction
            api_key: Optional API key override
            base_url: Optional base URL override
            model: Optional model name override
            use_xml_protocol: If True, use XML tags; if False (default), use OpenAI function calling
        """
        base_config = llm_config or {
            "model": LLM_MODEL_NAME,
            "generate_cfg": {
                "temperature": 0.1,
                "top_p": 0.95,
                "max_tokens": 10000,
            },
            "llm_timeout": 300.0,
        }
        self.llm_config = dict(base_config)
        if api_key:
            self.llm_config["api_key"] = api_key
        if base_url:
            self.llm_config["base_url"] = base_url
        if model:
            self.llm_config["model"] = model
        
        self.use_xml_protocol = use_xml_protocol
        
        # Initialize shared memory bank
        self.memory_bank = MemoryBank()

        # Initialize sub-agents with use_xml_protocol
        self.planner = WebWeaverPlanner(
            self.llm_config, 
            self.memory_bank, 
            function_list=function_list, 
            instruction=instruction,
            use_xml_protocol=use_xml_protocol,
        )
        self.writer = WebWeaverWriter(
            self.llm_config, 
            self.memory_bank, 
            instruction=instruction,
            use_xml_protocol=use_xml_protocol,
        )

        logger.debug("WebWeaver Dual-Agent Framework initialized.")
        logger.debug(f"Planner Tools: {self.planner.function_list}")
        logger.debug(f"Writer Tools: {self.writer.function_list}")
        logger.debug(f"Tool Calling Mode: {'XML Protocol' if use_xml_protocol else 'Function Calling'}")

    async def run(
            self, 
            question: str,
            progress_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute WebWeaver's complete dual-agent workflow.
        
        Args:
            question: Research question
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict with final_report, final_outline, and metadata
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
        
        await emit({"type": "status", "status": "starting", "phase": "planner"})

        # Phase 1: Run Planner
        try:
            final_outline = await asyncio.wait_for(
                self.planner.run(question, progress_callback=progress_callback),
                timeout=AGENT_TIMEOUT
            )
            logger.debug("--- Planner Phase Complete ---")
            logger.debug(f"Final Outline:\n{final_outline}")
            logger.debug(f"Memory Bank contains {self.memory_bank.size()} items.")
            
            await emit({
                "type": "phase_complete", 
                "phase": "planner",
                "outline": final_outline,
                "memory_bank_size": self.memory_bank.size(),
            })
        except Exception as e:
            logger.error(f"Planner Agent failed: {e}")
            await emit({"type": "error", "phase": "planner", "error": str(e)})
            return {
                "question": question,
                "final_report": "",
                "error": f"Planner phase error: {e}"
            }

        await emit({"type": "status", "status": "starting", "phase": "writer"})

        # Phase 2: Run Writer
        try:
            final_report = await asyncio.wait_for(
                self.writer.run(question, final_outline, progress_callback=progress_callback),
                timeout=AGENT_TIMEOUT
            )
            logger.debug("--- Writer Phase Complete ---")
            
            await emit({
                "type": "phase_complete",
                "phase": "writer",
                "report": final_report,
            })
        except Exception as e:
            logger.error(f"Writer Agent failed: {e}")
            await emit({"type": "error", "phase": "writer", "error": str(e)})
            return {
                "question": question,
                "final_report": "",
                "error": f"Writer phase error: {e}"
            }

        end_time = time.time()
        
        result = {
            "question": question,
            "final_outline": final_outline,
            "final_report": final_report,
            "memory_bank_size": self.memory_bank.size(),
            "total_time_seconds": end_time - start_time
        }
        
        await emit({
            "type": "complete",
            "result": result,
        })

        return result


async def main():
    """Main function for testing WebWeaver agent."""
    from webresearcher.log import set_log_level
    set_log_level("DEBUG")
    
    llm_config = {
        "model": LLM_MODEL_NAME,
        "generate_cfg": {
            "temperature": 0.1,
            "top_p": 0.95,
            "max_tokens": 10000,
        },
        "llm_timeout": 300.0,
    }
    agent = WebWeaverAgent(llm_config=llm_config)
    print(LLM_API_KEY)
    question =  "刘翔破纪录时候是多少岁?"
    result = await agent.run(question)
    print(result)


if __name__ == '__main__':
    asyncio.run(main())
