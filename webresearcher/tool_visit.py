import json
from typing import List, Union
import requests
from webresearcher.base import BaseTool
from openai import OpenAI
import time
import tiktoken
import re
from markdownify import MarkdownConverter
from webresearcher.prompt import get_extractor_prompt
from webresearcher.log import logger
from webresearcher.config import (
    JINA_API_KEY,
    LLM_API_KEY,
    LLM_BASE_URL,
    SUMMARY_MODEL_NAME,
    VISIT_SERVER_MAX_RETRIES,
)


def truncate_to_tokens(text: str, max_tokens: int = 95000) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


def extract_readable_text(html: str) -> str:
    """Convert raw HTML into plain text."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    text = soup.get_text(separator="\n")
    cleaned_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(cleaned_lines)


def parse_html_to_markdown(html: str, url: str) -> str:
    """Parse HTML to markdown format.
    
    Args:
        html: HTML content to convert
        url: Source URL (used for special handling of Wikipedia pages)
        
    Returns:
        Markdown formatted text
    """
    from bs4 import BeautifulSoup
    
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string if soup.title else "No Title"
    
    # Remove javascript, style blocks, and hyperlinks
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    
    # Remove other common irrelevant elements
    for element in soup.find_all(["nav", "footer", "aside", "form", "figure", "header"]):
        element.decompose()
    
    # Special handling for Wikipedia pages
    if "wikipedia.org" in url:
        body_elm = soup.find("div", {"id": "mw-content-text"})
        title_elm = soup.find("span", {"class": "mw-page-title-main"})
        
        if body_elm:
            main_title = title_elm.string if title_elm else title
            webpage_text = f"# {main_title}\n\n" + MarkdownConverter().convert_soup(body_elm)
        else:
            webpage_text = MarkdownConverter().convert_soup(soup)
    else:
        webpage_text = MarkdownConverter().convert_soup(soup)
    
    # Clean up excessive newlines
    webpage_text = re.sub(r"\r\n", "\n", webpage_text)
    webpage_text = re.sub(r"\n{2,}", "\n\n", webpage_text).strip()
    
    # Add title if not already present
    if not webpage_text.startswith("# "):
        webpage_text = f"# {title}\n\n{webpage_text}"
    
    return webpage_text

OSS_JSON_FORMAT = """# Response Formats
## visit_content
{"properties":{"rational":{"type":"string","description":"Locate the **specific sections/data** directly related to the 
user's goal within the webpage content"},"evidence":{"type":"string","description":"Identify and extract the **most 
relevant information** from the content, never miss any important information, output the **full original context** 
of the content as far as possible, it can be more than three paragraphs.","summary":{"type":"string","description":
"Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the 
information to the goal."}}}}"""


class Visit(BaseTool):
    # The `description` tells the agent the functionality of this tool.
    name = 'visit'
    description = 'Visit webpage(s) and return the summary of the content.'
    # The `parameters` tell the agent what input parameters the tool has.
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": ["string", "array"],
                "items": {
                    "type": "string"
                    },
                "minItems": 1,
                "description": "The URL(s) of the webpage(s) to visit. Can be a single URL or an array of URLs."
        },
        "goal": {
                "type": "string",
                "description": "The goal of the visit for webpage(s)."
        }
        },
        "required": ["url", "goal"]
    }
    # The `call` method is the main function of the tool.
    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            url = params["url"]
            goal = params["goal"]
        except:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields"

        if isinstance(url, str):
            response = self.readpage_jina(url, goal)
        else:
            response = []
            assert isinstance(url, List)
            start_time = time.time()
            for u in url: 
                if time.time() - start_time > 900:
                    cur_response = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                    cur_response += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                    cur_response += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
                else:
                    try:
                        cur_response = self.readpage_jina(u, goal)
                    except Exception as e:
                        cur_response = f"Error fetching {u}: {str(e)}"
                response.append(cur_response)
            response = "\n=======\n".join(response)
        response = response.strip()
        logger.debug(f'[Visit] url: {url},\nSummary Length: {len(response)};\nresponse: {response[:500]}...')
        return response
        
    def call_server(self, msgs, max_retries=2):
        client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL,
        )
        for attempt in range(max_retries):
            try:
                chat_response = client.chat.completions.create(
                    model=SUMMARY_MODEL_NAME,
                    messages=msgs,
                    temperature=0.7
                )
                content = chat_response.choices[0].message.content
                if content:
                    try:
                        json.loads(content)
                    except:
                        # extract json from string 
                        left = content.find('{')
                        right = content.rfind('}') 
                        if left != -1 and right != -1 and left <= right: 
                            content = content[left:right+1]
                    return content
            except Exception as e:
                logger.debug(f"API call attempt {attempt + 1} failed: {str(e)}")
                if attempt == (max_retries - 1):
                    return ""
                continue


    def jina_readpage(self, url: str) -> str:
        """
        Read webpage content using Jina service.
        
        Args:
            url: The URL to read
            
        Returns:
            str: The webpage content or error message
        """
        # Check if Jina API key is available
        if not JINA_API_KEY or JINA_API_KEY.strip() == "":
            logger.debug(f"[visit] Jina API key not configured, falling back to local fetch")
            return self.local_fetch_url(url)
        
        max_retries = 1  # 减少重试次数从 2 到 1
        timeout = 50
        
        for attempt in range(max_retries):
            headers = {
                "Authorization": f"Bearer {JINA_API_KEY}",
            }
            try:
                response = requests.get(
                    f"https://r.jina.ai/{url}",
                    headers=headers,
                    timeout=timeout
                )
                if response.status_code == 200:
                    webpage_content = response.text
                    return webpage_content
                else:
                    logger.debug(f"Jina API error response: {response.text}")
                    if attempt == max_retries - 1:
                        # 最后一次失败，降级到本地抓取
                        logger.debug(f"[visit] Jina API failed, falling back to local fetch")
                        return self.local_fetch_url(url)
                    raise ValueError("jina readpage error")
            except Exception as e:
                logger.debug(f"[visit] Jina fetch attempt {attempt + 1} failed: {e}")
                time.sleep(0.5)
                if attempt == max_retries - 1:
                    # 最后一次失败，降级到本地抓取
                    logger.debug(f"[visit] Jina API failed, falling back to local fetch")
                    return self.local_fetch_url(url)
                
        return "[visit] Failed to read page."
    
    def local_fetch_url(self, url: str) -> str:
        """
        Fallback method: Fetch URL content locally without Jina.
        
        Args:
            url: The URL to fetch
            
        Returns:
            str: Markdown formatted webpage content or error message
        """
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; URLCrawler/1.0; +https://example.com/bot)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
            
            logger.debug(f"[visit] Local fetching URL: {url}")
            response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get("content-type", "").lower()
            allowed_keywords = ("text", "html", "xml")
            if not any(keyword in content_type for keyword in allowed_keywords):
                logger.warning(f"[visit] Unsupported content type: {content_type}")
                return f"[visit] Unsupported content type: {content_type}"
            
            # Decode content
            html_content = response.content.decode(response.encoding or "utf-8", errors="replace")
            
            # Convert to markdown
            markdown_content = parse_html_to_markdown(html_content, url)
            logger.debug(f"[visit] Local fetch successful, content length: {len(markdown_content)}")
            return markdown_content
            
        except requests.exceptions.Timeout:
            logger.warning(f"[visit] Timeout while fetching {url}")
            return f"[visit] Timeout while fetching {url}"
        except requests.exceptions.HTTPError as e:
            logger.warning(f"[visit] HTTP error while fetching {url}: {e}")
            return f"[visit] HTTP error: {e}"
        except Exception as e:
            logger.warning(f"[visit] Failed to fetch {url}: {e}")
            return f"[visit] Failed to fetch page: {e}"

    def html_readpage_jina(self, url: str) -> str:
        max_attempts = 1  # 减少重试次数从 2 到 1
        for attempt in range(max_attempts):
            content = self.jina_readpage(url)
            service = "jina or local"     
            logger.debug(f"Using service: {service}")
            if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
                return content
        return "[visit] Failed to read page."

    def readpage_jina(self, url: str, goal: str) -> str:
        """
        Attempt to read webpage content by alternating between jina and aidata services.
        
        Args:
            url: The URL to read
            goal: The goal/purpose of reading the page
            
        Returns:
            str: The webpage content or error message
        """
   
        summary_page_func = self.call_server
        max_retries = VISIT_SERVER_MAX_RETRIES

        content = self.html_readpage_jina(url)

        if content and not content.startswith("[visit] Failed to read page.") and content != "[visit] Empty content." and not content.startswith("[document_parser]"):
            content = truncate_to_tokens(content, max_tokens=95000)
            extractor_prompt_template = get_extractor_prompt(goal)
            messages = [{"role":"user","content": extractor_prompt_template.format(webpage_content=content, goal=goal)}]
            parse_retry_times = 0
            raw = summary_page_func(messages, max_retries=max_retries)
            summary_retries = 1
            while len(raw) < 10 and summary_retries >= 0:
                truncate_length = int(0.7 * len(content)) if summary_retries > 0 else 25000
                status_msg = (
                    f"[visit] Summary url[{url}] " 
                    f"attempt {3 - summary_retries + 1}/3, "
                    f"content length: {len(content)}, "
                    f"truncating to {truncate_length} chars"
                ) if summary_retries > 0 else (
                    f"[visit] Summary url[{url}] failed after 3 attempts, "
                    f"final truncation to 25000 chars"
                )
                logger.debug(status_msg)
                content = content[:truncate_length]
                extractor_prompt_template = get_extractor_prompt(goal)
                extraction_prompt = extractor_prompt_template.format(
                    webpage_content=content,
                    goal=goal
                )
                messages = [{"role": "user", "content": extraction_prompt}]
                raw = summary_page_func(messages, max_retries=max_retries)
                summary_retries -= 1

            parse_retry_times = 1
            if isinstance(raw, str):
                raw = raw.replace("```json", "").replace("```", "").strip()
            while parse_retry_times < 3:
                try:
                    raw = json.loads(raw)
                    break
                except:
                    raw = summary_page_func(messages, max_retries=max_retries)
                    parse_retry_times += 1
            
            if parse_retry_times >= 3:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
                useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
            else:
                useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
                useful_information += "Evidence in page: \n" + str(raw["evidence"]) + "\n\n"
                useful_information += "Summary: \n" + str(raw["summary"]) + "\n\n"

            if len(useful_information) < 10 and summary_retries < 0:
                logger.debug("[visit] Could not generate valid summary after maximum retries")
                useful_information = "[visit] Failed to read page"
            
            return useful_information

        # If no valid content was obtained after all retries
        else:
            useful_information = "The useful information in {url} for user goal {goal} as follows: \n\n".format(url=url, goal=goal)
            useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
            useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
            return useful_information

# add demo
if __name__ == '__main__':
    logger.debug(f"JINA_API_KEY: {JINA_API_KEY}")
    visit_tool = Visit()
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    goal = "Explain the history of artificial intelligence."
    params = {
        "url": url,
        "goal": goal
    }
    response = visit_tool.call(params)
    logger.debug(f"Response: {response}")
