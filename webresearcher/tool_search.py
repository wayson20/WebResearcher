from typing import Dict, List, Optional, Union
import http.client
import json
import re
from webresearcher.log import logger
from webresearcher.base import BaseTool
from webresearcher.config import SERPER_API_KEY
from baidusearch.baidusearch import search as baidu_search


class Search(BaseTool):
    name = "search"
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call. max 5 queries."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Array of query strings. Include multiple complementary search queries in a single call. max 5 queries."
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        pass  # No parent __init__ needed

    def _clean_text(self, text: str) -> str:
        """清理文本：删除多余空白字符和空行"""
        if not text:
            return ""
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        text = text.strip()
        return text

    def baidu_search_fallback(self, query: str, num_results: int = 10) -> str:
        """百度搜索"""
        try:
            results = baidu_search(query, num_results=num_results)
            if not results:
                return f"No results found for '{query}'. Try with a more general query."
            
            web_snippets = []
            for idx, r in enumerate(results, 1):
                title = self._clean_text(r.get('title', ''))
                url = r.get('url', '')
                abstract = self._clean_text(r.get('abstract', ''))
                
                snippet = f"{idx}. [{title}]({url})"
                if abstract:
                    snippet += f"\n{abstract}"
                web_snippets.append(snippet)
            
            content = f"A Baidu search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(web_snippets)
            return content
        except Exception as e:
            logger.error(f"Baidu search error: {e}")
            return f"Baidu search failed for '{query}': {str(e)}"

    def google_search_with_serp(self, query: str):
        def contains_chinese_basic(text: str) -> bool:
            return any('\u4E00' <= char <= '\u9FFF' for char in text)

        conn = http.client.HTTPSConnection("google.serper.dev")
        if contains_chinese_basic(query):
            payload = json.dumps({
                "q": query,
                "location": "China",
                "gl": "cn",
                "hl": "zh-cn"
            })

        else:
            payload = json.dumps({
                "q": query,
                "location": "United States",
                "gl": "us",
                "hl": "en"
            })
        if not SERPER_API_KEY:
            return ""
        headers = {
            'X-API-KEY': SERPER_API_KEY,
            'Content-Type': 'application/json'
        }
        res = None
        for i in range(5):
            try:
                conn.request("POST", "/search", payload, headers)
                res = conn.getresponse()
                break
            except Exception as e:
                print(e)
                if i == 4:
                    return ""
                continue

        data = res.read()
        results = json.loads(data.decode("utf-8"))

        try:
            if "organic" not in results:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

            web_snippets = list()
            idx = 0
            if "organic" in results:
                for page in results["organic"]:
                    idx += 1
                    date_published = ""
                    if "date" in page:
                        date_published = "\nDate published: " + page["date"]

                    source = ""
                    if "source" in page:
                        source = "\nSource: " + page["source"]

                    snippet = ""
                    if "snippet" in page:
                        snippet = "\n" + page["snippet"]

                    redacted_version = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                    redacted_version = redacted_version.replace("Your browser can't play this video.", "")
                    web_snippets.append(redacted_version)

            content = f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n" + "\n\n".join(
                web_snippets)
            return content
        except:
            return ""

    def search_with_serp(self, query: str):
        """优先使用Serper API，如果不可用则降级为百度搜索"""
        if SERPER_API_KEY:
            result = self.google_search_with_serp(query)
            if result:
                return result
        return self.baidu_search_fallback(query)

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            query = params["query"]
        except:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"
        if isinstance(query, str):
            # 单个查询
            response = self.search_with_serp(query)
        else:
            # 多个查询
            assert isinstance(query, List)
            responses = []
            for q in query:
                responses.append(self.search_with_serp(q))
            response = "\n=======\n".join(responses)
        logger.debug(f"[Search] query: {query},\nresponse: {response[:500]}...")
        return response


# add demo
if __name__ == '__main__':
    print("SERPER_API_KEY:", SERPER_API_KEY)
    tool = Search()
    params = {
        "query": ["What is AI?", "北京是什么?"]
    }
    r = tool.call(params)
    print(r)
