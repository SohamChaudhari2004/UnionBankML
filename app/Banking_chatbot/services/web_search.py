import asyncio
from typing import List, Dict, Any
import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import AsyncDDGS
from core.config import settings
from models.retrieval import RetrievalResult, SourceType, WebSearchResult


class WebSearchService:
    def __init__(self):
        self.search_client = AsyncDDGS()
        self.http_client = httpx.AsyncClient(timeout=10.0)
        self.max_results = settings.DDG_SEARCH_MAX_RESULTS

    async def search(self, query: str) -> List[WebSearchResult]:
        """
        Perform a web search using DuckDuckGo
        
        Args:
            query: Search query
            
        Returns:
            List of WebSearchResult objects
        """
        # Append banking-related terms to the query for better results
        banking_query = f"{query} banking finance"
        
        try:
            # Perform the search
            results = await self.search_client.text(
                banking_query,
                max_results=self.max_results
            )
            
            web_results = []
            for result in results:
                web_results.append(
                    WebSearchResult(
                        title=result.get("title", ""),
                        snippet=result.get("body", ""),
                        url=result.get("href", "")
                    )
                )
            
            return web_results
        except Exception as e:
            print(f"Error in web search: {str(e)}")
            return []

    async def fetch_page_content(self, url: str) -> str:
        """
        Fetch and extract relevant content from a webpage
        
        Args:
            url: URL to fetch
            
        Returns:
            Extracted text content
        """
        try:
            response = await self.http_client.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style", "header", "footer", "nav"]):
                script.extract()
            
            # Get text content
            text = soup.get_text(separator="\n", strip=True)
            
            # Clean up text
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = "\n".join(lines)
            
            # Limit content length
            return text[:5000]
        except Exception as e:
            print(f"Error fetching page content: {str(e)}")
            return ""

    async def retrieve(self, query: str) -> List[RetrievalResult]:
        """
        Retrieve information from the web for a query
        
        Args:
            query: Query text
            
        Returns:
            List of RetrievalResult objects
        """
        web_results = await self.search(query)
        
        if not web_results:
            return []
        
        retrieval_results = []
        
        # Process top results and fetch content
        for result in web_results:
            # Combine snippet with title
            content = f"{result.title}\n\n{result.snippet}"
            
            # Try to fetch more detailed content for the top result
            if len(retrieval_results) == 0:
                page_content = await self.fetch_page_content(result.url)
                if page_content:
                    content = f"{result.title}\n\n{page_content}"
            
            retrieval_results.append(
                RetrievalResult(
                    source_type=SourceType.WEB,
                    content=content,
                    metadata={
                        "title": result.title,
                        "url": result.url
                    }
                )
            )
        
        return retrieval_results