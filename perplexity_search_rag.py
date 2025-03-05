from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import asyncio
import httpx
import os
from dotenv import load_dotenv

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI

load_dotenv()

@dataclass
class SearchResult:
    """Store search results from Perplexity API."""
    title: str
    content: str
    url: str
    score: float
    timestamp: str

@dataclass
class PerplexityDeps:
    openai_client: AsyncOpenAI
    perplexity_api_key: str = os.getenv("PERPLEXITY_API_KEY")
    search_history: List[SearchResult] = None

    def __post_init__(self):
        if self.search_history is None:
            self.search_history = []

# Initialize the model
llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

system_prompt = """You are an expert AI research assistant that uses Perplexity's Sonar-Deep-Research model for web searches.
Your goal is to provide comprehensive, accurate, and up-to-date information about AI frameworks and tools.

Key Responsibilities:
1. Perform deep web searches using Perplexity API
2. Extract relevant information from search results
3. Synthesize information into coherent responses
4. Maintain search history for context
5. Cite sources and provide URLs for verification

Guidelines:
- Always verify information from multiple sources
- Prioritize official documentation and reliable sources
- Include code examples when relevant
- Explain complex concepts clearly
- Be transparent about source reliability

Remember: Your responses should be accurate, well-researched, and include citations to sources."""

perplexity_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PerplexityDeps,
    retries=2
)

async def search_perplexity(query: str, api_key: str) -> List[Dict[str, Any]]:
    """Perform a search using Perplexity's Sonar-Deep-Research model."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.perplexity.ai/search",
                headers=headers,
                json={
                    "query": query,
                    "model": "sonar-deep-research",
                    "max_results": 5,
                    "include_domains": [
                        "python.langchain.com",
                        "ai.pydantic.dev",
                        "python.langgraph.com",
                        "docs.crewai.com",
                        "github.com"
                    ]
                }
            )
            response.raise_for_status()
            return response.json()["results"]
        except Exception as e:
            print(f"Error searching Perplexity: {e}")
            return []

@perplexity_expert.tool
async def perform_deep_search(
    ctx: RunContext[PerplexityDeps],
    query: str,
    use_history: bool = True
) -> str:
    """
    Perform a deep web search using Perplexity API and format the results.
    
    Args:
        ctx: The context including the OpenAI client and Perplexity API key
        query: The search query
        use_history: Whether to include search history in results
        
    Returns:
        Formatted string containing search results and citations
    """
    try:
        # Perform the search
        results = await search_perplexity(query, ctx.deps.perplexity_api_key)
        
        if not results:
            return "No search results found."
            
        # Format results and update history
        formatted_results = []
        for result in results:
            search_result = SearchResult(
                title=result["title"],
                content=result["content"],
                url=result["url"],
                score=result.get("score", 0.0),
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
            # Add to search history
            ctx.deps.search_history.append(search_result)
            if len(ctx.deps.search_history) > 10:
                ctx.deps.search_history = ctx.deps.search_history[-10:]
            
            result_text = f"""
# {search_result.title}

{search_result.content}

URL: {search_result.url}
Relevance Score: {search_result.score:.3f}
"""
            formatted_results.append(result_text)
            
        # Add search history if requested
        if use_history and ctx.deps.search_history:
            history_text = "\n\nRecent Search History:\n"
            for hist in ctx.deps.search_history[-3:]:  # Last 3 searches
                history_text += f"- {hist.title} ({hist.url})\n"
            formatted_results.append(history_text)
            
        return "\n\n---\n\n".join(formatted_results)
        
    except Exception as e:
        print(f"Error performing search: {e}")
        return f"Error performing search: {str(e)}"

@perplexity_expert.tool
async def get_search_history(ctx: RunContext[PerplexityDeps]) -> str:
    """
    Get the search history with results.
    
    Returns:
        Formatted string containing search history
    """
    if not ctx.deps.search_history:
        return "No search history available."
        
    history = []
    for result in ctx.deps.search_history:
        history.append(f"""
# {result.title}
URL: {result.url}
Timestamp: {result.timestamp}
""")
        
    return "\n---\n".join(history) 