from __future__ import annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client
from typing import List, Dict, Any

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI

system_prompt = """
You are an expert at multiple AI frameworks and tools including:
- Pydantic AI - a Python AI agent framework
- LangChain - a framework for developing applications powered by language models
- LangGraph - a library for building stateful, multi-actor applications with LLMs
- CrewAI - a framework for orchestrating role-playing autonomous AI agents
- And other documentation sources that have been ingested

Your job is to assist with questions about any of these frameworks and tools.
You have access to all their documentation through a RAG system.

Don't ask the user before taking an action, just do it. Always make sure you look at the documentation 
with the provided tools before answering the user's question unless you have already.

When looking at documentation, consider checking multiple sources if the question spans multiple frameworks.
Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
"""

pydantic_ai_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

@pydantic_ai_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str, sources: List[str] = None) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        sources: Optional list of source names to filter by (e.g. ["pydantic_ai_docs", "langchain"])
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Get the embedding for the query
        query_embedding = await get_embedding(user_query, ctx.deps.openai_client)
        
        # Prepare filter
        filter_dict = {'source': 'pydantic_ai_docs'}  # Default filter
        if sources:
            filter_dict = {'source': {'$in': sources}}  # Filter for specific sources
        
        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': filter_dict
            }
        ).execute()
        
        if not result.data:
            return "No relevant documentation found."
            
        # Format the results
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']} (Source: {doc['metadata']['source']})

{doc['content']}

URL: {doc['url']}
"""
            formatted_chunks.append(chunk_text)
            
        # Join all chunks with a separator
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@pydantic_ai_expert.tool
async def list_documentation_sources(ctx: RunContext[PydanticAIDeps]) -> Dict[str, List[str]]:
    """
    Retrieve a list of all available documentation sources and their URLs.
    
    Returns:
        Dict[str, List[str]]: Dictionary mapping source names to lists of their URLs
    """
    try:
        # Query Supabase for all sources and their URLs
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url, metadata->source') \
            .execute()
        
        if not result.data:
            return {}
            
        # Group URLs by source
        sources = {}
        for doc in result.data:
            source = doc['metadata']['source']
            if source not in sources:
                sources[source] = []
            sources[source].append(doc['url'])
            
        # Remove duplicates and sort URLs for each source
        for source in sources:
            sources[source] = sorted(set(sources[source]))
            
        return sources
        
    except Exception as e:
        print(f"Error retrieving documentation sources: {e}")
        return {}

@pydantic_ai_expert.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining all its chunks.
    
    Args:
        ctx: The context including the Supabase client
        url: The URL or path of the page to retrieve
        
    Returns:
        str: The complete page content with all chunks combined in order
    """
    try:
        # Query Supabase for all chunks of this URL, ordered by chunk_number
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number, metadata') \
            .eq('url', url) \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        # Format the page with its title and source
        page_title = result.data[0]['title'].split(' - ')[0]  # Get the main title
        source = result.data[0]['metadata']['source']
        formatted_content = [f"# {page_title} (Source: {source})\n"]
        
        # Add each chunk's content
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"