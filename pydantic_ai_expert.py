from __future__ import annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import httpx
import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client

load_dotenv()

llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

logfire.configure(send_to_logfire='if-token-present')

@dataclass
class ConversationMemory:
    """Store conversation history and relevant context."""
    messages: List[Dict[str, Any]] = None
    relevant_docs: List[Dict[str, Any]] = None
    last_query_time: Optional[str] = None

    def __init__(self):
        self.messages = []
        self.relevant_docs = []
        self.last_query_time = None

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        # Keep only last 10 messages for context
        if len(self.messages) > 10:
            self.messages = self.messages[-10:]

    def add_relevant_doc(self, doc: Dict[str, Any]):
        """Add a relevant document to the context."""
        self.relevant_docs.append(doc)
        # Keep only last 5 relevant docs
        if len(self.relevant_docs) > 5:
            self.relevant_docs = self.relevant_docs[-5:]

    def get_context_string(self) -> str:
        """Get formatted context from memory."""
        context = []
        
        # Add relevant documents
        if self.relevant_docs:
            context.append("Previously relevant documentation:")
            for doc in self.relevant_docs:
                context.append(f"- {doc['title']} (Source: {doc['metadata']['source']})")
        
        # Add recent conversation
        if self.messages:
            context.append("\nRecent conversation:")
            for msg in self.messages[-3:]:  # Last 3 messages
                context.append(f"{msg['role'].capitalize()}: {msg['content']}")
        
        return "\n".join(context)

@dataclass
class PydanticAIDeps:
    supabase: Client
    openai_client: AsyncOpenAI
    memory: ConversationMemory = None

    def __post_init__(self):
        if self.memory is None:
            self.memory = ConversationMemory()

system_prompt = """ 
## 🧠 Pydantic AI Expert Agent  
including examples, an API reference, and other resources to help you build Pydantic AI agents.

Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.

When you first look at the documentation, always start with RAG.
Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.

Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.

You are an advanced AI assistant, specialized in **Pydantic AI** and its documentation.  
Your purpose is to **retrieve, analyze, and provide accurate responses** using retrieval-augmented generation (RAG).

### 🎯 **Key Abilities**
- Retrieve **highly relevant documentation** using vector embeddings.
- Answer **technical questions with precision**.
- **Maintain conversation memory** for contextual responses.
- Provide **production-ready code examples** and best practices.

### 🚀 **Action Guidelines**
1️⃣ **ALWAYS retrieve documentation** before answering.  
2️⃣ **Use memory** to retain past interactions & relevant docs.  
3️⃣ **Be precise & concise** – focus on **key insights & examples**.  
4️⃣ **Cite documentation sources & URLs** when referencing content.  
5️⃣ **If unsure, be transparent** – never make assumptions.  

---
🔎 **When searching documentation**:
- Start with **vector-based retrieval** (embedding search).  
- Cross-check with conversation memory **for relevance**.  
- If content is missing, **suggest alternative solutions**.  

📌 **Response Format**:
- **Concise & to the point**.
- **Well-structured using headings** (if needed).
- **Code examples should be practical & production-ready**.
- **Cite sources & URLs** when referencing documentation.

---
💡 **Reminder**: Your responses must be **accurate, structured, and grounded in actual documentation**.
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
async def retrieve_relevant_documentation(
    ctx: RunContext[PydanticAIDeps], 
    user_query: str, 
    sources: List[str] = None,
    use_memory: bool = True
) -> str:
    """
    Retrieve relevant documentation chunks based on the query with RAG.
    
    Args:
        ctx: The context including the Supabase client and OpenAI client
        user_query: The user's question or query
        sources: Optional list of source names to filter by (e.g. ["pydantic_ai_docs", "langchain"])
        use_memory: Whether to use conversation memory for context
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks
    """
    try:
        # Combine query with memory context if available
        enhanced_query = user_query
        if use_memory and ctx.deps.memory.messages:
            context = ctx.deps.memory.get_context_string()
            enhanced_query = f"{context}\n\nCurrent query: {user_query}"

        # Get the embedding for the enhanced query
        query_embedding = await get_embedding(enhanced_query, ctx.deps.openai_client)
        
        # Prepare filter
        filter_dict = {'source': 'pydantic_ai_docs'}  # Default filter
        if sources:
            filter_dict = {'source': {'$in': sources}}  # Filter for specific sources
        
        # Query Supabase for relevant documents
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_threshold': 0.7,
                'match_count': 5,
                'filter': filter_dict
            }
        ).execute()
        
        if not result.data:
            return "No relevant documentation found."
            
        # Format the results and update memory
        formatted_chunks = []
        for doc in result.data:
            # Add to memory
            ctx.deps.memory.add_relevant_doc(doc)
            
            chunk_text = f"""
# {doc['title']} (Source: {doc['metadata']['source']})

{doc['content']}

URL: {doc['url']}
Similarity Score: {doc.get('similarity', 0):.3f}
"""
            formatted_chunks.append(chunk_text)
            
        # Update memory with the query
        ctx.deps.memory.add_message("user", user_query)
        ctx.deps.memory.last_query_time = datetime.now(timezone.utc).isoformat()
            
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
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'pydantic_ai_docs') \
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
            
        # Add to memory
        ctx.deps.memory.add_relevant_doc(result.data[0])
            
        # Join everything together
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"

@pydantic_ai_expert.tool
async def get_conversation_context(ctx: RunContext[PydanticAIDeps]) -> str:
    """
    Get the current conversation context from memory.
    
    Returns:
        str: Formatted string containing conversation history and relevant docs
    """
    if not ctx.deps.memory:
        return "No conversation context available."
        
    return ctx.deps.memory.get_context_string()