from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import asyncio
import os
from dotenv import load_dotenv

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from openai import AsyncOpenAI
from supabase import Client

from pydantic_ai_expert import pydantic_ai_expert, PydanticAIDeps
from perplexity_search_rag import perplexity_expert, PerplexityDeps

load_dotenv()

@dataclass
class HybridResult:
    """Store results from both RAG systems."""
    agentic_result: str
    perplexity_result: str
    source: str
    confidence: float
    timestamp: str

@dataclass
class HybridDeps:
    supabase: Client
    openai_client: AsyncOpenAI
    perplexity_api_key: str = os.getenv("PERPLEXITY_API_KEY")
    result_history: List[HybridResult] = None

    def __post_init__(self):
        if self.result_history is None:
            self.result_history = []

# Initialize the model
llm = os.getenv('LLM_MODEL', 'gpt-4o-mini')
model = OpenAIModel(llm)

system_prompt = """You are an advanced hybrid RAG system that combines two powerful components:

1. Agentic RAG: Retrieves information from a curated database of documentation
2. Perplexity Search RAG: Performs real-time web searches using Perplexity's Sonar-Deep-Research model

Your responsibilities:

1. Coordinate both RAG systems effectively
2. Analyze and compare results from both sources
3. Merge information intelligently based on:
   - Relevance to the query
   - Information completeness
   - Source reliability
   - Result freshness
4. Provide comprehensive responses that leverage the strengths of both systems

Guidelines for merging results:

- Prioritize official documentation from the database when available
- Use web search results to fill gaps or provide updates
- Cross-reference information between sources
- Include code examples from both sources when relevant
- Cite all sources clearly
- Explain any discrepancies between sources
- Use confidence scores to weight information

Remember: Your goal is to provide the most accurate, complete, and up-to-date information by leveraging both systems effectively."""

hybrid_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=HybridDeps,
    retries=2
)

async def analyze_results(
    agentic_result: str,
    perplexity_result: str,
    query: str,
    openai_client: AsyncOpenAI
) -> Tuple[str, float]:
    """Analyze and compare results from both systems."""
    analysis_prompt = f"""Analyze these two results and determine which is more relevant and complete:

Query: {query}

Agentic RAG Result:
{agentic_result}

Perplexity Search Result:
{perplexity_result}

Return a JSON object with:
1. "better_source": Either "agentic" or "perplexity"
2. "confidence": A float between 0 and 1
3. "reasoning": A brief explanation of the choice"""

    try:
        response = await openai_client.chat.completions.create(
            model=llm,
            messages=[
                {"role": "system", "content": "You are an expert at analyzing and comparing search results."},
                {"role": "user", "content": analysis_prompt}
            ],
            response_format={ "type": "json_object" }
        )
        
        result = response.choices[0].message.content
        analysis = eval(result)
        return analysis["better_source"], analysis["confidence"]
        
    except Exception as e:
        print(f"Error analyzing results: {e}")
        return "agentic", 0.5  # Default to agentic with neutral confidence

@hybrid_expert.tool
async def query_hybrid_rag(
    ctx: RunContext[HybridDeps],
    query: str,
    merge_results: bool = True
) -> str:
    """
    Query both RAG systems and merge results intelligently.
    
    Args:
        ctx: The context including clients for both systems
        query: The user's question
        merge_results: Whether to merge results or return both separately
        
    Returns:
        Formatted string containing the final response
    """
    try:
        # Set up dependencies for both systems
        agentic_deps = PydanticAIDeps(
            supabase=ctx.deps.supabase,
            openai_client=ctx.deps.openai_client
        )
        
        perplexity_deps = PerplexityDeps(
            openai_client=ctx.deps.openai_client,
            perplexity_api_key=ctx.deps.perplexity_api_key
        )
        
        # Query both systems in parallel
        agentic_task = pydantic_ai_expert.retrieve_relevant_documentation(
            RunContext(deps=agentic_deps),
            query,
            use_memory=True
        )
        
        perplexity_task = perplexity_expert.perform_deep_search(
            RunContext(deps=perplexity_deps),
            query,
            use_history=True
        )
        
        agentic_result, perplexity_result = await asyncio.gather(
            agentic_task,
            perplexity_task
        )
        
        # Analyze results
        better_source, confidence = await analyze_results(
            agentic_result,
            perplexity_result,
            query,
            ctx.deps.openai_client
        )
        
        # Create hybrid result
        hybrid_result = HybridResult(
            agentic_result=agentic_result,
            perplexity_result=perplexity_result,
            source=better_source,
            confidence=confidence,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        # Add to history
        ctx.deps.result_history.append(hybrid_result)
        if len(ctx.deps.result_history) > 10:
            ctx.deps.result_history = ctx.deps.result_history[-10:]
        
        if not merge_results:
            return f"""
# Agentic RAG Results:
{agentic_result}

---

# Perplexity Search Results:
{perplexity_result}

---

Analysis:
- Preferred Source: {better_source}
- Confidence: {confidence:.2f}
"""
        
        # Return the better result
        main_result = agentic_result if better_source == "agentic" else perplexity_result
        other_result = perplexity_result if better_source == "agentic" else agentic_result
        
        return f"""
# Combined Results (Confidence: {confidence:.2f})

{main_result}

Additional Information:
{other_result}
"""
        
    except Exception as e:
        print(f"Error querying hybrid RAG: {e}")
        return f"Error querying hybrid RAG: {str(e)}"

@hybrid_expert.tool
async def get_result_history(ctx: RunContext[HybridDeps]) -> str:
    """
    Get the history of hybrid results.
    
    Returns:
        Formatted string containing result history
    """
    if not ctx.deps.result_history:
        return "No result history available."
        
    history = []
    for result in ctx.deps.result_history:
        history.append(f"""
Timestamp: {result.timestamp}
Chosen Source: {result.source}
Confidence: {result.confidence:.2f}
""")
        
    return "\n---\n".join(history) 