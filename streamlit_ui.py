from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os

import streamlit as st
import json
import logfire
from supabase import Client
from openai import AsyncOpenAI

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)

# Import the RAG experts
from pydantic_ai_expert import pydantic_ai_expert, PydanticAIDeps
from perplexity_search_rag import perplexity_expert, PerplexityDeps
from hybrid_rag import hybrid_expert, HybridDeps

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""
    role: Literal['user', 'model']
    timestamp: str
    content: str

def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)          

async def run_agent_with_streaming(user_input: str, rag_type: str = "hybrid"):
    """
    Run the selected RAG agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Prepare dependencies based on RAG type
    if rag_type == "agentic":
        agent = pydantic_ai_expert
        deps = PydanticAIDeps(
            supabase=supabase,
            openai_client=openai_client
        )
    elif rag_type == "perplexity":
        agent = perplexity_expert
        deps = PerplexityDeps(
            openai_client=openai_client,
            perplexity_api_key=os.getenv("PERPLEXITY_API_KEY")
        )
    else:  # hybrid
        agent = hybrid_expert
        deps = HybridDeps(
            supabase=supabase,
            openai_client=openai_client,
            perplexity_api_key=os.getenv("PERPLEXITY_API_KEY")
        )

    # Run the agent in a stream
    async with agent.run_stream(
        user_input,
        deps=deps,
        message_history=st.session_state.messages[:-1],  # pass entire conversation so far
    ) as result:
        # We'll gather partial text to show incrementally
        partial_text = ""
        message_placeholder = st.empty()

        # Render partial text as it arrives
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        # Now that the stream is finished, we have a final result.
        # Add new messages from this run, excluding user-prompt messages
        filtered_messages = [msg for msg in result.new_messages() 
                           if not (hasattr(msg, 'parts') and 
                                 any(part.part_kind == 'user-prompt' for part in msg.parts))]
        st.session_state.messages.extend(filtered_messages)

        # Add the final response to the messages
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )

async def main():
    st.title("Hybrid RAG System")
    st.write("Ask questions about AI frameworks and tools. Our hybrid system combines local documentation with real-time web search.")

    # Add RAG type selector in the sidebar
    st.sidebar.title("RAG Configuration")
    rag_type = st.sidebar.radio(
        "Select RAG Type",
        ["hybrid", "agentic", "perplexity"],
        help="Choose which RAG system to use for queries"
    )

    # Add confidence threshold slider
    if rag_type == "hybrid":
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            help="Minimum confidence score to merge results"
        )

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages from the conversation so far
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("What would you like to know about AI frameworks and tools?")

    if user_input:
        # We append a new request to the conversation explicitly
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )
        
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Actually run the agent now, streaming the text
            await run_agent_with_streaming(user_input, rag_type)

if __name__ == "__main__":
    asyncio.run(main())
