from __future__ import annotations

import os
import sys
import json
import asyncio
import requests
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from pathlib import Path
import PyPDF2
from xml.etree import ElementTree

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

@dataclass
class DocumentSource:
    name: str  # e.g., "langchain", "crewai"
    sitemap_url: Optional[str] = None  # For web documentation
    pdf_paths: Optional[List[str]] = None  # For PDF files
    base_url: Optional[str] = None  # Base URL for the documentation

@dataclass
class ProcessedChunk:
    url: str  # For PDFs, this will be the file path
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using GPT-4."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  # Send first 1000 chars for context
            ],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}
    
async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        if "invalid_api_key" in str(e):
            print("Error: Invalid API key provided. Please check your OpenAI API key.")
        else:
            print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on errorector on error

async def process_chunk(chunk: str, chunk_number: int, url: str, source: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)
    
    # Get embedding
    embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
        "source": source,
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path if url.startswith('http') else url
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,
        metadata=metadata,
        embedding=embedding
    )


async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

def get_urls_from_sitemap(sitemap_url: str) -> List[str]:
    """Get URLs from a sitemap."""
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Extract all URLs from the sitemap
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

async def process_and_store_document(url: str, markdown: str, source: str):
    """Process a document and store its chunks in parallel."""
    # Split into chunks
    chunks = chunk_text(markdown)
    
    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, i, url, source) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    
    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)

async def crawl_documentation(source: DocumentSource, max_concurrent: int = 5):
    """Crawl documentation from a sitemap URL."""
    if not source.sitemap_url:
        print(f"No sitemap URL provided for {source.name}")
        return

    # Get URLs from sitemap
    urls = get_urls_from_sitemap(source.sitemap_url)
    if not urls:
        print(f"No URLs found for {source.name}")
        return

    print(f"Found {len(urls)} URLs to crawl for {source.name}")

    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id=f"session_{source.name}"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_document(url, result.markdown_v2.raw_markdown, source.name)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

async def process_pdf(pdf_path: str, source: str):
    """Process a PDF file and store its chunks."""
    try:
        # Read PDF
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            # Extract text from each page
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n\n"
        
        # Process and store the extracted text
        await process_and_store_document(pdf_path, text, source)
        print(f"Successfully processed PDF: {pdf_path}")
        
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")

async def process_pdfs(source: DocumentSource):
    """Process all PDFs for a source."""
    if not source.pdf_paths:
        return

    for pdf_path in source.pdf_paths:
        await process_pdf(pdf_path, source.name)

async def main():
    # Define documentation sources
    sources = [
        DocumentSource(
            name="pydantic_ai_docs",
            sitemap_url="https://ai.pydantic.dev/sitemap.xml"
        ),
        DocumentSource(
            name="langgraph",
            sitemap_url="https://python.langgraph.com/sitemap.xml"
        ),
        DocumentSource(
            name="crewai",
            sitemap_url="https://docs.crewai.com/sitemap.xml"
        ),
        # Add more documentation sources as needed
    ]

    # Add PDF sources
    pdf_sources = [
        DocumentSource(
            name="local_pdfs",
            pdf_paths=[
                "docs/2005.11401v4.pdf",
                "docs/2312.10997v5.pdf"
                # Add more PDF paths as needed
            ]
        )
    ]

    # Process all documentation sources
    for source in sources:
        print(f"\nProcessing documentation for {source.name}...")
        await crawl_documentation(source)

    # Process all PDF sources
    for source in pdf_sources:
        print(f"\nProcessing PDFs for {source.name}...")
        await process_pdfs(source)

if __name__ == "__main__":
    asyncio.run(main()) 