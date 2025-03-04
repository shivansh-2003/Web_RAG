# Multi-Source Documentation RAG System

This project implements a RAG (Retrieval-Augmented Generation) system that can ingest and query multiple documentation sources, including web-based documentation and PDF files. It uses Pydantic AI for the agent framework, Crawl4AI for web crawling, and Supabase for vector storage.

## Features

- Crawl and ingest documentation from multiple sources:
  - Web-based documentation via sitemaps (e.g., Pydantic AI, LangChain, LangGraph, CrewAI)
  - Local PDF files
- Process and chunk documents intelligently, preserving code blocks and paragraph structure
- Generate embeddings using OpenAI's text-embedding-3-small model
- Store documents and embeddings in Supabase with vector search capabilities
- Query across all sources using semantic search
- Interactive Streamlit UI for querying the documentation

## Prerequisites

- Python 3.8+
- Supabase account with vector extension enabled
- OpenAI API key
- The required Python packages (see requirements.txt)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables in `.env`:
```
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_supabase_key
OPENAI_API_KEY=your_openai_key
LLM_MODEL=gpt-4o-mini  # or your preferred model
```

4. Set up the Supabase database:
- Create a new Supabase project
- Run the SQL commands from `site_pages.sql` in the Supabase SQL editor

## Usage

### Ingesting Documentation

1. Run the ingestion script:
```bash
python ingest_docs.py
```

This will:
- Crawl all configured documentation sites
- Process any configured PDF files
- Store the chunks in Supabase with embeddings

To add new documentation sources or PDFs, edit the sources list in `ingest_docs.py`:

```python
sources = [
    DocumentSource(
        name="pydantic_ai_docs",
        sitemap_url="https://ai.pydantic.dev/sitemap.xml"
    ),
    # Add more documentation sources here
]

pdf_sources = [
    DocumentSource(
        name="local_pdfs",
        pdf_paths=[
            "docs/document1.pdf",
            "docs/document2.pdf"
        ]
    )
]
```

### Querying Documentation

1. Start the Streamlit UI:
```bash
streamlit run streamlit_ui.py
```

2. Open your browser to the provided URL (typically http://localhost:8501)

3. Enter your questions about any of the ingested documentation sources

The system will:
- Convert your question to an embedding
- Find the most relevant documentation chunks
- Use the Pydantic AI agent to generate a comprehensive answer

## Architecture

The system consists of several key components:

1. **Document Ingestion** (`ingest_docs.py`):
   - Crawls web documentation using Crawl4AI
   - Processes PDF files using PyPDF2
   - Chunks documents intelligently
   - Generates embeddings using OpenAI
   - Stores everything in Supabase

2. **Vector Storage** (Supabase):
   - Uses pgvector for efficient vector similarity search
   - Stores document chunks, metadata, and embeddings
   - Provides fast retrieval capabilities

3. **Agent Framework** (`pydantic_ai_expert.py`):
   - Built with Pydantic AI
   - Provides tools for documentation retrieval and querying
   - Generates coherent responses using the retrieved context

4. **User Interface** (`streamlit_ui.py`):
   - Built with Streamlit
   - Provides an intuitive chat interface
   - Displays responses in real-time

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 