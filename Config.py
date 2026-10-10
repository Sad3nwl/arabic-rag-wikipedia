"""
Central configuration shared across the RAG pipeline (ingestion, retrieval, generation).
Keeping everything in one place makes it much easier to tweak and experiment.
"""
import os
# ============ General settings ============
# Local directory where the persistent vector store (Chroma) is saved
PERSIST_DIR = "chroma_db"
# Name of the collection inside the Chroma database
COLLECTION_NAME = "wikipedia_rag"
# ============ Chunking settings ============
# Size of each text chunk (roughly in characters)
CHUNK_SIZE = 800
# Overlap between consecutive chunks, so we don't lose context at chunk boundaries
CHUNK_OVERLAP = 100
# ============ Model settings ============
# Embedding model: turns text into a numeric vector representing its meaning.
# Runs locally on your machine (free, no API key). This model supports Arabic.
# NOTE: if you change it, delete the chroma_db folder and rebuild the database.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# LLM used to generate the final answer (served by Groq, free tier)
LLM_MODEL = "openai/gpt-oss-120b"
# Lower temperature = answers stick closer to the retrieved sources (less "creative")
LLM_TEMPERATURE = 0.2
# Number of chunks the retriever returns to build the answer from
RETRIEVER_TOP_K = 4
# Wikipedia languages to try, in priority order.
# Arabic only — no fallback to English. If a topic doesn't exist in Arabic,
# it will simply be skipped (see the "No article found" message in ingest.py).
WIKI_LANGS = ["ar"]
def check_api_key():
    """
    Make sure the Groq API key is set before running anything that needs the LLM.
    Get a free key at https://console.groq.com/keys
    """
    if not os.environ.get("GROQ_API_KEY"):
        raise EnvironmentError(
            "You must set the GROQ_API_KEY environment variable (e.g. in your .env file).\n"
            "Get a free key at https://console.groq.com/keys"
        )