#made by: sad3wnl 
import os
# =========  General settings ========
# Local directory where the persistent vector store (Chroma) is saved
PERSIST_DIR = "chroma_db"
# Name of the collection inside the Chroma database
COLLECTION_NAME = "wikipedia_rag"
# ============ Chunking settings ============
# Size of each text chunk (roughly in characters)
CHUNK_SIZE = 800
# Overlap between consecutive chunks, so we don't lose context at chunk boundaries
CHUNK_OVERLAP = 100
# =========== Model settings ============
# Embedding model: turns text into a numeric vector representing its meaning
EMBEDDING_MODEL = "text-embedding-3-small"
# LLM used to generate the final answer
LLM_MODEL = "gpt-4o-mini"
# Lower temperature = answers stick closer to the retrieved sources (less "creative")
LLM_TEMPERATURE = 0.2
# Number of chunks the retriever returns to build the answer from
RETRIEVER_TOP_K = 4
# Default Wikipedia language (  "en", "ar")
WIKI_LANGS = ["ar", "en"]
def check_api_key():
# Make sure the OpenAI API key is set before running anything that needs a model.
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "You must set the OPENAI_API_KEY environment variable before running this.\n"
            "Linux/Mac:   export OPENAI_API_KEY='sk-...'\n"
            "Windows PS:  $env:OPENAI_API_KEY='sk-...'"
        )
