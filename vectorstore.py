"""
vectorstore.py
--------------
Steps 3 & 4 of the pipeline: turns text chunks into embeddings (numeric
vectors that represent meaning, computed locally for free), and stores them in a local Chroma
vector database so they can be searched later.
"""

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from Config import PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL
def build_vectorstore(chunks: list[Document]) -> Chroma:
    """
    Embeds each chunk and stores it in a persistent local Chroma database.
    Returns the vectorstore object, ready to be queried right away.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    print(f"💾 Stored {len(chunks)} chunk(s) in Chroma at '{PERSIST_DIR}'")
    return vectorstore
def load_vectorstore() -> Chroma:
    """
    Loads an existing Chroma database from disk, without re-embedding
    anything. Use this on later runs, once build_vectorstore() has
    already been called once.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    print(f"📂 Loaded existing Chroma database from '{PERSIST_DIR}'")
    return vectorstore