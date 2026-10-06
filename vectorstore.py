from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from Config import PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL, check_api_key