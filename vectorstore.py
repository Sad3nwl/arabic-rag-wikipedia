from langchain.schema import Document
# الأداة اللي بتتصل بـ OpenAI وتحول أي نص لمتجه رقمي (vector)
from langchain_openai import OpenAIEmbeddings
# قاعدة بيانات المتجهات نفسها (محلية، بتتخزن على جهازك)
from langchain_community.vectorstores import Chroma
from Config import PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL, check_api_key