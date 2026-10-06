from langchain.schema import Document
# الأداة اللي بتتصل بـ OpenAI وتحول أي نص لمتجه رقمي (vector)
from langchain_openai import OpenAIEmbeddings
# قاعدة بيانات المتجهات نفسها (محلية، بتتخزن على جهازك)
from langchain_community.vectorstores import Chroma
# مكان التخزين، اسم المجموعة، اسم نموذج الـ embedding، ودالة التأكد من وجود API key
from Config import PERSIST_DIR, COLLECTION_NAME, EMBEDDING_MODEL, check_api_key

def build_vectorstore(chunks: list[Document]) -> Chroma:
    """
       Embeds each chunk and stores it in a persistent local Chroma database.
       Returns the vectorstore object, ready to be queried right away.
       بتاخد الأجزاء (من chunking.py)، وبترجع قاعدة بيانات جاهزة للبحث فيها.
       """
    check_api_key()