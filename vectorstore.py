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
    # بننشئ "المحوّل" — الأداة اللي بتاخد نص وترجعلك متجه رقمي يمثل معناه. لسا ما حولنا أي شي هون، بس جهزنا الأداة.
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    """Chroma بتسوي شغلتين تلقائياً:
بتاخد كل جزء من chunks، وبتستخدم embeddings تحوله لمتجه رقمي
بتخزن كل متجه + النص الأصلي + الـ metadata بقاعدة بيانات محلية، بمجلد اسمه chroma_db (من PERSIST_DIR)"""
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )