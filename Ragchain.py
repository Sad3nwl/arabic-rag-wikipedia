"""given a user question, retrieve the most
relevant chunks from the vector store, then pass them to an LLM along
with the question so it can generate a grounded answer."""
# أداة اللي بتتصل بنموذج GPT (المحادثة) من OpenAI
from langchain_openai import ChatOpenAI
# "سلسلة" جاهزة من LangChain بتربط الـ retriever مع الـ LLM تلقائياً، بدون ما تكتب المنطق يدوياً
from langchain.chains import RetrievalQA
# – قالب جاهز نحط فيه التعليمات اللي بدنا نرسلها مع كل سؤال
from langchain.prompts import PromptTemplate
# اسم الموديل، درجة الحرارة، عدد الأجزاء المسترجعة، ودالة فحص الـ API key
from Config import LLM_MODEL, LLM_TEMPERATURE, RETRIEVER_TOP_K, check_api_key
