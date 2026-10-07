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
# هاد بالظبط اللي بيمنع الـ LLM إنه "يخترع" معلومات مش موجودة عندنا.
PROMPT_TEMPLATE = """You are a helpful assistant that answers questions using
ONLY the context provided below. If the answer isn't in the context, say you
don't have enough information — do not make anything up.
Context:
{context}
Question: {question}
Answer:"""
# بتاخد قاعدة البيانات الجاهزة (الناتجة من vectorstore.py)، وبترجع "سلسلة" (chain) جاهزة تقدر تسألها أسئلة مباشرة.
def build_rag_chain(vectorstore):
    check_api_key()
    # بنجهز نموذج المحادثة. temperature=0.2 (من config.py) معناها: خليه "محافظ" بإجاباته، يلتزم بالنص المعطى بدل ما "يبدع" أو يحيد عنه.
    llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    # بنحول قاعدة البيانات لـ "باحث" (retriever) — أداة وظيفتها الوحيدة: تاخد سؤال وترجع أقرب k أجزاء له (٤ أجزاء، من RETRIEVER_TOP_K).
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})
    # بنحول النص الخام اللي كتبناه فوق لـ "قالب" رسمي تقدر LangChain تتعامل معه، وبنحدد صراحة: "هاد القالب بياخد متغيرين: context و question".
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )




