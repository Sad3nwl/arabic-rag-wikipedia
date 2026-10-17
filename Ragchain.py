"""
 given a user question, retrieve the most
relevant chunks from the vector store, then pass them to an LLM along
with the question so it can generate a grounded answer.
"""
# الأداة اللي بتتواصل مع نماذج Groq (الـ LLM اللي بيكتب الجواب).
from langchain_groq import ChatGroq
# "سلسلة" جاهزة من LangChain بتربط البحث مع الـ LLM تلقائياً، فما بنكتب المنطق بإيدينا.
from langchain.chains import RetrievalQA
# قالب بنحط فيه التعليمات اللي بنرسلها للنموذج مع كل سؤال.
from langchain.prompts import PromptTemplate
# بنجيب من Config.py: اسم الموديل، درجة الإبداع، عدد الأجزاء اللي بنجيبها، ودالة فحص المفتاح.
from Config import LLM_MODEL, LLM_TEMPERATURE, RETRIEVER_TOP_K, check_api_key
# The prompt tells the LLM exactly how to behave: answer ONLY from the
# provided context, and admit it when the answer isn't there.
"""هاد نص خام (مش كود بيتنفذ)، وله ٣ أجزاء:
التعليمات: "جاوب بس من السياق اللي تحت، ولو الجواب مش موجود قول إنه ما عندك معلومات كافية ولا تخترع". هاد أهم سطر بالمشروع لأنه بيقلل الـ hallucination.
{context}: فراغ رح يتعبّى بالأجزاء اللي بيجيبها الـ retriever.
{question}: فراغ رح يتعبّى بسؤال المستخدم.
Answer: بنختم فيها عشان النموذج يكمّل بعدها بالجواب.
(تلات علامات تنصيص) بتسمح بنص على أكتر من سطر."""
PROMPT_TEMPLATE = """You are a helpful assistant that answers questions using
ONLY the context provided below. If the answer isn't in the context, say you
don't have enough information — do not make anything up.
Context:
{context}
Question: {question}
Answer:"""
# بتاخد قاعدة البيانات الجاهزة (الناتجة من vectorstore.py)، وبترجع "سلسلة" تقدر تسألها أسئلة.
def build_rag_chain(vectorstore):
    """
    Builds a question-answering chain on top of an existing vectorstore.
    Returns a chain object with a `.invoke({"query": ...})` method.
    """
 # بتتأكد إنه GROQ_API_KEY موجود. لو مش موجود بتوقف فوراً برسالة واضحة، بدل خطأ غامض لاحقاً.
    check_api_key()
    llm = ChatGroq(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_TOP_K})
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return chain
def ask(chain, question: str) -> dict:
    """
    Asks a question through the chain and returns a clean dict with
    the answer text and the list of sources it was based on.
    """
    result = chain.invoke({"query": question})
    sources = [
        {"title": doc.metadata.get("source"), "url": doc.metadata.get("url")}
        for doc in result["source_documents"]
    ]
    return {
        "answer": result["result"],
        "sources": sources, }
