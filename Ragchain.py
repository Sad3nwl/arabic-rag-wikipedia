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
"""هون بنربط كل القطع اللي جهزناها مع بعض بسطر وحيد:
1)llm – مين رح يولّد الجواب
2)retriever – مين رح يجيب النصوص
3)chain_type="stuff" – يعني "خد كل الأجزاء المسترجعة واحشرها (stuff) كلها جوا الـ {context} دفعة وحدة" (أبسط استراتيجية، مناسبة لما الأجزاء مش كتار كتار)
4)return_source_documents=True – خليه يرجعلنا كمان من وين جاب المعلومة، مش بس الجواب
5)chain_type_kwargs={"prompt": prompt} – استخدم القالب اللي كتبناه إحنا، مش القالب الافتراضي لـ LangChain"""
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return chain
# دالة "واجهة بسيطة" — بتاخد السلسلة والسؤال، وبترجع نتيجة منظمة وواضحة، بدل ما تضطر تتعامل مع شكل الإخراج المعقّد تبع LangChain مباشرة.
def ask(chain, question: str) -> dict:
    # هون فعلياً بيصير كل شي: الاسترجاع + بناء الـ prompt + إرسالها للـ LLM + استقبال الجواب. السطر الوحيد اللي "بيشتغل" بكل الدالة.
    result = chain.invoke({"query": question})
    """result["source_documents"] فيها كل الأجزاء الأصلية (الـ Document objects) اللي استخدمها الـ LLM.
     هون بنلف عليهم ونسحب منهم بس اسم المقالة والرابط (مش النص الكامل)، ونحطهم بشكل نظيف كـ لستة dict."""
    sources = [
        {"title": doc.metadata.get("source"), "url": doc.metadata.get("url")}
        for doc in result["source_documents"]
    ]
    # بنرجّع dict بسيط فيه بس شيئين: نص الجواب، ولستة المصادر — هاد الشكل اللي رح تستخدمه مباشرة بواجهة Streamlit بعدين.
    return {
        "answer": result["result"],
        "sources": sources,
    }





