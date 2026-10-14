"""
Central configuration shared across the RAG pipeline (ingestion, retrieval, generation).
Keeping everything in one place makes it much easier to tweak and experiment.
"""
# مكتبة بايثون الجاهزة للتعامل مع نظام التشغيل. استخدمناها بس عشان نقرا متغيرات البيئة (مفتاح Groq) بآخر الملف.
import os
# ============ General settings ============
# اسم المجلد اللي بتنحفظ فيه قاعدة البيانات على جهازك. أول ما تبني القاعدة بيتعمل مجلد بهالاسم جنب المشروع، وإذا بدك تبدأ من الصفر بتمسحه.
PERSIST_DIR = "chroma_db"
# اسم "الجدول" جوا قاعدة Chroma. لازم يكون نفسه عند البناء وعند التحميل، وإلا ما بتوصل لنفس البيانات.
COLLECTION_NAME = "wikipedia_rag"
# ============ Chunking settings ============
# الحد الأقصى لحجم كل جزء من المقالة، تقريباً 800 حرف
CHUNK_SIZE = 800
# كل جزء بياخد آخر 100 حرف من الجزء اللي قبله. فايدة التداخل إنه جملة مهمة ما بتنقطع نص نص وتضيع بين جزئين.
CHUNK_OVERLAP = 100
# ============ Model settings ============
# Embedding model: turns text into a numeric vector representing its meaning.
"""اسم النموذج اللي بيحوّل النص لأرقام تمثل معناه. الاسم مقسوم:
sentence-transformers/: الجهة اللي نشرته.
paraphrase-multilingual: بيفهم لغات كتير، بينها العربي.
MiniLM-L12-v2: حجمه صغير وسريع."""
# NOTE: if you change it, delete the chroma_db folder and rebuild the database.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# اسم النموذج اللي بيكتب الجواب النهائي، وهو شغّال عند Groq. Groq بتسحب موديلات من وقت للتاني (متل ما صار مع الموديل الأول)، فإذا طلع خطأ model_not_found، هاد السطر الوحيد اللي بتغيّره.
LLM_MODEL = "openai/gpt-oss-120b"
# درجة "الإبداع" من 0 لـ 1. كل ما قرّبت من الصفر، كان الجواب أكتر التزاماً بالنص. وبالـ RAG بدنا دقة، فاخترنا رقم صغير.
LLM_TEMPERATURE = 0.2
# عدد الأجزاء اللي بنجيبها لكل سؤال. أقل من هيك ممكن تفوتنا معلومة، وأكتر ممكن يتلخبط النموذج بمعلومات مش مرتبطة.
RETRIEVER_TOP_K = 4
# Wikipedia languages to try, in priority order.
# Arabic only — no fallback to English. If a topic doesn't exist in Arabic,
# it will simply be skipped (see the "No article found" message in ingest.py).
WIKI_LANGS = ["ar"]
def check_api_key():
    """
    Make sure the Groq API key is set before running anything that needs the LLM.
    Get a free key at https://console.groq.com/keys
    """
    if not os.environ.get("GROQ_API_KEY"):
        raise EnvironmentError(
            "You must set the GROQ_API_KEY environment variable (e.g. in your .env file).\n"
            "Get a free key at https://console.groq.com/keys"
        )
