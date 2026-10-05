"""
chunking.py
-----------
Step 2 of the pipeline: takes full-length articles (as LangChain Documents,
usually coming from ingest.py) and splits them into small, overlapping chunks
that are easier and cheaper to embed and search later.
"""
from langchain.schema import Document
from Config import CHUNK_SIZE, CHUNK_OVERLAP
# التقطيع الفعلي RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
def split_documents(documents: list[Document]) -> list[Document]:
    """
    Splits full-length articles into small overlapping chunks.
    Each resulting chunk is still a Document, and inherits the same
    metadata (source, url, lang) as the article it came from.
    """
    splitter = RecursiveCharacterTextSplitter(
        # الحد الأقصى لعدد الحروف بكل جزء
        chunk_size=CHUNK_SIZE,
        # كل جزء بياخد آخر ١٠٠ حرف من الجزء اللي قبله (عشان ما نقطع جملة مهمة نص نص وتضيع)
        chunk_overlap=CHUNK_OVERLAP,
        #ترتيب أولوية "أماكن القطع المفضلة": فقرة كاملة (\n\n) → سطر جديد (\n) → نهاية جملة (. ) → مسافة (" ") → أي مكان ("").
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    # تقطع النص حسب الـ separators
    # تنسخ نفس الـ metadata تبع كل مقالة أصلية لكل جزء طلع منها — هيك ما بتفقد معرفة "هاد الجزء جاي من وين"
    chunks = splitter.split_documents(documents)
    # طباعة بس عشان تتابع (كم مقالة دخلت، كم جزء طلع)، وبعدين إرجاع اللستة النهائية للي استدعى الدالة.
    print(f" Split {len(documents)} article(s) into {len(chunks)} chunk(s)")
    return chunks