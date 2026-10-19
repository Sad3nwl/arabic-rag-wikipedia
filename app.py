# os: للتعامل مع الملفات والمجلدات (بنستخدمها لنفحص إذا مجلد القاعدة موجود).
import os
# streamlit as st: مكتبة الواجهة. as st اختصار عشان نكتب st.title بدل streamlit.title.
import streamlit as st
# load_dotenv: دالة بتقرا ملف .env وبتحوّل محتواه لمتغيرات بيئة.
from dotenv import load_dotenv
# Load GROQ_API_KEY from a local .env file (if there is one)
# بتنفّذ القراءة فعلياً. لازم تيجي قبل استيراد ملفات المشروع، لأنه check_api_key بتدور على المفتاح بمتغيرات البيئة. إذا ما في ملف .env (مثل ما بالاستضافة)، ما بتكسّر، وبتاخد المفتاح من الـ Secrets.
load_dotenv()
# PERSIST_DIR: اسم مجلد القاعدة.
from Config import PERSIST_DIR
# fetch_wikipedia_pages: جلب المقالات.
from ingest import fetch_wikipedia_pages
# split_documents: التقطيع.
from chunking import split_documents
# build_vectorstore وload_vectorstore: بناء القاعدة أو فتح الموجودة.
from vectorstore import build_vectorstore, load_vectorstore
# build_rag_chain وask: تجهيز السلسلة وسؤالها.
from Ragchain import build_rag_chain, ask
# ---------- Page setup ----------
# بتحدد عنوان التبويب بالمتصفح، وأيقونته، وشكل التخطيط (centered يعني المحتوى بنص الشاشة). لازم تكون أول أمر Streamlit بالملف، وإلا بيطلع خطأ.
st.set_page_config(page_title="Wikipedia RAG", page_icon="📖", layout="centered")
# Make chat messages render right-to-left so Arabic text looks right
"""حقن CSS صغير بالصفحة:
[data-testid="stChatMessage"]: بيستهدف فقاعات الشات.
direction: rtl: اتجاه الكتابة من اليمين لليسار.
text-align: right: محاذاة لليمين.
unsafe_allow_html=True: بدونها Streamlit بيعرض الوسم كنص عادي بدل ما ينفّذه."""
st.markdown(
    """
    <style>
    [data-testid="stChatMessage"] { direction: rtl; text-align: right; }
    </style>
    """,
    unsafe_allow_html=True,
)
"""أهم فكرة بـ Streamlit: كل ما تضغط زر أو تكتب سؤال، الملف كله بيعيد التشغيل من الأول للآخر. فأي متغير عادي بيضيع. أما st.session_state فهو "ذاكرة" بتضل بين التشغيلات.
chain: السلسلة الجاهزة (فاضية None بالبداية).
messages: لستة المحادثة (فاضية بالبداية).
الشرط if ... not in بيمنع إعادة تصفيرهم كل مرة، فبنعمل القيمة الأولية بس أول مرة."""
if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
# If a database was already built in a previous run, load it automatically
"""٣ شروط لازم تتحقق مع بعض:
ما في سلسلة جاهزة بعد.
مجلد chroma_db موجود (isdir).
المجلد مش فاضي (os.listdir بترجع لستة، واللستة الفاضية بتعتبر False)."""
if st.session_state.chain is None and os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    # بنفتح القاعدة الموجودة (load_vectorstore) وبنبني السلسلة فوقها، وبنخزنها بالذاكرة. إذا صار أي خطأ (مثل خطأ الذاكرة os error 1455 اللي طلع معك)، بيعرض تحذير بدل ما يكسّر الصفحة كلها.
    try:
        st.session_state.chain = build_rag_chain(load_vectorstore())
    except Exception as e:
        st.warning(f"Found an existing database but couldn't load it: {e}")
# ---------- Sidebar: build the knowledge base ----------
# كل سطر مزاح جواها بيظهر بالشريط الجانبي.
with st.sidebar:
    st.header("📚 Knowledge base")
    # مربع نص متعدد الأسطر. value هو النص اللي بيظهر مسبقاً (الموضوع الافتراضي عندك)، وheight ارتفاعه بالبكسل. اللي بتكتبه بيتخزّن بـ topics_text.
    topics_text = st.text_area(
        "Wikipedia topics (one per line)",
        value="كريستيانو رونالدو",
        height=150,
    )
    # st.button بترجع True بس بالتشغيل اللي بعد الضغط مباشرة. type="primary" بيخلي الزر ملوّن.
    if st.button("Build knowledge base", type="primary"):
        """بيحوّل النص للستة مواضيع:
splitlines() بيقسم النص لأسطر.
t.strip() بيشيل المسافات من أطراف كل سطر.
if t.strip() بيتجاهل الأسطر الفاضية"""
        topics = [t.strip() for t in topics_text.splitlines() if t.strip()]
# إذا اللستة فاضية بيطلع تحذير. غير هيك بنكمل.
        if not topics:
            st.warning("Please enter at least one topic.")
        else:
            try:
                # st.spinner بيعرض دوّامة تحميل ورسالة طول ما الكود اللي جواه شغّال. هون بنجيب المقالات.
                with st.spinner("Fetching articles from Wikipedia..."):
                    docs = fetch_wikipedia_pages(topics)
# إذا ما انجابت ولا مقالة، رسالة خطأ. وإلا بنكمل.
                if not docs:
                    st.error("No articles were found for these topics.")
                else:
                    # بنقطّع، وبعدين بنحوّل لأرقام وبنخزّن. الخطوة التانية هي الأبطأ، خصوصاً أول مرة (تحميل النموذج).
                    with st.spinner("Splitting articles into chunks..."):
                        chunks = split_documents(docs)

                    with st.spinner("Creating embeddings and storing them..."):
                        vectorstore = build_vectorstore(chunks)
                        """بنبني السلسلة وبنخزنها بالذاكرة.
بنفضّي المحادثة القديمة (لأنه القاعدة تغيّرت).
بنعرض رسالة نجاح بعدد المقالات والأجزاء."""
                    st.session_state.chain = build_rag_chain(vectorstore)
                    st.session_state.messages = []
                    st.success(f"Indexed {len(docs)} article(s) into {len(chunks)} chunks.")
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                # سطر صغير توضيحي، وخط فاصل، وسطر اسمك. (هدول جوا الـ with st.sidebar فبيظهروا بالشريط الجانبي.)
    st.caption("New topics are added to the existing database, not replaced.")
    st.divider()
    st.caption("This project was made by Sadeen Abdelalrahman.")
# ---------- Main area: chat ---------
st.title("📖 Wikipedia RAG")
st.caption("Ask questions and get answers grounded in the Wikipedia articles you indexed.")
st.caption("Made by Sadeen Abdelalrahman(sad3nwl)")
# Show the conversation so far
# بنلف على كل رسالة محفوظة.
for msg in st.session_state.messages:
    # بيرسم فقاعة شات. msg["role"] إما "user" أو "assistant"، وStreamlit بيحط أيقونة مناسبة لكل واحد.
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        """msg.get("sources") بترجع None لرسائل المستخدم (ما فيها مصادر)، فالشرط ما بيتحقق.
st.expander بيعمل قسم قابل للطي.
لكل مصدر بنكتب سطر ماركداون على شكل رابط: - [العنوان](الرابط)."""
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- [{s['title']}]({s['url']})")
                    # إذا ما في سلسلة جاهزة، بنعرض رسالة بتقول "ابنِ القاعدة أول". وإلا بنعرض الشات.
if st.session_state.chain is None:
    st.info("Build a knowledge base from the sidebar to start asking questions.")
else:
    question = st.chat_input("Ask a question...")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    result = ask(st.session_state.chain, question)
                # The same article can appear several times (several chunks), so de-duplicate
                unique_sources = list({s["url"]: s for s in result["sources"]}.values())
                st.markdown(result["answer"])
                with st.expander("Sources"):
                    for s in unique_sources:
                        st.markdown(f"- [{s['title']}]({s['url']})")
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.messages.append(
                    {"role": "assistant", "content": result["answer"], "sources": unique_sources}
                )
            except Exception as e:
                st.error(f"Couldn't get an answer: {e}")
