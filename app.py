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
st.set_page_config(page_title="Wikipedia RAG", page_icon="📖", layout="centered")
# Make chat messages render right-to-left so Arabic text looks right
st.markdown(
    """
    <style>
    [data-testid="stChatMessage"] { direction: rtl; text-align: right; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "chain" not in st.session_state:
    st.session_state.chain = None
if "messages" not in st.session_state:
    st.session_state.messages = []
# If a database was already built in a previous run, load it automatically
if st.session_state.chain is None and os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    try:
        st.session_state.chain = build_rag_chain(load_vectorstore())
    except Exception as e:
        st.warning(f"Found an existing database but couldn't load it: {e}")
# ---------- Sidebar: build the knowledge base ----------
with st.sidebar:
    st.header("📚 Knowledge base")
    topics_text = st.text_area(
        "Wikipedia topics (one per line)",
        value="كريستيانو رونالدو",
        height=150,
    )
    
    if st.button("Build knowledge base", type="primary"):
        topics = [t.strip() for t in topics_text.splitlines() if t.strip()]

        if not topics:
            st.warning("Please enter at least one topic.")
        else:
            try:
                with st.spinner("Fetching articles from Wikipedia..."):
                    docs = fetch_wikipedia_pages(topics)

                if not docs:
                    st.error("No articles were found for these topics.")
                else:
                    with st.spinner("Splitting articles into chunks..."):
                        chunks = split_documents(docs)

                    with st.spinner("Creating embeddings and storing them..."):
                        vectorstore = build_vectorstore(chunks)
                    st.session_state.chain = build_rag_chain(vectorstore)
                    st.session_state.messages = []
                    st.success(f"Indexed {len(docs)} article(s) into {len(chunks)} chunks.")
            except Exception as e:
                st.error(f"Something went wrong: {e}")
    st.caption("New topics are added to the existing database, not replaced.")
    st.divider()
    st.caption("This project was made by Sadeen Abdelalrahman.")
# ---------- Main area: chat ---------
st.title("📖 Wikipedia RAG")
st.caption("Ask questions and get answers grounded in the Wikipedia articles you indexed.")
st.caption("Made by Sadeen Abdelalrahman(sad3nwl)")
# Show the conversation so far
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- [{s['title']}]({s['url']})")
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
