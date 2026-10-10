"""
 given a user question, retrieve the most
relevant chunks from the vector store, then pass them to an LLM along
with the question so it can generate a grounded answer.
"""
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from Config import LLM_MODEL, LLM_TEMPERATURE, RETRIEVER_TOP_K, check_api_key
# The prompt tells the LLM exactly how to behave: answer ONLY from the
# provided context, and admit it when the answer isn't there.
PROMPT_TEMPLATE = """You are a helpful assistant that answers questions using
ONLY the context provided below. If the answer isn't in the context, say you
don't have enough information — do not make anything up.
Context:
{context}
Question: {question}
Answer:"""
def build_rag_chain(vectorstore):
    """
    Builds a question-answering chain on top of an existing vectorstore.
    Returns a chain object with a `.invoke({"query": ...})` method.
    """
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
        "sources": sources,
    }