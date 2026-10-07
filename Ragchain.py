"""given a user question, retrieve the most
relevant chunks from the vector store, then pass them to an LLM along
with the question so it can generate a grounded answer."""
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from Config import LLM_MODEL, LLM_TEMPERATURE, RETRIEVER_TOP_K, check_api_key
