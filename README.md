<h1 align="center">Arabic RAG over Wikipedia</h1>

<p align="center">
  Ask questions about Arabic Wikipedia articles and get answers grounded in the source text, with a link to every article used.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" alt="LangChain" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Chroma-FF6446?style=for-the-badge" alt="Chroma" />
  <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Groq-F55036?style=for-the-badge" alt="Groq" />
  <img src="https://img.shields.io/badge/Wikipedia-000000?style=for-the-badge&logo=wikipedia&logoColor=white" alt="Wikipedia" />
</p>

A step-by-step **Retrieval-Augmented Generation (RAG)** app that fetches Arabic Wikipedia articles, chunks and embeds them, stores them in a Chroma vector database, and answers questions through a Streamlit chat interface. Every answer comes with links to the Wikipedia articles it was based on.

Built with **LangChain**, **Chroma**, **Streamlit**, local **Hugging Face embeddings**, and an LLM served by **Groq** (free tier, no credit card).

<!-- Add a screenshot of the app here:
![App screenshot](docs/screenshot.png)
-->

## Table of contents

- [Features](#features)
- [How it works](#how-it-works)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [Usage](#usage)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Author](#author)

## Features

- Fetches Arabic Wikipedia articles by topic
- Splits articles into overlapping chunks that respect paragraph and sentence boundaries
- Multilingual embeddings that run **locally and for free** (support Arabic)
- Persistent local vector store, so you only embed your data once
- Answers grounded in the retrieved text, with the model told to say so when the answer isn't there
- Source links shown under every answer
- Chat interface with right-to-left rendering for Arabic

## How it works

```
 Offline (once per set of topics)
 ────────────────────────────────
 Wikipedia ──► Chunking ──► Embeddings ──► Chroma vector store
 (ingest.py)  (chunking.py)  (vectorstore.py)  (chroma_db/)

 Online (every question)
 ───────────────────────
 Question ──► Retriever ──► LLM (Groq) ──► Answer + sources
 (app.py)    (top 4 chunks)  (rag_chain.py)
```

1. **Ingest:** fetch full articles from Wikipedia as LangChain `Document`s.
2. **Chunk:** split each article into ~800-character pieces with 100 characters of overlap.
3. **Embed:** turn each chunk into a vector that represents its meaning.
4. **Store:** save the vectors in a local Chroma database.
5. **Retrieve:** embed the question and find the closest chunks.
6. **Generate:** give the chunks and the question to the LLM and ask it to answer using only that context.

## Tech stack

| | Technology | Role in the project |
|---|---|---|
| <img src="https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" /> | **Python** | Programming language |
| <img src="https://img.shields.io/badge/-LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white" alt="LangChain" /> | **LangChain** | Orchestrates chunking, retrieval and the answer chain |
| <img src="https://img.shields.io/badge/-Chroma-FF6446?style=flat-square" alt="Chroma" /> | **Chroma** | Local vector database |
| <img src="https://img.shields.io/badge/-Sentence--Transformers-2F80ED?style=flat-square" alt="Sentence Transformers" /> | **Sentence Transformers** | Local multilingual embeddings (supports Arabic) |
| <img src="https://img.shields.io/badge/-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" /> | **PyTorch** | Runs the embedding model |
| <img src="https://img.shields.io/badge/-Groq-F55036?style=flat-square" alt="Groq" /> | **Groq** | Fast LLM inference (free tier) |
| <img src="https://img.shields.io/badge/-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit" /> | **Streamlit** | Chat interface |
  <img src="https://img.shields.io/badge/Wikipedia-000000?style=for-the-badge&logo=wikipedia&logoColor=white" alt="Wikipedia" /> | **Wikipedia** package | Source of the articles |

## Project structure

```
arabic-rag-wikipedia/
├── app.py            # Streamlit interface that ties everything together
├── config.py         # All settings in one place (models, chunk size, language...)
├── ingest.py         # Step 1: fetch articles from Wikipedia
├── chunking.py       # Step 2: split articles into chunks
├── vectorstore.py    # Steps 3-4: embeddings + Chroma storage
├── rag_chain.py      # Steps 5-6: retrieval + answer generation
├── requirements.txt  # Python dependencies
├── .gitignore
└── .env              # Your API key (not committed)
```

## Getting started

### Prerequisites

- Python 3.10 or newer
- A free Groq API key: create one at [console.groq.com/keys](https://console.groq.com/keys)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Sad3nwl/arabic-rag-wikipedia.git
cd arabic-rag-wikipedia

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows (PowerShell)
# source .venv/bin/activate       # macOS / Linux

# 3. Install dependencies
python -m pip install -r requirements.txt
```

> The first install downloads large packages (such as PyTorch), and the first time you build a knowledge base the embedding model (a few hundred MB) is downloaded once and cached.

### Configure your API key

Create a file named `.env` in the project folder:

```
GROQ_API_KEY=gsk_your_key_here
```

Never commit this file. It is already listed in `.gitignore`.

### Run

```bash
python -m streamlit run app.py
```

Then open <http://localhost:8501> if the browser doesn't open automatically.

## Usage

1. In the sidebar, enter one or more Wikipedia topics (one per line), for example `الذكاء الاصطناعي`.
2. Click **Build knowledge base** and wait for it to finish indexing.
3. Ask questions in the chat box. Each answer includes a **Sources** section with links to the articles used.

Topics you add later are appended to the existing database. To start from scratch, delete the `chroma_db/` folder.

## Configuration

Everything lives in [`config.py`](config.py):

| Setting | Default | Description |
|---|---|---|
| `WIKI_LANGS` | `["ar"]` | Wikipedia languages to try, in priority order (e.g. `["ar", "en"]` falls back to English) |
| `CHUNK_SIZE` | `800` | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `100` | Characters shared between consecutive chunks |
| `RETRIEVER_TOP_K` | `4` | Number of chunks retrieved per question |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Local multilingual embedding model |
| `LLM_MODEL` | `openai/gpt-oss-120b` | Model used to generate answers on Groq |
| `LLM_TEMPERATURE` | `0.2` | Lower values keep answers closer to the sources |
| `PERSIST_DIR` | `chroma_db` | Where the vector database is stored |

**Important:** if you change `EMBEDDING_MODEL`, delete the `chroma_db/` folder and rebuild the knowledge base. Vectors from different models are not compatible. Changing `LLM_MODEL` is safe at any time.

## Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| `JSONDecodeError: Expecting value` while fetching | The old `wikipedia` package uses plain HTTP and a bare request. `ingest.py` replaces its request function to fix this. If it persists, check for a VPN, firewall or antivirus blocking requests. |
| `429 Too Many Requests` from Wikipedia | You sent too many requests in a short time. `ingest.py` retries automatically; if it still fails, wait a few minutes and build the knowledge base once. |
| `model_not_found` / 404 from Groq | Groq retires models regularly. Pick a current one from [console.groq.com/docs/models](https://console.groq.com/docs/models) and update `LLM_MODEL`. |
| `429` from Groq | You reached the free-tier limit. Wait a moment and try again. |
| `os error 1455` (paging file too small) on Windows | Not enough memory to load PyTorch and the embedding model. Close heavy programs and increase the Windows virtual memory (paging file), then restart. |
| `ModuleNotFoundError` for a project file | File names must be lowercase with underscores: `config.py`, `ingest.py`, `rag_chain.py`, `vectorstore.py`, `chunking.py`. |
| `GROQ_API_KEY` error | The `.env` file is missing, in the wrong folder, or the variable name is wrong. Restart the app after editing it. |

## Author

<p align="center">
  Made by <a href="https://github.com/Sad3nwl"><b>Sadeen Abdelalrahman</b></a>
</p>
