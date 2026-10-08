"""
ingest.py
---------
Handles the first stages of the pipeline (offline — runs once, or whenever you want
to refresh the data):
  1. Fetch articles from Wikipedia
  2. Split them into small chunks          <- added in the next step
  3. Turn chunks into embeddings & store   <- added later
"""

import requests
import wikipedia
from langchain.schema import Document

from Config import WIKI_LANGS

# --- Patch for the old, unmaintained `wikipedia` package ---
# Two separate problems commonly hit with this library:
#   1. It hardcodes API_URL with "http://" (not "https://").
#   2. Its bare `requests.get()` call with a generic User-Agent sometimes gets
#      silently blocked or stripped by antivirus "web shield" features,
#      VPNs, or corporate firewalls — you get back a 200 response with an
#      EMPTY body, which fails to parse as JSON:
#      JSONDecodeError("Expecting value: line 1 column 1 (char 0)")
# Rather than trust the library's own networking code, we replace its
# internal `_wiki_request` function with our own: a real Session, a
# descriptive User-Agent, a timeout, and a forced HTTPS URL.
_session = requests.Session()
_session.headers.update({
    "User-Agent": "arabic-rag-wikipedia/1.0 (contact: you@example.com)",
    "Accept": "application/json",
})


def _patched_wiki_request(params):
    params["format"] = "json"
    if "action" not in params:
        params["action"] = "query"
    url = wikipedia.wikipedia.API_URL.replace("http://", "https://")
    response = _session.get(url, params=params, timeout=20)
    response.raise_for_status()
    return response.json()


wikipedia.wikipedia._wiki_request = _patched_wiki_request
# --- end patch ---


def _fetch_single_page(topic: str, lang: str):
    """
    Tries to fetch one article in one language.
    Returns the wikipedia page object, or None if it doesn't exist in this language.
    """
    wikipedia.set_lang(lang)
    try:
        return wikipedia.page(topic, auto_suggest=False)
    except wikipedia.exceptions.DisambiguationError as e:
        # The topic has multiple meanings — just take the first suggestion
        print(f"⚠️ '{topic}' is ambiguous in '{lang}', using: {e.options[0]}")
        return wikipedia.page(e.options[0], auto_suggest=False)
    except wikipedia.exceptions.PageError:
        return None


def fetch_wikipedia_pages(topics: list[str], langs: list[str] = WIKI_LANGS) -> list[Document]:
    """
    Takes a list of topics and returns a list of LangChain Documents.
    For each topic, tries the languages in `langs` in order (e.g. ["ar", "en"]) and
    stops at the first language where the article actually exists.
    """
    documents: list[Document] = []

    for topic in topics:
        page = None
        used_lang = None

        for lang in langs:
            page = _fetch_single_page(topic, lang)
            if page is not None:
                used_lang = lang
                break

        if page is None:
            print(f"❌ No article found for '{topic}' in any of {langs}")
            continue

        documents.append(
            Document(
                page_content=page.content,
                metadata={"source": page.title, "url": page.url, "lang": used_lang},
            )
        )
        print(f"✅ Fetched: {page.title} [{used_lang}] ({len(page.content)} chars)")

    return documents