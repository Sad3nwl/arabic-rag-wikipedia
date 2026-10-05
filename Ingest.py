
"""Handles the first stages of the pipeline (offline — runs once, or whenever you want
to refresh the data):
1. Fetch articles from Wikipedia """
import wikipedia
from langchain.schema import Document
from Config import WIKI_LANGS
# جيب صفحة وحدة بلغة وحدة
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
#     جرب كل اللغات لموضوع معين لحد ما تلاقي نتيجة
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
            print(f"No article found for '{topic}' in any of {langs}")
            continue
        documents.append(
            Document(
                page_content=page.content,
                metadata={"source": page.title, "url": page.url, "lang": used_lang},
            )
        )
        print(f"Fetched: {page.title} [{used_lang}] ({len(page.content)} chars)")
    return documents
if __name__ == "__main__":
    # Quick sanity check before moving on
    sample_docs = fetch_wikipedia_pages(["الذكاء الاصطناعي", "Quantum supremacy"])
    for d in sample_docs:
        print("-", d.metadata["source"], f"[{d.metadata['lang']}]", "->", d.metadata["url"])
