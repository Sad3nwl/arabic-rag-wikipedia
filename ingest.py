"""
Handles the first stages of the pipeline (offline — runs once, or whenever you want
to refresh the data):
  1. Fetch articles from Wikipedia
  2. Split them into small chunks          <- added in the next step
  3. Turn chunks into embeddings & store   <- added later
"""
"""requests: لإرسال طلبات للإنترنت.
wikipedia: المكتبة اللي بتجيب المقالات.
Document: القالب الموحّد اللي بيفهمه LangChain (نص + معلومات عنه).
WIKI_LANGS: لستة اللغات من ملف الإعدادات (عندك ["ar"]
"""
import requests
import wikipedia
from langchain.schema import Document
from Config import WIKI_LANGS
# بننشئ "جلسة" وحدة تُستخدم لكل الطلبات. فايدتها إنها بتعيد استخدام نفس الاتصال (أسرع)، وبتخلينا نحدد إعدادات مرة وحدة. الـ _ بأول الاسم معناها "للاستخدام الداخلي بهاد الملف".
_session = requests.Session()
"""بنضيف معلومات بتنرسل مع كل طلب:
User-Agent: بيعرّف ويكيبيديا مين اللي عم يطلب. هي بتطلب تعريف واضح، وهون فيه إيميل وهمي (you@example.com) والأفضل تحط رابط الريبو بدله.
Accept: application/json: بنقول للسيرفر "بدي الرد بصيغة JSON"
"""
_session.headers.update({
    "User-Agent": "arabic-rag-wikipedia/1.0 (contact: you@example.com)",
    "Accept": "application/json",
})
# دالة بتاخد params (قاموس فيه تفاصيل الطلب اللي المكتبة بتبنيها، مثل عنوان المقالة).
def _patched_wiki_request(params):
  # بنطلب من ويكيبيديا ترد بـ JSON.
    params["format"] = "json"
  # واجهة ويكيبيديا لازم تعرف شو العملية المطلوبة. إذا المكتبة ما حددتها، بنحط query (استعلام) كقيمة افتراضية
    if "action" not in params:
        params["action"] = "query"
      # المكتبة بتخزن عنوان الـ API بمتغير اسمه API_URL (مثلاً http://ar.wikipedia.org/w/api.php). بنقرأه وبنبدّل http بـ https عشان الاتصال يكون آمن.
    url = wikipedia.wikipedia.API_URL.replace("http://", "https://")
  """هون الطلب الفعلي:
get: نوع الطلب (جيب بيانات).
params: بتتحول لجزء من الرابط (?titles=...&format=json).
timeout=20: إذا ما رد السيرفر خلال 20 ثانية، بنوقف بدل ما يعلق للأبد.
"""
    response = _session.get(url, params=params, timeout=20)
# إذا كان الرد خطأ (مثل 429 أو 500)، بترمي خطأ واضح. بدونها كان الكود بيكمل ويحاول يقرأ رد فاضي، وبيطلع الخطأ الغامض JSONDecodeError.
    response.raise_for_status()
# بتحوّل الرد النصي لقاموس بايثون وبترجعه للمكتبة.
    return response.json()
# هاد "الترقيع" نفسه. المكتبة عندها دالة داخلية اسمها _wiki_request بتستخدمها لكل طلب، وإحنا بنبدّلها بدالتنا. لاحظ إنه ما في أقواس بعد _patched_wiki_request: بنخزّن الدالة نفسها، مش بننادي عليها.
wikipedia.wikipedia._wiki_request = _patched_wiki_request
# --- end patch ---
"""بتجرب تجيب موضوع واحد بلغة واحدة، وفي ٣ احتمالات:

نجحت: بترجع المقالة.
الاسم له أكتر من معنى: بتاخد أول اقتراح.
المقالة مش موجودة: بترجع None وما بتكسّر البرنامج."""
# بتاخد موضوع ولغة. الـ : str تلميح للقارئ عن نوع القيمة (ما بيأثر على التنفيذ).
def _fetch_single_page(topic: str, lang: str):
    """
    Tries to fetch one article in one language.
    Returns the wikipedia page object, or None if it doesn't exist in this language.
    """
    wikipedia.set_lang(lang)
    try:
      # الـ auto_suggest=False بيمنع المكتبة من "تخمين" عنوان قريب، فبتاخد بالضبط اللي كتبته.
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
# بتلف على كل موضوع، وبتجرب اللغات بالترتيب لحد ما تلاقي مقالة:
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
