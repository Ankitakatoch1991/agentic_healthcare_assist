"""Medical search for the brief: DuckDuckGo restricted to MedlinePlus or WHO, then FAISS RAG.

DuckDuckGoSearchRun.run() returns snippet text without URLs, so this uses that tool's
api_wrapper.results (the same client run() calls) and keeps the links for citations.
"""

import re

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from agent.prompts import RAG_PROMPT, fill, get_embeddings, get_llm, message_text


def _chunks(text: str, size: int = 500) -> list:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    return [text[i : i + size] for i in range(0, len(text), size)]


def _rows_for(topic: str) -> tuple:
    """Run DuckDuckGoSearchRun. Fall back to one site at a time if the OR query is empty."""
    tool = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper(time=None, max_results=5))
    queries = [
        f"{topic} site:medlineplus.gov OR site:who.int",
        f"{topic} site:medlineplus.gov",
        f"{topic} site:who.int",
    ]
    for query in queries:
        try:
            rows = tool.api_wrapper.results(query, max_results=5) or []
        except Exception:
            rows = []
        rows = [row for row in rows if row.get("snippet") or row.get("link")]
        if rows:
            return query, rows
    return queries[0], []


def medical_search(topic: str) -> dict:
    """Search, embed the snippets, retrieve top matches, and answer with citations."""
    query, rows = _rows_for(topic)
    if not rows:
        return {
            "answer": "No MedlinePlus or WHO pages were found for that topic.",
            "sources": [],
            "query": query,
        }
    documents = []
    sources = []
    for row in rows:
        title = row.get("title") or "Source"
        link = row.get("link") or ""
        sources.append({"title": title, "link": link})
        for chunk in _chunks(f"{title}. {row.get('snippet', '')}"):
            documents.append(Document(page_content=chunk, metadata={"source": link or title}))
    store = FAISS.from_documents(documents, get_embeddings())
    hits = store.similarity_search(topic, k=min(4, len(documents)))
    context = "\n\n".join(
        f"Source: {hit.metadata.get('source', '')}\n{hit.page_content}" for hit in hits
    )
    answer = message_text(get_llm().invoke(fill(RAG_PROMPT, question=topic, context=context)))
    return {"answer": answer, "sources": sources, "query": query}
