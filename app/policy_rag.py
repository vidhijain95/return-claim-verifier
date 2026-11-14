"""Policy RAG: chunk the return-policy markdown, index it, retrieve the most relevant sections.

Retrieval here is TF-IDF + cosine similarity (no API, deterministic, easy to explain).
To upgrade to dense retrieval, replace ``Retriever.search`` with an embedding model + vector
store (Chroma / FAISS); the rest of the system only depends on the ``search`` contract.
"""
from __future__ import annotations

import re
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config


def chunk_markdown(text: str) -> list[dict]:
    """Split on '## ' headings. Each chunk = one policy section."""
    chunks: list[dict] = []
    title, buf = None, []
    for line in text.splitlines():
        if line.startswith("## "):
            if title is not None:
                chunks.append({"section": title, "text": "\n".join(buf).strip()})
            title, buf = line[3:].strip(), []
        elif title is not None:
            buf.append(line)
    if title is not None:
        chunks.append({"section": title, "text": "\n".join(buf).strip()})
    return chunks


class Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([f"{c['section']}. {c['text']}" for c in chunks])

    def search(self, query: str, k: int = 2) -> list[dict]:
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.matrix).ravel()
        order = sims.argsort()[::-1][:k]
        return [
            {"section": self.chunks[i]["section"], "text": self.chunks[i]["text"], "score": round(float(sims[i]), 3)}
            for i in order
            if sims[i] > 0
        ]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever(chunk_markdown(config.POLICY_PATH.read_text(encoding="utf-8")))


_WINDOW_RE = re.compile(r"-\s*([a-z_ ]+?):\s*(\d+)\s*days", re.I)


def parse_return_windows(text: str) -> dict[str, int]:
    """Extract '- electronics: 7 days' style lines from a retrieved chunk."""
    return {m.group(1).strip().lower(): int(m.group(2)) for m in _WINDOW_RE.finditer(text)}
