"""Policy RAG: chunk the return-policy markdown, index it, retrieve the most relevant sections.

Retrieval is TF-IDF + cosine similarity, written in plain Python (no scikit-learn): deterministic,
no API, easy to explain, and it runs on locked-down machines that block compiled packages.
To upgrade to dense retrieval, replace ``Retriever.search`` with an embedding model + vector
store (Chroma / FAISS); the rest of the system only depends on the ``search`` contract.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

from . import config

STOPWORDS = frozenset("""
a about above after again all also am an and any are as at be because been before being below between both
but by can could did do does doing down during each few for from further had has have having he her here hers
him his how i if in into is it its itself just me more most my no nor not now of off on once only or other our
ours out over own same she should so some such than that the their theirs them then there these they this those
through to too under until up very was we were what when where which while who whom why will with would you your
""".split())

_TOKEN_RE = re.compile(r"\b\w\w+\b")


def tokenize(text: str) -> list[str]:
    """Lowercase words (2+ chars, stop words removed) plus adjacent-word bigrams."""
    words = [w for w in _TOKEN_RE.findall(text.lower()) if w not in STOPWORDS]
    return words + [f"{a} {b}" for a, b in zip(words, words[1:])]


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
        docs = [Counter(tokenize(f"{c['section']}. {c['text']}")) for c in chunks]
        n = len(docs)
        df = Counter(term for d in docs for term in d)
        # smoothed inverse document frequency: rare terms count more
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}
        self.vectors = [self._vectorize(d) for d in docs]

    def _vectorize(self, counts: Counter) -> dict[str, float]:
        """Sub-linear tf * idf, L2-normalised. Terms never seen in the policy are ignored."""
        vec = {t: (1 + math.log(c)) * self.idf[t] for t, c in counts.items() if t in self.idf}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return {t: v / norm for t, v in vec.items()} if norm else {}

    def search(self, query: str, k: int = 2) -> list[dict]:
        q = self._vectorize(Counter(tokenize(query)))
        scored = []
        for chunk, vec in zip(self.chunks, self.vectors):
            sim = sum(w * vec.get(t, 0.0) for t, w in q.items())  # cosine: both sides are unit length
            if sim > 0:
                scored.append((sim, chunk))
        scored.sort(key=lambda s: -s[0])
        return [{"section": c["section"], "text": c["text"], "score": round(sim, 3)} for sim, c in scored[:k]]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever(chunk_markdown(config.POLICY_PATH.read_text(encoding="utf-8")))


_WINDOW_RE = re.compile(r"-\s*([a-z_ ]+?):\s*(\d+)\s*days", re.I)


def parse_return_windows(text: str) -> dict[str, int]:
    """Extract '- electronics: 7 days' style lines from a retrieved chunk."""
    return {m.group(1).strip().lower(): int(m.group(2)) for m in _WINDOW_RE.finditer(text)}