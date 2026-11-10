"""Guideline retrieval (RAG retrieval half). Modular + testable on its own.

Pipeline: ingest (txt/md/json/pdf) -> split into overlapping sentence-window chunks -> vectorise ->
cosine-similarity search. Vectoriser = TF-IDF (word 1-2 grams): no model download, deterministic,
fully offline. The `Vectorizer` is isolated so a sentence-embedding model can replace it later.
If the best score is below `min_score`, the result says INSUFFICIENT_EVIDENCE -- the system never
fabricates a guideline passage or citation: every returned passage is a real stored chunk with source id.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = os.path.dirname(__file__)


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text)) if s.strip()]


def chunk_text(text: str, source_id: str, prefix: str, window: int = 4, stride: int = 2) -> List[dict]:
    sents = split_sentences(text)
    out = []
    for i in range(0, max(1, len(sents) - window + stride), stride):
        part = " ".join(sents[i:i + window])
        if part:
            out.append({"chunk_id": f"{source_id}-{prefix}-{i}", "source_id": source_id, "text": part,
                        "provenance": "ingested document"})
    return out


def read_document(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:  # optional dependency
            raise RuntimeError("install pypdf to ingest PDFs") from e
        return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    with open(path, encoding="utf-8") as f:
        return f.read()


class GuidelineRetriever:
    def __init__(self, min_score: float = 0.12, sources: Optional[Dict[str, dict]] = None):
        self.min_score = min_score
        self.chunks: List[dict] = []
        self.sources = sources or {}
        self._vec: Optional[TfidfVectorizer] = None
        self._mat = None

    def add_chunks(self, chunks: List[dict]) -> None:
        self.chunks.extend(chunks)

    def add_document(self, path: str, source_id: str) -> int:
        new = chunk_text(read_document(path), source_id, os.path.basename(path))
        self.add_chunks(new)
        return len(new)

    @classmethod
    def default(cls) -> "GuidelineRetriever":
        from mapcos.knowledge_graph.graph import LocalGraph
        r = cls(sources=LocalGraph().sources)
        with open(os.path.join(HERE, "corpus", "seed_passages.json")) as f:
            r.add_chunks(json.load(f))
        r.build()
        return r

    def build(self) -> None:
        if not self.chunks:
            raise ValueError("no chunks to index")
        self._vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self._mat = self._vec.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query: str, k: int = 3) -> dict:
        if self._vec is None:
            self.build()
        q = self._vec.transform([query])
        scores = (self._mat @ q.T).toarray().ravel()
        order = np.argsort(-scores)[:k]
        hits = []
        for i in order:
            if scores[i] <= 0:
                continue
            c = self.chunks[i]
            src = self.sources.get(c["source_id"], {})
            hits.append({"chunk_id": c["chunk_id"], "score": round(float(scores[i]), 4), "text": c["text"],
                         "provenance": c.get("provenance"),
                         "citation": {"id": c["source_id"], "title": src.get("title"), "venue": src.get("venue"),
                                      "date": src.get("date"), "url": src.get("url")}})
        top = hits[0]["score"] if hits else 0.0
        sufficient = top >= self.min_score
        return {"query": query, "sufficient_evidence": sufficient,
                "status": "OK" if sufficient else "INSUFFICIENT_EVIDENCE",
                "passages": hits if sufficient else [],
                "message": None if sufficient else "No guideline passage matched this query with enough confidence."}
