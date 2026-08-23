"""
rag/retriever.py
Evidence retrieval using in-memory cosine similarity over sentence-transformer embeddings.

Retrieves support tickets and operational reports relevant to an investigation.
No FAISS — corpus is small enough for direct cosine similarity.

Design note: sentence-transformers model is loaded once at module level
and reused across requests.
"""

from __future__ import annotations
import os
import csv
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

DATA_DIR = Path(__file__).parent.parent / "data" / "generated"
TICKETS_PATH = DATA_DIR / "support_tickets.csv"

# Lazy-loaded model (only on first use)
_model = None
_corpus = []  # List of {id, text, region, category, embedding}


def _load_model():
    global _model
    if _model is None and HAS_SENTENCE_TRANSFORMERS:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _load_corpus():
    """Loads and embeds the support tickets corpus on first use."""
    global _corpus
    if _corpus:
        return _corpus

    if not TICKETS_PATH.exists():
        return []

    rows = []
    with open(TICKETS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    model = _load_model()
    if model is None:
        # Fallback: no embeddings, just keyword matching
        _corpus = [{"id": r["id"], "text": r["text"], "region": r["region"], "category": r["category"], "embedding": None} for r in rows]
        return _corpus

    texts = [r["text"] for r in rows]
    embeddings = model.encode(texts, normalize_embeddings=True)

    _corpus = [
        {
            "id": rows[i]["id"],
            "text": rows[i]["text"],
            "region": rows[i]["region"],
            "category": rows[i]["category"],
            "embedding": embeddings[i],
        }
        for i in range(len(rows))
    ]
    return _corpus


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized cosine similarity (vectors already L2-normalized)."""
    return float(np.dot(a, b))


def _keyword_match_score(query: str, text: str) -> float:
    """Simple keyword overlap score as fallback when embeddings unavailable."""
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    if not query_words:
        return 0.0
    return len(query_words & text_words) / len(query_words)


def retrieve_evidence(
    query: str,
    region_id: Optional[str] = None,
    top_k: int = 5,
    min_score: float = 0.3,
) -> Dict[str, Any]:
    """
    Retrieves most relevant support tickets for a given query.

    Args:
        query: Investigation context string (e.g. "warehouse staffing delay cancellation")
        region_id: optional filter to restrict to a specific region
        top_k: maximum number of results
        min_score: minimum similarity score to include

    Returns:
        - results: list of {id, text, region, category, score, evidence_type}
        - retrieval_method: "embedding" | "keyword"
        - query: str
    """
    corpus = _load_corpus()
    model = _load_model()

    if not corpus:
        return {
            "results": [],
            "retrieval_method": "none",
            "query": query,
            "note": "Support ticket corpus not found. Run data/generate_synthetic.py first.",
        }

    # Filter by region if specified
    candidates = corpus
    if region_id:
        candidates = [c for c in corpus if c.get("region") == region_id]
        if not candidates:
            candidates = corpus  # Fall back to all if region not found

    results = []

    if model is not None and candidates[0].get("embedding") is not None:
        # Embedding-based retrieval
        query_embedding = model.encode([query], normalize_embeddings=True)[0]
        for item in candidates:
            score = _cosine_similarity(query_embedding, item["embedding"])
            results.append({
                "id": item["id"],
                "text": item["text"],
                "region": item["region"],
                "category": item["category"],
                "score": round(score, 4),
                "evidence_type": "SUPPORTS" if score > 0.5 else "RELATED",
            })
        retrieval_method = "embedding"
    else:
        # Keyword fallback
        for item in candidates:
            score = _keyword_match_score(query, item["text"])
            results.append({
                "id": item["id"],
                "text": item["text"],
                "region": item["region"],
                "category": item["category"],
                "score": round(score, 4),
                "evidence_type": "SUPPORTS" if score > 0.4 else "RELATED",
            })
        retrieval_method = "keyword"

    # Sort by score and filter
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    results = [r for r in results if r["score"] >= min_score][:top_k]

    return {
        "results": results,
        "retrieval_method": retrieval_method,
        "query": query,
        "corpus_size": len(corpus),
        "candidates_searched": len(candidates),
    }
