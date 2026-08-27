"""
telemetry/tracker.py
Tracks latency, token usage, and estimated cost per investigation.

Stored in SQLite. Exposed via /telemetry/{investigation_id}.
"""

from __future__ import annotations
import sqlite3
import uuid
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

DB_PATH = Path(__file__).parent / "telemetry.db"

# Model pricing (approximate, as of 2024-2025)
COST_PER_1K_TOKENS = {
    "llama3-8b-8192": 0.00005,   # $0.05 per 1M tokens
    "llama3-70b-8192": 0.00059,
    "mixtral-8x7b-32768": 0.00024,
    "gemini-2.5-flash": 0.00015,  # $0.15 per 1M tokens blended
}


def _get_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS telemetry (
                id TEXT PRIMARY KEY,
                investigation_id TEXT,
                region_id TEXT,
                pipeline_latency_ms REAL,
                llm_latency_ms REAL,
                rag_latency_ms REAL,
                total_latency_ms REAL,
                tokens_prompt INTEGER,
                tokens_completion INTEGER,
                tokens_total INTEGER,
                model TEXT,
                llm_used INTEGER,
                estimated_cost_usd REAL,
                created_at TEXT,
                metadata TEXT
            )
        """)
        conn.commit()


class TelemetryTimer:
    """Context manager for timing pipeline stages."""
    def __init__(self):
        self.stages: Dict[str, float] = {}
        self._starts: Dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        self._starts[name] = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - self._starts[name]) * 1000
            self.stages[name] = round(elapsed, 2)

    def total(self) -> float:
        return round(sum(self.stages.values()), 2)


def record_telemetry(
    investigation_id: str,
    region_id: str,
    pipeline_latency_ms: float,
    llm_latency_ms: float = 0.0,
    rag_latency_ms: float = 0.0,
    tokens_prompt: int = 0,
    tokens_completion: int = 0,
    model: Optional[str] = None,
    llm_used: bool = False,
    metadata: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Records telemetry for an investigation run."""
    init_db()

    total_tokens = tokens_prompt + tokens_completion
    cost_per_1k = COST_PER_1K_TOKENS.get(model or "", 0.00005)
    estimated_cost = round(total_tokens / 1000 * cost_per_1k, 8)
    total_latency = pipeline_latency_ms + llm_latency_ms + rag_latency_ms

    telemetry_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO telemetry (
                id, investigation_id, region_id,
                pipeline_latency_ms, llm_latency_ms, rag_latency_ms, total_latency_ms,
                tokens_prompt, tokens_completion, tokens_total,
                model, llm_used, estimated_cost_usd, created_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                telemetry_id, investigation_id, region_id,
                pipeline_latency_ms, llm_latency_ms, rag_latency_ms, round(total_latency, 2),
                tokens_prompt, tokens_completion, total_tokens,
                model, int(llm_used), estimated_cost, now,
                json.dumps(metadata or {}),
            ),
        )
        conn.commit()

    return {
        "telemetry_id": telemetry_id,
        "investigation_id": investigation_id,
        "region_id": region_id,
        "latency": {
            "pipeline_ms": pipeline_latency_ms,
            "llm_ms": llm_latency_ms,
            "rag_ms": rag_latency_ms,
            "total_ms": round(total_latency, 2),
        },
        "tokens": {
            "prompt": tokens_prompt,
            "completion": tokens_completion,
            "total": total_tokens,
        },
        "model": model,
        "llm_used": llm_used,
        "estimated_cost_usd": estimated_cost,
        "created_at": now,
    }


def get_telemetry(investigation_id: str) -> Optional[Dict]:
    """Fetches telemetry for a specific investigation."""
    init_db()
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM telemetry WHERE investigation_id = ? ORDER BY created_at DESC LIMIT 1",
            (investigation_id,),
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    d["metadata"] = json.loads(d.get("metadata") or "{}")
    d["llm_used"] = bool(d["llm_used"])
    return d


def get_aggregate_telemetry() -> Dict[str, Any]:
    """Returns aggregate telemetry stats across all investigations."""
    init_db()
    with _get_connection() as conn:
        stats = conn.execute("""
            SELECT
                COUNT(*) as total_runs,
                AVG(total_latency_ms) as avg_latency_ms,
                MAX(total_latency_ms) as max_latency_ms,
                SUM(tokens_total) as total_tokens,
                SUM(estimated_cost_usd) as total_cost_usd,
                SUM(llm_used) as llm_used_count
            FROM telemetry
        """).fetchone()

    return dict(stats) if stats else {}
