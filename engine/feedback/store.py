"""
feedback/store.py
SQLite-backed feedback logging for human-in-the-loop improvement.

Schema:
  feedback(
    id TEXT PRIMARY KEY,
    investigation_id TEXT,
    region_id TEXT,
    persona_id TEXT,
    verdict TEXT,
    user_verdict TEXT,  -- what the human thinks the verdict should be
    driver_selected TEXT,  -- which root cause the human picks as correct
    rating TEXT,  -- 'correct' | 'incorrect' | 'partially_correct'
    comment TEXT,
    created_at TEXT
  )
"""

from __future__ import annotations
import sqlite3
import uuid
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

DB_PATH = Path(__file__).parent / "feedback.db"


def _get_connection():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Creates the feedback table if it doesn't exist."""
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                investigation_id TEXT,
                region_id TEXT,
                persona_id TEXT,
                verdict TEXT,
                user_verdict TEXT,
                driver_selected TEXT,
                rating TEXT,
                comment TEXT,
                created_at TEXT
            )
        """)
        conn.commit()


def store_feedback(
    investigation_id: str,
    region_id: str,
    persona_id: str,
    verdict: str,
    user_verdict: Optional[str] = None,
    driver_selected: Optional[str] = None,
    rating: str = "correct",
    comment: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Stores user feedback on an investigation result.

    Args:
        investigation_id: ID of the investigation being rated
        region_id: region the investigation covers
        persona_id: persona who submitted the feedback
        verdict: system-generated verdict
        user_verdict: what the user thinks the verdict should be
        driver_selected: KPI the user thinks is the real root cause
        rating: "correct" | "incorrect" | "partially_correct"
        comment: free-text comment

    Returns:
        The stored feedback record.
    """
    init_db()
    feedback_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO feedback (id, investigation_id, region_id, persona_id, verdict,
                                  user_verdict, driver_selected, rating, comment, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feedback_id,
                investigation_id,
                region_id,
                persona_id,
                verdict,
                user_verdict,
                driver_selected,
                rating,
                comment,
                now,
            ),
        )
        conn.commit()

    return {
        "feedback_id": feedback_id,
        "investigation_id": investigation_id,
        "region_id": region_id,
        "persona_id": persona_id,
        "verdict": verdict,
        "user_verdict": user_verdict,
        "driver_selected": driver_selected,
        "rating": rating,
        "comment": comment,
        "created_at": now,
    }


def get_feedback_stats() -> Dict[str, Any]:
    """Returns aggregate feedback statistics."""
    init_db()
    with _get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        by_rating = conn.execute(
            "SELECT rating, COUNT(*) as count FROM feedback GROUP BY rating"
        ).fetchall()
        by_region = conn.execute(
            "SELECT region_id, COUNT(*) as count FROM feedback GROUP BY region_id"
        ).fetchall()

    return {
        "total_feedback": total,
        "by_rating": {row["rating"]: row["count"] for row in by_rating},
        "by_region": {row["region_id"]: row["count"] for row in by_region},
    }


def get_feedback_for_investigation(investigation_id: str) -> List[Dict]:
    """Returns all feedback for a specific investigation."""
    init_db()
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM feedback WHERE investigation_id = ? ORDER BY created_at DESC",
            (investigation_id,),
        ).fetchall()
    return [dict(row) for row in rows]
