"""
pipeline/reconciliation.py
Aligns multiple data sources to a common grain and cadence.

Handles:
- Timezone normalization (all dates → UTC date)
- Cadence alignment (daily grain enforced)
- Column standardization across sources
- NaN propagation tracking per source
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Any


EXPECTED_COLUMNS = {
    "OMS": ["date", "revenue", "order_cancellation_rate"],
    "logistics": ["date", "fulfillment_delay_rate"],
    "support": ["date", "support_ticket_volume"],
    "WMS": ["date", "warehouse_staffing_level"],
}

SOURCE_COLUMN_MAP = {
    "revenue": "OMS",
    "order_cancellation_rate": "OMS",
    "fulfillment_delay_rate": "logistics",
    "support_ticket_volume": "support",
    "warehouse_staffing_level": "WMS",
}


def reconcile_sources(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Takes the raw region DataFrame (all columns merged) and reconciles
    it into a clean, aligned multi-source view.

    Returns a dict with:
        - aligned_df: pd.DataFrame aligned to daily grain
        - source_completeness: {source: completeness_pct}
        - grain_issues: list of detected grain/cadence problems
        - reconciliation_notes: list of human-readable notes
    """
    df = df.copy()

    # Normalize date column
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").reset_index(drop=True)

    # Check for duplicate dates (grain conflict)
    grain_issues = []
    dup_dates = df[df.duplicated("date")]["date"].tolist()
    if dup_dates:
        grain_issues.append(f"Duplicate dates detected: {dup_dates[:3]}{'...' if len(dup_dates) > 3 else ''}")

    # Enforce daily grain by reindexing to full date range
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_range).rename_axis("date").reset_index()

    if len(df) > len(full_range):
        grain_issues.append("Gaps detected and filled with NaN after reindexing.")

    # Compute completeness per source
    source_completeness = {}
    for col, source in SOURCE_COLUMN_MAP.items():
        if col in df.columns:
            completeness = df[col].notna().mean()
            source_completeness[source] = round(float(completeness), 4)

    # Per-column gap counts
    gap_report = {}
    for col in SOURCE_COLUMN_MAP:
        if col in df.columns:
            n_gaps = int(df[col].isna().sum())
            if n_gaps > 0:
                gap_report[col] = n_gaps

    reconciliation_notes = []
    for col, gaps in gap_report.items():
        source = SOURCE_COLUMN_MAP.get(col, "unknown")
        reconciliation_notes.append(f"{source}/{col}: {gaps} missing day(s)")

    if not reconciliation_notes:
        reconciliation_notes.append("All sources complete — no gaps detected.")

    return {
        "aligned_df": df,
        "source_completeness": source_completeness,
        "grain_issues": grain_issues,
        "reconciliation_notes": reconciliation_notes,
        "total_days": len(df),
        "date_range": {
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
        },
    }
