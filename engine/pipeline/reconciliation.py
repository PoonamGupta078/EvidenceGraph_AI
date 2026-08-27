"""
pipeline/reconciliation.py

Performs ID-based joins and aggregations across heterogeneous enterprise
sources and aligns them to a common daily grain for a specific region.

Sources:
- OMS: order-level
- TMS/Logistics: shipment-level
- WMS: SKU x warehouse x day level
- Support: ticket-level
- Marketing: campaign-level

Output:
- One reconciled daily DataFrame
- Source completeness
- Grain/data-quality issues
- Reconciliation notes
- Metric lineage
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


REGION_WAREHOUSES = {
    "region_a": "WH-A",
    "region_b": "WH-B",
    "region_c": "WH-C",
    "region_d": "WH-D",
    "region_e": "WH-E",
}


SOURCE_COMPLETENESS_MAP = {
    "OMS": [
        "revenue",
        "order_cancellation_rate",
    ],
    "logistics": [
        "fulfillment_delay_rate",
    ],
    "WMS": [
        "warehouse_staffing_level",
    ],
    "support": [
        "support_ticket_volume",
    ],
    "marketing": [
        "marketing_spend",
        "seasonal_index",
    ],
}


def _require_columns(
    df: pd.DataFrame,
    required: list[str],
    source_name: str,
) -> None:
    """Raise a clear error when required source columns are missing."""

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{source_name} is missing required column(s): "
            f"{', '.join(missing)}"
        )


def reconcile_sources(
    oms_df: pd.DataFrame,
    logistics_df: pd.DataFrame,
    wms_df: pd.DataFrame,
    support_df: pd.DataFrame,
    marketing_df: pd.DataFrame,
    region_id: str,
) -> Dict[str, Any]:
    """
    Reconcile heterogeneous source data into a daily region-level view.

    The function performs:
        1. Region/warehouse filtering
        2. ID-based joins where required
        3. Source-specific aggregation
        4. Daily calendar alignment
        5. Completeness and grain checks
        6. Metric lineage construction

    Args:
        oms_df: Order Management System data.
        logistics_df: Shipment/logistics data.
        wms_df: Warehouse Management System data.
        support_df: Support ticket data.
        marketing_df: Marketing data.
        region_id: Region identifier such as "region_a".

    Returns:
        Dictionary containing the reconciled daily DataFrame and
        reconciliation metadata.
    """

    # ------------------------------------------------------------------
    # 0. Validate region
    # ------------------------------------------------------------------

    if region_id not in REGION_WAREHOUSES:
        raise ValueError(
            f"Unknown region_id '{region_id}'. "
            f"Expected one of: {list(REGION_WAREHOUSES)}"
        )

    wh_id = REGION_WAREHOUSES[region_id]

    # ------------------------------------------------------------------
    # 1. Validate required source columns
    # ------------------------------------------------------------------

    _require_columns(
        oms_df,
        [
            "region_id",
            "order_id",
            "order_date",
            "quantity",
            "unit_price",
            "discount",
            "cancelled",
        ],
        "OMS",
    )

    _require_columns(
        logistics_df,
        [
            "order_id",
            "shipment_id",
            "warehouse_id",
            "promised_date",
            "actual_delivery_date",
        ],
        "Logistics",
    )

    _require_columns(
        wms_df,
        [
            "warehouse_id",
            "product_id",
            "date",
            "staffing",
        ],
        "WMS",
    )

    _require_columns(
        support_df,
        [
            "order_id",
            "ticket_id",
            "ticket_date",
        ],
        "Support",
    )

    _require_columns(
        marketing_df,
        [
            "region_id",
            "date",
            "marketing_spend",
            "seasonal_index",
        ],
        "Marketing",
    )

    # ------------------------------------------------------------------
    # 2. Filter each source to the requested region
    # ------------------------------------------------------------------

    oms_filtered = oms_df[
        oms_df["region_id"] == region_id
    ].copy()

    wms_filtered = wms_df[
        wms_df["warehouse_id"] == wh_id
    ].copy()

    logistics_filtered = logistics_df[
        logistics_df["warehouse_id"] == wh_id
    ].copy()

    marketing_filtered = marketing_df[
        marketing_df["region_id"] == region_id
    ].copy()

    # Support has no explicit region field in the stated source contract.
    # Therefore link support records to the region through OMS order_id.
    oms_orders_set = set(
        oms_filtered["order_id"]
        .dropna()
        .unique()
    )

    support_filtered = support_df[
        support_df["order_id"].isin(oms_orders_set)
    ].copy()

    # ------------------------------------------------------------------
    # 3. Convert dates
    # ------------------------------------------------------------------

    oms_filtered["order_date"] = pd.to_datetime(
        oms_filtered["order_date"],
        errors="coerce",
    )

    wms_filtered["date"] = pd.to_datetime(
        wms_filtered["date"],
        errors="coerce",
    )

    logistics_filtered["promised_date"] = pd.to_datetime(
        logistics_filtered["promised_date"],
        errors="coerce",
    )

    logistics_filtered["actual_delivery_date"] = pd.to_datetime(
        logistics_filtered["actual_delivery_date"],
        errors="coerce",
    )

    support_filtered["ticket_date"] = pd.to_datetime(
        support_filtered["ticket_date"],
        errors="coerce",
    )

    marketing_filtered["date"] = pd.to_datetime(
        marketing_filtered["date"],
        errors="coerce",
    )

    # ------------------------------------------------------------------
    # 4. WMS -> daily staffing
    # ------------------------------------------------------------------
    #
    # Source grain:
    # warehouse x product x day
    #
    # We must aggregate all product records for the warehouse/day.
    # Dropping duplicates by warehouse/date would incorrectly discard
    # product-level observations.
    # ------------------------------------------------------------------

    wms_valid = wms_filtered.dropna(
        subset=["date", "staffing"]
    ).copy()

    wms_daily = (
        wms_valid
        .groupby("date")["staffing"]
        .mean()
        .rename("warehouse_staffing_level")
    )

    # ------------------------------------------------------------------
    # 5. OMS -> daily metrics
    # ------------------------------------------------------------------

    oms_valid = oms_filtered.dropna(
        subset=["order_date", "order_id"]
    ).copy()

    # Revenue contribution at order level.
    oms_valid["order_revenue"] = (
        pd.to_numeric(
            oms_valid["quantity"],
            errors="coerce",
        )
        * pd.to_numeric(
            oms_valid["unit_price"],
            errors="coerce",
        )
        - pd.to_numeric(
            oms_valid["discount"],
            errors="coerce",
        )
    )

    # Completed-order revenue.
    oms_completed = oms_valid[
        oms_valid["cancelled"] == 0
    ]

    revenue_daily = (
        oms_completed
        .groupby("order_date")["order_revenue"]
        .sum()
        .rename("revenue")
    )

    # Cancellation rate.
    cancel_daily = (
        oms_valid
        .groupby("order_date")["cancelled"]
        .mean()
        .mul(100.0)
        .rename("order_cancellation_rate")
    )

    # Average unit price for PVM.
    unit_price_daily = (
        oms_valid
        .groupby("order_date")["unit_price"]
        .mean()
        .rename("unit_price")
    )

    # Total quantity for PVM volume analysis.
    quantity_daily = (
        oms_valid
        .groupby("order_date")["quantity"]
        .sum()
        .rename("quantity")
    )

    # ------------------------------------------------------------------
    # 6. Logistics -> daily fulfillment delay
    # ------------------------------------------------------------------
    #
    # Logistics is shipment-level, while the pipeline uses OMS order_date
    # as the common analytical date basis.
    #
    # Join using order_id.
    # ------------------------------------------------------------------

    oms_order_dates = (
        oms_valid[
            ["order_id", "order_date"]
        ]
        .drop_duplicates("order_id")
    )

    logistics_merged = logistics_filtered.merge(
        oms_order_dates,
        on="order_id",
        how="inner",
        validate="many_to_one",
    )

    # A shipment can only be classified as delayed when both dates exist.
    valid_delivery_dates = (
        logistics_merged["actual_delivery_date"].notna()
        & logistics_merged["promised_date"].notna()
    )

    logistics_merged["is_delayed"] = np.nan

    logistics_merged.loc[
        valid_delivery_dates,
        "is_delayed",
    ] = (
        logistics_merged.loc[
            valid_delivery_dates,
            "actual_delivery_date",
        ]
        > logistics_merged.loc[
            valid_delivery_dates,
            "promised_date",
        ]
    ).astype(float)

    delay_daily = (
        logistics_merged
        .dropna(subset=["order_date", "is_delayed"])
        .groupby("order_date")["is_delayed"]
        .mean()
        .mul(100.0)
        .rename("fulfillment_delay_rate")
    )

    # ------------------------------------------------------------------
    # 7. Support -> daily ticket volume
    # ------------------------------------------------------------------

    support_daily = (
        support_filtered
        .dropna(subset=["ticket_date", "ticket_id"])
        .groupby("ticket_date")["ticket_id"]
        .nunique()
        .rename("support_ticket_volume")
    )

    # ------------------------------------------------------------------
    # 8. Marketing -> daily metrics
    # ------------------------------------------------------------------

    marketing_valid = marketing_filtered.dropna(
        subset=["date"]
    ).copy()

    marketing_daily = (
        marketing_valid
        .groupby("date")["marketing_spend"]
        .sum()
        .rename("marketing_spend")
    )

    seasonal_daily = (
        marketing_valid
        .groupby("date")["seasonal_index"]
        .mean()
        .rename("seasonal_index")
    )

    # ------------------------------------------------------------------
    # 9. Construct common daily calendar
    # ------------------------------------------------------------------

    source_indices = [
        wms_daily.index,
        revenue_daily.index,
        delay_daily.index,
        support_daily.index,
        marketing_daily.index,
    ]

    non_empty_indices = [
        index
        for index in source_indices
        if len(index) > 0
    ]

    if not non_empty_indices:
        # Do not invent dates when there is no source data.
        aligned_df = pd.DataFrame(
            columns=[
                "date",
                "revenue",
                "order_cancellation_rate",
                "fulfillment_delay_rate",
                "support_ticket_volume",
                "warehouse_staffing_level",
                "marketing_spend",
                "seasonal_index",
                "unit_price",
                "quantity",
            ]
        )

        return {
            "aligned_df": aligned_df,
            "source_completeness": {
                source: 0.0
                for source in SOURCE_COMPLETENESS_MAP
            },
            "grain_issues": [],
            "reconciliation_notes": [
                "No usable records found across the supplied sources."
            ],
            "metric_lineage": _metric_lineage(),
            "total_days": 0,
            "date_range": {
                "start": None,
                "end": None,
            },
        }

    global_start = min(
        index.min()
        for index in non_empty_indices
    )

    global_end = max(
        index.max()
        for index in non_empty_indices
    )

    date_range_idx = pd.date_range(
        start=global_start,
        end=global_end,
        freq="D",
    )

    aligned_df = (
        pd.DataFrame(index=date_range_idx)
        .rename_axis("date")
    )

    # ------------------------------------------------------------------
    # 10. Join daily metrics
    # ------------------------------------------------------------------

    daily_series = [
        revenue_daily,
        cancel_daily,
        delay_daily,
        support_daily,
        wms_daily,
        marketing_daily,
        seasonal_daily,
        unit_price_daily,
        quantity_daily,
    ]

    for daily_metric in daily_series:
        aligned_df = aligned_df.join(
            daily_metric,
            how="left",
        )

    aligned_df = aligned_df.reset_index()

    # ------------------------------------------------------------------
    # 11. Source completeness
    # ------------------------------------------------------------------

    source_completeness: Dict[str, float] = {}

    for source, cols in SOURCE_COMPLETENESS_MAP.items():

        completeness_pcts = []

        for col in cols:

            if col in aligned_df.columns:
                completeness_pcts.append(
                    float(
                        aligned_df[col].notna().mean()
                    )
                )

        if completeness_pcts:
            source_completeness[source] = round(
                min(completeness_pcts),
                4,
            )
        else:
            source_completeness[source] = 0.0

    # ------------------------------------------------------------------
    # 12. Grain / duplicate checks
    # ------------------------------------------------------------------

    grain_issues = []

    dup_orders = oms_filtered[
        "order_id"
    ].duplicated().sum()

    if dup_orders > 0:
        grain_issues.append(
            f"OMS: {dup_orders} duplicate order_id record(s)"
        )

    dup_shipments = logistics_filtered[
        "shipment_id"
    ].duplicated().sum()

    if dup_shipments > 0:
        grain_issues.append(
            f"TMS: {dup_shipments} duplicate shipment_id record(s)"
        )

    dup_wms = wms_filtered.duplicated(
        [
            "warehouse_id",
            "product_id",
            "date",
        ]
    ).sum()

    if dup_wms > 0:
        grain_issues.append(
            f"WMS: {dup_wms} duplicate "
            "warehouse-product-day record(s)"
        )

    dup_tickets = support_filtered[
        "ticket_id"
    ].duplicated().sum()

    if dup_tickets > 0:
        grain_issues.append(
            f"Support: {dup_tickets} duplicate ticket_id record(s)"
        )

    # Marketing is campaign-level, but campaign records may legitimately
    # span multiple dates. Therefore do not flag campaign_id duplication
    # by itself as a grain violation.

    # ------------------------------------------------------------------
    # 13. Missing-day notes
    # ------------------------------------------------------------------

    reconciliation_notes = []

    tracked_columns = [
        "revenue",
        "order_cancellation_rate",
        "fulfillment_delay_rate",
        "support_ticket_volume",
        "warehouse_staffing_level",
    ]

    for col in tracked_columns:

        if col not in aligned_df.columns:
            continue

        n_gaps = int(
            aligned_df[col].isna().sum()
        )

        if n_gaps > 0:
            reconciliation_notes.append(
                f"{col}: {n_gaps} missing day(s)"
            )

    if not reconciliation_notes:
        reconciliation_notes.append(
            "All tracked sources complete - no gaps detected."
        )

    # ------------------------------------------------------------------
    # 14. Metric lineage
    # ------------------------------------------------------------------

    return {
        "aligned_df": aligned_df,
        "source_completeness": source_completeness,
        "grain_issues": grain_issues,
        "reconciliation_notes": reconciliation_notes,
        "metric_lineage": _metric_lineage(),
        "total_days": len(aligned_df),
        "date_range": {
            "start": (
                str(aligned_df["date"].min().date())
                if len(aligned_df) > 0
                else None
            ),
            "end": (
                str(aligned_df["date"].max().date())
                if len(aligned_df) > 0
                else None
            ),
        },
    }


def _metric_lineage() -> Dict[str, Dict[str, Any]]:
    """
    Returns the lineage contract for metrics generated by reconciliation.
    """

    return {
        "revenue": {
            "source": "OMS",
            "source_grain": "order",
            "aggregation": (
                "SUM(quantity * unit_price - discount) "
                "for non-cancelled orders"
            ),
        },
        "order_cancellation_rate": {
            "source": "OMS",
            "source_grain": "order",
            "aggregation": "MEAN(cancelled) * 100",
        },
        "fulfillment_delay_rate": {
            "source": "TMS",
            "source_grain": "shipment",
            "join": "TMS.order_id → OMS.order_id",
            "aggregation": "MEAN(is_delayed) * 100",
            "date_basis": "OMS.order_date",
        },
        "support_ticket_volume": {
            "source": "Support",
            "source_grain": "ticket",
            "join": "Support.order_id → OMS.order_id",
            "aggregation": "COUNT(DISTINCT ticket_id)",
        },
        "warehouse_staffing_level": {
            "source": "WMS",
            "source_grain": (
                "warehouse × product × day"
            ),
            "aggregation": (
                "MEAN(staffing) across products "
                "for each warehouse-day"
            ),
        },
        "marketing_spend": {
            "source": "Marketing",
            "source_grain": "campaign × day",
            "aggregation": "SUM(marketing_spend)",
        },
        "seasonal_index": {
            "source": "Marketing",
            "source_grain": "campaign × day",
            "aggregation": "MEAN(seasonal_index)",
        },
        "unit_price": {
            "source": "OMS",
            "source_grain": "order",
            "aggregation": "MEAN(unit_price)",
        },
        "quantity": {
            "source": "OMS",
            "source_grain": "order",
            "aggregation": "SUM(quantity)",
        },
    }