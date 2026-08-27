"""
generate_synthetic.py
Generates 90-day synthetic relational database tables for all regions/scenarios.
Produces transactional OMS, logistics shipments, WMS logs, and support tickets.

Tables generated in data/generated/:
  - oms.csv: order-level transaction details
  - logistics.csv: shipment-level transit logs
  - wms.csv: daily warehouse staffing and capacity metrics
  - support.csv: ticket-level customer issues
  - marketing.csv: daily marketing spend per region

Usage:
    python data/generate_synthetic.py
"""

import os
import uuid
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
OUT_DIR = Path(__file__).parent / "generated"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Setup Base Entity Pools
# ---------------------------------------------------------------------------
CUSTOMERS = [f"C{i:04d}" for i in range(1, 1500)]
PRODUCTS = [f"P{i:03d}" for i in range(1, 200)]
CARRIERS = ["CARRIER-01", "CARRIER-02", "CARRIER-03", "CARRIER-04"]

REGION_WAREHOUSES = {
    "region_a": "WH-A",
    "region_b": "WH-B",
    "region_c": "WH-C",
    "region_d": "WH-D",
    "region_e": "WH-E",
}

def date_range(days=90):
    return pd.date_range(end="2024-03-31", periods=days, freq="D")

def generate_relational_data():
    print("Generating synthetic relational enterprise world...")

    all_dates = date_range(90)
    n_days = len(all_dates)

    # Output list buffers
    oms_records = []
    logistics_records = []
    wms_records = []
    support_records = []
    marketing_records = []

    # Keep track of daily variables for regions to simulate causal chains
    regions = ["region_a", "region_b", "region_c", "region_d", "region_e"]
    
    # Store daily variables to compute lags easily
    daily_vars = {r: {
        "staffing": np.zeros(n_days),
        "delay_prob": np.zeros(n_days),
        "ticket_prob": np.zeros(n_days),
        "cancel_prob": np.zeros(n_days),
    } for r in regions}

    # Pre-populate daily curves for operational regions (A, B, C, D, E)
    for r in regions:
        # Region D only has sparse history (last 11 days of the 90 day range)
        start_idx = 79 if r == "region_d" else 0

        for d in range(start_idx, n_days):
            # 1. Staffing Level
            if r == "region_a":
                staff = 95.0 if d < 45 else 68.0
            elif r == "region_b":
                staff = 93.0 if d < 45 else 66.0
            elif r == "region_c":
                staff = 90.0
            elif r == "region_d":
                staff = 85.0
            else: # region_e
                staff = 92.0
            
            staff_noise = np.random.normal(0, 1.5)
            daily_vars[r]["staffing"][d] = max(50, min(100, staff + staff_noise))

            # 2. Logistics Delay Probability (lags staffing by 2 days)
            staff_lag_idx = max(start_idx, d - 2)
            staff_lag = daily_vars[r]["staffing"][staff_lag_idx]
            
            if r in ["region_a", "region_b"]:
                base_delay = 0.08 + (95.0 - staff_lag) * 0.007
            else:
                base_delay = 0.09
            
            daily_vars[r]["delay_prob"][d] = max(0.02, min(0.85, base_delay + np.random.normal(0, 0.02)))

            # 3. Support Ticket Probability (lags delay by 1 day)
            delay_lag_idx = max(start_idx, d - 1)
            delay_lag = daily_vars[r]["delay_prob"][delay_lag_idx]
            
            if r in ["region_a", "region_b"]:
                ticket_p = 0.05 + delay_lag * 0.4
            else:
                ticket_p = 0.06
            
            daily_vars[r]["ticket_prob"][d] = max(0.02, min(0.9, ticket_p + np.random.normal(0, 0.02)))

            # 4. Cancellation Probability (lags ticket by 1 day)
            ticket_lag_idx = max(start_idx, d - 1)
            ticket_lag = daily_vars[r]["ticket_prob"][ticket_lag_idx]
            
            if r in ["region_a", "region_b"]:
                cancel_p = 0.02 + ticket_lag * 0.35
            else:
                cancel_p = 0.03
            
            daily_vars[r]["cancel_prob"][d] = max(0.01, min(0.6, cancel_p + np.random.normal(0, 0.01)))

    # Order/Shipment ID counter
    order_counter = 100000
    shipment_counter = 500000
    ticket_counter = 900000

    # ---------------------------------------------------------------------------
    # Day-by-Day Transaction Generator
    # ---------------------------------------------------------------------------
    for d_idx, date in enumerate(all_dates):
        date_str = date.strftime("%Y-%m-%d")

        for r in regions:
            # Region D sparse check
            if r == "region_d" and d_idx < 79:
                continue

            wh_id = REGION_WAREHOUSES[r]

            # 1. Marketing campaign spend (Region E shock, Region B normal, etc.)
            mkt_spend = 1000.0
            if r == "region_e":
                mkt_spend = 5000.0 if d_idx < 40 else 2000.0
            elif r == "region_b":
                mkt_spend = 2500.0  # slightly higher marketing baseline
            
            mkt_spend = max(0.0, mkt_spend + np.random.normal(0, 100))
            
            # 2. Determine Order Volume (Demand)
            base_orders = 80
            price = 450.0  # Base unit price
            discount = 0.0
            seasonal_index = 1.0

            if r == "region_e":
                # Price shock at Day 30 (+12% price -> -18% volume)
                price = 500.0 if d_idx < 30 else 560.0
                
                # Marketing spend shock at Day 40 (5000 -> 2000, -60% spend -> -10% volume)
                mkt_factor = 1.0 if d_idx < 40 else 0.9
                
                # Seasonal dip at Day 55–70 (seasonal factor 0.9 -> -10% volume)
                if 55 <= d_idx <= 70:
                    seasonal_index = 0.9
                
                price_pct = (price - 500.0) / 500.0
                price_factor = 1.0 - 1.5 * price_pct  # -1.5 price elasticity
                
                # Apply multipliers to volume
                vol_target = base_orders * price_factor * mkt_factor * seasonal_index
                n_orders = int(np.random.poisson(vol_target))
            else:
                # Normal demand
                n_orders = int(np.random.poisson(base_orders))
                
                # Region B promo discount after Day 47
                if r == "region_b" and d_idx >= 47:
                    discount = 0.15 # 15% promo discount

            # Append to marketing records after variables are resolved
            marketing_records.append({
                "date": date_str,
                "region_id": r,
                "marketing_spend": round(mkt_spend, 2),
                "seasonal_index": round(seasonal_index, 2),
                "campaign_id": f"CAMP-{r.upper()}-{date.strftime('%y%m%d')}"
            })

            # Generate orders, shipments and support tickets
            for _ in range(n_orders):
                order_counter += 1
                shipment_counter += 1
                
                order_id = f"ORD-{order_counter}"
                shipment_id = f"SHP-{shipment_counter}"
                cust_id = np.random.choice(CUSTOMERS)
                prod_id = np.random.choice(PRODUCTS)
                
                qty = int(np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1]))
                
                # Cancellation logic
                cancel_prob = daily_vars[r]["cancel_prob"][d_idx]
                is_cancelled = np.random.random() < cancel_prob
                
                # Write OMS
                oms_records.append({
                    "order_id": order_id,
                    "customer_id": cust_id,
                    "product_id": prod_id,
                    "region_id": r,
                    "order_date": date_str,
                    "quantity": qty,
                    "unit_price": round(price, 2),
                    "discount": round(discount * price * qty, 2),
                    "refund": round(price * qty if is_cancelled else 0.0, 2),
                    "cancelled": 1 if is_cancelled else 0,
                })

                # Write Logistics (unless region_c data drop applies)
                # Region C data-drop scenario: drop shipments completely between days 30-50
                is_region_c_drop = (r == "region_c" and 30 <= d_idx <= 50)
                
                if not is_region_c_drop:
                    delay_prob = daily_vars[r]["delay_prob"][d_idx]
                    is_delayed = np.random.random() < delay_prob
                    
                    promised_dt = date + pd.Timedelta(days=3)
                    
                    if is_delayed:
                        actual_dt = promised_dt + pd.Timedelta(days=int(np.random.randint(1, 6)))
                    else:
                        actual_dt = promised_dt + pd.Timedelta(days=int(np.random.choice([-1, 0])))

                    logistics_records.append({
                        "shipment_id": shipment_id,
                        "order_id": order_id,
                        "warehouse_id": wh_id,
                        "carrier_id": np.random.choice(CARRIERS),
                        "promised_date": promised_dt.strftime("%Y-%m-%d"),
                        "actual_delivery_date": actual_dt.strftime("%Y-%m-%d")
                    })

                # Write Support Tickets
                ticket_prob = daily_vars[r]["ticket_prob"][d_idx]
                if np.random.random() < ticket_prob:
                    ticket_counter += 1
                    t_id = f"TCK-{ticket_counter}"
                    
                    # Sentiment and Issue classification based on scenario
                    sentiment = 0.5 + np.random.normal(0, 0.15)
                    issue = "general_inquiry"
                    
                    if is_cancelled:
                        issue = "cancellation"
                        sentiment = 0.1 + np.random.normal(0, 0.08)
                    elif not is_region_c_drop and is_delayed:
                        issue = "delivery_delay"
                        sentiment = 0.2 + np.random.normal(0, 0.1)
                    elif r == "region_e" and d_idx >= 30:
                        issue = "price_complaint"
                        sentiment = 0.3 + np.random.normal(0, 0.12)
                    elif r == "region_b" and d_idx >= 47:
                        issue = "promo_inquiry"
                        sentiment = 0.8 + np.random.normal(0, 0.05)

                    support_records.append({
                        "ticket_id": t_id,
                        "order_id": order_id,
                        "customer_id": cust_id,
                        "region_id": r,
                        "ticket_date": date_str,
                        "issue_type": issue,
                        "sentiment": round(max(0.0, min(1.0, sentiment)), 2)
                    })

            # 3. Warehouse WMS logs for the day (one per warehouse-product combination)
            staff_level = daily_vars[r]["staffing"][d_idx]
            stockout = 1 if (staff_level < 75.0 and np.random.random() < 0.15) else 0

            # Sample WMS logs for top 5 active products in the warehouse to avoid massive file size
            for prod in PRODUCTS[:25]:
                wms_records.append({
                    "warehouse_id": wh_id,
                    "product_id": prod,
                    "date": date_str,
                    "inventory": int(np.random.randint(100, 2000) if not stockout else np.random.randint(0, 10)),
                    "staffing": round(staff_level, 2),
                    "capacity": 1000,
                    "stockout": stockout
                })

    # Save all datasets to data/generated/
    df_oms = pd.DataFrame(oms_records)
    df_oms.to_csv(OUT_DIR / "oms.csv", index=False)
    print(f"[OK] OMS: {len(df_oms)} orders saved to {OUT_DIR / 'oms.csv'}")

    df_logistics = pd.DataFrame(logistics_records)
    df_logistics.to_csv(OUT_DIR / "logistics.csv", index=False)
    print(f"[OK] Logistics: {len(df_logistics)} shipments saved to {OUT_DIR / 'logistics.csv'}")

    df_wms = pd.DataFrame(wms_records)
    df_wms.to_csv(OUT_DIR / "wms.csv", index=False)
    print(f"[OK] WMS: {len(df_wms)} staffing records saved to {OUT_DIR / 'wms.csv'}")

    df_support = pd.DataFrame(support_records)
    df_support.to_csv(OUT_DIR / "support.csv", index=False)
    print(f"[OK] Support: {len(df_support)} tickets saved to {OUT_DIR / 'support.csv'}")

    df_marketing = pd.DataFrame(marketing_records)
    df_marketing.to_csv(OUT_DIR / "marketing.csv", index=False)
    print(f"[OK] Marketing: {len(df_marketing)} records saved to {OUT_DIR / 'marketing.csv'}")

    # Build RAG corpus directly from generated support records (for evidence lineage)
    generate_rag_corpus_from_support(df_support, df_logistics)
    # Generate metadata files
    generate_source_metadata()
    generate_scenario_metadata()


# ─── ISSUE TEXT TEMPLATES (populated from generated ticket fields) ──────────
ISSUE_TEXT_TEMPLATES = {
    "delivery_delay": [
        "Customer {cid} reported order {oid} delayed significantly — {days}+ days past promised delivery.",
        "Delivery for order {oid} is overdue. Customer {cid} flagged missing shipment.",
        "Order {oid} stuck in processing. Warehouse capacity likely insufficient.",
    ],
    "cancellation": [
        "Customer {cid} cancelled order {oid} after extended wait for fulfillment.",
        "Order {oid} cancelled by {cid} due to unacceptable delivery delays.",
        "Cancellation received for {oid}. Customer cited repeated delays.",
    ],
    "price_complaint": [
        "Customer {cid} complained about recent price increase on order {oid}.",
        "Order {oid} query: customer {cid} finds new pricing uncompetitive.",
        "Price sensitivity flagged by {cid} — reduced order size on {oid}.",
    ],
    "promo_inquiry": [
        "Customer {cid} inquired about active discount code for order {oid}.",
        "Promo code applied to {oid} by customer {cid}. Positive feedback.",
        "Order {oid}: customer {cid} satisfied with discount offer.",
    ],
    "general_inquiry": [
        "Customer {cid} submitted general query regarding order {oid}.",
        "Order {oid} status check by customer {cid}.",
    ],
}


def generate_rag_corpus_from_support(df_support: pd.DataFrame, df_logistics: pd.DataFrame):
    """
    Builds support_tickets.csv (the RAG retrieval corpus) directly from
    the generated support.csv records. Every corpus document is traceable
    back to a real generated support ticket via ticket_id and order_id.
    """
    corpus_records = []
    logistics_lookup = df_logistics.set_index("order_id") if "order_id" in df_logistics.columns else None

    for _, row in df_support.iterrows():
        issue = row.get("issue_type", "general_inquiry")
        templates = ISSUE_TEXT_TEMPLATES.get(issue, ISSUE_TEXT_TEMPLATES["general_inquiry"])
        
        template_idx = sum(ord(c) for c in str(row["ticket_id"])) % len(templates)
        template = templates[template_idx]
        
        days_late = 0
        if "delay" in issue and logistics_lookup is not None:
            order_id = row["order_id"]
            if order_id in logistics_lookup.index:
                log_row = logistics_lookup.loc[order_id]
                if isinstance(log_row, pd.DataFrame):
                    log_row = log_row.iloc[0]
                try:
                    actual = pd.to_datetime(log_row.get("actual_delivery_date"))
                    promised = pd.to_datetime(log_row.get("promised_date"))
                    if pd.notna(actual) and pd.notna(promised):
                        days_late = max(0, (actual - promised).days)
                except Exception:
                    pass
        if days_late == 0 and "delay" in issue:
            days_late = 5  # fallback
            
        text = template.format(
            cid=row["customer_id"],
            oid=row["order_id"],
            days=days_late,
        )
        corpus_records.append({
            # Traceability fields — link directly to support.csv and oms.csv
            "id": row["ticket_id"],
            "source": "SUPPORT",
            "source_id": row["ticket_id"],
            "order_id": row["order_id"],
            "region": row.get("region_id", ""),
            "date": row["ticket_date"],
            "text": text,
            "category": issue,
            "sentiment": row.get("sentiment", 0.5),
        })

    df_corpus = pd.DataFrame(corpus_records)
    df_corpus.to_csv(OUT_DIR / "support_tickets.csv", index=False)
    print(f"[OK] RAG corpus (traced): {len(df_corpus)} records -> {OUT_DIR / 'support_tickets.csv'}")


def generate_source_metadata():
    """
    Produces source_metadata.csv: formal registration of each enterprise source.
    The data_reality_check engine reads this to evaluate freshness and cadence.
    This is NOT ground truth — it describes source structure, not scenario outcomes.
    """
    import json
    # Simulate last refresh at end of the synthetic period
    LAST_REFRESH_BASE = "2024-03-31"
    sources = [
        {
            "source": "OMS",
            "table": "oms.csv",
            "grain": "order",
            "refresh_frequency": "daily",
            "last_refresh": f"{LAST_REFRESH_BASE}T23:00:00Z",
            "expected_lag_hours": 1,
            "status": "FRESH",
            "primary_key": "order_id",
            "join_keys": "order_id",
            "owner": "Commerce Platform",
        },
        {
            "source": "TMS",
            "table": "logistics.csv",
            "grain": "shipment",
            "refresh_frequency": "15min",
            "last_refresh": f"{LAST_REFRESH_BASE}T23:45:00Z",
            "expected_lag_hours": 0.25,
            "status": "FRESH",
            "primary_key": "shipment_id",
            "join_keys": "order_id, warehouse_id",
            "owner": "Logistics Platform",
        },
        {
            "source": "WMS",
            "table": "wms.csv",
            "grain": "warehouse_product_day",
            "refresh_frequency": "daily",
            "last_refresh": f"{LAST_REFRESH_BASE}T06:00:00Z",
            "expected_lag_hours": 18,
            "status": "STALE",
            "primary_key": "warehouse_id+product_id+date",
            "join_keys": "warehouse_id",
            "owner": "Warehouse Operations",
        },
        {
            "source": "Support",
            "table": "support.csv",
            "grain": "ticket",
            "refresh_frequency": "realtime",
            "last_refresh": f"{LAST_REFRESH_BASE}T23:58:00Z",
            "expected_lag_hours": 0,
            "status": "FRESH",
            "primary_key": "ticket_id",
            "join_keys": "order_id, customer_id",
            "owner": "Customer Experience",
        },
        {
            "source": "Marketing",
            "table": "marketing.csv",
            "grain": "region_day",
            "refresh_frequency": "daily",
            "last_refresh": f"{LAST_REFRESH_BASE}T18:00:00Z",
            "expected_lag_hours": 6,
            "status": "FRESH",
            "primary_key": "campaign_id",
            "join_keys": "region_id, date",
            "owner": "Growth & Marketing",
        },
    ]
    df_meta = pd.DataFrame(sources)
    df_meta.to_csv(OUT_DIR / "source_metadata.csv", index=False)
    print(f"[OK] Source metadata: {len(df_meta)} sources -> {OUT_DIR / 'source_metadata.csv'}")


def generate_scenario_metadata():
    """
    Produces scenario_metadata.json: formal record of ground truth per region.

    IMPORTANT: This file is used ONLY by:
      - evaluate.py (test harness)
      - Demo scenario selector (UI)
    It is NEVER loaded by the intelligence pipeline (main.py, reconciliation, confidence gate).
    The pipeline must independently discover the verdict from raw source data.
    """
    import json
    metadata = {
        "region_a": {
            "scenario": "operational_disruption",
            "label": "Staffing Chain — Pacific NW",
            "description": "Warehouse staffing dropped 28% at Day 45, triggering a causal chain through delivery delays, support tickets, and cancellations to revenue loss.",
            "primary_driver": "warehouse_staffing_level",
            "causal_chain": [
                "warehouse_staffing_level",
                "fulfillment_delay_rate",
                "support_ticket_volume",
                "order_cancellation_rate",
                "revenue"
            ],
            "shock_day": 45,
            "expected_verdict": "ACT",
            "demo_use": True,
        },
        "region_b": {
            "scenario": "contradictory_evidence",
            "label": "Promo Compensation — Southwest",
            "description": "Same operational failure as Region A, but a 15% promotional discount launched at Day 47 compensated revenue, creating contradictory signals.",
            "primary_driver": "fulfillment_delay_rate",
            "compensating_factor": "promo_discount",
            "shock_day": 45,
            "promo_day": 47,
            "expected_verdict": "INVESTIGATE",
            "demo_use": True,
        },
        "region_c": {
            "scenario": "data_quality_failure",
            "label": "TMS Outage — Northeast",
            "description": "Logistics (TMS) records are absent for Days 30-50 (21-day gap). The engine cannot reconcile shipment completeness and must abstain.",
            "missing_source": "TMS",
            "gap_start_day": 30,
            "gap_end_day": 50,
            "expected_verdict": "ABSTAIN",
            "abstain_reason": "logistics source completeness 77%",
            "demo_use": True,
        },
        "region_d": {
            "scenario": "sparse_history",
            "label": "Newly Launched Market — Midwest",
            "description": "Region D is a newly launched market with only 11 days of transaction history, below the 14-day minimum required for statistical confidence.",
            "history_days": 11,
            "minimum_required_days": 14,
            "expected_verdict": "ABSTAIN",
            "abstain_reason": "insufficient history",
            "demo_use": True,
        },
        "region_e": {
            "scenario": "multi_factor_pvm",
            "label": "Price + Marketing + Seasonality — Southeast",
            "description": "Revenue decline driven by three distinct economic factors: price elasticity (Day 30), marketing spend cut (Day 40), and seasonal dip (Days 55-70). Operational KPIs remain healthy.",
            "primary_driver": "unit_price",
            "distinct_factors": ["unit_price", "marketing_spend", "seasonal_index"],
            "price_shock_day": 30,
            "marketing_cut_day": 40,
            "seasonal_dip_start": 55,
            "seasonal_dip_end": 70,
            "expected_verdict": "ACT",
            "demo_use": True,
        },
    }
    out_path = OUT_DIR / "scenario_metadata.json"
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[OK] Scenario metadata (evaluation-only): {len(metadata)} regions -> {out_path}")
    print("     NOTE: scenario_metadata.json is for evaluate.py / demo only. NEVER loaded by the pipeline.")


if __name__ == "__main__":
    generate_relational_data()
    print("\n[SUCCESS] Relational synthetic database generated successfully.")
