"""
generate_synthetic.py
Generates 90-day synthetic time-series data for all 5 regions/scenarios.
Produces internally consistent causal chains baked into the data.

Usage:
    python data/generate_synthetic.py
Outputs CSV files to data/generated/
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
OUT_DIR = Path(__file__).parent / "generated"
OUT_DIR.mkdir(exist_ok=True)


def date_range(days=90):
    return pd.date_range(end="2024-03-31", periods=days, freq="D")


# ---------------------------------------------------------------------------
# REGION A: Staffing → Delay → Support → Cancellation → Revenue chain (ACT)
# ---------------------------------------------------------------------------
def generate_region_a():
    dates = date_range(90)
    n = len(dates)

    # Staffing drops at day 45 (shock event)
    staffing = np.ones(n) * 95.0
    staffing[45:] = 68.0  # 27-point drop
    staffing += np.random.normal(0, 2, n)

    # Fulfillment delay rate lags staffing by 2 days
    delay = np.zeros(n)
    for i in range(n):
        staff_lag = staffing[max(0, i - 2)]
        delay[i] = max(0, 8 + (95 - staff_lag) * 0.6 + np.random.normal(0, 1.5))

    # Support tickets lag delay by 1 day
    tickets = np.zeros(n)
    for i in range(n):
        delay_lag = delay[max(0, i - 1)]
        tickets[i] = max(0, 50 + delay_lag * 3.2 + np.random.normal(0, 5))

    # Cancellation rate lags tickets by 1 day
    cancellation = np.zeros(n)
    for i in range(n):
        ticket_lag = tickets[max(0, i - 1)]
        cancellation[i] = max(0, 2 + ticket_lag * 0.08 + np.random.normal(0, 0.5))

    # Revenue lags cancellation by 1 day
    revenue = np.zeros(n)
    baseline_revenue = 500_000
    for i in range(n):
        cancel_lag = cancellation[max(0, i - 1)]
        revenue[i] = max(0, baseline_revenue * (1 - cancel_lag / 100) - cancel_lag * 800 + np.random.normal(0, 8000))

    df = pd.DataFrame({
        "date": dates,
        "region": "region_a",
        "warehouse_staffing_level": staffing.round(2),
        "fulfillment_delay_rate": delay.round(2),
        "support_ticket_volume": tickets.round(0).astype(int),
        "order_cancellation_rate": cancellation.round(2),
        "revenue": revenue.round(2),
        "staffing_source": "WMS",
        "logistics_source": "logistics",
        "support_source": "support",
        "oms_source": "OMS",
    })
    df.to_csv(OUT_DIR / "region_a.csv", index=False)
    print(f"✓ Region A: {len(df)} rows → {OUT_DIR / 'region_a.csv'}")
    return df


# ---------------------------------------------------------------------------
# REGION B: Same delay pattern BUT compensating promo — INVESTIGATE (contradiction)
# ---------------------------------------------------------------------------
def generate_region_b():
    dates = date_range(90)
    n = len(dates)

    # Same staffing drop
    staffing = np.ones(n) * 93.0
    staffing[45:] = 66.0
    staffing += np.random.normal(0, 2, n)

    # Same delay chain
    delay = np.zeros(n)
    for i in range(n):
        staff_lag = staffing[max(0, i - 2)]
        delay[i] = max(0, 8 + (93 - staff_lag) * 0.6 + np.random.normal(0, 1.5))

    tickets = np.zeros(n)
    for i in range(n):
        delay_lag = delay[max(0, i - 1)]
        tickets[i] = max(0, 48 + delay_lag * 3.0 + np.random.normal(0, 5))

    cancellation = np.zeros(n)
    for i in range(n):
        ticket_lag = tickets[max(0, i - 1)]
        cancellation[i] = max(0, 2 + ticket_lag * 0.075 + np.random.normal(0, 0.5))

    # PROMO kicks in at day 47 — compensates revenue (this is the contradiction)
    promo_discount = np.zeros(n)
    promo_discount[47:] = 15.0  # 15% promo

    revenue = np.zeros(n)
    baseline_revenue = 480_000
    for i in range(n):
        cancel_lag = cancellation[max(0, i - 1)]
        promo_boost = promo_discount[i] * 1200  # promo drives volume despite cancellations
        revenue[i] = max(0, baseline_revenue * (1 - cancel_lag / 200) + promo_boost + np.random.normal(0, 9000))

    df = pd.DataFrame({
        "date": dates,
        "region": "region_b",
        "warehouse_staffing_level": staffing.round(2),
        "fulfillment_delay_rate": delay.round(2),
        "support_ticket_volume": tickets.round(0).astype(int),
        "order_cancellation_rate": cancellation.round(2),
        "revenue": revenue.round(2),
        "promo_discount_pct": promo_discount.round(2),
        "staffing_source": "WMS",
        "logistics_source": "logistics",
        "support_source": "support",
        "oms_source": "OMS",
    })
    df.to_csv(OUT_DIR / "region_b.csv", index=False)
    print(f"✓ Region B: {len(df)} rows → {OUT_DIR / 'region_b.csv'}")
    return df


# ---------------------------------------------------------------------------
# REGION C: Data-quality ABSTAIN — corrupted logistics window
# ---------------------------------------------------------------------------
def generate_region_c():
    dates = date_range(90)
    n = len(dates)

    staffing = np.ones(n) * 90.0 + np.random.normal(0, 2, n)

    # Deliberately corrupt logistics data days 30–50 (NaN injection)
    delay = 10.0 + np.random.normal(0, 1.5, n)
    delay[30:51] = np.nan  # 21-day gap in logistics source

    tickets = 60 + np.random.normal(0, 5, n)
    cancellation = 3.0 + np.random.normal(0, 0.4, n)
    revenue = 490_000 + np.random.normal(0, 10000, n)

    df = pd.DataFrame({
        "date": dates,
        "region": "region_c",
        "warehouse_staffing_level": staffing.round(2),
        "fulfillment_delay_rate": delay.round(2),  # has NaNs
        "support_ticket_volume": tickets.round(0).astype(int),
        "order_cancellation_rate": cancellation.round(2),
        "revenue": revenue.round(2),
        "data_quality_flag": ["CORRUPT" if 30 <= i <= 50 else "OK" for i in range(n)],
    })
    df.to_csv(OUT_DIR / "region_c.csv", index=False)
    print(f"✓ Region C: {len(df)} rows → {OUT_DIR / 'region_c.csv'}")
    return df


# ---------------------------------------------------------------------------
# REGION D: Sparse history ABSTAIN — only 11 days of data
# ---------------------------------------------------------------------------
def generate_region_d():
    dates = date_range(11)  # Only 11 days — below 14-day minimum
    n = len(dates)

    staffing = 85.0 + np.random.normal(0, 3, n)
    delay = 12.0 + np.random.normal(0, 2, n)
    tickets = 55 + np.random.normal(0, 8, n)
    cancellation = 3.5 + np.random.normal(0, 0.6, n)
    revenue = 460_000 + np.random.normal(0, 12000, n)

    df = pd.DataFrame({
        "date": dates,
        "region": "region_d",
        "warehouse_staffing_level": staffing.round(2),
        "fulfillment_delay_rate": delay.round(2),
        "support_ticket_volume": tickets.round(0).astype(int),
        "order_cancellation_rate": cancellation.round(2),
        "revenue": revenue.round(2),
    })
    df.to_csv(OUT_DIR / "region_d.csv", index=False)
    print(f"✓ Region D: {len(df)} rows (sparse) → {OUT_DIR / 'region_d.csv'}")
    return df


# ---------------------------------------------------------------------------
# REGION E: Multi-factor PVM — price + marketing + seasonal independent drivers (ACT)
# ---------------------------------------------------------------------------
def generate_region_e():
    dates = date_range(90)
    n = len(dates)

    # Healthy operations — staffing, delays, tickets all normal
    staffing = 92.0 + np.random.normal(0, 2, n)
    delay = 9.0 + np.random.normal(0, 1.2, n)
    tickets = 52 + np.random.normal(0, 4, n)
    cancellation = 2.5 + np.random.normal(0, 0.3, n)

    # Revenue drop driven by 3 INDEPENDENT factors:
    # 1. Price increase at day 30 (+12% price → -8% volume)
    price_effect = np.zeros(n)
    price_effect[30:] = -38_000  # revenue loss from elasticity

    # 2. Marketing spend cut at day 40 (no promo)
    marketing_effect = np.zeros(n)
    marketing_effect[40:] = -22_000

    # 3. Seasonal dip (natural Q1 trough at days 55–70)
    seasonal_effect = np.zeros(n)
    seasonal_effect[55:71] = -15_000

    baseline_revenue = 510_000
    revenue = np.zeros(n)
    for i in range(n):
        revenue[i] = (baseline_revenue
                      + price_effect[i]
                      + marketing_effect[i]
                      + seasonal_effect[i]
                      + np.random.normal(0, 7000))

    df = pd.DataFrame({
        "date": dates,
        "region": "region_e",
        "warehouse_staffing_level": staffing.round(2),
        "fulfillment_delay_rate": delay.round(2),
        "support_ticket_volume": tickets.round(0).astype(int),
        "order_cancellation_rate": cancellation.round(2),
        "revenue": revenue.round(2),
        "price_effect_usd": price_effect.round(2),
        "marketing_effect_usd": marketing_effect.round(2),
        "seasonal_effect_usd": seasonal_effect.round(2),
    })
    df.to_csv(OUT_DIR / "region_e.csv", index=False)
    print(f"✓ Region E: {len(df)} rows → {OUT_DIR / 'region_e.csv'}")
    return df


# ---------------------------------------------------------------------------
# Synthetic support tickets corpus (for RAG)
# ---------------------------------------------------------------------------
def generate_support_tickets():
    tickets = [
        {"id": "T001", "region": "region_a", "date": "2024-02-20", "text": "My order has been delayed for 5 days with no update. Extremely frustrated.", "category": "delivery_delay"},
        {"id": "T002", "region": "region_a", "date": "2024-02-21", "text": "Package showing as processing for a week, warehouse says understaffed.", "category": "delivery_delay"},
        {"id": "T003", "region": "region_a", "date": "2024-02-22", "text": "Cancelled my order after waiting 8 days. First time I've had to do this.", "category": "cancellation"},
        {"id": "T004", "region": "region_a", "date": "2024-02-23", "text": "Agent told me there are staffing issues at the fulfillment center. Unacceptable.", "category": "delivery_delay"},
        {"id": "T005", "region": "region_a", "date": "2024-02-24", "text": "Three orders cancelled this month due to delays. Switching to competitor.", "category": "cancellation"},
        {"id": "T006", "region": "region_b", "date": "2024-02-20", "text": "Got a 15% promo code today, applying it right away!", "category": "promo"},
        {"id": "T007", "region": "region_b", "date": "2024-02-21", "text": "Delay was frustrating but the promo made up for it somewhat.", "category": "mixed"},
        {"id": "T008", "region": "region_b", "date": "2024-02-22", "text": "Happy with the discount but wish delivery was faster.", "category": "mixed"},
        {"id": "T009", "region": "region_c", "date": "2024-02-15", "text": "No tracking updates for 10 days. Logistics system seems broken.", "category": "data_issue"},
        {"id": "T010", "region": "region_e", "date": "2024-02-20", "text": "Prices went up a lot this month. Had to reduce my order size.", "category": "price"},
        {"id": "T011", "region": "region_e", "date": "2024-02-25", "text": "Used to get promo emails, haven't received anything in weeks.", "category": "marketing"},
        {"id": "T012", "region": "region_e", "date": "2024-03-01", "text": "Post-holiday spending is always lower. Normal for this time of year.", "category": "seasonal"},
        {"id": "T013", "region": "region_a", "date": "2024-02-25", "text": "Warehouse in area reportedly had worker shortage last month.", "category": "delivery_delay"},
        {"id": "T014", "region": "region_a", "date": "2024-02-26", "text": "5 of my 7 recent orders had delays exceeding 3 days.", "category": "delivery_delay"},
        {"id": "T015", "region": "region_b", "date": "2024-02-23", "text": "Despite delays, kept my order because of the great discount.", "category": "promo"},
    ]
    df = pd.DataFrame(tickets)
    df.to_csv(OUT_DIR / "support_tickets.csv", index=False)
    print(f"✓ Support tickets corpus: {len(df)} rows → {OUT_DIR / 'support_tickets.csv'}")
    return df


if __name__ == "__main__":
    print("Generating synthetic data for all 5 regions...\n")
    generate_region_a()
    generate_region_b()
    generate_region_c()
    generate_region_d()
    generate_region_e()
    generate_support_tickets()
    print("\n✅ All synthetic data generated successfully.")
