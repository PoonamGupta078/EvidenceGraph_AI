"""
evaluate.py
Evaluates the core pipeline against all 5 synthetic scenarios to validate correctness.
Ground truth labels are ONLY defined here — they are never passed to the pipeline.
"""

import sys
import pandas as pd
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from pipeline.reconciliation import reconcile_sources
from pipeline.data_reality_check import check_data_reality
from pipeline.materiality import detect_materiality
from pipeline.evidence_graph import build_evidence_graph
from pipeline.root_cause import rank_root_causes
from pipeline.confidence import compute_confidence
from pipeline.challenge_engine import run_challenge
from pipeline.pvm_decomposition import decompose_pvm

DATA_DIR = Path("data/generated")

def load_scenarios() -> list:
    """Load scenario metadata (ground truth) dynamically from generated json."""
    metadata_path = DATA_DIR / "scenario_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"[ERROR] '{metadata_path}' missing. Run generate_synthetic.py")
    
    with open(metadata_path, "r") as f:
        scenarios = json.load(f)
        
    return [
        (region_id, data["scenario"], data["expected_verdict"], data.get("primary_driver"))
        for region_id, data in scenarios.items()
    ]

ENTERPRISE_SOURCES = ["oms", "logistics", "wms", "support", "marketing"]


def load_sources() -> dict:
    """Loads all 5 enterprise source tables. Ground truth is NOT included."""
    dfs = {}
    for src in ENTERPRISE_SOURCES:
        path = DATA_DIR / f"{src}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"[ERROR] Enterprise source '{src}.csv' missing. "
                "Run: python data/generate_synthetic.py"
            )
        dfs[src] = pd.read_csv(path)
    return dfs


def evaluate_pipeline():
    print("=" * 60)
    print("EVIDENCEGRAPH AI -- PIPELINE EVALUATION")
    print("=" * 60 + "\n")

    # Load all enterprise sources once
    try:
        sources = load_sources()
        print(f"[OK] Loaded {len(sources)} enterprise source tables:\n"
              f"     OMS({len(sources['oms'])} rows), "
              f"Logistics({len(sources['logistics'])} rows), "
              f"WMS({len(sources['wms'])} rows), "
              f"Support({len(sources['support'])} rows), "
              f"Marketing({len(sources['marketing'])} rows)\n")
    except FileNotFoundError as e:
        print(str(e))
        return

    pass_count = 0
    fail_count = 0

    scenarios = load_scenarios()
    for region_id, scenario, expected_verdict, expected_driver in scenarios:
        print(f"Evaluating: {region_id.upper()} (Scenario: {scenario})")

        # --- PIPELINE RECEIVES ONLY RAW ENTERPRISE OBSERVATIONS ---
        # Ground truth (expected_verdict) is not passed in
        reconciled = reconcile_sources(
            oms_df=sources["oms"],
            logistics_df=sources["logistics"],
            wms_df=sources["wms"],
            support_df=sources["support"],
            marketing_df=sources["marketing"],
            region_id=region_id,
        )
        aligned_df = reconciled["aligned_df"]

        print(f"  >> Reconciled {reconciled['total_days']} days "
              f"({reconciled['date_range']['start']} to {reconciled['date_range']['end']})")
        print(f"  >> Source Completeness: "
              + ", ".join(f"{k}={v:.0%}" for k, v in reconciled["source_completeness"].items()))
        if reconciled["reconciliation_notes"]:
            for note in reconciled["reconciliation_notes"][:2]:
                print(f"  >> Note: {note}")

        # Reality Check
        reality = check_data_reality(aligned_df, reconciled["source_completeness"], reconciled["total_days"])
        print(f"  >> Data Quality Score: {reality['quality_score']:.2f} (Passed: {reality['passes']})")

        if not reality["passes"]:
            final_verdict = "ABSTAIN"
            print(f"  >> Final Verdict: ABSTAIN (Expected: {expected_verdict})")
            print(f"  >> Reason: {reality.get('abstain_reason', 'data quality gate failed')}")
            match = "PASS" if final_verdict == expected_verdict else "FAIL"
            print(f"  >> Test Result: [{match}]\n")
            if match == "PASS":
                pass_count += 1
            else:
                fail_count += 1
            continue

        # Materiality
        materiality = detect_materiality(aligned_df)
        print(f"  >> Material KPIs: {materiality['material_kpis']}")
        print(f"  >> Signal Strength: {materiality['signal_strength']:.2f}")

        # PVM decomposition (for Region E or any revenue-material case)
        pvm = None
        if scenario == "multi_factor_pvm" or "revenue" in materiality["material_kpis"]:
            pvm = decompose_pvm(aligned_df)

        # Evidence Graph
        graph = build_evidence_graph(aligned_df, materiality["material_kpis"], region_id, scenario, pvm_result=pvm)
        top_driver = graph["driver_ranking"][0] if graph["driver_ranking"] else None
        print(f"  >> Top Driver: {top_driver['kpi'] if top_driver else 'None'} "
              f"(Score: {top_driver['score']:.3f})" if top_driver else "  >> Top Driver: None")

        # Root Cause
        root_causes = rank_root_causes(aligned_df, graph["driver_ranking"], materiality["material_kpis"], scenario, pvm_result=pvm)

        # Confidence Gate
        confidence = compute_confidence(
            quality_score=reality["quality_score"],
            signal_strength=materiality["signal_strength"],
            correlation_matrix=graph["correlation_matrix"],
            root_causes=root_causes["root_causes"],
            causal_chain=root_causes["causal_chain"],
            effect_sizes=root_causes["effect_sizes"],
            scenario=scenario,
        )

        # Challenge Engine
        challenge = run_challenge(
            aligned_df,
            region_id,
            {
                "verdict": confidence["verdict"],
                "primary_cause": root_causes.get("primary_cause"),
                "material_kpis": materiality["material_kpis"],
            },
            scenario=scenario,
        )

        # Compare against ground truth (only evaluated here, never inside the pipeline)
        final_verdict = challenge.get("verdict_adjustment") or confidence["verdict"]

        print(f"  >> Confidence Score: {confidence['score']:.2f}")
        print(f"  >> Sub-Scores:")
        for k, v in confidence["sub_scores"].items():
            print(f"       {k}: {v:.2f}")
        print(f"  >> Challenge Engine: {challenge.get('challenge_summary', 'N/A')}")
        print(f"  >> Final Verdict: {final_verdict} (Expected: {expected_verdict})")

        actual_driver = root_causes.get("primary_cause")
        actual_kpi = actual_driver.get("kpi") if isinstance(actual_driver, dict) else actual_driver
        driver_match = True
        if expected_driver:
            driver_match = (actual_kpi == expected_driver)
            print(f"  >> Primary Cause: {actual_kpi} (Expected: {expected_driver})")
            print(f"  >> Driver Match: [{'PASS' if driver_match else 'FAIL'}]")

        match = "PASS" if (final_verdict == expected_verdict and driver_match) else "FAIL"
        print(f"  >> Test Result: [{match}]\n")
        if match == "PASS":
            pass_count += 1
        else:
            fail_count += 1

    print("=" * 60)
    print(f"RESULTS: {pass_count} PASS / {fail_count} FAIL out of {len(scenarios)} scenarios")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_pipeline()
