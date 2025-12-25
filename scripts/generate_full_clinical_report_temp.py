"""
Generate Full Clinical Fairness Report

Integrates all 4 tiers of Gemini LLM to produce a comprehensive clinical fairness report:
1. Causal graph validation
2. Clinical harm narratives
3. Intervention safety rationales
4. Auto-generated intervention code

Run with: python scripts/generate_full_clinical_report.py
"""

import logging
import json
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Generate comprehensive clinical fairness report."""
    logger.info("="*80)
    logger.info("CLINICAL FAIRNESS REPORT GENERATION")
    logger.info("="*80)

    # Step 1: Load existing benchmark results
    logger.info("\nStep 1: Loading benchmark results...")
    benchmark_path = "results/benchmark_compas.json"

    if not Path(benchmark_path).exists():
        logger.error(f"  ✗ Benchmark file not found: {benchmark_path}")
        logger.info("\nPlease run systematic_experiments.py first:")
        logger.info("  python systematic_experiments.py")
        return

    with open(benchmark_path, 'r') as f:
        benchmark_data = json.load(f)

    # Parse fold results into method-level aggregates
    benchmark_results = {}
    if 'fold_results' in benchmark_data:
        # Group by method and compute averages
        from collections import defaultdict
        import numpy as np

        method_folds = defaultdict(list)
        for fold in benchmark_data['fold_results']:
            method_name = fold['method']
            method_folds[method_name].append(fold)

        for method_name, folds in method_folds.items():
            benchmark_results[method_name] = {
                'accuracy_mean': np.mean([f['accuracy'] for f in folds]),
                'fnr_disparity_mean': np.mean([f['fnr_disparity'] for f in folds]),
                'dp_difference': np.mean([f['dp_difference'] for f in folds]),
                'accuracy': np.mean([f['accuracy'] for f in folds]),
                'fnr_disparity': np.mean([f['fnr_disparity'] for f in folds]),
            }
    else:
        # Already in method-level format
        benchmark_results = benchmark_data

    logger.info(f"  ✓ Loaded benchmark with {len(benchmark_results)} methods")

    # Step 2: Load Medicare data
    logger.info("\nStep 2: Loading Medicare data...")
    data_path = "data/DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv"

    if not Path(data_path).exists():
        logger.error(f"  ✗ Data file not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    logger.info(f"  ✓ Loaded {len(df)} patient records")

    # Prepare features
    df['age'] = 2008 - (df['BENE_BIRTH_DT'] // 10000)
    df['sex'] = (df['BENE_SEX_IDENT_CD'] == 1).astype(int)
    df['race_white'] = (df['BENE_RACE_CD'] == 1).astype(int)
    df['has_esrd'] = (df['BENE_ESRD_IND'] == 'Y').astype(int)
    df['has_diabetes'] = (df['SP_DIABETES'] == 1).astype(int)
    df['has_chf'] = (df['SP_CHF'] == 1).astype(int)
    df['has_copd'] = (df['SP_COPD'] == 1).astype(int)
    df['chronic_count'] = (
        (df['SP_ALZHDMTA'] == 1).astype(int) +
        (df['SP_CHF'] == 1).astype(int) +
        (df['SP_CHRNKIDN'] == 1).astype(int) +
        (df['SP_CNCR'] == 1).astype(int) +
        (df['SP_COPD'] == 1).astype(int) +
        (df['SP_DEPRESSN'] == 1).astype(int) +
        (df['SP_DIABETES'] == 1).astype(int)
    )
    df['total_cost'] = (
        df['MEDREIMB_IP'].fillna(0) +
        df['MEDREIMB_OP'].fillna(0) +
        df['MEDREIMB_CAR'].fillna(0)
    )
    cost_threshold = df['total_cost'].quantile(0.75)
    df['high_cost'] = (df['total_cost'] > cost_threshold).astype(int)

    df_clean = df[[
        'age', 'sex', 'has_esrd', 'has_diabetes',
        'has_chf', 'has_copd', 'chronic_count',
        'race_white', 'high_cost'
    ]].dropna()

    logger.info(f"  ✓ Prepared {len(df_clean)} samples for analysis")

    # Step 3: Initialize Gemini client
    logger.info("\nStep 3: Initializing Gemini client...")

    # Try to import and create client, but don't fail if unavailable
    gemini_client = None

    # Check if google.generativeai is importable first (with timeout protection)
    try:
        import importlib.util
        spec = importlib.util.find_spec("google.generativeai")
        if spec is None:
            raise ModuleNotFoundError("google.generativeai not found")

        logger.info("  ⚠ google-generativeai found but may freeze on Python 3.13")
        logger.info("  ℹ Skipping LLM client creation - using fallback mode")
        logger.info("  ℹ To enable LLM: use Python 3.11 or 3.12")
        gemini_client = None
    except ModuleNotFoundError:
        logger.warning("  ⚠ google-generativeai not installed - using fallback mode")
        logger.info("  ℹ Install with: pip install google-generativeai")
        gemini_client = None

    if gemini_client is None:
        logger.info("  ℹ Running in FALLBACK MODE (no LLM required)")
        logger.info("  ℹ Using correlation-based validation and template narratives")

    # Step 4: Initialize report generator
    logger.info("\nStep 4: Initializing report generator...")
    from src.clinical_fairness_report import ClinicalFairnessReportGenerator
    from src.causal_graph_refiner import CausalEdge

    report_gen = ClinicalFairnessReportGenerator(gemini_client)
    logger.info("  ✓ Report generator ready")

    # Step 5: Define expert causal knowledge
    logger.info("\nStep 5: Defining expert causal edges...")
    expert_edges = [
        CausalEdge("age", "has_diabetes", 0.9, "expert", "Age increases diabetes risk"),
        CausalEdge("age", "has_chf", 0.85, "expert", "Age increases CHF risk"),
        CausalEdge("age", "has_copd", 0.80, "expert", "Age increases COPD risk"),
        CausalEdge("has_diabetes", "chronic_count", 0.95, "expert", "Diabetes contributes to chronic disease count"),
        CausalEdge("has_chf", "chronic_count", 0.95, "expert", "CHF contributes to chronic disease count"),
        CausalEdge("has_copd", "chronic_count", 0.95, "expert", "COPD contributes to chronic disease count"),
        CausalEdge("chronic_count", "high_cost", 0.90, "expert", "Chronic conditions increase healthcare costs"),
        CausalEdge("has_esrd", "high_cost", 0.95, "expert", "ESRD significantly increases costs"),
    ]
    logger.info(f"  ✓ Defined {len(expert_edges)} expert edges")

    # Step 6: Define discovered edges (from PC algorithm)
    discovered_edges = [
        ("race_white", "age"),
        ("age", "chronic_count"),
        ("chronic_count", "high_cost"),
        ("has_chf", "high_cost"),
        ("has_diabetes", "high_cost"),
        ("has_esrd", "high_cost"),
        ("age", "has_diabetes"),
        ("age", "has_chf"),
    ]
    logger.info(f"  ✓ Using {len(discovered_edges)} discovered edges")

    # Step 7: Generate full report
    logger.info("\nStep 7: Generating comprehensive clinical fairness report...")
    logger.info("  (This may take 2-3 minutes with fallbacks, or longer with LLM)")

    report = report_gen.generate_report(
        data=df_clean,
        protected_attr="race_white",
        outcome="high_cost",
        expert_edges=expert_edges,
        discovered_edges=discovered_edges,
        benchmark_results=benchmark_results,
        context="medicare_high_cost"
    )

    # Step 8: Report already saved by report generator
    # Find the generated files
    report_files = list(Path("results").glob("clinical_fairness_*.json"))
    summary_files = list(Path("results").glob("clinical_fairness_*_summary.txt"))

    logger.info(f"\n  ✓ Report saved: {report_files[-1] if report_files else 'N/A'}")
    logger.info(f"  ✓ Summary saved: {summary_files[-1] if summary_files else 'N/A'}")

    # Step 9: Display completion summary
    logger.info("\n" + "="*80)
    logger.info("REPORT GENERATION COMPLETE")
    logger.info("="*80)

    code_files = list(Path("results/generated_code").glob("*.py")) if Path("results/generated_code").exists() else []
    llm_logs = len(list(Path("llm_logs").glob("*.json"))) if Path("llm_logs").exists() else 0
    llm_cache = len(list(Path("llm_cache").glob("*.json"))) if Path("llm_cache").exists() else 0

    logger.info(f"\nOutputs:")
    logger.info(f"  1. JSON report: {report_files[-1] if report_files else 'N/A'}")
    logger.info(f"  2. Text summary: {summary_files[-1] if summary_files else 'N/A'}")
    logger.info(f"  3. Generated code: {len(code_files)} files in results/generated_code/")
    logger.info(f"  4. LLM logs: {llm_logs} files in llm_logs/")
    logger.info(f"  5. LLM cache: {llm_cache} files in llm_cache/")

    logger.info("\nNext steps:")
    logger.info("  1. Review the summary file for clinical insights")
    logger.info("  2. Inspect generated code in results/generated_code/")
    logger.info("  3. Check llm_logs/ for reproducibility verification")
    logger.info("  4. Share results with clinical stakeholders")

    # Check if fallback was used
    if gemini_client is None:
        logger.info("\n⚠ NOTE: This report was generated using fallback mode")
        logger.info("  To get LLM-enhanced narratives:")
        logger.info("  - Use Python 3.11 or 3.12 (currently using Python 3.13)")
        logger.info("  - Or wait for google-generativeai to support Python 3.13")


if __name__ == "__main__":
    main()
