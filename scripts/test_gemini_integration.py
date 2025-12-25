"""
Test Script: Gemini LLM Integration with Medicare Data

Demonstrates all 4 tiers of LLM integration:
1. Causal graph refinement
2. Bias interpretation
3. Intervention rationale
4. Code generation

Run with: python test_gemini_integration.py
"""

import logging
import pandas as pd
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run Gemini integration test with Medicare data."""
    logger.info("="*80)
    logger.info("GEMINI LLM INTEGRATION TEST")
    logger.info("="*80)

    # Step 1: Load configuration
    logger.info("\nStep 1: Loading Gemini configuration...")
    from src.gemini_client import create_gemini_client

    try:
        gemini_client = create_gemini_client()
        logger.info(f"  ✓ Gemini client initialized: {gemini_client.model}")
    except Exception as e:
        logger.error(f"  ✗ Failed to create Gemini client: {e}")
        logger.info("\nPlease ensure config/api_keys.yaml has valid gemini_api_key")
        return

    # Step 2: Load Medicare data
    logger.info("\nStep 2: Loading Medicare data...")
    data_path = "data/DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv"

    if not Path(data_path).exists():
        logger.error(f"  ✗ Data file not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    logger.info(f"  ✓ Loaded {len(df)} patient records")

    # Prepare features (same as systematic_experiments.py)
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

    # Sample for faster testing
    df_sample = df_clean.sample(n=min(5000, len(df_clean)), random_state=42)
    logger.info(f"  ✓ Using {len(df_sample)} samples for testing")

    # Step 3: Test TIER 1 - Causal Graph Refinement
    logger.info("\n" + "="*80)
    logger.info("TIER 1: Causal Graph Refinement")
    logger.info("="*80)

    from src.causal_graph_refiner import CausalEdge, CausalGraphRefiner

    # Create some expert edges
    expert_edges = [
        CausalEdge("age", "has_diabetes", 0.9, "expert", "Age increases diabetes risk"),
        CausalEdge("age", "has_chf", 0.85, "expert", "Age increases CHF risk"),
        CausalEdge("has_diabetes", "chronic_count", 0.95, "expert", "Diabetes contributes to chronic disease count"),
    ]

    # Discovered edges from PC algorithm
    discovered_edges = [
        ("race_white", "age"),
        ("age", "chronic_count"),
        ("chronic_count", "high_cost"),
        ("has_chf", "high_cost"),
    ]

    refiner = CausalGraphRefiner(gemini_client, domain="medicare_high_cost")
    validation_result = refiner.refine_causal_graph(
        expert_edges=expert_edges,
        discovered_edges=discovered_edges,
        data=df_sample,
        protected_attr="race_white",
        outcome="high_cost"
    )

    logger.info(f"\n  Results:")
    logger.info(f"  - Total validated edges: {validation_result['summary']['total_edges']}")
    logger.info(f"  - Expert edges: {validation_result['summary']['expert_edges']}")
    logger.info(f"  - Validated discovered: {validation_result['summary']['validated_discovered_edges']}")

    # Step 4: Test TIER 2 - Bias Interpretation
    logger.info("\n" + "="*80)
    logger.info("TIER 2: Bias Interpretation")
    logger.info("="*80)

    from src.bias_interpreter import BiasInterpreter

    interpreter = BiasInterpreter(gemini_client)

    # Example fairness metrics
    harm_report = interpreter.interpret_bias_clinically(
        fnr_disparity=0.026,  # From Medicare results
        demographic_parity_diff=0.026,
        protected_attr="race_white",
        outcome="high_cost",
        context="medicare_high_cost",
        dataset_size=len(df_sample)
    )

    logger.info(f"\n  Clinical Harm Narrative:")
    logger.info(f"  {harm_report.narrative}")

    if harm_report.preventable_outcomes_estimate:
        logger.info(f"\n  Estimated Impact:")
        logger.info(f"  - Affected patients: {harm_report.affected_patients_estimate}")
        logger.info(f"  - Preventable outcomes: {harm_report.preventable_outcomes_estimate}")

    # Step 5: Test TIER 3 - Intervention Rationale
    logger.info("\n" + "="*80)
    logger.info("TIER 3: Intervention Rationale")
    logger.info("="*80)

    from src.intervention_recommender import InterventionRecommender

    recommender = InterventionRecommender(gemini_client)

    rationale = recommender.generate_intervention_rationale(
        intervention_name="Fairlearn (Equalized Odds)",
        outcome="high_cost",
        sensitive_attr="race_white",
        context="medicare_high_cost",
        accuracy_tradeoff=0.057  # From Medicare results
    )

    logger.info(f"\n  Safety Assessment:")
    logger.info(f"  - Clinical safety score: {rationale.safety_assessment.clinical_safety_score}")
    logger.info(f"  - Implementation complexity: {rationale.safety_assessment.implementation_complexity}")
    logger.info(f"  - Deployment recommendation: {rationale.safety_assessment.deployment_recommendation}")

    logger.info(f"\n  Safety Narrative:")
    logger.info(f"  {rationale.safety_narrative}")

    # Step 6: Test TIER 4 - Code Generation
    logger.info("\n" + "="*80)
    logger.info("TIER 4: Code Generation")
    logger.info("="*80)

    from src.code_generator import CodeGenerator

    generator = CodeGenerator(gemini_client)

    generated = generator.generate_intervention_code(
        intervention_name="Reweighing",
        data=df_sample,
        sensitive_attr="race_white",
        outcome="high_cost",
        use_template_fallback=True
    )

    logger.info(f"\n  Validation Report:")
    logger.info(f"  - Syntax valid: {generated.validation_report.syntax_valid}")
    logger.info(f"  - Security safe: {generated.validation_report.security_safe}")
    logger.info(f"  - Implements intervention: {generated.validation_report.implements_intervention}")
    logger.info(f"  - Passes unit tests: {generated.validation_report.passes_unit_tests}")
    logger.info(f"  - Fallback used: {generated.validation_report.fallback_used}")

    logger.info(f"\n  Dependencies: {', '.join(generated.dependencies)}")

    # Summary: Full Report Generation
    logger.info("\n" + "="*80)
    logger.info("FULL REPORT GENERATION")
    logger.info("="*80)
    logger.info("\nTo generate a full clinical fairness report, run:")
    logger.info("  python scripts/generate_full_clinical_report.py")

    # Check LLM logs
    logger.info("\n" + "="*80)
    logger.info("LOGS AND CACHE")
    logger.info("="*80)

    log_count = len(list(Path("llm_logs").glob("*.json")))
    cache_count = len(list(Path("llm_cache").glob("*.json")))

    logger.info(f"\n  LLM call logs: {log_count} files in llm_logs/")
    logger.info(f"  Cached responses: {cache_count} files in llm_cache/")

    logger.info("\n" + "="*80)
    logger.info("TEST COMPLETE")
    logger.info("="*80)
    logger.info("\nAll 4 tiers successfully demonstrated!")
    logger.info("\nNext steps:")
    logger.info("  1. Review llm_logs/ for reproducibility")
    logger.info("  2. Check llm_cache/ for cost optimization")
    logger.info("  3. Run full report generation with benchmark results")


if __name__ == "__main__":
    main()
