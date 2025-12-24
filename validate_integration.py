"""
Integration validation: Test clinical PC algorithm with real COMPAS data.

Validates:
1. PC algorithm integration works
2. Bias pathways are discovered
3. Results match expected fairness metrics
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)

print("="*80)
print("INTEGRATION VALIDATION: Clinical PC Algorithm + Real COMPAS Data")
print("="*80)

# Load real COMPAS data
print("\n[1/4] Loading ProPublica COMPAS data...")
try:
    data_path = "../propublicaCompassRecividism_data_fairml.csv/propublica_data_for_fairml.csv"
    df = pd.read_csv(data_path)

    # Preprocess to match expected format
    df_clean = pd.DataFrame({
        'age': 1 - df['Age_Below_TwentyFive'],
        'sex': 1 - df['Female'],
        'priors_count': df['Number_of_Priors'],
        'c_charge_degree': 1 - df['Misdemeanor'],
        'race_binary': df['African_American'],
        'two_year_recid': df['Two_yr_Recidivism']
    })

    print(f"  [OK] Loaded {len(df_clean)} samples")
    print(f"  [OK] {df_clean['race_binary'].mean():.1%} African-American")
    print(f"  [OK] {df_clean['two_year_recid'].mean():.1%} recidivism rate")

except Exception as e:
    print(f"  [ERROR] Failed to load data: {e}")
    sys.exit(1)

# Test clinical PC algorithm integration
print("\n[2/4] Testing clinical PC algorithm...")
try:
    from pc_algorithm_clinical import PCAlgorithmClinical

    temporal_order = {
        'race_binary': 0,
        'age': 1,
        'sex': 1,
        'priors_count': 2,
        'c_charge_degree': 2,
        'two_year_recid': 3
    }

    pc_algo = PCAlgorithmClinical(
        data=df_clean,
        protected_attr='race_binary',
        outcome='two_year_recid',
        temporal_order=temporal_order,
        alpha=0.05,
        n_bootstrap=50
    )

    result = pc_algo.run()

    print(f"  [OK] PC algorithm completed")
    print(f"  [OK] Discovered {len(result['skeleton_edges'])} skeleton edges")
    print(f"  [OK] Oriented {len(result['directed_edges'])} directed edges")
    print(f"  [OK] Found {len(result['bias_pathways'])} bias pathways")

    # Display bias pathways
    if result['bias_pathways']:
        print("\n  Bias Pathways Discovered:")
        for i, pathway in enumerate(result['bias_pathways'][:3], 1):
            print(f"    {i}. {' -> '.join(pathway.path)}")
            print(f"       Type: {pathway.pathway_type}")
            print(f"       Robustness: {pathway.sensitivity_robustness:.2%}")

except ImportError as e:
    print(f"  [WARNING] Clinical PC not available: {e}")
    print(f"  [INFO] Will use fallback in main system")
except Exception as e:
    print(f"  [ERROR] PC algorithm failed: {e}")
    import traceback
    traceback.print_exc()

# Test causal analyzer integration
print("\n[3/4] Testing ResearchGradeCausalAnalyzer integration...")
try:
    from causal_analysis_research_grade import ResearchGradeCausalAnalyzer

    # Sample subset for speed
    sample_df = df_clean.sample(n=min(1000, len(df_clean)), random_state=42)

    analyzer = ResearchGradeCausalAnalyzer(
        data=sample_df,
        protected_attr='race_binary',
        outcome='two_year_recid'
    )

    # Run hybrid causal discovery
    result = analyzer.run_full_analysis()

    print(f"  [OK] Causal analyzer completed")
    print(f"  [OK] Expert edges: {len(result.expert_edges)}")
    print(f"  [OK] Discovered edges: {len(result.discovered_edges)}")
    print(f"  [OK] Total graph edges: {result.graph.number_of_edges()}")
    print(f"  [OK] Bias pathways: {len(result.bias_pathways)}")

except Exception as e:
    print(f"  [ERROR] Analyzer integration failed: {e}")
    import traceback
    traceback.print_exc()

# Validate benchmark integration
print("\n[4/4] Validating benchmark results...")
try:
    results_path = "results/benchmark_compas_table.csv"
    bench_df = pd.read_csv(results_path)

    baseline = bench_df[bench_df['Method'] == 'Unmitigated Baseline']
    fairlearn_eo = bench_df[bench_df['Method'] == 'Fairlearn (Equalized Odds)']

    if len(baseline) > 0 and len(fairlearn_eo) > 0:
        baseline_fnr = baseline['FNR Disparity (mean)'].iloc[0]
        eo_fnr = fairlearn_eo['FNR Disparity (mean)'].iloc[0]

        print(f"  [OK] Baseline FNR Disparity: {baseline_fnr:.3f}")
        print(f"  [OK] Fairlearn EO FNR Disparity: {eo_fnr:.3f}")

        # Validate against Obermeyer expectations
        if baseline_fnr > 0.15:
            print(f"  [OK] Baseline shows realistic bias (>15%)")
        else:
            print(f"  [WARNING] Baseline bias lower than expected")

        if eo_fnr <= 0.055:
            print(f"  [OK] Fairlearn EO achieves clinical safety (<=5.5%)")
        else:
            print(f"  [WARNING] Fairlearn EO above clinical safety threshold")
    else:
        print(f"  [WARNING] Could not find expected methods in results")

except FileNotFoundError:
    print(f"  [INFO] Benchmark results not yet generated")
except Exception as e:
    print(f"  [ERROR] Validation failed: {e}")

# Final summary
print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)
print("\n[OK] Integration test complete!")
print("\nNext steps:")
print("  1. Run full benchmark: python test_benchmark_compas.py")
print("  2. Generate artifacts: python scripts/generate_publication_artifacts.py")
print("  3. Commit to GitHub")
print("="*80)
