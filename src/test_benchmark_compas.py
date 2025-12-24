"""
Test systematic benchmarking on synthetic COMPAS dataset.

This demonstrates the experimental validation pipeline:
1. Synthetic COMPAS dataset (n=5000) with realistic bias patterns
2. 5-fold stratified cross-validation
3. Baselines: Unmitigated, Fairlearn (DP), Fairlearn (EO), AIF360 Reweighing
4. Statistical testing with paired t-tests
"""

import sys
sys.path.append('src')

from systematic_experiments import SystematicBenchmarkExperiments, Dataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    print("="*80)
    print("SYSTEMATIC BENCHMARK: SYNTHETIC COMPAS DATASET")
    print("="*80)
    print("\nThis benchmark validates fairness interventions on a synthetic recidivism")
    print("prediction task with realistic bias patterns based on ProPublica's COMPAS study.")
    print("\nProtocol:")
    print("  - Dataset: Synthetic COMPAS (n=5000, 51% African-American)")
    print("  - Task: Binary recidivism prediction")
    print("  - Protected attribute: race_binary")
    print("  - Features: age, sex, priors_count, charge_degree")
    print("  - Bias: Direct discrimination + systemic correlation with priors")
    print("  - CV: 5-fold stratified cross-validation")
    print("  - Baselines: Unmitigated, Fairlearn (DP), Fairlearn (EO), AIF360 Reweighing")
    print("  - Metrics: Accuracy, FNR disparity, DP difference, EO difference")
    print("  - Statistical testing: Paired t-tests (alpha=0.05) with Cohen's d")
    print("\n" + "="*80 + "\n")

    # Initialize experiment runner
    runner = SystematicBenchmarkExperiments(seed=42, n_folds=5)

    # Run benchmark on synthetic COMPAS
    print("Running benchmark (this may take 2-3 minutes)...\n")
    report = runner.run_benchmarks(dataset=Dataset.COMPAS)

    # Export results
    json_path, csv_path, stats_path = runner.export_results(report)

    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\nDataset: {report.dataset}")
    print(f"Samples: {report.n_samples}")
    print(f"Folds: {report.n_folds}")
    print(f"Sensitive Attribute: {report.sensitive_attribute}")
    print(f"Outcome: {report.outcome_variable}")
    print(f"Data Hash: {report.data_hash}")

    print("\n" + "-"*80)
    print("TABLE 1: METHOD COMPARISON")
    print("-"*80)
    print(f"{'Method':<35} {'Accuracy':<12} {'FNR Disp':<12} {'DP Diff':<12} {'Safety':<10}")
    print("-"*80)

    for br in sorted(report.aggregated_results, key=lambda x: x.fnr_disparity_mean):
        print(f"{br.method:<35} "
              f"{br.accuracy_mean:.3f}±{br.accuracy_std:.3f}  "
              f"{br.fnr_disparity_mean:.3f}±{br.fnr_disparity_std:.3f}  "
              f"{br.dp_diff_mean:.3f}±{br.dp_diff_std:.3f}  "
              f"{br.clinical_safety_score:<10}")

    print("\n" + "-"*80)
    print("TABLE 2: STATISTICAL SIGNIFICANCE (vs Unmitigated Baseline)")
    print("-"*80)
    print(f"{'Method':<35} {'Metric':<20} {'p-value':<10} {'Cohen d':<10} {'Winner':<20}")
    print("-"*80)

    for st in report.statistical_tests:
        sig_marker = "***" if st.significant else "ns"
        print(f"{st.method_b:<35} "
              f"{st.metric:<20} "
              f"{st.p_value:.4f} {sig_marker:<5} "
              f"{st.cohens_d:>8.3f}  "
              f"{st.winner:<20}")

    print("\n" + "="*80)
    print("CLINICAL INTERPRETATION")
    print("="*80)

    # Find best method by FNR disparity (clinical safety metric)
    best_method = min(report.aggregated_results, key=lambda x: x.fnr_disparity_mean)
    baseline_method = next(br for br in report.aggregated_results if "Baseline" in br.method)

    print(f"\nBest method by FNR disparity: {best_method.method}")
    print(f"  FNR Disparity: {best_method.fnr_disparity_mean:.4f} (threshold: 0.05)")
    print(f"  Clinical Safety: {best_method.clinical_safety_score}")
    print(f"  Accuracy: {best_method.accuracy_mean:.4f}")

    print(f"\nBaseline comparison:")
    print(f"  Baseline FNR Disparity: {baseline_method.fnr_disparity_mean:.4f}")
    print(f"  Improvement: {baseline_method.fnr_disparity_mean - best_method.fnr_disparity_mean:.4f}")

    if best_method.fnr_disparity_mean <= 0.05:
        print("\n[OK] SAFE FOR DEPLOYMENT: FNR disparity within 5% threshold")
    elif best_method.fnr_disparity_mean <= 0.10:
        print("\n[WARNING] CONDITIONAL DEPLOYMENT: FNR disparity 5-10%, requires monitoring")
    else:
        print("\n[NOT SAFE] HIGH RISK: FNR disparity exceeds 10%, do not deploy")

    print("\n" + "="*80)
    print("EXPORTED FILES")
    print("="*80)
    print(f"Full results (JSON): {json_path}")
    print(f"Method comparison (CSV): {csv_path}")
    print(f"Statistical tests (CSV): {stats_path}")
    print("\nReady for FAccT/NeurIPS ML4H submission!")
    print("="*80 + "\n")
