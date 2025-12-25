"""
End-to-End Clinical Fairness Report Generator

Integrates all 4 Gemini LLM tiers to produce comprehensive fairness analysis:
1. Causal graph refinement with clinical plausibility
2. Bias interpretation with clinical harm translation
3. Intervention rationale with safety assessments
4. Code generation with validation

Output: Clinical fairness report ready for publication/deployment
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd

from src.gemini_client import create_gemini_client, GeminiClient
from src.causal_graph_refiner import CausalGraphRefiner, CausalEdge
from src.bias_interpreter import BiasInterpreter
from src.intervention_recommender import InterventionRecommender
from src.code_generator import CodeGenerator

logger = logging.getLogger(__name__)


# ============================================================================
# Report Data Models
# ============================================================================

@dataclass
class ClinicalFairnessReport:
    """Complete clinical fairness analysis report."""
    # Metadata
    report_id: str
    timestamp: str
    dataset_info: Dict

    # Tier 1: Causal Analysis
    causal_graph_validation: Dict

    # Tier 2: Bias Interpretation
    bias_harm_narratives: Dict

    # Tier 3: Intervention Recommendations
    intervention_rationales: Dict

    # Tier 4: Implementation Code
    generated_code: Dict

    # Summary
    executive_summary: Dict
    deployment_recommendations: List[str]


# ============================================================================
# Report Generator
# ============================================================================

class ClinicalFairnessReportGenerator:
    """
    Generates comprehensive clinical fairness reports using Gemini LLM.

    Orchestrates all 4 tiers of LLM integration to produce actionable
    clinical fairness analysis with implementation code.
    """

    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        """
        Initialize report generator.

        Args:
            gemini_client: Configured Gemini client (can be None for fallback mode)
        """
        self.gemini_client = gemini_client

        # Initialize tier components (they handle None gracefully with fallbacks)
        self.causal_refiner = CausalGraphRefiner(self.gemini_client)
        self.bias_interpreter = BiasInterpreter(self.gemini_client)
        self.intervention_recommender = InterventionRecommender(self.gemini_client)
        self.code_generator = CodeGenerator(self.gemini_client)

        # Output directories
        self.output_dir = Path("results")
        self.code_output_dir = Path("results/generated_code")
        self.output_dir.mkdir(exist_ok=True)
        self.code_output_dir.mkdir(exist_ok=True)

    def generate_report(
        self,
        data: pd.DataFrame,
        protected_attr: str,
        outcome: str,
        expert_edges: List[CausalEdge],
        discovered_edges: List[Tuple[str, str]],
        benchmark_results: Dict,
        context: str = "medicare_high_cost"
    ) -> ClinicalFairnessReport:
        """
        Generate comprehensive clinical fairness report.

        Args:
            data: Clinical dataset
            protected_attr: Protected attribute name
            outcome: Outcome variable name
            expert_edges: Expert-provided causal edges
            discovered_edges: PC algorithm discovered edges
            benchmark_results: Fairness intervention benchmark results
            context: Clinical context

        Returns:
            ClinicalFairnessReport with all analyses
        """
        logger.info("="*80)
        logger.info("GENERATING CLINICAL FAIRNESS REPORT")
        logger.info("="*80)

        report_id = f"clinical_fairness_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Dataset info
        dataset_info = {
            "n_samples": len(data),
            "n_features": len(data.columns),
            "features": list(data.columns),
            "protected_attr": protected_attr,
            "outcome": outcome,
            "outcome_prevalence": float(data[outcome].mean()),
            "context": context
        }

        logger.info(f"Dataset: {len(data)} samples, {len(data.columns)} features")

        # TIER 1: Causal Graph Refinement
        logger.info("\n[TIER 1] Refining causal graph with clinical plausibility...")
        causal_validation = self.causal_refiner.refine_causal_graph(
            expert_edges=expert_edges,
            discovered_edges=discovered_edges,
            data=data,
            protected_attr=protected_attr,
            outcome=outcome
        )
        logger.info(f"  -> Validated {len(causal_validation['edges'])} causal edges")

        # TIER 2: Bias Interpretation
        logger.info("\n[TIER 2] Generating clinical harm narratives...")
        bias_narratives = {}

        for method_name, method_results in benchmark_results.items():
            if 'fnr_disparity' in method_results:
                harm_report = self.bias_interpreter.interpret_bias_clinically(
                    fnr_disparity=method_results['fnr_disparity'],
                    demographic_parity_diff=method_results.get('dp_difference', 0.0),
                    protected_attr=protected_attr,
                    outcome=outcome,
                    context=context,
                    dataset_size=len(data)
                )
                bias_narratives[method_name] = asdict(harm_report)

        logger.info(f"  -> Generated {len(bias_narratives)} harm narratives")

        # TIER 3: Intervention Rationale
        logger.info("\n[TIER 3] Generating intervention rationales...")
        intervention_rationales = {}

        for method_name in benchmark_results.keys():
            # Calculate accuracy tradeoff if baseline available
            accuracy_tradeoff = None
            if 'Unmitigated Baseline' in benchmark_results:
                baseline_acc = benchmark_results['Unmitigated Baseline'].get('accuracy', 0)
                method_acc = benchmark_results[method_name].get('accuracy', 0)
                accuracy_tradeoff = baseline_acc - method_acc

            rationale = self.intervention_recommender.generate_intervention_rationale(
                intervention_name=method_name,
                outcome=outcome,
                sensitive_attr=protected_attr,
                context=context,
                accuracy_tradeoff=accuracy_tradeoff
            )
            intervention_rationales[method_name] = asdict(rationale)

        logger.info(f"  -> Generated {len(intervention_rationales)} intervention rationales")

        # TIER 4: Code Generation (for recommended interventions)
        logger.info("\n[TIER 4] Generating validated implementation code...")
        generated_code = {}

        # Only generate code for recommended interventions
        for method_name, rationale_dict in intervention_rationales.items():
            deployment_rec = rationale_dict['safety_assessment']['deployment_recommendation']

            if deployment_rec in ['recommended', 'conditional']:
                try:
                    code_result = self.code_generator.generate_intervention_code(
                        intervention_name=method_name,
                        data=data,
                        sensitive_attr=protected_attr,
                        outcome=outcome
                    )
                    generated_code[method_name] = asdict(code_result)

                    # Save code to file
                    code_file = self.code_output_dir / f"{method_name.replace(' ', '_').lower()}.py"
                    with open(code_file, 'w') as f:
                        f.write(code_result.code)
                        f.write("\n\n# " + "="*70 + "\n")
                        f.write(f"# Validation Report\n")
                        f.write(f"# Syntax Valid: {code_result.validation_report.syntax_valid}\n")
                        f.write(f"# Security Safe: {code_result.validation_report.security_safe}\n")
                        f.write(f"# Functional Tests Pass: {code_result.validation_report.passes_unit_tests}\n")
                        f.write("# " + "="*70 + "\n\n")
                        f.write(code_result.usage_example)

                    logger.info(f"  -> Generated code for {method_name} ({code_file})")

                except Exception as e:
                    logger.warning(f"  -> Code generation failed for {method_name}: {e}")

        # Executive Summary
        executive_summary = self._generate_executive_summary(
            dataset_info,
            causal_validation,
            bias_narratives,
            intervention_rationales,
            generated_code
        )

        # Deployment Recommendations
        deployment_recommendations = self._generate_deployment_recommendations(
            intervention_rationales,
            bias_narratives
        )

        # Create report
        report = ClinicalFairnessReport(
            report_id=report_id,
            timestamp=datetime.now().isoformat(),
            dataset_info=dataset_info,
            causal_graph_validation=causal_validation,
            bias_harm_narratives=bias_narratives,
            intervention_rationales=intervention_rationales,
            generated_code=generated_code,
            executive_summary=executive_summary,
            deployment_recommendations=deployment_recommendations
        )

        # Save report
        self._save_report(report)

        logger.info("\n" + "="*80)
        logger.info("REPORT GENERATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Report ID: {report_id}")
        logger.info(f"Output directory: {self.output_dir}")

        return report

    def _generate_executive_summary(
        self,
        dataset_info: Dict,
        causal_validation: Dict,
        bias_narratives: Dict,
        intervention_rationales: Dict,
        generated_code: Dict
    ) -> Dict:
        """Generate executive summary of findings."""
        # Find best intervention (lowest FNR disparity with clinical safety)
        best_intervention = None
        best_fnr = float('inf')

        for method, narrative in bias_narratives.items():
            if method == "Unmitigated Baseline":
                continue

            fnr = narrative.get('fnr_disparity', float('inf'))
            if fnr < best_fnr:
                best_fnr = fnr
                best_intervention = method

        # Count validated edges
        validated_edges = [e for e in causal_validation['edges'] if e['edge_type'] == 'validated']

        return {
            "dataset_summary": f"{dataset_info['n_samples']} patients, {dataset_info['n_features']} features",
            "causal_analysis": f"{len(causal_validation['edges'])} total causal edges ({len(validated_edges)} LLM-validated)",
            "fairness_interventions_evaluated": len(intervention_rationales),
            "recommended_intervention": best_intervention,
            "best_fnr_disparity": f"{best_fnr:.1%}",
            "implementation_code_generated": len(generated_code),
            "clinical_safety_status": "SAFE" if best_fnr < 0.05 else "REVIEW_REQUIRED"
        }

    def _generate_deployment_recommendations(
        self,
        intervention_rationales: Dict,
        bias_narratives: Dict
    ) -> List[str]:
        """Generate actionable deployment recommendations."""
        recommendations = []

        # Find recommended interventions
        for method, rationale in intervention_rationales.items():
            deployment = rationale['safety_assessment']['deployment_recommendation']

            if deployment == 'recommended':
                recommendations.append(
                    f"RECOMMENDED: Deploy {method} - {rationale['safety_narrative'][:100]}..."
                )
            elif deployment == 'conditional':
                recommendations.append(
                    f"CONDITIONAL: {method} requires monitoring - {rationale['limitations'][0] if rationale['limitations'] else 'Review required'}"
                )

        # Add harm mitigation priority
        worst_harm = None
        worst_fnr = 0

        for method, narrative in bias_narratives.items():
            if method == "Unmitigated Baseline":
                fnr = narrative.get('fnr_disparity', 0)
                if fnr > worst_fnr:
                    worst_fnr = fnr
                    worst_harm = narrative.get('narrative', '')

        if worst_harm:
            recommendations.insert(0, f"PRIORITY: Address baseline bias - {worst_harm[:150]}...")

        return recommendations

    def _save_report(self, report: ClinicalFairnessReport):
        """Save report to JSON file."""
        report_path = self.output_dir / f"{report.report_id}.json"

        with open(report_path, 'w') as f:
            json.dump(asdict(report), f, indent=2)

        logger.info(f"Saved report: {report_path}")

        # Also save human-readable summary
        summary_path = self.output_dir / f"{report.report_id}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CLINICAL FAIRNESS ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Report ID: {report.report_id}\n")
            f.write(f"Generated: {report.timestamp}\n\n")

            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*80 + "\n")
            for key, value in report.executive_summary.items():
                f.write(f"{key}: {value}\n")

            f.write("\n\nDEPLOYMENT RECOMMENDATIONS\n")
            f.write("-"*80 + "\n")
            for i, rec in enumerate(report.deployment_recommendations, 1):
                f.write(f"{i}. {rec}\n")

            f.write("\n\nCLINICAL HARM NARRATIVES\n")
            f.write("-"*80 + "\n")
            for method, narrative in report.bias_harm_narratives.items():
                f.write(f"\n{method}:\n")
                f.write(f"{narrative.get('narrative', 'N/A')}\n")

        logger.info(f"Saved summary: {summary_path}")


def generate_clinical_fairness_report(
    data: pd.DataFrame,
    protected_attr: str,
    outcome: str,
    expert_edges: List[CausalEdge],
    discovered_edges: List[Tuple[str, str]],
    benchmark_results: Dict,
    context: str = "medicare_high_cost",
    gemini_client: Optional[GeminiClient] = None
) -> ClinicalFairnessReport:
    """
    Convenience function to generate clinical fairness report.

    Args:
        data: Clinical dataset
        protected_attr: Protected attribute
        outcome: Outcome variable
        expert_edges: Expert causal edges
        discovered_edges: Discovered causal edges
        benchmark_results: Benchmark results
        context: Clinical context
        gemini_client: Gemini client (optional)

    Returns:
        ClinicalFairnessReport
    """
    generator = ClinicalFairnessReportGenerator(gemini_client)
    return generator.generate_report(
        data=data,
        protected_attr=protected_attr,
        outcome=outcome,
        expert_edges=expert_edges,
        discovered_edges=discovered_edges,
        benchmark_results=benchmark_results,
        context=context
    )
