"""
Intervention Engine Module

Recommends and simulates fairness interventions based on detected bias.
Supports multiple intervention strategies including:
- Reweighing
- Resampling
- Adversarial Debiasing
- Prejudice Remover
- Calibrated Equalized Odds
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class InterventionRecommendation:
    """
    Represents a single fairness intervention recommendation.

    Attributes:
        name: Intervention method name
        category: Type of intervention (preprocessing/inprocessing/postprocessing)
        priority: Recommendation priority (1=highest)
        description: Human-readable description
        expected_impact: Expected fairness improvement (qualitative)
        complexity: Implementation complexity (low/medium/high)
        preserves_accuracy: Whether method tends to preserve model accuracy
        clinical_rationale: Clinical justification from Gemini 3
    """
    name: str
    category: str
    priority: int
    description: str
    expected_impact: str
    complexity: str
    preserves_accuracy: bool
    parameters: Dict[str, Any]
    clinical_rationale: str = ""


class InterventionEngine:
    """
    Recommends and evaluates fairness interventions based on bias analysis.

    Provides both automated recommendations and customizable intervention strategies.
    """

    def __init__(self, bias_threshold: float = 0.1):
        """
        Initialize the intervention engine.

        Args:
            bias_threshold: Threshold for fairness metrics (e.g., 0.1 for 10% disparity)
        """
        self.bias_threshold = bias_threshold
        self.available_interventions = self._initialize_interventions()

    def _initialize_interventions(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize catalog of available fairness interventions.

        Returns:
            Dictionary mapping intervention names to their properties
        """
        return {
            "Reweighing": {
                "category": "preprocessing",
                "description": "Reweights training samples to remove bias before model training",
                "pros": ["Simple to implement", "Model-agnostic", "Preserves all data"],
                "cons": ["May not address complex bias patterns"],
                "complexity": "low",
                "preserves_accuracy": True,
                "best_for": ["demographic_parity"]
            },
            "Resampling": {
                "category": "preprocessing",
                "description": "Balances training data by oversampling/undersampling groups",
                "pros": ["Straightforward", "Works with any model"],
                "cons": ["May lose information", "Can overfit minority groups"],
                "complexity": "low",
                "preserves_accuracy": False,
                "best_for": ["demographic_parity", "equal_opportunity"]
            },
            "Adversarial Debiasing": {
                "category": "inprocessing",
                "description": "Uses adversarial learning to remove bias during training",
                "pros": ["Sophisticated", "Can handle complex bias"],
                "cons": ["Requires deep learning", "Harder to tune"],
                "complexity": "high",
                "preserves_accuracy": False,
                "best_for": ["demographic_parity", "equalized_odds"]
            },
            "Prejudice Remover": {
                "category": "inprocessing",
                "description": "Adds regularization term to objective function to reduce discrimination",
                "pros": ["Integrated into training", "Theoretically grounded"],
                "cons": ["Model-specific", "Requires hyperparameter tuning"],
                "complexity": "medium",
                "preserves_accuracy": True,
                "best_for": ["demographic_parity"]
            },
            "Calibrated Equalized Odds": {
                "category": "postprocessing",
                "description": "Adjusts model predictions to satisfy equalized odds",
                "pros": ["Model-agnostic", "Optimizes specific fairness criteria"],
                "cons": ["Requires calibration set", "May reduce accuracy"],
                "complexity": "medium",
                "preserves_accuracy": False,
                "best_for": ["equalized_odds", "equal_opportunity"]
            },
            "Reject Option Classification": {
                "category": "postprocessing",
                "description": "Adjusts predictions in uncertainty region to improve fairness",
                "pros": ["Flexible", "Targets uncertain predictions"],
                "cons": ["Requires probability scores", "May affect calibration"],
                "complexity": "medium",
                "preserves_accuracy": True,
                "best_for": ["equalized_odds"]
            },
            "Fairness Constraints": {
                "category": "inprocessing",
                "description": "Adds fairness constraints to model optimization",
                "pros": ["Direct control over fairness-accuracy tradeoff"],
                "cons": ["Requires specialized solver", "Implementation complexity"],
                "complexity": "high",
                "preserves_accuracy": False,
                "best_for": ["demographic_parity", "equalized_odds"]
            }
        }

    def suggest_interventions(
        self,
        bias_report: Dict[str, Any],
        max_recommendations: int = 3,
        prioritize_accuracy: bool = True
    ) -> List[InterventionRecommendation]:
        """
        Suggest fairness interventions based on bias analysis.

        Args:
            bias_report: Output from bias_detection.compute_fairness_metrics
            max_recommendations: Maximum number of interventions to recommend
            prioritize_accuracy: Whether to prefer accuracy-preserving methods

        Returns:
            List of InterventionRecommendation objects, sorted by priority

        Example:
            >>> engine = InterventionEngine()
            >>> recommendations = engine.suggest_interventions(bias_metrics)
            >>> for rec in recommendations:
            ...     print(f"{rec.name}: {rec.description}")
        """
        logger.info("Generating intervention recommendations")

        # Analyze which fairness criteria are violated
        violations = self._identify_violations(bias_report)

        if not violations:
            logger.info("No fairness violations detected")
            return []

        # Score and rank interventions
        scored_interventions = []

        for name, properties in self.available_interventions.items():
            score = self._score_intervention(
                name,
                properties,
                violations,
                prioritize_accuracy
            )
            scored_interventions.append((name, properties, score))

        # Sort by score (higher is better)
        scored_interventions.sort(key=lambda x: x[2], reverse=True)

        # Create recommendations
        recommendations = []

        # Extract sensitive_attr and outcome from bias_report
        sensitive_attr = "protected_attribute"
        outcome = "outcome"
        if bias_report.get('group_metrics'):
            groups = list(bias_report['group_metrics'].keys())
            if groups:
                sensitive_attr = "sensitive_attribute"

        for i, (name, properties, score) in enumerate(scored_interventions[:max_recommendations]):
            rec = InterventionRecommendation(
                name=name,
                category=properties['category'],
                priority=i + 1,
                description=properties['description'],
                expected_impact=self._estimate_impact(name, violations),
                complexity=properties['complexity'],
                preserves_accuracy=properties['preserves_accuracy'],
                parameters=self._get_default_parameters(name),
                clinical_rationale=self._generate_clinical_rationale(name, sensitive_attr, outcome)
            )
            recommendations.append(rec)

        logger.info(f"Generated {len(recommendations)} recommendations")
        return recommendations

    def _identify_violations(self, bias_report: Dict[str, Any]) -> Dict[str, float]:
        """
        Identify which fairness criteria are violated.

        Args:
            bias_report: Bias metrics from detector

        Returns:
            Dictionary mapping criterion name to violation severity (0-1)
        """
        violations = {}

        # Check demographic parity
        if 'demographic_parity' in bias_report:
            dp_diff = bias_report['demographic_parity'].get('difference', 0)
            if dp_diff > self.bias_threshold:
                violations['demographic_parity'] = min(dp_diff, 1.0)

        # Check equalized odds
        if 'equalized_odds' in bias_report:
            eo_diff = bias_report['equalized_odds'].get('average_difference', 0)
            if eo_diff > self.bias_threshold:
                violations['equalized_odds'] = min(eo_diff, 1.0)

        # Check equal opportunity
        if 'equal_opportunity' in bias_report:
            eop_diff = bias_report['equal_opportunity'].get('difference', 0)
            if eop_diff > self.bias_threshold:
                violations['equal_opportunity'] = min(eop_diff, 1.0)

        # Check predictive parity
        if 'predictive_parity' in bias_report:
            pp_diff = bias_report['predictive_parity'].get('difference', 0)
            if pp_diff > self.bias_threshold:
                violations['predictive_parity'] = min(pp_diff, 1.0)

        return violations

    def _score_intervention(
        self,
        name: str,
        properties: Dict[str, Any],
        violations: Dict[str, float],
        prioritize_accuracy: bool
    ) -> float:
        """
        Score an intervention based on how well it addresses violations.

        Args:
            name: Intervention name
            properties: Intervention properties
            violations: Detected fairness violations
            prioritize_accuracy: Whether to prefer accuracy-preserving methods

        Returns:
            Score (higher is better)
        """
        score = 0.0

        # Match with violations
        best_for = properties.get('best_for', [])
        for criterion, severity in violations.items():
            if criterion in best_for:
                score += severity * 10  # Weight by violation severity

        # Complexity penalty
        complexity_penalty = {
            'low': 0,
            'medium': -2,
            'high': -5
        }
        score += complexity_penalty.get(properties['complexity'], 0)

        # Accuracy preservation bonus
        if prioritize_accuracy and properties.get('preserves_accuracy', False):
            score += 3

        return score

    def _estimate_impact(self, intervention_name: str, violations: Dict[str, float]) -> str:
        """
        Estimate the expected impact of an intervention.

        Args:
            intervention_name: Name of the intervention
            violations: Current fairness violations

        Returns:
            Qualitative impact description
        """
        properties = self.available_interventions[intervention_name]
        best_for = properties.get('best_for', [])

        matched_violations = [v for v in violations.keys() if v in best_for]

        if len(matched_violations) >= 2:
            return "High - Addresses multiple fairness criteria"
        elif len(matched_violations) == 1:
            return "Medium - Addresses primary fairness violation"
        else:
            return "Low - May provide partial improvement"

    def _get_default_parameters(self, intervention_name: str) -> Dict[str, Any]:
        """
        Get default parameters for an intervention.

        Args:
            intervention_name: Name of the intervention

        Returns:
            Dictionary of default parameters
        """
        defaults = {
            "Reweighing": {
                "privileged_groups": "auto",
                "unprivileged_groups": "auto"
            },
            "Resampling": {
                "strategy": "uniform",
                "sampling_rate": 1.0
            },
            "Adversarial Debiasing": {
                "adversary_loss_weight": 0.1,
                "num_epochs": 50,
                "batch_size": 128
            },
            "Prejudice Remover": {
                "eta": 1.0,
                "sensitive_attr": "auto"
            },
            "Calibrated Equalized Odds": {
                "cost_constraint": "weighted",
                "seed": 42
            },
            "Reject Option Classification": {
                "metric": "statistical_parity",
                "threshold": 0.5
            },
            "Fairness Constraints": {
                "constraint_type": "demographic_parity",
                "epsilon": 0.01
            }
        }

        return defaults.get(intervention_name, {})

    def _generate_clinical_rationale(
        self,
        intervention_name: str,
        sensitive_attr: str,
        outcome: str
    ) -> str:
        """
        Generate clinical rationale using Gemini 3.

        Args:
            intervention_name: Name of the intervention
            sensitive_attr: Protected attribute name
            outcome: Outcome variable name

        Returns:
            Clinical justification (1 sentence)
        """
        from causal_analysis import _init_gemini, _call_gemini_with_retry

        gemini_model = _init_gemini()

        if not gemini_model:
            return f"{intervention_name} may reduce bias in {outcome} predictions."

        prompt = f"""Why is {intervention_name} appropriate for mitigating bias in {outcome} prediction with sensitive attribute {sensitive_attr}?

Consider:
1. Clinical safety implications
2. Model interpretability for healthcare providers
3. Regulatory compliance (HIPAA, FDA)

Respond in exactly 1 sentence."""

        response = _call_gemini_with_retry(gemini_model, prompt)
        return response.strip() if response else f"{intervention_name} addresses systematic disparities while maintaining clinical validity."

    def simulate_intervention(
        self,
        intervention_name: str,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sensitive_attr: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Simulate the effect of an intervention (placeholder).

        In production, this would actually apply the intervention and
        re-evaluate fairness metrics.

        Args:
            intervention_name: Name of intervention to apply
            model: Original model
            X_train, y_train: Training data
            X_test, y_test: Test data
            sensitive_attr: Protected attribute
            parameters: Intervention-specific parameters

        Returns:
            Dictionary with simulated results including new fairness metrics
        """
        logger.info(f"Simulating intervention: {intervention_name}")

        # TODO: Implement actual intervention simulation using AIF360/Fairlearn
        # For now, return mock results

        mock_results = {
            "intervention": intervention_name,
            "status": "simulated (placeholder)",
            "original_metrics": {
                "demographic_parity_difference": 0.25,
                "equalized_odds_difference": 0.20,
                "accuracy": 0.75
            },
            "new_metrics": {
                "demographic_parity_difference": 0.08,
                "equalized_odds_difference": 0.09,
                "accuracy": 0.73
            },
            "improvement": {
                "demographic_parity": 0.17,
                "equalized_odds": 0.11,
                "accuracy_loss": -0.02
            },
            "recommendation": "Accept - Significant fairness improvement with minimal accuracy loss"
        }

        logger.warning("Returning mock simulation results. Implement actual intervention logic.")
        return mock_results


def suggest_interventions(bias_report: Dict[str, Any]) -> List[str]:
    """
    Convenience function to suggest interventions.

    Args:
        bias_report: Bias metrics from detector

    Returns:
        List of intervention names

    Example:
        >>> interventions = suggest_interventions(bias_metrics)
        >>> print("Recommended interventions:", interventions)
    """
    engine = InterventionEngine()
    recommendations = engine.suggest_interventions(bias_report)
    return [rec.name for rec in recommendations]


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)

    # Mock bias report
    mock_bias_report = {
        'demographic_parity': {
            'difference': 0.25,
            'ratio': 0.70
        },
        'equalized_odds': {
            'tpr_difference': 0.18,
            'fpr_difference': 0.12,
            'average_difference': 0.15
        },
        'equal_opportunity': {
            'difference': 0.18,
            'ratio': 0.75
        },
        'predictive_parity': {
            'difference': 0.08,
            'ratio': 0.90
        }
    }

    print("Mock Bias Report:")
    print(f"  Demographic Parity Difference: {mock_bias_report['demographic_parity']['difference']}")
    print(f"  Equalized Odds Difference: {mock_bias_report['equalized_odds']['average_difference']}")
    print()

    # Get recommendations
    engine = InterventionEngine(bias_threshold=0.1)
    recommendations = engine.suggest_interventions(mock_bias_report, max_recommendations=5)

    print("=" * 70)
    print("INTERVENTION RECOMMENDATIONS")
    print("=" * 70)
    print()

    for rec in recommendations:
        print(f"Priority {rec.priority}: {rec.name}")
        print(f"  Category: {rec.category}")
        print(f"  Description: {rec.description}")
        print(f"  Expected Impact: {rec.expected_impact}")
        print(f"  Complexity: {rec.complexity}")
        print(f"  Preserves Accuracy: {rec.preserves_accuracy}")
        print(f"  Parameters: {rec.parameters}")
        print()
