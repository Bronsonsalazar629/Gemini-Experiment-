"""
TIER 3: Intervention Rationale - Safety-First Recommendations

Generates clinical safety-focused rationale for fairness interventions.

Evaluation criteria:
1. Clinical safety (no patient harm)
2. Implementation feasibility (EHR integration, clinician trust)
3. Interpretability (explainable to medical staff)
4. Evidence base (peer-reviewed validation)
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from src.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class SafetyAssessment:
    """Safety assessment for intervention."""
    clinical_safety_score: float  # 0-1
    implementation_complexity: str  # "low", "medium", "high"
    clinician_trust_impact: str  # "positive", "neutral", "negative"
    deployment_recommendation: str  # "recommended", "conditional", "not_recommended"
    evidence_base: str  # "strong", "moderate", "limited", "experimental"


@dataclass
class InterventionRationale:
    """Complete rationale for fairness intervention."""
    intervention_name: str
    safety_narrative: str
    implementation_narrative: str
    interpretability_narrative: str
    safety_assessment: SafetyAssessment
    limitations: List[str]
    monitoring_requirements: List[str]


# ============================================================================
# Safety Knowledge Base
# ============================================================================

INTERVENTION_SAFETY_KB = {
    "Reweighing": {
        "clinical_safety": "Adjusts training sample weights without modifying model architecture or prediction logic, preserving interpretability and clinical decision pathways",
        "implementation": "Requires no EHR system changes; single-line code modification in training pipeline; compatible with existing clinical workflows",
        "interpretability": "Fully interpretable - weights are explicit and auditable; clinicians can review which patient subgroups receive higher/lower training emphasis",
        "safety_score": 0.95,
        "complexity": "low",
        "trust_impact": "positive",
        "deployment": "recommended",
        "evidence_base": "strong",
        "limitations": [
            "Does not address underlying data collection biases",
            "Effectiveness depends on sufficient sample size in minority groups",
            "May not resolve biases from unmeasured confounders"
        ],
        "monitoring": [
            "Track FNR disparity monthly across protected subgroups",
            "Monitor accuracy degradation by demographic stratification",
            "Alert if disparity exceeds 5% clinical safety threshold"
        ]
    },
    "Fairlearn (Equalized Odds)": {
        "clinical_safety": "Constrains model to equalize true positive and false positive rates across demographic groups while maintaining clinical accuracy thresholds",
        "implementation": "Requires model retraining with fairness constraints; compatible with scikit-learn pipelines; moderate integration effort",
        "interpretability": "Moderately interpretable - fairness constraints are explicit, but multiplier adjustments may not be intuitive to all clinicians",
        "safety_score": 0.90,
        "complexity": "medium",
        "trust_impact": "positive",
        "deployment": "recommended",
        "evidence_base": "strong",
        "limitations": [
            "May reduce overall accuracy by 2-5% to achieve fairness",
            "Requires careful threshold calibration for clinical context",
            "Performance varies with class imbalance severity"
        ],
        "monitoring": [
            "Monthly FNR/FPR disparity validation across demographics",
            "Quarterly model recalibration with updated patient data",
            "Continuous accuracy monitoring with alert thresholds"
        ]
    },
    "Fairlearn (Demographic Parity)": {
        "clinical_safety": "Equalizes positive prediction rates across demographic groups, ensuring equal access to clinical interventions",
        "implementation": "Medium complexity; requires modification to training pipeline; compatible with standard ML frameworks",
        "interpretability": "Interpretable - prediction rate parity is straightforward concept for clinical staff",
        "safety_score": 0.85,
        "complexity": "medium",
        "trust_impact": "neutral",
        "deployment": "conditional",
        "evidence_base": "moderate",
        "limitations": [
            "May not address outcome disparities (only prediction rate equity)",
            "Can reduce precision in higher-prevalence demographic groups",
            "Assumes equal base rates across groups (often violated in practice)"
        ],
        "monitoring": [
            "Track prediction rate parity across protected attributes",
            "Monitor actual clinical outcomes by demographic subgroup",
            "Review false discovery rates and positive predictive values"
        ]
    },
    "AIF360 Reweighing": {
        "clinical_safety": "IBM AIF360 implementation of sample reweighing with comprehensive bias metrics; preserves model interpretability",
        "implementation": "Low-medium complexity; requires AIF360 library installation; well-documented integration path",
        "interpretability": "Highly interpretable - provides detailed bias metrics dashboard; weights are transparent and auditable",
        "safety_score": 0.92,
        "complexity": "low",
        "trust_impact": "positive",
        "deployment": "recommended",
        "evidence_base": "strong",
        "limitations": [
            "Requires additional AIF360 dependency in deployment environment",
            "Limited to preprocessing stage (cannot correct in-process biases)",
            "May not handle intersectional fairness (e.g., race × sex)"
        ],
        "monitoring": [
            "Monthly comprehensive fairness metrics dashboard",
            "Disparate impact ratio tracking (0.8-1.2 acceptable range)",
            "Statistical parity difference and equal opportunity monitoring"
        ]
    },
    "Adversarial Debiasing": {
        "clinical_safety": "HIGH RISK: Uses adversarial neural networks that significantly reduce model interpretability; difficult for clinicians to validate predictions",
        "implementation": "High complexity; requires deep learning infrastructure, GPU resources, and specialized ML expertise; not suitable for resource-constrained settings",
        "interpretability": "Poor interpretability - black-box neural network architecture; predictions cannot be traced to clinical features",
        "safety_score": 0.40,
        "complexity": "high",
        "trust_impact": "negative",
        "deployment": "not_recommended",
        "evidence_base": "experimental",
        "limitations": [
            "Black-box model structure reduces clinical trust and validation",
            "Difficult to audit causal pathways and feature importance",
            "Unstable training dynamics; requires extensive hyperparameter tuning",
            "Not suitable for clinical decision support without extensive validation"
        ],
        "monitoring": [
            "Requires extensive pre-deployment validation studies",
            "Real-time prediction monitoring with human oversight",
            "Quarterly explainability audits and model behavior reviews",
            "Prospective clinical trial validation recommended"
        ]
    },
    "Unmitigated Baseline": {
        "clinical_safety": "No fairness intervention applied; preserves existing algorithmic biases from training data",
        "implementation": "No implementation required - current state",
        "interpretability": "Baseline interpretability (varies by model type)",
        "safety_score": 0.60,
        "complexity": "low",
        "trust_impact": "neutral",
        "deployment": "not_recommended",
        "evidence_base": "N/A",
        "limitations": [
            "Perpetuates existing healthcare disparities",
            "May violate ethical principles of justice and equity",
            "Does not address known biases in training data"
        ],
        "monitoring": [
            "Baseline metrics for comparison with interventions",
            "Track disparity trends over time",
            "Document harm from unaddressed bias"
        ]
    }
}


# ============================================================================
# Intervention Recommender
# ============================================================================

class InterventionRecommender:
    """
    Generates safety-focused rationale for fairness interventions.

    Uses:
    - Knowledge base for established methods (fast, validated)
    - Gemini LLM for novel interventions (flexible, comprehensive)
    """

    def __init__(self, gemini_client: GeminiClient):
        """
        Initialize recommender.

        Args:
            gemini_client: Configured Gemini API client
        """
        self.gemini_client = gemini_client

    def generate_intervention_rationale(
        self,
        intervention_name: str,
        outcome: str,
        sensitive_attr: str,
        context: str = "medicare_high_cost",
        accuracy_tradeoff: Optional[float] = None
    ) -> InterventionRationale:
        """
        Generate comprehensive rationale for intervention.

        Args:
            intervention_name: Name of fairness intervention
            outcome: Clinical outcome variable
            sensitive_attr: Protected attribute
            context: Clinical context
            accuracy_tradeoff: Accuracy reduction (if known)

        Returns:
            InterventionRationale with safety assessment
        """
        logger.info(f"Generating rationale for {intervention_name}")

        # Try knowledge base first
        if intervention_name in INTERVENTION_SAFETY_KB:
            return self._generate_kb_rationale(
                intervention_name,
                outcome,
                sensitive_attr,
                context,
                accuracy_tradeoff
            )
        else:
            # Use Gemini for novel interventions
            return self._generate_llm_rationale(
                intervention_name,
                outcome,
                sensitive_attr,
                context,
                accuracy_tradeoff
            )

    def _generate_kb_rationale(
        self,
        intervention_name: str,
        outcome: str,
        sensitive_attr: str,
        context: str,
        accuracy_tradeoff: Optional[float]
    ) -> InterventionRationale:
        """Generate rationale using knowledge base."""
        kb_entry = INTERVENTION_SAFETY_KB[intervention_name]

        # Build narratives
        safety_narrative = kb_entry["clinical_safety"]

        implementation_narrative = f"{kb_entry['implementation']} " \
                                  f"Evidence base: {kb_entry['evidence_base']} (peer-reviewed validation)."

        interpretability_narrative = kb_entry["interpretability"]

        # Add accuracy tradeoff context if available
        if accuracy_tradeoff is not None and accuracy_tradeoff > 0:
            safety_narrative += f" Note: May reduce overall accuracy by {accuracy_tradeoff:.1%}, " \
                               f"but this tradeoff is ethically justified to achieve clinical fairness."

        # Create safety assessment
        assessment = SafetyAssessment(
            clinical_safety_score=kb_entry["safety_score"],
            implementation_complexity=kb_entry["complexity"],
            clinician_trust_impact=kb_entry["trust_impact"],
            deployment_recommendation=kb_entry["deployment"],
            evidence_base=kb_entry["evidence_base"]
        )

        return InterventionRationale(
            intervention_name=intervention_name,
            safety_narrative=safety_narrative,
            implementation_narrative=implementation_narrative,
            interpretability_narrative=interpretability_narrative,
            safety_assessment=assessment,
            limitations=kb_entry["limitations"],
            monitoring_requirements=kb_entry["monitoring"]
        )

    def _generate_llm_rationale(
        self,
        intervention_name: str,
        outcome: str,
        sensitive_attr: str,
        context: str,
        accuracy_tradeoff: Optional[float]
    ) -> InterventionRationale:
        """Generate rationale using Gemini LLM."""
        system_instruction = """You are a clinical AI safety officer at a hospital's
AI governance board. Your role is to evaluate fairness interventions for deployment
in clinical settings, focusing on patient safety and clinical interpretability."""

        prompt = f"""
Evaluate this fairness intervention for clinical deployment:

INTERVENTION: {intervention_name}
CONTEXT: {context}
- Outcome: {outcome}
- Protected attribute: {sensitive_attr}
- Accuracy tradeoff: {f"{accuracy_tradeoff:.1%}" if accuracy_tradeoff else "unknown"}

TASK: Provide safety assessment in these dimensions:

1. CLINICAL SAFETY (2 sentences):
   - Does this intervention modify model logic in ways that reduce interpretability?
   - Could it cause patient harm through misclassification?

2. IMPLEMENTATION FEASIBILITY (2 sentences):
   - EHR system integration complexity
   - Impact on clinician workflow and trust

3. INTERPRETABILITY (2 sentences):
   - Can clinicians understand how the intervention works?
   - Are predictions explainable to patients and medical staff?

4. DEPLOYMENT RECOMMENDATION:
   - "recommended": Safe for immediate deployment
   - "conditional": Safe with monitoring requirements
   - "not_recommended": Safety concerns prevent deployment

5. LIMITATIONS (3 bullet points):
   - Technical limitations
   - Clinical limitations
   - Ethical considerations

6. MONITORING REQUIREMENTS (3 bullet points):
   - What metrics to track
   - How often to review
   - Alert thresholds

Return JSON format:
{{
  "safety_narrative": "...",
  "implementation_narrative": "...",
  "interpretability_narrative": "...",
  "deployment_recommendation": "recommended|conditional|not_recommended",
  "limitations": ["...", "...", "..."],
  "monitoring": ["...", "...", "..."]
}}
"""

        try:
            response = self.gemini_client.call_with_retry(
                prompt,
                temperature=0.3,
                system_instruction=system_instruction
            )

            # Parse JSON response
            import json
            response_text = response.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]
            elif response_text.startswith("```"):
                response_text = response_text[3:-3]

            data = json.loads(response_text)

            # Create assessment
            assessment = SafetyAssessment(
                clinical_safety_score=0.70,  # Default medium score for LLM-generated
                implementation_complexity="medium",
                clinician_trust_impact="neutral",
                deployment_recommendation=data.get("deployment_recommendation", "conditional"),
                evidence_base="experimental"
            )

            return InterventionRationale(
                intervention_name=intervention_name,
                safety_narrative=data.get("safety_narrative", "Unknown safety profile"),
                implementation_narrative=data.get("implementation_narrative", "Implementation complexity unknown"),
                interpretability_narrative=data.get("interpretability_narrative", "Interpretability assessment required"),
                safety_assessment=assessment,
                limitations=data.get("limitations", ["Unknown limitations"]),
                monitoring_requirements=data.get("monitoring", ["Requires monitoring plan"])
            )

        except Exception as e:
            logger.error(f"LLM rationale generation failed: {e}")
            return self._generate_generic_fallback(intervention_name)

    def _generate_generic_fallback(self, intervention_name: str) -> InterventionRationale:
        """Generic fallback for unknown interventions."""
        assessment = SafetyAssessment(
            clinical_safety_score=0.50,
            implementation_complexity="unknown",
            clinician_trust_impact="neutral",
            deployment_recommendation="conditional",
            evidence_base="unknown"
        )

        return InterventionRationale(
            intervention_name=intervention_name,
            safety_narrative=f"{intervention_name}: Safety profile unknown. Requires validation.",
            implementation_narrative="Implementation complexity not assessed.",
            interpretability_narrative="Interpretability assessment required.",
            safety_assessment=assessment,
            limitations=["Unknown intervention - full safety assessment required"],
            monitoring_requirements=["Comprehensive monitoring plan required before deployment"]
        )

    def compare_interventions(
        self,
        interventions: List[str],
        context: str = "medicare_high_cost"
    ) -> Dict[str, InterventionRationale]:
        """
        Generate comparative analysis of multiple interventions.

        Args:
            interventions: List of intervention names
            context: Clinical context

        Returns:
            Dictionary mapping intervention name to rationale
        """
        rationales = {}

        for intervention in interventions:
            rationales[intervention] = self.generate_intervention_rationale(
                intervention,
                outcome="high_cost",
                sensitive_attr="race",
                context=context
            )

        return rationales


def generate_intervention_rationale(
    intervention_name: str,
    outcome: str,
    sensitive_attr: str,
    gemini_client: GeminiClient,
    context: str = "medicare_high_cost",
    accuracy_tradeoff: Optional[float] = None
) -> InterventionRationale:
    """
    Convenience function for intervention rationale generation.

    Args:
        intervention_name: Intervention method name
        outcome: Clinical outcome
        sensitive_attr: Protected attribute
        gemini_client: Gemini API client
        context: Clinical context
        accuracy_tradeoff: Accuracy reduction

    Returns:
        InterventionRationale with safety assessment
    """
    recommender = InterventionRecommender(gemini_client)
    return recommender.generate_intervention_rationale(
        intervention_name,
        outcome,
        sensitive_attr,
        context,
        accuracy_tradeoff
    )
