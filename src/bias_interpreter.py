"""
TIER 2: Bias Interpretation - Clinical Harm Translation

Translates statistical fairness metrics into real-world clinical harm narratives
that clinicians and patients can understand.

Focus: Human impact, not statistics
Principles: Justice, beneficence, non-maleficence
Output: Actionable narratives for policy intervention
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field

from src.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


# ============================================================================
# Data Models
# ============================================================================

class ClinicalHarmNarrative(BaseModel):
    """Structured clinical harm narrative."""
    harm_description: str = Field(description="Specific clinical harm (deaths, amputations, etc)")
    affected_population: str = Field(description="Which patients are harmed")
    ethical_violation: str = Field(description="Which medical ethics principle is violated")
    relative_risk: float = Field(ge=1.0, description="Relative risk multiplier")
    actionable_intervention: str = Field(description="Concrete intervention to address harm")


@dataclass
class BiasHarmReport:
    """Complete bias harm assessment."""
    fnr_disparity: float
    dp_difference: float
    protected_attr: str
    outcome: str
    context: str
    narrative: str
    affected_patients_estimate: Optional[int] = None
    preventable_outcomes_estimate: Optional[int] = None


# ============================================================================
# Clinical Harm Templates
# ============================================================================

HARM_TEMPLATES = {
    "diabetic_amputation": {
        "baseline_risk": 0.12,  # 12% amputation rate in Medicare diabetics
        "harm_unit": "preventable amputations",
        "ethical_principle": "justice (equitable access to limb-saving care)",
        "template": """
In diabetic care, a False Negative Rate (FNR) disparity of {fnr:.1%} means
that {group} patients are {relative_risk:.1f}x more likely to have preventable
amputations because they are incorrectly classified as low-risk for specialist
referral. This violates the ethical principle of {principle} by denying equitable
access to podiatry and vascular surgery interventions.
"""
    },
    "medicare_high_cost": {
        "baseline_risk": 0.25,  # 25% high-cost threshold
        "harm_unit": "missed high-risk patients",
        "ethical_principle": "justice (fair resource allocation)",
        "template": """
A False Negative Rate (FNR) disparity of {fnr:.1%} means that {group} patients
are {relative_risk:.1f}x more likely to be incorrectly classified as low-cost,
leading to inadequate care management and preventable hospitalizations. This
violates the principle of {principle} by systematically underestimating care
needs for vulnerable populations.
"""
    },
    "hospital_readmission": {
        "baseline_risk": 0.18,  # 18% readmission rate
        "harm_unit": "preventable readmissions",
        "ethical_principle": "beneficence (prevent patient harm)",
        "template": """
A False Negative Rate (FNR) disparity of {fnr:.1%} indicates {group} patients
are {relative_risk:.1f}x more likely to experience preventable readmissions
due to inadequate discharge planning and follow-up. This violates {principle}
by failing to provide equivalent post-discharge support.
"""
    }
}


# ============================================================================
# Bias Interpreter
# ============================================================================

class BiasInterpreter:
    """
    Translates fairness metrics into clinical harm narratives.

    Uses:
    - Predefined templates for known contexts (fast, consistent)
    - Gemini LLM for novel contexts (flexible, comprehensive)
    - Always focuses on human impact, not statistics
    """

    def __init__(self, gemini_client: GeminiClient):
        """
        Initialize interpreter.

        Args:
            gemini_client: Configured Gemini API client
        """
        self.gemini_client = gemini_client

    def interpret_bias_clinically(
        self,
        fnr_disparity: float,
        demographic_parity_diff: float,
        protected_attr: str,
        outcome: str,
        context: str = "medicare_high_cost",
        dataset_size: Optional[int] = None
    ) -> BiasHarmReport:
        """
        Generate clinical harm narrative from fairness metrics.

        Args:
            fnr_disparity: False Negative Rate disparity
            demographic_parity_diff: Demographic parity difference
            protected_attr: Protected attribute (e.g., "race", "sex")
            outcome: Outcome variable
            context: Clinical context
            dataset_size: Number of patients (for impact estimates)

        Returns:
            BiasHarmReport with narrative and impact estimates
        """
        logger.info(f"Generating clinical harm narrative for {context}")

        # Try template-based approach first (fast, consistent)
        if context in HARM_TEMPLATES:
            narrative = self._generate_template_narrative(
                fnr_disparity,
                demographic_parity_diff,
                protected_attr,
                outcome,
                context
            )
        else:
            # Use Gemini for novel contexts
            narrative = self._generate_llm_narrative(
                fnr_disparity,
                demographic_parity_diff,
                protected_attr,
                outcome,
                context
            )

        # Calculate impact estimates
        affected_patients = None
        preventable_outcomes = None
        if dataset_size and context in HARM_TEMPLATES:
            template = HARM_TEMPLATES[context]
            baseline_risk = template["baseline_risk"]

            # Estimate affected patients
            affected_patients = int(dataset_size * 0.5)  # Assume 50% minority

            # Estimate preventable outcomes
            preventable_outcomes = int(
                affected_patients * baseline_risk * fnr_disparity
            )

        return BiasHarmReport(
            fnr_disparity=fnr_disparity,
            dp_difference=demographic_parity_diff,
            protected_attr=protected_attr,
            outcome=outcome,
            context=context,
            narrative=narrative,
            affected_patients_estimate=affected_patients,
            preventable_outcomes_estimate=preventable_outcomes
        )

    def _generate_template_narrative(
        self,
        fnr_disparity: float,
        dp_diff: float,
        protected_attr: str,
        outcome: str,
        context: str
    ) -> str:
        """Generate narrative using predefined template."""
        template_info = HARM_TEMPLATES[context]

        # Determine affected group
        if "race" in protected_attr.lower():
            group = "Black" if context == "diabetic_amputation" else "Non-White"
        elif "sex" in protected_attr.lower() or "gender" in protected_attr.lower():
            group = "female"
        else:
            group = "disadvantaged"

        # Calculate relative risk
        baseline = template_info["baseline_risk"]
        relative_risk = (baseline + fnr_disparity) / baseline if baseline > 0 else 1.5

        # Format narrative
        narrative = template_info["template"].format(
            fnr=fnr_disparity,
            group=group,
            relative_risk=relative_risk,
            principle=template_info["ethical_principle"]
        ).strip()

        # Add demographic parity context if significant
        if dp_diff > 0.05:
            narrative += f"\n\nAdditionally, a demographic parity difference of {dp_diff:.1%} " \
                        f"indicates systematic over-representation of {group} patients in " \
                        f"negative outcome predictions, suggesting algorithmic bias."

        return narrative

    def _generate_llm_narrative(
        self,
        fnr_disparity: float,
        dp_diff: float,
        protected_attr: str,
        outcome: str,
        context: str
    ) -> str:
        """Generate narrative using Gemini LLM."""
        system_instruction = """You are a clinical AI ethicist specializing in translating
statistical bias into real-world patient harm. Your role is to help clinicians and
policymakers understand the human impact of algorithmic bias in healthcare."""

        prompt = f"""
Translate this fairness metric into real-world clinical harm:

CONTEXT: {context}
- Protected attribute: {protected_attr}
- Outcome: {outcome}
- False Negative Rate (FNR) disparity: {fnr_disparity:.1%}
- Demographic parity difference: {dp_diff:.1%}

TASK: Explain in 2-3 sentences:
1. What SPECIFIC clinical harm this causes (be concrete: deaths, amputations,
   hospitalizations, delays in care, etc.)
2. Which patient population is harmed (be specific about demographics)
3. Why this violates medical ethics principles (justice, beneficence,
   non-maleficence, autonomy)

RULES:
- DO NOT mention statistics or percentages
- Focus on human impact: "patients experience..." not "rates differ by..."
- Be specific about clinical outcomes, not abstract
- Mention ethical principles explicitly
- Keep under 100 words

Example (for diabetic amputation):
"Black diabetic patients are 1.5 times more likely to lose limbs because AI
incorrectly flags them as low-risk for podiatry referral. This denies them
access to preventive foot care that saves White patients' limbs, violating
the principle of justice (fair access to care) and beneficence (prevent harm)."
"""

        try:
            response = self.gemini_client.call_with_retry(
                prompt,
                temperature=0.7,  # Higher for more natural language
                system_instruction=system_instruction
            )
            return response.strip()

        except Exception as e:
            logger.error(f"LLM narrative generation failed: {e}")
            # Fallback to generic template
            return self._generate_generic_fallback(
                fnr_disparity, protected_attr, outcome
            )

    def _generate_generic_fallback(
        self,
        fnr_disparity: float,
        protected_attr: str,
        outcome: str
    ) -> str:
        """Generic fallback when both templates and LLM fail."""
        group = "minority" if "race" in protected_attr.lower() else "disadvantaged"
        return (
            f"An FNR disparity of {fnr_disparity:.1%} means {group} patients are "
            f"systematically more likely to have negative outcomes ({outcome}) due "
            f"to incorrect risk classification. This violates the ethical principle "
            f"of justice by denying equitable healthcare access."
        )

    def generate_intervention_narrative(
        self,
        baseline_fnr: float,
        intervention_fnr: float,
        intervention_name: str,
        context: str = "medicare_high_cost"
    ) -> str:
        """
        Generate narrative explaining improvement from intervention.

        Args:
            baseline_fnr: FNR disparity before intervention
            intervention_fnr: FNR disparity after intervention
            intervention_name: Name of intervention method
            context: Clinical context

        Returns:
            Human-readable narrative of improvement
        """
        improvement = baseline_fnr - intervention_fnr
        pct_reduction = (improvement / baseline_fnr * 100) if baseline_fnr > 0 else 0

        system_instruction = """You are a clinical AI ethicist explaining fairness
interventions to hospital administrators and policymakers."""

        prompt = f"""
Explain this fairness intervention in plain language:

CONTEXT: {context}
- Intervention: {intervention_name}
- Baseline FNR disparity: {baseline_fnr:.1%}
- After intervention FNR disparity: {intervention_fnr:.1%}
- Improvement: {improvement:.1%} ({pct_reduction:.0f}% reduction)

TASK: In 2 sentences, explain:
1. What clinical outcomes improved (be specific)
2. How many patients benefit (use relative terms like "significantly fewer")

Example:
"{intervention_name} reduces preventable amputations by {pct_reduction:.0f}%,
meaning significantly fewer Black diabetic patients lose limbs due to delayed
specialist referrals. This brings care equity closer to the clinical safety
threshold."

Keep under 50 words. Focus on outcomes, not methods.
"""

        try:
            response = self.gemini_client.call_with_retry(
                prompt,
                temperature=0.5,
                system_instruction=system_instruction
            )
            return response.strip()

        except Exception as e:
            logger.error(f"Intervention narrative generation failed: {e}")
            return (
                f"{intervention_name} reduces bias by {pct_reduction:.0f}%, "
                f"improving fairness from {baseline_fnr:.1%} to {intervention_fnr:.1%} "
                f"FNR disparity."
            )


def interpret_bias_clinically(
    fnr_disparity: float,
    demographic_parity_diff: float,
    protected_attr: str,
    outcome: str,
    gemini_client: GeminiClient,
    context: str = "medicare_high_cost",
    dataset_size: Optional[int] = None
) -> BiasHarmReport:
    """
    Convenience function for clinical bias interpretation.

    Args:
        fnr_disparity: FNR disparity metric
        demographic_parity_diff: DP difference metric
        protected_attr: Protected attribute
        outcome: Outcome variable
        gemini_client: Gemini API client
        context: Clinical context
        dataset_size: Dataset size for impact estimates

    Returns:
        BiasHarmReport with clinical narrative
    """
    interpreter = BiasInterpreter(gemini_client)
    return interpreter.interpret_bias_clinically(
        fnr_disparity,
        demographic_parity_diff,
        protected_attr,
        outcome,
        context,
        dataset_size
    )
