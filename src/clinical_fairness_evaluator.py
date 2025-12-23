"""
Clinical Fairness Evaluator - Healthcare Safety-Oriented Metrics

This module implements fairness evaluation with clinical safety as the primary
objective. Designed for high-stakes medical decision support systems where
false negatives can harm patient health.

Key Principles:
1. False Negative Rate (FNR) parity is PRIMARY (patient safety)
2. Calibration is SECONDARY (trust in predictions)
3. Demographic parity is TERTIARY (access equity)
4. Acknowledge impossibility theorems (Kleinberg et al. 2018)
5. Provide deployment verdicts with risk assessment

References:
- Kleinberg, J., et al. (2018). Inherent Trade-Offs in Algorithmic Fairness.
- Obermeyer, Z., et al. (2019). Dissecting racial bias in an algorithm.
- KDIGO (2012). Clinical Practice Guideline for CKD.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class RiskLevel(Enum):
    """Clinical risk assessment levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DeploymentVerdict(Enum):
    """Deployment safety verdict."""
    SAFE = "[OK] SAFE"
    CONDITIONAL = "[WARNING] CONDITIONAL"
    NOT_SAFE = "[NOT SAFE]"


@dataclass
class GroupClinicalMetrics:
    """Clinical performance metrics for a demographic group."""
    group_name: str
    n_samples: int

    # Classification metrics
    fnr: float  # False Negative Rate (CRITICAL for patient safety)
    fpr: float  # False Positive Rate
    tpr: float  # True Positive Rate (Recall/Sensitivity)
    tnr: float  # True Negative Rate (Specificity)
    ppv: float  # Positive Predictive Value (Precision)
    npv: float  # Negative Predictive Value

    # Clinical interpretation
    missed_cases: int          # False negatives (patients denied care)
    false_alarms: int          # False positives (unnecessary referrals)
    total_high_need: int       # Actual positive cases
    total_low_need: int        # Actual negative cases
    pct_missed: float          # Percentage of high-need patients missed

    # Calibration
    calibration_error: float   # Expected Calibration Error (ECE)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'group_name': self.group_name,
            'n_samples': int(self.n_samples),
            'fnr': float(round(self.fnr, 4)),
            'fpr': float(round(self.fpr, 4)),
            'tpr': float(round(self.tpr, 4)),
            'ppv': float(round(self.ppv, 4)),
            'missed_cases': int(self.missed_cases),
            'total_high_need': int(self.total_high_need),
            'pct_missed': float(round(self.pct_missed, 2)),
            'calibration_error': float(round(self.calibration_error, 4))
        }


@dataclass
class FairnessDisparity:
    """Measures disparity between groups for a specific metric."""
    metric_name: str
    reference_group: str
    comparisons: Dict[str, float]  # {group: disparity_value}
    max_disparity: float
    exceeds_threshold: bool
    threshold: float
    clinical_impact: str


@dataclass
class ClinicalFairnessReport:
    """Comprehensive fairness audit with clinical interpretation."""
    # Group-level metrics
    group_metrics: Dict[str, GroupClinicalMetrics]

    # Disparity analysis
    fnr_disparity: FairnessDisparity
    calibration_disparity: FairnessDisparity
    demographic_parity_disparity: FairnessDisparity

    # Impossibility theorem acknowledgment
    impossibility_statement: str
    chosen_priority: str

    # Deployment decision
    verdict: DeploymentVerdict
    risk_level: RiskLevel
    next_steps: List[str]

    # Metadata
    n_total_samples: int
    sensitive_attribute: str
    outcome_name: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export."""
        from datetime import datetime

        return {
            'summary': {
                'verdict': self.verdict.value,
                'risk_level': self.risk_level.value,
                'n_samples': self.n_total_samples,
                'sensitive_attribute': self.sensitive_attribute
            },
            'group_metrics': {
                name: metrics.to_dict()
                for name, metrics in self.group_metrics.items()
            },
            'disparities': {
                'fnr': {
                    'max_disparity': float(self.fnr_disparity.max_disparity),
                    'threshold': float(self.fnr_disparity.threshold),
                    'exceeds_threshold': bool(self.fnr_disparity.exceeds_threshold),
                    'clinical_impact': self.fnr_disparity.clinical_impact
                },
                'calibration': {
                    'max_disparity': float(self.calibration_disparity.max_disparity),
                    'threshold': float(self.calibration_disparity.threshold),
                    'exceeds_threshold': bool(self.calibration_disparity.exceeds_threshold)
                },
                'demographic_parity': {
                    'max_disparity': float(self.demographic_parity_disparity.max_disparity),
                    'threshold': float(self.demographic_parity_disparity.threshold),
                    'exceeds_threshold': bool(self.demographic_parity_disparity.exceeds_threshold)
                }
            },
            'impossibility_theorem': {
                'statement': self.impossibility_statement,
                'chosen_priority': self.chosen_priority
            },
            'next_steps': self.next_steps,
            'timestamp': self.timestamp
        }


# ============================================================================
# CLINICALLY-CENTRIC FAIRNESS EVALUATOR
# ============================================================================

class ClinicallyCentricFairnessEvaluator:
    """
    Fairness evaluator prioritizing clinical safety over statistical parity.

    Metric Hierarchy:
    1. PRIMARY: False Negative Rate (FNR) parity - patient safety
    2. SECONDARY: Calibration - trust and reliability
    3. TERTIARY: Demographic parity - access equity

    Thresholds based on clinical acceptability, not arbitrary 0.1 cutoffs.
    """

    def __init__(
        self,
        max_fnr_disparity: float = 0.05,
        max_calibration_error: float = 0.08,
        max_demographic_parity_diff: float = 0.10,
        n_calibration_bins: int = 5
    ):
        """
        Initialize evaluator with clinically-motivated thresholds.

        Args:
            max_fnr_disparity: Maximum acceptable FNR difference (default 5%)
            max_calibration_error: Maximum ECE per group (default 8%)
            max_demographic_parity_diff: Maximum demographic parity gap
            n_calibration_bins: Number of bins for calibration analysis
        """
        self.max_fnr_disparity = max_fnr_disparity
        self.max_calibration_error = max_calibration_error
        self.max_demographic_parity_diff = max_demographic_parity_diff
        self.n_calibration_bins = n_calibration_bins

        logger.info(f"Initialized ClinicallyCentricFairnessEvaluator")
        logger.info(f"  Max FNR disparity: {max_fnr_disparity:.1%}")
        logger.info(f"  Max calibration error: {max_calibration_error:.1%}")

    def _compute_group_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        group_name: str
    ) -> GroupClinicalMetrics:
        """
        Compute comprehensive clinical metrics for a single group.

        Args:
            y_true: True labels (1 = needs referral, 0 = does not need)
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            group_name: Name of demographic group

        Returns:
            GroupClinicalMetrics with all computed values
        """
        n = len(y_true)

        # Confusion matrix components
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        # Counts for clinical interpretation
        total_high_need = int(np.sum(y_true == 1))
        total_low_need = int(np.sum(y_true == 0))
        missed_cases = int(fn)
        false_alarms = int(fp)

        # Rates
        fnr = fn / total_high_need if total_high_need > 0 else 0.0
        fpr = fp / total_low_need if total_low_need > 0 else 0.0
        tpr = tp / total_high_need if total_high_need > 0 else 0.0
        tnr = tn / total_low_need if total_low_need > 0 else 0.0

        # Predictive values
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        pct_missed = (fn / total_high_need * 100) if total_high_need > 0 else 0.0

        # Calibration
        calibration_error = self._compute_expected_calibration_error(
            y_true, y_pred_proba
        )

        return GroupClinicalMetrics(
            group_name=group_name,
            n_samples=n,
            fnr=fnr,
            fpr=fpr,
            tpr=tpr,
            tnr=tnr,
            ppv=ppv,
            npv=npv,
            missed_cases=missed_cases,
            false_alarms=false_alarms,
            total_high_need=total_high_need,
            total_low_need=total_low_need,
            pct_missed=pct_missed,
            calibration_error=calibration_error
        )

    def _compute_expected_calibration_error(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> float:
        """
        Compute Expected Calibration Error (ECE) using equal-width binning.

        ECE = Σ (n_bin / n_total) * |acc_bin - conf_bin|

        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities

        Returns:
            Expected Calibration Error (0 to 1)
        """
        n = len(y_true)
        bin_edges = np.linspace(0, 1, self.n_calibration_bins + 1)

        ece = 0.0
        for i in range(self.n_calibration_bins):
            bin_mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])

            if i == self.n_calibration_bins - 1:  # Last bin includes right edge
                bin_mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba <= bin_edges[i + 1])

            n_bin = np.sum(bin_mask)

            if n_bin > 0:
                bin_accuracy = np.mean(y_true[bin_mask])
                bin_confidence = np.mean(y_pred_proba[bin_mask])
                ece += (n_bin / n) * abs(bin_accuracy - bin_confidence)

        return ece

    def _analyze_fnr_disparity(
        self,
        group_metrics: Dict[str, GroupClinicalMetrics]
    ) -> FairnessDisparity:
        """
        Analyze False Negative Rate disparity across groups.

        Clinical Impact: FNR disparity means some racial/ethnic groups have
        higher rates of missed diagnoses or denied specialist care.
        """
        fnrs = {name: metrics.fnr for name, metrics in group_metrics.items()}

        # Use group with lowest FNR as reference (best outcome)
        reference_group = min(fnrs, key=fnrs.get)
        reference_fnr = fnrs[reference_group]

        # Compute disparity for each group relative to reference
        comparisons = {}
        for group_name, fnr in fnrs.items():
            if group_name != reference_group:
                comparisons[group_name] = fnr - reference_fnr

        max_disparity = max(comparisons.values()) if comparisons else 0.0
        exceeds_threshold = max_disparity > self.max_fnr_disparity

        # Clinical impact statement
        if exceeds_threshold:
            worst_group = max(comparisons, key=comparisons.get)
            missed_diff = group_metrics[worst_group].pct_missed - group_metrics[reference_group].pct_missed
            clinical_impact = (
                f"CRITICAL: {worst_group} patients have {missed_diff:.1f}% higher "
                f"rate of missed high-need cases than {reference_group} patients. "
                f"This disparity threatens patient safety and health equity."
            )
        else:
            clinical_impact = "FNR disparity within acceptable clinical threshold."

        return FairnessDisparity(
            metric_name="False Negative Rate",
            reference_group=reference_group,
            comparisons=comparisons,
            max_disparity=max_disparity,
            exceeds_threshold=exceeds_threshold,
            threshold=self.max_fnr_disparity,
            clinical_impact=clinical_impact
        )

    def _analyze_calibration_disparity(
        self,
        group_metrics: Dict[str, GroupClinicalMetrics]
    ) -> FairnessDisparity:
        """
        Analyze calibration error disparity across groups.

        Clinical Impact: Poor calibration means predictions are not trustworthy.
        Calibration disparity means trust varies by demographic group.
        """
        calibration_errors = {
            name: metrics.calibration_error
            for name, metrics in group_metrics.items()
        }

        max_error = max(calibration_errors.values())
        exceeds_threshold = max_error > self.max_calibration_error

        # Use best-calibrated group as reference
        reference_group = min(calibration_errors, key=calibration_errors.get)

        comparisons = {
            name: error - calibration_errors[reference_group]
            for name, error in calibration_errors.items()
            if name != reference_group
        }

        max_disparity = max(comparisons.values()) if comparisons else 0.0

        return FairnessDisparity(
            metric_name="Calibration Error (ECE)",
            reference_group=reference_group,
            comparisons=comparisons,
            max_disparity=max_disparity,
            exceeds_threshold=exceeds_threshold,
            threshold=self.max_calibration_error,
            clinical_impact="Poor calibration undermines clinical trust in predictions."
        )

    def _analyze_demographic_parity(
        self,
        group_metrics: Dict[str, GroupClinicalMetrics]
    ) -> FairnessDisparity:
        """
        Analyze demographic parity (positive prediction rate) across groups.

        Note: This is TERTIARY priority. Per Kleinberg et al. (2018),
        demographic parity and equalized odds cannot both hold unless
        base rates are equal.
        """
        positive_rates = {}
        for name, metrics in group_metrics.items():
            total_positive_pred = metrics.missed_cases + (metrics.total_high_need - metrics.missed_cases)
            positive_rates[name] = total_positive_pred / metrics.n_samples if metrics.n_samples > 0 else 0.0

        reference_group = list(positive_rates.keys())[0]  # Arbitrary reference

        comparisons = {
            name: rate - positive_rates[reference_group]
            for name, rate in positive_rates.items()
            if name != reference_group
        }

        max_disparity = max(abs(v) for v in comparisons.values()) if comparisons else 0.0
        exceeds_threshold = max_disparity > self.max_demographic_parity_diff

        return FairnessDisparity(
            metric_name="Demographic Parity",
            reference_group=reference_group,
            comparisons=comparisons,
            max_disparity=max_disparity,
            exceeds_threshold=exceeds_threshold,
            threshold=self.max_demographic_parity_diff,
            clinical_impact="Disparity in referral rates may indicate access barriers."
        )

    def _determine_deployment_verdict(
        self,
        fnr_disparity: FairnessDisparity,
        calibration_disparity: FairnessDisparity,
        demographic_parity: FairnessDisparity
    ) -> Tuple[DeploymentVerdict, RiskLevel, List[str]]:
        """
        Determine if model is safe for clinical deployment.

        Decision tree:
        1. FNR disparity > threshold → NOT SAFE (patient safety risk)
        2. Calibration error > threshold → CONDITIONAL (trust issue)
        3. DP disparity > threshold → CONDITIONAL (access equity concern)
        4. All pass → SAFE
        """
        next_steps = []

        # CRITICAL: FNR disparity threatens patient safety
        if fnr_disparity.exceeds_threshold:
            verdict = DeploymentVerdict.NOT_SAFE
            risk_level = RiskLevel.CRITICAL
            next_steps.extend([
                f"URGENT: FNR disparity is {fnr_disparity.max_disparity:.1%}, exceeds {fnr_disparity.threshold:.1%} threshold",
                "Apply fairness intervention: Reweighing or Equalized Odds Postprocessing",
                "Re-evaluate with clinician oversight before deployment",
                "Consider separate models per group if disparity persists"
            ])

        # HIGH: Calibration issues undermine clinical trust
        elif calibration_disparity.exceeds_threshold:
            verdict = DeploymentVerdict.CONDITIONAL
            risk_level = RiskLevel.HIGH
            next_steps.extend([
                f"Calibration error exceeds {calibration_disparity.threshold:.1%} threshold",
                "Apply temperature scaling or Platt scaling per group",
                "Deploy with probability score disclaimers",
                "Require clinician review of high-uncertainty cases"
            ])

        # MEDIUM: Access equity concern
        elif demographic_parity.exceeds_threshold:
            verdict = DeploymentVerdict.CONDITIONAL
            risk_level = RiskLevel.MEDIUM
            next_steps.extend([
                f"Demographic parity gap: {demographic_parity.max_disparity:.1%}",
                "Investigate structural barriers (insurance, geography)",
                "Deploy with monitoring dashboard tracking referral rates",
                "Quarterly fairness audits required"
            ])

        # All metrics pass
        else:
            verdict = DeploymentVerdict.SAFE
            risk_level = RiskLevel.LOW
            next_steps.extend([
                "All fairness metrics within acceptable thresholds",
                "Deploy with standard monitoring (monthly fairness checks)",
                "Document baseline metrics for ongoing surveillance"
            ])

        return verdict, risk_level, next_steps

    def comprehensive_fairness_audit(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_pred_proba: pd.Series,
        sensitive_attr: pd.Series,
        outcome_name: str = "clinical_decision"
    ) -> ClinicalFairnessReport:
        """
        Perform comprehensive fairness audit with clinical interpretation.

        Args:
            y_true: Ground truth labels (1 = positive outcome needed)
            y_pred: Model predictions (1 = positive prediction)
            y_pred_proba: Model probability estimates (0 to 1)
            sensitive_attr: Protected attribute (e.g., race, gender)
            outcome_name: Name of clinical outcome (for reporting)

        Returns:
            ClinicalFairnessReport with verdict, disparities, and next steps
        """
        from datetime import datetime

        logger.info("=" * 70)
        logger.info("CLINICAL FAIRNESS AUDIT")
        logger.info("=" * 70)

        # Convert to numpy arrays
        y_true_arr = y_true.values if hasattr(y_true, 'values') else np.array(y_true)
        y_pred_arr = y_pred.values if hasattr(y_pred, 'values') else np.array(y_pred)
        y_pred_proba_arr = y_pred_proba.values if hasattr(y_pred_proba, 'values') else np.array(y_pred_proba)
        sensitive_arr = sensitive_attr.values if hasattr(sensitive_attr, 'values') else np.array(sensitive_attr)

        # Compute metrics per group
        unique_groups = np.unique(sensitive_arr)
        group_metrics = {}

        for group in unique_groups:
            mask = (sensitive_arr == group)
            metrics = self._compute_group_metrics(
                y_true_arr[mask],
                y_pred_arr[mask],
                y_pred_proba_arr[mask],
                str(group)
            )
            group_metrics[str(group)] = metrics

            logger.info(f"\nGroup: {group}")
            logger.info(f"  n = {metrics.n_samples}")
            logger.info(f"  FNR = {metrics.fnr:.1%} ({metrics.missed_cases}/{metrics.total_high_need} missed)")
            logger.info(f"  Calibration Error = {metrics.calibration_error:.1%}")

        # Analyze disparities (hierarchy: FNR > Calibration > DP)
        fnr_disparity = self._analyze_fnr_disparity(group_metrics)
        calibration_disparity = self._analyze_calibration_disparity(group_metrics)
        demographic_parity = self._analyze_demographic_parity(group_metrics)

        # Impossibility theorem statement
        impossibility_statement = (
            "Per Kleinberg et al. (2018), demographic parity and equalized odds "
            "cannot both be satisfied unless base rates are equal across groups. "
            "We prioritize equalized odds (specifically FNR parity) because "
            "false negative disparity threatens patient safety."
        )

        # Deployment verdict
        verdict, risk_level, next_steps = self._determine_deployment_verdict(
            fnr_disparity, calibration_disparity, demographic_parity
        )

        logger.info("\n" + "=" * 70)
        logger.info(f"VERDICT: {verdict.value}")
        logger.info(f"RISK LEVEL: {risk_level.value}")
        logger.info("=" * 70)

        report = ClinicalFairnessReport(
            group_metrics=group_metrics,
            fnr_disparity=fnr_disparity,
            calibration_disparity=calibration_disparity,
            demographic_parity_disparity=demographic_parity,
            impossibility_statement=impossibility_statement,
            chosen_priority="Equalized Odds (FNR Parity) - Patient Safety First",
            verdict=verdict,
            risk_level=risk_level,
            next_steps=next_steps,
            n_total_samples=len(y_true_arr),
            sensitive_attribute=sensitive_attr.name if hasattr(sensitive_attr, 'name') else "protected_attribute",
            outcome_name=outcome_name,
            timestamp=datetime.now().isoformat()
        )

        return report


# ============================================================================
# EXPORT UTILITIES
# ============================================================================

def export_fairness_report(report: ClinicalFairnessReport, output_path: str) -> None:
    """Export fairness report to JSON."""
    import json
    from pathlib import Path

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(report.to_dict(), f, indent=2)

    logger.info(f"Exported fairness report to {output_path}")


if __name__ == "__main__":
    # Test with demo data
    logging.basicConfig(level=logging.INFO)

    from pathlib import Path

    data_path = Path(__file__).parent.parent / "data" / "sample" / "demo_data.csv"

    if data_path.exists():
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.linear_model import LogisticRegression

        # Load data
        data = pd.read_csv(data_path)

        # Encode categorical
        data_encoded = data.copy()
        for col in ['race', 'gender', 'insurance_type']:
            if col in data.columns:
                le = LabelEncoder()
                data_encoded[f'{col}_encoded'] = le.fit_transform(data[col])

        # Train simple model
        feature_cols = [
            'age', 'race_encoded', 'gender_encoded', 'creatinine_level',
            'chronic_conditions', 'insurance_type_encoded', 'prior_visits',
            'distance_to_hospital'
        ]

        X = data_encoded[feature_cols]
        y = data_encoded['referral']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = pd.Series(model.predict(X_test), index=X_test.index)
        y_pred_proba = pd.Series(model.predict_proba(X_test)[:, 1], index=X_test.index)

        # Get race for test set
        race_test = data.loc[X_test.index, 'race']

        # Evaluate fairness
        evaluator = ClinicallyCentricFairnessEvaluator(
            max_fnr_disparity=0.05,
            max_calibration_error=0.08,
            max_demographic_parity_diff=0.10
        )

        report = evaluator.comprehensive_fairness_audit(
            y_true=y_test,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            sensitive_attr=race_test,
            outcome_name="kidney_specialist_referral"
        )

        print("\n" + "=" * 70)
        print("CLINICAL FAIRNESS AUDIT SUMMARY")
        print("=" * 70)
        print(f"\nVerdict: {report.verdict.value}")
        print(f"Risk Level: {report.risk_level.value}")
        print(f"\nFNR Disparity: {report.fnr_disparity.max_disparity:.1%}")
        print(f"  Threshold: {report.fnr_disparity.threshold:.1%}")
        print(f"  Exceeds: {report.fnr_disparity.exceeds_threshold}")
        print(f"\n{report.fnr_disparity.clinical_impact}")

        print(f"\nNext Steps:")
        for i, step in enumerate(report.next_steps, 1):
            print(f"  {i}. {step}")

        # Export
        output_path = Path(__file__).parent.parent / "results" / "clinical_fairness_audit.json"
        export_fairness_report(report, str(output_path))

    else:
        print(f"Demo data not found at {data_path}")
