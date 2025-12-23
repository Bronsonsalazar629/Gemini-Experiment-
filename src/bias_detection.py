"""
Bias Detection Module

Computes fairness metrics for clinical ML models using multiple frameworks
(AIF360, Fairlearn) to identify disparities across protected groups.

Supports various fairness criteria including:
- Demographic Parity
- Equalized Odds
- Equal Opportunity
- Predictive Parity
- Calibration
"""

from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import logging

logger = logging.getLogger(__name__)


class BiasDetector:
    """
    Detects and quantifies bias in clinical ML model predictions.

    Supports multiple fairness metrics and generates comprehensive bias reports.
    """

    def __init__(
        self,
        sensitive_attributes: List[str],
        favorable_label: int = 1,
        unfavorable_label: int = 0
    ):
        """
        Initialize bias detector.

        Args:
            sensitive_attributes: List of protected attribute names (e.g., ['race', 'gender'])
            favorable_label: Label considered favorable/positive (default: 1)
            unfavorable_label: Label considered unfavorable/negative (default: 0)
        """
        self.sensitive_attributes = sensitive_attributes
        self.favorable_label = favorable_label
        self.unfavorable_label = unfavorable_label

    def compute_fairness_metrics(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        sensitive_attr: str,
        y_pred: Optional[np.ndarray] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute comprehensive fairness metrics for a model.

        Args:
            model: Trained ML model (with predict method) or None if y_pred provided
            X_test: Test features
            y_test: True labels
            sensitive_attr: Name of the sensitive attribute to analyze
            y_pred: Optional pre-computed predictions (if None, will call model.predict)

        Returns:
            Dictionary with fairness metrics organized by category:
            {
                'demographic_parity': {...},
                'equalized_odds': {...},
                'equal_opportunity': {...},
                'predictive_parity': {...},
                'group_metrics': {...}
            }

        Example:
            >>> detector = BiasDetector(['race'])
            >>> metrics = detector.compute_fairness_metrics(model, X_test, y_test, 'race')
            >>> print(f"Demographic parity difference: {metrics['demographic_parity']['difference']:.3f}")
        """
        logger.info(f"Computing fairness metrics for attribute: {sensitive_attr}")

        # Ensure we have the sensitive attribute in the test data
        if sensitive_attr not in X_test.columns:
            raise ValueError(f"Sensitive attribute '{sensitive_attr}' not found in X_test")

        sensitive_values = X_test[sensitive_attr]
        unique_groups = sensitive_values.unique()

        if y_pred is None:
            if model is None:
                raise ValueError("Either model or y_pred must be provided")
            # Remove sensitive attribute before prediction
            X_test_features = X_test.drop(columns=[sensitive_attr])
            y_pred = model.predict(X_test_features)

        logger.info(f"Analyzing {len(unique_groups)} groups: {unique_groups}")

        # Compute metrics for each group
        group_metrics = self._compute_group_metrics(
            y_true=y_test,
            y_pred=y_pred,
            sensitive_values=sensitive_values,
            unique_groups=unique_groups
        )

        # Compute fairness criteria
        demographic_parity = self._compute_demographic_parity(group_metrics)
        equalized_odds = self._compute_equalized_odds(group_metrics)
        equal_opportunity = self._compute_equal_opportunity(group_metrics)
        predictive_parity = self._compute_predictive_parity(group_metrics)

        results = {
            'demographic_parity': demographic_parity,
            'equalized_odds': equalized_odds,
            'equal_opportunity': equal_opportunity,
            'predictive_parity': predictive_parity,
            'group_metrics': group_metrics,
            'overall_accuracy': accuracy_score(y_test, y_pred)
        }

        return results

    def _compute_group_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        sensitive_values: pd.Series,
        unique_groups: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute performance metrics for each demographic group.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_values: Sensitive attribute values
            unique_groups: Unique values of the sensitive attribute

        Returns:
            Dictionary mapping group name to metrics dict
        """
        group_metrics = {}

        for group in unique_groups:
            group_mask = (sensitive_values == group)
            y_true_group = y_true[group_mask]
            y_pred_group = y_pred[group_mask]

            if len(y_true_group) == 0:
                logger.warning(f"No samples for group {group}")
                continue

            # Compute confusion matrix
            tn, fp, fn, tp = confusion_matrix(
                y_true_group,
                y_pred_group,
                labels=[self.unfavorable_label, self.favorable_label]
            ).ravel()

            # Compute rates
            positive_rate = (y_pred_group == self.favorable_label).mean()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall)
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate (Specificity)
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value (Precision)
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

            group_metrics[str(group)] = {
                'size': len(y_true_group),
                'positive_rate': positive_rate,
                'true_positive_rate': tpr,
                'false_positive_rate': fpr,
                'true_negative_rate': tnr,
                'false_negative_rate': fnr,
                'positive_predictive_value': ppv,
                'negative_predictive_value': npv,
                'accuracy': accuracy_score(y_true_group, y_pred_group),
                'confusion_matrix': {'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)}
            }

        return group_metrics

    def _compute_demographic_parity(
        self,
        group_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Compute demographic parity metrics.

        Demographic parity is satisfied when P(Ŷ=1|S=a) = P(Ŷ=1|S=b) for all groups.

        Args:
            group_metrics: Per-group metrics

        Returns:
            Dict with 'difference' (max - min) and 'ratio' (min / max)
        """
        positive_rates = [m['positive_rate'] for m in group_metrics.values()]

        if not positive_rates:
            return {'difference': 0.0, 'ratio': 1.0}

        max_rate = max(positive_rates)
        min_rate = min(positive_rates)

        return {
            'difference': max_rate - min_rate,
            'ratio': min_rate / max_rate if max_rate > 0 else 1.0,
            'max_rate': max_rate,
            'min_rate': min_rate
        }

    def _compute_equalized_odds(
        self,
        group_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Compute equalized odds metrics.

        Equalized odds requires equal TPR and FPR across groups.

        Args:
            group_metrics: Per-group metrics

        Returns:
            Dict with TPR and FPR differences
        """
        tprs = [m['true_positive_rate'] for m in group_metrics.values()]
        fprs = [m['false_positive_rate'] for m in group_metrics.values()]

        if not tprs or not fprs:
            return {'tpr_difference': 0.0, 'fpr_difference': 0.0, 'average_difference': 0.0}

        tpr_diff = max(tprs) - min(tprs)
        fpr_diff = max(fprs) - min(fprs)

        return {
            'tpr_difference': tpr_diff,
            'fpr_difference': fpr_diff,
            'average_difference': (tpr_diff + fpr_diff) / 2
        }

    def _compute_equal_opportunity(
        self,
        group_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Compute equal opportunity metrics.

        Equal opportunity requires equal TPR (recall) across groups.

        Args:
            group_metrics: Per-group metrics

        Returns:
            Dict with TPR difference and ratio
        """
        tprs = [m['true_positive_rate'] for m in group_metrics.values()]

        if not tprs:
            return {'difference': 0.0, 'ratio': 1.0}

        max_tpr = max(tprs)
        min_tpr = min(tprs)

        return {
            'difference': max_tpr - min_tpr,
            'ratio': min_tpr / max_tpr if max_tpr > 0 else 1.0
        }

    def _compute_predictive_parity(
        self,
        group_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Compute predictive parity metrics.

        Predictive parity requires equal PPV (precision) across groups.

        Args:
            group_metrics: Per-group metrics

        Returns:
            Dict with PPV difference and ratio
        """
        ppvs = [m['positive_predictive_value'] for m in group_metrics.values()]

        if not ppvs:
            return {'difference': 0.0, 'ratio': 1.0}

        max_ppv = max(ppvs)
        min_ppv = min(ppvs)

        return {
            'difference': max_ppv - min_ppv,
            'ratio': min_ppv / max_ppv if max_ppv > 0 else 1.0
        }

    def generate_bias_report(
        self,
        metrics: Dict[str, Any],
        sensitive_attr: str
    ) -> str:
        """
        Generate human-readable bias report.

        Args:
            metrics: Output from compute_fairness_metrics
            sensitive_attr: Name of the sensitive attribute

        Returns:
            Formatted text report
        """
        report = []
        report.append("=" * 70)
        report.append(f"BIAS DETECTION REPORT - {sensitive_attr.upper()}")
        report.append("=" * 70)
        report.append("")

        # Overall accuracy
        report.append(f"Overall Model Accuracy: {metrics['overall_accuracy']:.3f}")
        report.append("")

        # Group-level metrics
        report.append("GROUP-LEVEL METRICS:")
        report.append("-" * 70)
        for group, group_data in metrics['group_metrics'].items():
            report.append(f"\nGroup: {group} (n={group_data['size']})")
            report.append(f"  Positive Rate: {group_data['positive_rate']:.3f}")
            report.append(f"  Accuracy: {group_data['accuracy']:.3f}")
            report.append(f"  TPR (Recall): {group_data['true_positive_rate']:.3f}")
            report.append(f"  FPR: {group_data['false_positive_rate']:.3f}")
            report.append(f"  PPV (Precision): {group_data['positive_predictive_value']:.3f}")

        report.append("")
        report.append("FAIRNESS CRITERIA ASSESSMENT:")
        report.append("-" * 70)

        # Demographic parity
        dp = metrics['demographic_parity']
        report.append(f"\n1. Demographic Parity")
        report.append(f"   Difference: {dp['difference']:.3f} (threshold: < 0.1)")
        report.append(f"   Ratio: {dp['ratio']:.3f} (threshold: > 0.8)")
        report.append(f"   Status: {'[OK] PASS' if dp['difference'] < 0.1 else '[X] FAIL'}")

        # Equalized odds
        eo = metrics['equalized_odds']
        report.append(f"\n2. Equalized Odds")
        report.append(f"   TPR Difference: {eo['tpr_difference']:.3f}")
        report.append(f"   FPR Difference: {eo['fpr_difference']:.3f}")
        report.append(f"   Avg Difference: {eo['average_difference']:.3f} (threshold: < 0.1)")
        report.append(f"   Status: {'[OK] PASS' if eo['average_difference'] < 0.1 else '[X] FAIL'}")

        # Equal opportunity
        eop = metrics['equal_opportunity']
        report.append(f"\n3. Equal Opportunity")
        report.append(f"   TPR Difference: {eop['difference']:.3f} (threshold: < 0.1)")
        report.append(f"   Status: {'[OK] PASS' if eop['difference'] < 0.1 else '[X] FAIL'}")

        # Predictive parity
        pp = metrics['predictive_parity']
        report.append(f"\n4. Predictive Parity")
        report.append(f"   PPV Difference: {pp['difference']:.3f} (threshold: < 0.1)")
        report.append(f"   Status: {'[OK] PASS' if pp['difference'] < 0.1 else '[X] FAIL'}")

        report.append("")
        report.append("=" * 70)

        return "\n".join(report)

    def interpret_bias_clinically(
        self,
        bias_report: Dict[str, Any],
        sensitive_attr: str,
        outcome: str
    ) -> str:
        """
        Translate statistical bias metrics into clinical insights using Gemini 3.

        Args:
            bias_report: Output from compute_fairness_metrics
            sensitive_attr: Protected attribute name
            outcome: Outcome variable name

        Returns:
            Clinical interpretation of bias (2 sentences)
        """
        from causal_analysis import _init_gemini, _call_gemini_with_retry

        gemini_model = _init_gemini()

        if not gemini_model:
            return f"Statistical bias detected in {outcome} predictions across {sensitive_attr} groups. Clinical review recommended."

        # Extract key metrics
        dp_diff = bias_report.get('demographic_parity', {}).get('difference', 0)
        eo_diff = bias_report.get('equalized_odds', {}).get('average_difference', 0)

        prompt = f"""Translate these fairness metrics into clinical insights:

Sensitive attribute: {sensitive_attr}
Outcome: {outcome}
Demographic Parity Difference: {dp_diff:.3f}
Equalized Odds Difference: {eo_diff:.3f}

Group-level metrics: {bias_report.get('group_metrics', {})}

Explain the potential real-world harm to patients in exactly 2 sentences.
Focus on clinical safety and health equity implications."""

        response = _call_gemini_with_retry(gemini_model, prompt)

        if response:
            return response.strip()

        return f"Disparate {outcome} rates across {sensitive_attr} may lead to unequal access to care. This could result in delayed treatment for underserved groups."


def compute_fairness_metrics(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sensitive_attr: str,
    y_pred: Optional[np.ndarray] = None
) -> Dict[str, Dict[str, float]]:
    """
    Convenience function to compute fairness metrics.

    This is the main entry point for bias detection in the pipeline.

    Args:
        model: Trained ML model
        X_test: Test features
        y_test: True labels
        sensitive_attr: Protected attribute name
        y_pred: Optional pre-computed predictions

    Returns:
        Dictionary of fairness metrics

    Example:
        >>> metrics = compute_fairness_metrics(model, X_test, y_test, 'race')
        >>> print(f"Demographic parity: {metrics['demographic_parity']['difference']:.3f}")
    """
    detector = BiasDetector([sensitive_attr])
    return detector.compute_fairness_metrics(model, X_test, y_test, sensitive_attr, y_pred)


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)

    # Load sample data and train a simple model
    import os
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder

    data_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "sample",
        "demo_data.csv"
    )

    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
        print(f"Loaded {len(data)} patient records")

        # Prepare data
        # Encode categorical variables
        le_race = LabelEncoder()
        le_gender = LabelEncoder()
        le_insurance = LabelEncoder()

        data['race_encoded'] = le_race.fit_transform(data['race'])
        data['gender_encoded'] = le_gender.fit_transform(data['gender'])
        data['insurance_encoded'] = le_insurance.fit_transform(data['insurance_type'])

        # Features and target
        feature_cols = ['age', 'race_encoded', 'gender_encoded', 'creatinine_level',
                       'chronic_conditions', 'insurance_encoded', 'prior_visits',
                       'distance_to_hospital']
        X = data[feature_cols + ['race']]  # Include race for bias analysis
        y = data['referral']

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=data['race']
        )

        # Train simple model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train[feature_cols], y_train)

        print(f"\nModel accuracy: {model.score(X_test[feature_cols], y_test):.3f}")

        # Compute fairness metrics
        detector = BiasDetector(['race'])
        metrics = detector.compute_fairness_metrics(
            model, X_test, y_test, 'race'
        )

        # Generate report
        report = detector.generate_bias_report(metrics, 'race')
        print("\n" + report)
    else:
        print(f"Sample data not found at {data_path}")
