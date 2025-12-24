"""
Systematic Benchmarking for Clinical Fairness Interventions

Implements rigorous experimental validation comparing intervention methods:
- COMPAS recidivism dataset
- MIMIC-IV kidney referral subset (synthetic fallback)
- Baselines: Fairlearn (DP), Fairlearn (EO), AIF360 Reweighing
- 5-fold stratified cross-validation
- Statistical testing with paired t-tests

References:
- Agarwal et al. (2018) "A Reductions Approach to Fair Classification"
- Bellamy et al. (2018) "AI Fairness 360"
- Demšar (2006) "Statistical Comparisons of Classifiers"
"""

import logging
import hashlib
import json
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats

# Optional dependencies with graceful fallback
try:
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    logging.warning("Fairlearn not installed. Install with: pip install fairlearn")

try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.algorithms.preprocessing import Reweighing
    AIF360_AVAILABLE = True
except ImportError:
    AIF360_AVAILABLE = False
    logging.warning("AIF360 not installed. Install with: pip install aif360")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Dataset(Enum):
    """Supported benchmark datasets."""
    COMPAS = "compas"
    MIMIC_KIDNEY = "mimic_kidney"
    SYNTHETIC_DEMO = "synthetic_demo"


class Method(Enum):
    """Fairness intervention methods."""
    BASELINE = "Unmitigated Baseline"
    FAIRLEARN_DP = "Fairlearn (Demographic Parity)"
    FAIRLEARN_EO = "Fairlearn (Equalized Odds)"
    AIF360_REWEIGH = "AIF360 Reweighing"


@dataclass
class FoldResult:
    """Results from a single cross-validation fold."""
    fold_id: int
    method: str
    accuracy: float
    dp_difference: float  # max(P(Y_hat=1|A=a)) - min(P(Y_hat=1|A=a))
    eo_difference: float  # max(TPR_a - TPR_b, FPR_a - FPR_b)
    fnr_disparity: float  # max(FNR_a) - min(FNR_a)
    tpr_disparity: float
    fpr_disparity: float
    group_metrics: Dict[str, Dict[str, float]]


@dataclass
class BenchmarkResult:
    """Aggregated results across all folds for one method."""
    method: str
    dataset: str
    n_folds: int
    accuracy_mean: float
    accuracy_std: float
    dp_diff_mean: float
    dp_diff_std: float
    eo_diff_mean: float
    eo_diff_std: float
    fnr_disparity_mean: float
    fnr_disparity_std: float
    clinical_safety_score: str  # SAFE/CONDITIONAL/NOT_SAFE based on FNR threshold


@dataclass
class StatisticalComparison:
    """Statistical test results comparing two methods."""
    method_a: str
    method_b: str
    metric: str
    t_statistic: float
    p_value: float
    cohens_d: float  # Effect size
    significant: bool  # p < 0.05
    winner: str  # Which method is better for this metric


@dataclass
class ExperimentReport:
    """Complete experimental validation report."""
    dataset: str
    n_samples: int
    n_folds: int
    sensitive_attribute: str
    outcome_variable: str
    fold_results: List[FoldResult]
    aggregated_results: List[BenchmarkResult]
    statistical_tests: List[StatisticalComparison]
    data_hash: str
    timestamp: str
    seed: int

    def to_dict(self):
        """Convert to JSON-serializable dictionary."""
        return {
            'dataset': self.dataset,
            'n_samples': int(self.n_samples),
            'n_folds': int(self.n_folds),
            'sensitive_attribute': self.sensitive_attribute,
            'outcome_variable': self.outcome_variable,
            'fold_results': [asdict(fr) for fr in self.fold_results],
            'aggregated_results': [asdict(br) for br in self.aggregated_results],
            'statistical_tests': [asdict(st) for st in self.statistical_tests],
            'data_hash': self.data_hash,
            'timestamp': self.timestamp,
            'seed': int(self.seed)
        }


class SystematicBenchmarkExperiments:
    """
    Research-grade experimental validation of fairness interventions.

    Protocol:
    1. Load benchmark dataset (COMPAS, MIMIC-IV, or synthetic)
    2. 5-fold stratified cross-validation
    3. Train baselines: Unmitigated, Fairlearn (DP), Fairlearn (EO), AIF360 Reweighing
    4. Compute metrics per fold: Accuracy, DP diff, EO diff, FNR disparity
    5. Statistical testing: Paired t-tests with effect sizes
    6. Export publication-ready CSV
    """

    def __init__(self, seed: int = 42, n_folds: int = 5):
        """
        Initialize benchmark experiment runner.

        Args:
            seed: Random seed for reproducibility
            n_folds: Number of cross-validation folds
        """
        self.seed = seed
        self.n_folds = n_folds
        np.random.seed(seed)
        logger.info(f"Initialized SystematicBenchmarkExperiments (seed={seed}, folds={n_folds})")

    def _compute_data_hash(self, data: pd.DataFrame) -> str:
        """Compute MD5 hash of dataset for reproducibility."""
        data_str = data.to_csv(index=False)
        return hashlib.md5(data_str.encode()).hexdigest()[:16]

    def _load_dataset(self, dataset: Dataset) -> Tuple[pd.DataFrame, str, str]:
        """
        Load benchmark dataset.

        Returns:
            (data, sensitive_attribute, outcome_variable)
        """
        if dataset == Dataset.COMPAS:
            return self._load_medicare()  # Try Medicare first, falls back to COMPAS
        elif dataset == Dataset.MIMIC_KIDNEY:
            return self._load_mimic_kidney()
        elif dataset == Dataset.SYNTHETIC_DEMO:
            return self._load_synthetic_demo()
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def _load_medicare(self) -> Tuple[pd.DataFrame, str, str]:
        """
        Load CMS Medicare beneficiary data and predict high-cost outcomes.

        Uses real Medicare data to predict patients with high medical costs,
        examining fairness across race/sex demographics.
        """
        try:
            import os
            medicare_paths = [
                "data/DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv",
                "../data/DE1_0_2008_Beneficiary_Summary_File_Sample_1.csv",
            ]

            for path in medicare_paths:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    logger.info(f"Loaded Medicare from {path} (n={len(df)})")

                    # Calculate age from birth date
                    df['age'] = 2008 - (df['BENE_BIRTH_DT'] // 10000)

                    # Binary features
                    df['sex'] = (df['BENE_SEX_IDENT_CD'] == 1).astype(int)  # 1 = Male
                    df['race_white'] = (df['BENE_RACE_CD'] == 1).astype(int)  # Protected attr
                    df['has_esrd'] = (df['BENE_ESRD_IND'] == 'Y').astype(int)

                    # Chronic conditions (2=No, 1=Yes in data)
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

                    # Total medical costs
                    df['total_cost'] = (
                        df['MEDREIMB_IP'].fillna(0) +
                        df['MEDREIMB_OP'].fillna(0) +
                        df['MEDREIMB_CAR'].fillna(0)
                    )

                    # Outcome: High-cost patient (top 25% = high risk)
                    cost_threshold = df['total_cost'].quantile(0.75)
                    df['high_cost'] = (df['total_cost'] > cost_threshold).astype(int)

                    # Select features
                    df_clean = df[[
                        'age', 'sex', 'has_esrd', 'has_diabetes',
                        'has_chf', 'has_copd', 'chronic_count',
                        'race_white', 'high_cost'
                    ]].dropna()

                    logger.info(f"Medicare data: {len(df_clean)} samples, "
                               f"{df_clean['race_white'].mean():.1%} White, "
                               f"{df_clean['high_cost'].mean():.1%} high-cost patients")

                    return df_clean, 'race_white', 'high_cost'

            logger.warning("Medicare dataset not found. Trying COMPAS...")
            return self._load_compas()

        except Exception as e:
            logger.warning(f"Error loading Medicare: {e}. Trying COMPAS...")
            return self._load_compas()

    def _load_compas(self) -> Tuple[pd.DataFrame, str, str]:
        """
        Load COMPAS recidivism dataset (ProPublica FairML version).

        Falls back to synthetic if real data unavailable.
        """
        try:
            # Try loading from common paths
            import os
            compas_paths = [
                "data/propublica_data_for_fairml.csv",
                "../propublicaCompassRecividism_data_fairml.csv/propublica_data_for_fairml.csv",
                "data/compas-scores-two-years.csv",
                "../data/compas-scores-two-years.csv",
            ]

            for path in compas_paths:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    logger.info(f"Loaded COMPAS from {path} (n={len(df)})")

                    # Check if it's the FairML version (already preprocessed)
                    if 'Two_yr_Recidivism' in df.columns and 'African_American' in df.columns:
                        # FairML version - already preprocessed
                        df_clean = pd.DataFrame({
                            'age': 1 - df['Age_Below_TwentyFive'],  # Approximate age feature
                            'sex': 1 - df['Female'],  # 1 = Male
                            'priors_count': df['Number_of_Priors'],
                            'c_charge_degree': 1 - df['Misdemeanor'],  # 1 = Felony
                            'race_binary': df['African_American'],
                            'two_year_recid': df['Two_yr_Recidivism']
                        })

                        logger.info(f"Loaded ProPublica FairML COMPAS: {len(df_clean)} samples, "
                                   f"{df_clean['race_binary'].mean():.1%} African-American, "
                                   f"{df_clean['two_year_recid'].mean():.1%} recidivism rate")

                        return df_clean, 'race_binary', 'two_year_recid'

                    else:
                        # Standard ProPublica format - apply Dressel & Farid preprocessing
                        df = df[
                            (df['days_b_screening_arrest'] <= 30) &
                            (df['days_b_screening_arrest'] >= -30) &
                            (df['is_recid'] != -1) &
                            (df['c_charge_degree'] != 'O') &
                            (df['score_text'] != 'N/A')
                        ]

                        df = df[[
                            'age', 'sex', 'race', 'priors_count',
                            'c_charge_degree', 'two_year_recid'
                        ]].copy()

                        df['sex'] = (df['sex'] == 'Male').astype(int)
                        df['c_charge_degree'] = (df['c_charge_degree'] == 'F').astype(int)
                        df['race_binary'] = (df['race'] == 'African-American').astype(int)
                        df = df.drop('race', axis=1)

                        return df, 'race_binary', 'two_year_recid'

            logger.warning("COMPAS dataset not found. Using synthetic fallback.")
            return self._generate_synthetic_compas()

        except Exception as e:
            logger.warning(f"Error loading COMPAS: {e}. Using synthetic fallback.")
            return self._generate_synthetic_compas()

    def _generate_synthetic_compas(self) -> Tuple[pd.DataFrame, str, str]:
        """Generate synthetic COMPAS-like data."""
        n_samples = 5000

        # Generate correlated features
        race_binary = np.random.binomial(1, 0.51, n_samples)  # 51% African-American (real COMPAS)
        age = np.random.normal(34, 10, n_samples).clip(18, 70)
        sex = np.random.binomial(1, 0.81, n_samples)  # 81% male

        # Priors count correlated with race (systemic bias)
        priors_count = np.where(
            race_binary == 1,
            np.random.poisson(3.0, n_samples),
            np.random.poisson(2.2, n_samples)
        )

        c_charge_degree = np.random.binomial(1, 0.46, n_samples)  # 46% felony

        # Recidivism: legitimate correlation with priors + biased correlation with race
        recid_prob = (
            0.20 +
            0.05 * (priors_count / 10) +
            0.10 * race_binary +  # Direct discrimination
            0.05 * c_charge_degree +
            -0.02 * (age / 34)
        ).clip(0, 1)

        two_year_recid = np.random.binomial(1, recid_prob)

        df = pd.DataFrame({
            'age': age,
            'sex': sex,
            'race_binary': race_binary,
            'priors_count': priors_count,
            'c_charge_degree': c_charge_degree,
            'two_year_recid': two_year_recid
        })

        logger.info("Generated synthetic COMPAS dataset (n=5000)")
        return df, 'race_binary', 'two_year_recid'

    def _load_mimic_kidney(self) -> Tuple[pd.DataFrame, str, str]:
        """
        Load MIMIC-IV kidney referral subset.

        Falls back to synthetic if real data unavailable.
        """
        logger.warning("MIMIC-IV access not configured. Using synthetic fallback.")
        return self._generate_synthetic_mimic_kidney()

    def _generate_synthetic_mimic_kidney(self) -> Tuple[pd.DataFrame, str, str]:
        """Generate synthetic MIMIC-like kidney referral data."""
        n_samples = 1000

        # Demographics
        race = np.random.choice(['White', 'Black', 'Hispanic', 'Other'], n_samples, p=[0.60, 0.22, 0.12, 0.06])
        age = np.random.normal(62, 15, n_samples).clip(18, 95)
        gender = np.random.binomial(1, 0.48, n_samples)

        # Clinical features (Obermeyer pathway)
        creatinine_level = np.where(
            race == 'Black',
            np.random.normal(1.8, 0.6, n_samples),
            np.random.normal(1.5, 0.5, n_samples)
        ).clip(0.5, 5.0)

        chronic_conditions = np.random.poisson(2.3, n_samples)

        # Systemic barriers
        insurance_type = np.where(
            np.isin(race, ['Black', 'Hispanic']),
            np.random.choice(['Medicaid', 'Private', 'Medicare'], n_samples, p=[0.45, 0.25, 0.30]),
            np.random.choice(['Medicaid', 'Private', 'Medicare'], n_samples, p=[0.20, 0.50, 0.30])
        )

        prior_visits = np.random.poisson(8, n_samples)
        distance_to_hospital = np.random.exponential(12, n_samples)

        # Referral (biased by race through insurance/distance)
        referral_prob = (
            0.15 +
            0.30 * (creatinine_level / 2.5) +
            0.15 * (chronic_conditions / 5) +
            0.10 * (insurance_type == 'Private') +
            -0.08 * (distance_to_hospital / 20) +
            0.05 * (race == 'White')  # Direct bias
        ).clip(0, 1)

        referral = np.random.binomial(1, referral_prob)

        # Encode categoricals
        le_race = LabelEncoder()
        le_insurance = LabelEncoder()

        df = pd.DataFrame({
            'age': age,
            'race': le_race.fit_transform(race),
            'gender': gender,
            'creatinine_level': creatinine_level,
            'chronic_conditions': chronic_conditions,
            'insurance_type': le_insurance.fit_transform(insurance_type),
            'prior_visits': prior_visits,
            'distance_to_hospital': distance_to_hospital,
            'referral': referral
        })

        logger.info("Generated synthetic MIMIC kidney dataset (n=1000)")
        return df, 'race', 'referral'

    def _load_synthetic_demo(self) -> Tuple[pd.DataFrame, str, str]:
        """Load existing demo dataset."""
        import os
        demo_path = "data/sample/demo_data.csv"

        if os.path.exists(demo_path):
            df = pd.read_csv(demo_path)
            logger.info(f"Loaded demo dataset from {demo_path}")

            # Encode categorical variables
            le_race = LabelEncoder()
            le_gender = LabelEncoder()
            le_insurance = LabelEncoder()

            df['race'] = le_race.fit_transform(df['race'])
            df['gender'] = le_gender.fit_transform(df['gender'])
            df['insurance_type'] = le_insurance.fit_transform(df['insurance_type'])

            logger.info("Encoded categorical variables: race, gender, insurance_type")
        else:
            # Generate if missing
            logger.warning("Demo dataset not found. Generating new synthetic data.")
            df, _, _ = self._generate_synthetic_mimic_kidney()

        return df, 'race', 'referral'

    def _compute_fairness_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attr: np.ndarray
    ) -> Tuple[float, float, float, Dict[str, Dict[str, float]]]:
        """
        Compute comprehensive fairness metrics.

        Returns:
            (dp_difference, eo_difference, fnr_disparity, group_metrics)
        """
        unique_groups = np.unique(sensitive_attr)
        group_metrics = {}

        selection_rates = []
        tprs = []
        fprs = []
        fnrs = []

        for group in unique_groups:
            mask = sensitive_attr == group
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]

            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group, labels=[0, 1]).ravel()

            # Rates
            selection_rate = np.mean(y_pred_group)
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            selection_rates.append(selection_rate)
            tprs.append(tpr)
            fprs.append(fpr)
            fnrs.append(fnr)

            group_metrics[f"group_{group}"] = {
                'n_samples': int(np.sum(mask)),
                'selection_rate': float(selection_rate),
                'tpr': float(tpr),
                'fpr': float(fpr),
                'fnr': float(fnr),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            }

        # Demographic Parity: max selection rate difference
        dp_difference = float(np.max(selection_rates) - np.min(selection_rates))

        # Equalized Odds: max of TPR disparity and FPR disparity
        tpr_disparity = float(np.max(tprs) - np.min(tprs))
        fpr_disparity = float(np.max(fprs) - np.min(fprs))
        eo_difference = float(max(tpr_disparity, fpr_disparity))

        # FNR Disparity (clinical safety metric)
        fnr_disparity = float(np.max(fnrs) - np.min(fnrs))

        return dp_difference, eo_difference, fnr_disparity, group_metrics

    def _train_baseline(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray
    ) -> np.ndarray:
        """Train unmitigated logistic regression baseline."""
        model = LogisticRegression(random_state=self.seed, max_iter=1000)
        model.fit(X_train, y_train)
        return model.predict(X_test)

    def _train_fairlearn_dp(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sensitive_train: np.ndarray,
        X_test: np.ndarray
    ) -> np.ndarray:
        """Train Fairlearn with Demographic Parity constraint."""
        if not FAIRLEARN_AVAILABLE:
            logger.warning("Fairlearn not available. Using baseline.")
            return self._train_baseline(X_train, y_train, X_test)

        mitigator = ExponentiatedGradient(
            LogisticRegression(random_state=self.seed, max_iter=1000),
            constraints=DemographicParity(),
            eps=0.01
        )
        mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
        return mitigator.predict(X_test)

    def _train_fairlearn_eo(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sensitive_train: np.ndarray,
        X_test: np.ndarray
    ) -> np.ndarray:
        """Train Fairlearn with Equalized Odds constraint."""
        if not FAIRLEARN_AVAILABLE:
            logger.warning("Fairlearn not available. Using baseline.")
            return self._train_baseline(X_train, y_train, X_test)

        mitigator = ExponentiatedGradient(
            LogisticRegression(random_state=self.seed, max_iter=1000),
            constraints=EqualizedOdds(),
            eps=0.01
        )
        mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
        return mitigator.predict(X_test)

    def _train_aif360_reweigh(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sensitive_train: np.ndarray,
        X_test: np.ndarray,
        sensitive_test: np.ndarray
    ) -> np.ndarray:
        """Train AIF360 reweighing preprocessing."""
        if not AIF360_AVAILABLE:
            logger.warning("AIF360 not available. Using baseline.")
            return self._train_baseline(X_train, y_train, X_test)

        # Convert to AIF360 format
        df_train = pd.DataFrame(X_train)
        df_train['sensitive'] = sensitive_train
        df_train['label'] = y_train

        aif_train = BinaryLabelDataset(
            df=df_train,
            label_names=['label'],
            protected_attribute_names=['sensitive']
        )

        # Reweigh
        RW = Reweighing(unprivileged_groups=[{'sensitive': 0}], privileged_groups=[{'sensitive': 1}])
        aif_train_transformed = RW.fit_transform(aif_train)

        # Train with sample weights
        weights = aif_train_transformed.instance_weights
        model = LogisticRegression(random_state=self.seed, max_iter=1000)
        model.fit(X_train, y_train, sample_weight=weights)

        return model.predict(X_test)

    def run_benchmarks(
        self,
        dataset: Dataset = Dataset.SYNTHETIC_DEMO,
        methods: Optional[List[Method]] = None
    ) -> ExperimentReport:
        """
        Run systematic benchmarking experiments.

        Args:
            dataset: Which benchmark dataset to use
            methods: List of methods to compare (default: all available)

        Returns:
            ExperimentReport with aggregated results and statistical tests
        """
        if methods is None:
            methods = [Method.BASELINE, Method.FAIRLEARN_DP, Method.FAIRLEARN_EO, Method.AIF360_REWEIGH]

        logger.info(f"Starting benchmark on {dataset.value} with {len(methods)} methods")

        # Load dataset
        data, sensitive_attr, outcome = self._load_dataset(dataset)
        data_hash = self._compute_data_hash(data)

        # Prepare features
        X = data.drop(columns=[outcome, sensitive_attr]).values
        y = data[outcome].values
        sensitive = data[sensitive_attr].values

        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        logger.info(f"Dataset: n={len(data)}, features={X.shape[1]}, sensitive={sensitive_attr}, outcome={outcome}")

        # 5-fold stratified CV
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        fold_results = []

        for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Fold {fold_id + 1}/{self.n_folds}")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            sensitive_train, sensitive_test = sensitive[train_idx], sensitive[test_idx]

            for method in methods:
                # Train method
                if method == Method.BASELINE:
                    y_pred = self._train_baseline(X_train, y_train, X_test)
                elif method == Method.FAIRLEARN_DP:
                    y_pred = self._train_fairlearn_dp(X_train, y_train, sensitive_train, X_test)
                elif method == Method.FAIRLEARN_EO:
                    y_pred = self._train_fairlearn_eo(X_train, y_train, sensitive_train, X_test)
                elif method == Method.AIF360_REWEIGH:
                    y_pred = self._train_aif360_reweigh(X_train, y_train, sensitive_train, X_test, sensitive_test)
                else:
                    continue

                # Compute metrics
                accuracy = float(accuracy_score(y_test, y_pred))
                dp_diff, eo_diff, fnr_disp, group_metrics = self._compute_fairness_metrics(
                    y_test, y_pred, sensitive_test
                )

                # Extract disparities
                tpr_values = [gm['tpr'] for gm in group_metrics.values()]
                fpr_values = [gm['fpr'] for gm in group_metrics.values()]
                tpr_disparity = float(np.max(tpr_values) - np.min(tpr_values))
                fpr_disparity = float(np.max(fpr_values) - np.min(fpr_values))

                fold_result = FoldResult(
                    fold_id=fold_id,
                    method=method.value,
                    accuracy=accuracy,
                    dp_difference=dp_diff,
                    eo_difference=eo_diff,
                    fnr_disparity=fnr_disp,
                    tpr_disparity=tpr_disparity,
                    fpr_disparity=fpr_disparity,
                    group_metrics=group_metrics
                )
                fold_results.append(fold_result)

                logger.info(f"  {method.value}: Acc={accuracy:.3f}, FNR_disp={fnr_disp:.3f}, DP={dp_diff:.3f}")

        # Aggregate results per method
        aggregated_results = self._aggregate_results(fold_results, dataset.value)

        # Statistical testing
        statistical_tests = self._perform_statistical_tests(fold_results, methods)

        # Create report
        report = ExperimentReport(
            dataset=dataset.value,
            n_samples=len(data),
            n_folds=self.n_folds,
            sensitive_attribute=sensitive_attr,
            outcome_variable=outcome,
            fold_results=fold_results,
            aggregated_results=aggregated_results,
            statistical_tests=statistical_tests,
            data_hash=data_hash,
            timestamp=datetime.now().isoformat(),
            seed=self.seed
        )

        logger.info("Benchmark complete!")
        return report

    def _aggregate_results(self, fold_results: List[FoldResult], dataset: str) -> List[BenchmarkResult]:
        """Aggregate fold results by method."""
        methods = set(fr.method for fr in fold_results)
        aggregated = []

        for method in methods:
            method_folds = [fr for fr in fold_results if fr.method == method]

            accuracies = [fr.accuracy for fr in method_folds]
            dp_diffs = [fr.dp_difference for fr in method_folds]
            eo_diffs = [fr.eo_difference for fr in method_folds]
            fnr_disps = [fr.fnr_disparity for fr in method_folds]

            # Clinical safety score based on FNR disparity threshold (5%)
            mean_fnr_disp = float(np.mean(fnr_disps))
            if mean_fnr_disp <= 0.05:
                safety_score = "SAFE"
            elif mean_fnr_disp <= 0.10:
                safety_score = "CONDITIONAL"
            else:
                safety_score = "NOT_SAFE"

            result = BenchmarkResult(
                method=method,
                dataset=dataset,
                n_folds=len(method_folds),
                accuracy_mean=float(np.mean(accuracies)),
                accuracy_std=float(np.std(accuracies)),
                dp_diff_mean=float(np.mean(dp_diffs)),
                dp_diff_std=float(np.std(dp_diffs)),
                eo_diff_mean=float(np.mean(eo_diffs)),
                eo_diff_std=float(np.std(eo_diffs)),
                fnr_disparity_mean=mean_fnr_disp,
                fnr_disparity_std=float(np.std(fnr_disps)),
                clinical_safety_score=safety_score
            )
            aggregated.append(result)

        return aggregated

    def _perform_statistical_tests(
        self,
        fold_results: List[FoldResult],
        methods: List[Method]
    ) -> List[StatisticalComparison]:
        """
        Perform paired t-tests comparing methods.

        Tests baseline vs each fairness intervention on:
        - Accuracy (higher is better)
        - FNR disparity (lower is better)
        - DP difference (lower is better)
        """
        tests = []
        baseline_method = Method.BASELINE.value

        # Get baseline fold results
        baseline_folds = [fr for fr in fold_results if fr.method == baseline_method]
        if not baseline_folds:
            logger.warning("No baseline results found for statistical testing")
            return tests

        baseline_folds = sorted(baseline_folds, key=lambda x: x.fold_id)

        for method in methods:
            if method == Method.BASELINE:
                continue

            method_folds = [fr for fr in fold_results if fr.method == method.value]
            method_folds = sorted(method_folds, key=lambda x: x.fold_id)

            if len(method_folds) != len(baseline_folds):
                logger.warning(f"Fold mismatch for {method.value}, skipping statistical test")
                continue

            # Test accuracy (higher better)
            baseline_acc = np.array([fr.accuracy for fr in baseline_folds])
            method_acc = np.array([fr.accuracy for fr in method_folds])
            t_stat_acc, p_val_acc = stats.ttest_rel(method_acc, baseline_acc)

            # Handle zero variance (Cohen's d undefined)
            std_diff_acc = np.std(method_acc - baseline_acc)
            if std_diff_acc > 1e-10:
                cohens_d_acc = (np.mean(method_acc) - np.mean(baseline_acc)) / std_diff_acc
            else:
                cohens_d_acc = 0.0

            winner_acc = method.value if np.mean(method_acc) > np.mean(baseline_acc) else baseline_method

            tests.append(StatisticalComparison(
                method_a=baseline_method,
                method_b=method.value,
                metric="Accuracy",
                t_statistic=float(t_stat_acc),
                p_value=float(p_val_acc),
                cohens_d=float(cohens_d_acc),
                significant=bool(p_val_acc < 0.05),
                winner=winner_acc
            ))

            # Test FNR disparity (lower better)
            baseline_fnr = np.array([fr.fnr_disparity for fr in baseline_folds])
            method_fnr = np.array([fr.fnr_disparity for fr in method_folds])
            t_stat_fnr, p_val_fnr = stats.ttest_rel(baseline_fnr, method_fnr)  # Reversed for "lower is better"

            # Handle zero variance
            std_diff_fnr = np.std(baseline_fnr - method_fnr)
            if std_diff_fnr > 1e-10:
                cohens_d_fnr = (np.mean(baseline_fnr) - np.mean(method_fnr)) / std_diff_fnr
            else:
                cohens_d_fnr = 0.0

            winner_fnr = method.value if np.mean(method_fnr) < np.mean(baseline_fnr) else baseline_method

            tests.append(StatisticalComparison(
                method_a=baseline_method,
                method_b=method.value,
                metric="FNR Disparity",
                t_statistic=float(t_stat_fnr),
                p_value=float(p_val_fnr),
                cohens_d=float(cohens_d_fnr),
                significant=bool(p_val_fnr < 0.05),
                winner=winner_fnr
            ))

            # Test DP difference (lower better)
            baseline_dp = np.array([fr.dp_difference for fr in baseline_folds])
            method_dp = np.array([fr.dp_difference for fr in method_folds])
            t_stat_dp, p_val_dp = stats.ttest_rel(baseline_dp, method_dp)

            # Handle zero variance
            std_diff_dp = np.std(baseline_dp - method_dp)
            if std_diff_dp > 1e-10:
                cohens_d_dp = (np.mean(baseline_dp) - np.mean(method_dp)) / std_diff_dp
            else:
                cohens_d_dp = 0.0

            winner_dp = method.value if np.mean(method_dp) < np.mean(baseline_dp) else baseline_method

            tests.append(StatisticalComparison(
                method_a=baseline_method,
                method_b=method.value,
                metric="DP Difference",
                t_statistic=float(t_stat_dp),
                p_value=float(p_val_dp),
                cohens_d=float(cohens_d_dp),
                significant=bool(p_val_dp < 0.05),
                winner=winner_dp
            ))

        return tests

    def export_results(self, report: ExperimentReport, output_dir: str = "results"):
        """
        Export results in publication-ready formats.

        Generates:
        - JSON with full experimental details
        - CSV with aggregated method comparison
        - CSV with statistical test results
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Export full JSON
        json_path = os.path.join(output_dir, f"benchmark_{report.dataset}.json")
        with open(json_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info(f"Exported full results to {json_path}")

        # Export aggregated CSV (Table 1 format)
        csv_path = os.path.join(output_dir, f"benchmark_{report.dataset}_table.csv")
        aggregated_df = pd.DataFrame([asdict(br) for br in report.aggregated_results])

        # Reorder and format columns for publication
        aggregated_df = aggregated_df[[
            'method', 'accuracy_mean', 'accuracy_std',
            'fnr_disparity_mean', 'fnr_disparity_std',
            'dp_diff_mean', 'dp_diff_std',
            'clinical_safety_score'
        ]]

        aggregated_df.columns = [
            'Method', 'Accuracy (mean)', 'Accuracy (std)',
            'FNR Disparity (mean)', 'FNR Disparity (std)',
            'DP Difference (mean)', 'DP Difference (std)',
            'Clinical Safety'
        ]

        aggregated_df.to_csv(csv_path, index=False, float_format='%.4f')
        logger.info(f"Exported aggregated table to {csv_path}")

        # Export statistical tests CSV
        stats_path = os.path.join(output_dir, f"benchmark_{report.dataset}_stats.csv")
        stats_df = pd.DataFrame([asdict(st) for st in report.statistical_tests])
        stats_df.to_csv(stats_path, index=False, float_format='%.4f')
        logger.info(f"Exported statistical tests to {stats_path}")

        return json_path, csv_path, stats_path


# Example usage
if __name__ == "__main__":
    # Initialize experiment runner
    runner = SystematicBenchmarkExperiments(seed=42, n_folds=5)

    # Run benchmark on Medicare data (high-cost prediction)
    report = runner.run_benchmarks(dataset=Dataset.COMPAS)

    # Export results
    runner.export_results(report)

    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(f"Dataset: {report.dataset}")
    print(f"Samples: {report.n_samples}")
    print(f"Folds: {report.n_folds}")
    print(f"Sensitive Attribute: {report.sensitive_attribute}")
    print(f"Outcome: {report.outcome_variable}")
    print("\n" + "-"*80)
    print("AGGREGATED METRICS")
    print("-"*80)
    for br in report.aggregated_results:
        print(f"\n{br.method}")
        print(f"  Accuracy: {br.accuracy_mean:.4f} +/- {br.accuracy_std:.4f}")
        print(f"  FNR Disparity: {br.fnr_disparity_mean:.4f} +/- {br.fnr_disparity_std:.4f}")
        print(f"  DP Difference: {br.dp_diff_mean:.4f} +/- {br.dp_diff_std:.4f}")
        print(f"  Clinical Safety: {br.clinical_safety_score}")

    print("\n" + "-"*80)
    print("STATISTICAL SIGNIFICANCE TESTS (vs Baseline)")
    print("-"*80)
    for st in report.statistical_tests:
        sig_marker = "***" if st.significant else "ns"
        print(f"\n{st.method_b} vs {st.method_a} on {st.metric}")
        print(f"  p-value: {st.p_value:.4f} {sig_marker}")
        print(f"  Cohen's d: {st.cohens_d:.4f}")
        print(f"  Winner: {st.winner}")
    print("\n" + "="*80)
