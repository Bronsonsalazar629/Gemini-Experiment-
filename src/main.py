"""
Clinical Fairness Intervention System - Main Entry Point

Command-line interface and orchestration for the complete fairness analysis pipeline.

Usage:
    python main.py analyze --data data/sample/demo_data.csv --model model.pkl
    python main.py generate --intervention Reweighing
    python main.py pipeline --data data/sample/demo_data.csv --output results/
"""

import os
import sys
import argparse
import logging
import pandas as pd
import pickle
from pathlib import Path
from typing import Optional, Dict, Any

# Import our modules
from causal_analysis import CausalAnalyzer, infer_causal_graph
from bias_detection import BiasDetector, compute_fairness_metrics
from intervention_engine import InterventionEngine, suggest_interventions
from code_generator import CodeGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FairnessPipeline:
    """
    Orchestrates the complete fairness analysis and intervention pipeline.
    """

    def __init__(
        self,
        data_path: str,
        sensitive_attr: str = "race",
        outcome: str = "referral",
        config_path: Optional[str] = None
    ):
        """
        Initialize the fairness pipeline.

        Args:
            data_path: Path to clinical dataset CSV
            sensitive_attr: Protected attribute column name
            outcome: Outcome variable column name
            config_path: Optional path to configuration file
        """
        self.data_path = data_path
        self.sensitive_attr = sensitive_attr
        self.outcome = outcome
        self.config = self._load_config(config_path) if config_path else {}

        # Load data
        self.data = pd.read_csv(data_path)
        logger.info(f"Loaded {len(self.data)} records from {data_path}")

        # Initialize components
        self.causal_analyzer = None
        self.bias_detector = BiasDetector([sensitive_attr])
        self.intervention_engine = InterventionEngine()
        self.code_generator = CodeGenerator(
            api_key=self.config.get('gemini_api_key')
        )

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        import yaml
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
            return {}

    def run_causal_analysis(self) -> tuple:
        """
        Run causal graph inference.

        Returns:
            Tuple of (causal_graph, explanation)
        """
        logger.info("=" * 70)
        logger.info("STEP 1: CAUSAL ANALYSIS")
        logger.info("=" * 70)

        result = infer_causal_graph(
            self.data,
            self.sensitive_attr,
            self.outcome,
            use_cache=True,
            llm_config=self.config
        )

        # Unpack dict result
        graph = result['graph']
        explanation = result['llm_explanation']

        # Handle Unicode for Windows console
        explanation_safe = explanation.replace('\u2192', '->')
        print("\n" + explanation_safe)

        self.causal_analyzer = CausalAnalyzer(
            self.data,
            self.sensitive_attr,
            self.outcome
        )
        self.causal_analyzer.causal_graph = graph

        # Identify bias pathways
        pathways = self.causal_analyzer.identify_bias_pathways()
        if pathways:
            print("\nIdentified Bias Pathways:")
            for i, path in enumerate(pathways, 1):
                print(f"  {i}. {' -> '.join(path)}")

        return graph, explanation

    def run_bias_detection(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Run bias detection on a trained model.

        Args:
            model: Trained ML model
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of fairness metrics
        """
        logger.info("=" * 70)
        logger.info("STEP 2: BIAS DETECTION")
        logger.info("=" * 70)

        metrics = self.bias_detector.compute_fairness_metrics(
            model, X_test, y_test, self.sensitive_attr
        )

        # Generate and print report
        report = self.bias_detector.generate_bias_report(metrics, self.sensitive_attr)
        print("\n" + report)

        return metrics

    def run_intervention_recommendation(self, bias_metrics: Dict) -> list:
        """
        Generate intervention recommendations.

        Args:
            bias_metrics: Output from bias detection

        Returns:
            List of intervention recommendations
        """
        logger.info("=" * 70)
        logger.info("STEP 3: INTERVENTION RECOMMENDATIONS")
        logger.info("=" * 70)

        recommendations = self.intervention_engine.suggest_interventions(
            bias_metrics,
            max_recommendations=5
        )

        print("\nRecommended Interventions:")
        for rec in recommendations:
            print(f"\n{rec.priority}. {rec.name} ({rec.category})")
            print(f"   {rec.description}")
            print(f"   Expected Impact: {rec.expected_impact}")
            print(f"   Complexity: {rec.complexity}")
            print(f"   Preserves Accuracy: {rec.preserves_accuracy}")

        return recommendations

    def run_code_generation(self, intervention_name: str, output_path: Optional[str] = None):
        """
        Generate implementation code for an intervention.

        Args:
            intervention_name: Name of intervention to implement
            output_path: Optional path to save generated code
        """
        logger.info("=" * 70)
        logger.info("STEP 4: CODE GENERATION")
        logger.info("=" * 70)

        result = self.code_generator.generate_fix_code(intervention_name)

        print(f"\nGenerated Code for: {result.intervention_name}")
        print(f"Description: {result.description}")
        print(f"Estimated Runtime: {result.estimated_runtime}")
        print("\n" + "=" * 70)
        print("IMPORTS:")
        print("=" * 70)
        for imp in result.imports:
            print(imp)

        print("\n" + "=" * 70)
        print("CODE:")
        print("=" * 70)
        print(result.code)

        print("\n" + "=" * 70)
        print("USAGE EXAMPLE:")
        print("=" * 70)
        print(result.usage_example)

        # Optionally save to file
        if output_path:
            full_code = "\n".join(result.imports) + "\n\n" + result.code + "\n\n# " + "=" * 70 + "\n# USAGE EXAMPLE\n# " + "=" * 70 + "\n" + result.usage_example
            with open(output_path, 'w') as f:
                f.write(full_code)
            logger.info(f"Code saved to {output_path}")

    def run_full_pipeline(self, model, X_test: pd.DataFrame, y_test: pd.Series, output_dir: Optional[str] = None):
        """
        Run the complete fairness analysis pipeline.

        Args:
            model: Trained ML model to analyze
            X_test: Test features
            y_test: Test labels
            output_dir: Optional directory to save results
        """
        logger.info("\n" + "=" * 70)
        logger.info("CLINICAL FAIRNESS INTERVENTION SYSTEM - FULL PIPELINE")
        logger.info("=" * 70 + "\n")

        # Step 1: Causal Analysis
        graph, explanation = self.run_causal_analysis()

        # Step 2: Bias Detection
        bias_metrics = self.run_bias_detection(model, X_test, y_test)

        # Step 3: Intervention Recommendations
        recommendations = self.run_intervention_recommendation(bias_metrics)

        # Step 4: Code Generation for top recommendation
        if recommendations:
            top_intervention = recommendations[0].name
            output_path = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"{top_intervention.replace(' ', '_').lower()}_fix.py")

            self.run_code_generation(top_intervention, output_path)

        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETED")
        logger.info("=" * 70)


def train_demo_model(data: pd.DataFrame, test_size: float = 0.3):
    """
    Train a simple demo model for testing.

    Args:
        data: Clinical dataset
        test_size: Fraction of data for testing

    Returns:
        Tuple of (model, X_train, X_test, y_train, y_test)
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression

    # Encode categorical variables
    data = data.copy()
    le_race = LabelEncoder()
    le_gender = LabelEncoder()
    le_insurance = LabelEncoder()

    data['race_encoded'] = le_race.fit_transform(data['race'])
    data['gender_encoded'] = le_gender.fit_transform(data['gender'])
    data['insurance_encoded'] = le_insurance.fit_transform(data['insurance_type'])

    # Features (including race for bias analysis)
    feature_cols = [
        'age', 'race_encoded', 'gender_encoded', 'creatinine_level',
        'chronic_conditions', 'insurance_encoded', 'prior_visits',
        'distance_to_hospital'
    ]
    X = data[feature_cols + ['race']]
    y = data['referral']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=data['race']
    )

    # Train
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train[feature_cols], y_train)

    logger.info(f"Trained model accuracy: {model.score(X_test[feature_cols], y_test):.3f}")

    return model, X_train, X_test, y_train, y_test


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Clinical Fairness Intervention System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run full fairness pipeline')
    pipeline_parser.add_argument('--data', required=True, help='Path to clinical data CSV')
    pipeline_parser.add_argument('--model', help='Path to trained model pickle (optional, will train demo model if not provided)')
    pipeline_parser.add_argument('--sensitive-attr', default='race', help='Protected attribute name')
    pipeline_parser.add_argument('--outcome', default='referral', help='Outcome variable name')
    pipeline_parser.add_argument('--output', help='Output directory for results')
    pipeline_parser.add_argument('--config', help='Path to config YAML file')

    # Analyze command (causal + bias only)
    analyze_parser = subparsers.add_parser('analyze', help='Run causal and bias analysis')
    analyze_parser.add_argument('--data', required=True, help='Path to clinical data CSV')
    analyze_parser.add_argument('--model', help='Path to trained model pickle')
    analyze_parser.add_argument('--sensitive-attr', default='race', help='Protected attribute name')
    analyze_parser.add_argument('--outcome', default='referral', help='Outcome variable name')

    # Generate command (code generation only)
    generate_parser = subparsers.add_parser('generate', help='Generate intervention code')
    generate_parser.add_argument('--intervention', required=True, help='Intervention type')
    generate_parser.add_argument('--output', help='Output file path')

    args = parser.parse_args()

    if args.command == 'pipeline':
        # Run full pipeline
        pipeline = FairnessPipeline(
            args.data,
            args.sensitive_attr,
            args.outcome,
            args.config
        )

        # Load or train model
        if args.model and os.path.exists(args.model):
            with open(args.model, 'rb') as f:
                model = pickle.load(f)
            # Need to prepare test data...
            logger.error("Loading external models requires additional data preparation code")
            sys.exit(1)
        else:
            logger.info("Training demo model...")
            model, X_train, X_test, y_train, y_test = train_demo_model(pipeline.data)

        pipeline.run_full_pipeline(model, X_test, y_test, args.output)

    elif args.command == 'analyze':
        pipeline = FairnessPipeline(args.data, args.sensitive_attr, args.outcome)

        # Run causal analysis
        pipeline.run_causal_analysis()

        # Run bias detection if model provided
        if args.model and os.path.exists(args.model):
            with open(args.model, 'rb') as f:
                model = pickle.load(f)
            # Would need test data preparation
            logger.warning("Bias detection skipped - implement data preparation for external models")
        else:
            logger.info("Training demo model for bias analysis...")
            model, X_train, X_test, y_train, y_test = train_demo_model(pipeline.data)
            pipeline.run_bias_detection(model, X_test, y_test)

    elif args.command == 'generate':
        generator = CodeGenerator()
        generator_result = generator.generate_fix_code(args.intervention)

        print(f"\nGenerated Code for: {generator_result.intervention_name}")
        print("=" * 70)
        for imp in generator_result.imports:
            print(imp)
        print("\n" + generator_result.code)
        print("\n# USAGE EXAMPLE:")
        print(generator_result.usage_example)

        if args.output:
            full_code = "\n".join(generator_result.imports) + "\n\n" + generator_result.code
            with open(args.output, 'w') as f:
                f.write(full_code)
            logger.info(f"Code saved to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
