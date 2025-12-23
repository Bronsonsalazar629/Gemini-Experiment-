"""
Code Generator Module

Generates executable Python code for implementing fairness interventions
using LLM (Gemini 3 API) for intelligent code synthesis.

Supports generating code for:
- Preprocessing interventions (reweighing, resampling)
- Inprocessing interventions (adversarial debiasing, constraints)
- Postprocessing interventions (calibrated equalized odds)
- Complete pipeline code with evaluation
"""

from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GeneratedCode:
    """
    Represents generated intervention code.

    Attributes:
        intervention_name: Name of the intervention
        code: Generated Python code as string
        imports: Required import statements
        description: Code description
        usage_example: Example of how to use the code
        estimated_runtime: Estimated execution time
    """
    intervention_name: str
    code: str
    imports: List[str]
    description: str
    usage_example: str
    estimated_runtime: str


class CodeGenerator:
    """
    Generates Python code for fairness interventions using LLM assistance.

    Integrates with Gemini 3 API for intelligent code synthesis based on
    model architecture, data characteristics, and fairness requirements.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-3-pro"):
        """
        Initialize the code generator.

        Args:
            api_key: Gemini API key (if None, will look for environment variable)
            model: Gemini model to use for code generation
        """
        self.api_key = api_key
        self.model = model
        self.llm_client = None

        if api_key:
            self._initialize_llm_client()

    def _initialize_llm_client(self):
        """
        Initialize the LLM client for code generation.

        In production, this would set up the actual Gemini API client.
        """
        # TODO: Implement actual Gemini API client initialization
        # import google.generativeai as genai
        # genai.configure(api_key=self.api_key)
        # self.llm_client = genai.GenerativeModel(self.model)

        logger.info(f"LLM client initialized (MOCK) with model: {self.model}")
        self.llm_client = "mock_client"

    def generate_fix_code(
        self,
        intervention_type: str,
        model_code_snippet: Optional[str] = None,
        data_description: Optional[Dict[str, Any]] = None,
        fairness_requirements: Optional[Dict[str, Any]] = None
    ) -> GeneratedCode:
        """
        Generate Python code to implement a fairness intervention using Gemini 3.

        Args:
            intervention_type: Type of intervention (e.g., "Reweighing", "Adversarial Debiasing")
            model_code_snippet: Optional code snippet of the original model
            data_description: Optional description of the dataset
            fairness_requirements: Optional specific fairness constraints

        Returns:
            GeneratedCode object with complete implementation
        """
        import re
        from causal_analysis import _init_gemini, _call_gemini_with_retry, _load_api_config

        logger.info(f"Generating code for intervention: {intervention_type}")

        # Try to load API key from config if not set
        if not self.api_key:
            config = _load_api_config()
            self.api_key = config.get('gemini_api_key')

        # Initialize Gemini
        gemini_model = _init_gemini()

        if not gemini_model:
            logger.warning("Gemini not available, using template generation")
            return self._template_generate_code(intervention_type)

        # Extract context
        outcome = fairness_requirements.get('outcome', 'outcome') if fairness_requirements else 'outcome'
        sensitive_attr = fairness_requirements.get('sensitive_attr', 'sensitive_attribute') if fairness_requirements else 'sensitive_attribute'

        # Build prompt
        prompt = f"""Generate ONLY executable Python code using aif360 or fairlearn to apply {intervention_type} fairness intervention.

Requirements:
- Assume input is pandas DataFrame 'df' with columns for features and outcome '{outcome}'
- Sensitive attribute: '{sensitive_attr}'
- Include all necessary imports
- Handle both binary classification and clinical outcomes
- Add error handling for missing data
- Include docstrings
- Use sklearn-compatible API

Context:
- Intervention: {intervention_type}
- Use case: Clinical fairness in healthcare AI
{f"- Model type: {model_code_snippet[:200]}" if model_code_snippet else ""}

Return ONLY Python code with no markdown blocks, no explanations."""

        response = _call_gemini_with_retry(gemini_model, prompt)

        if not response:
            logger.error("Gemini code generation failed, using template")
            return self._template_generate_code(intervention_type)

        # Parse response
        code = response.strip()
        code = re.sub(r'```python\n?', '', code)
        code = re.sub(r'```', '', code)

        # Extract imports
        import_lines = [line for line in code.split('\n') if line.startswith('import ') or line.startswith('from ')]

        # Generate usage example
        usage_prompt = f"""Write a 5-line usage example for this intervention code:

```python
{code[:500]}
```

Show how to load data, apply the intervention, and train a model."""

        usage = _call_gemini_with_retry(gemini_model, usage_prompt)
        usage = usage.strip() if usage else "# Load data\n# Apply intervention\n# Train model"
        usage = re.sub(r'```python\n?|```', '', usage)

        return GeneratedCode(
            intervention_name=intervention_type,
            code=code,
            imports=import_lines,
            description=f"Gemini-generated {intervention_type} implementation",
            usage_example=usage,
            estimated_runtime="Varies by dataset size"
        )

    def _llm_generate_code(
        self,
        intervention_type: str,
        model_code_snippet: Optional[str],
        data_description: Optional[Dict[str, Any]],
        fairness_requirements: Optional[Dict[str, Any]]
    ) -> GeneratedCode:
        """
        Use LLM to generate intervention code.

        Args:
            intervention_type: Type of intervention
            model_code_snippet: Original model code
            data_description: Dataset characteristics
            fairness_requirements: Fairness constraints

        Returns:
            GeneratedCode object
        """
        # TODO: Implement actual Gemini API call
        # For now, return mock LLM-generated code

        prompt = self._build_code_generation_prompt(
            intervention_type,
            model_code_snippet,
            data_description,
            fairness_requirements
        )

        logger.info("Sending prompt to LLM (MOCK)")
        logger.debug(f"Prompt: {prompt[:200]}...")

        # Mock LLM response
        return self._template_generate_code(intervention_type)

    def _build_code_generation_prompt(
        self,
        intervention_type: str,
        model_code_snippet: Optional[str],
        data_description: Optional[Dict[str, Any]],
        fairness_requirements: Optional[Dict[str, Any]]
    ) -> str:
        """
        Build prompt for LLM code generation.

        Args:
            intervention_type: Type of intervention
            model_code_snippet: Original model code
            data_description: Dataset info
            fairness_requirements: Fairness constraints

        Returns:
            Formatted prompt string
        """
        prompt = f"""Generate production-ready Python code to implement {intervention_type} fairness intervention.

Requirements:
- Use AIF360 or Fairlearn libraries
- Include proper error handling
- Add comprehensive docstrings
- Generate complete, executable code
- Include example usage

"""

        if model_code_snippet:
            prompt += f"\nOriginal Model Code:\n{model_code_snippet}\n"

        if data_description:
            prompt += f"\nDataset Description:\n{data_description}\n"

        if fairness_requirements:
            prompt += f"\nFairness Requirements:\n{fairness_requirements}\n"

        prompt += """
Please generate:
1. Import statements
2. Main intervention function
3. Helper functions if needed
4. Usage example
5. Expected output description
"""

        return prompt

    def _template_generate_code(self, intervention_type: str) -> GeneratedCode:
        """
        Generate code using predefined templates.

        Args:
            intervention_type: Type of intervention

        Returns:
            GeneratedCode object
        """
        templates = {
            "Reweighing": self._generate_reweighing_code(),
            "Resampling": self._generate_resampling_code(),
            "Adversarial Debiasing": self._generate_adversarial_code(),
            "Calibrated Equalized Odds": self._generate_calibrated_eo_code(),
            "Prejudice Remover": self._generate_prejudice_remover_code(),
        }

        return templates.get(
            intervention_type,
            self._generate_generic_code(intervention_type)
        )

    def _generate_reweighing_code(self) -> GeneratedCode:
        """Generate code for Reweighing intervention."""

        imports = [
            "import pandas as pd",
            "import numpy as np",
            "from aif360.datasets import BinaryLabelDataset",
            "from aif360.algorithms.preprocessing import Reweighing",
            "from sklearn.linear_model import LogisticRegression"
        ]

        code = '''def apply_reweighing(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sensitive_attr: str,
    privileged_groups: list,
    unprivileged_groups: list
) -> tuple:
    """
    Apply Reweighing fairness intervention to training data.

    Args:
        X_train: Training features
        y_train: Training labels
        sensitive_attr: Name of protected attribute
        privileged_groups: List of dicts defining privileged groups
        unprivileged_groups: List of dicts defining unprivileged groups

    Returns:
        Tuple of (reweighted_X, reweighted_y, sample_weights)
    """
    # Combine features and labels
    train_data = X_train.copy()
    train_data['label'] = y_train

    # Convert to AIF360 dataset
    dataset = BinaryLabelDataset(
        df=train_data,
        label_names=['label'],
        protected_attribute_names=[sensitive_attr]
    )

    # Apply reweighing
    RW = Reweighing(
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )

    dataset_transformed = RW.fit_transform(dataset)

    # Extract sample weights
    sample_weights = dataset_transformed.instance_weights

    # Get transformed data
    X_reweighted = dataset_transformed.features
    y_reweighted = dataset_transformed.labels.ravel()

    return X_reweighted, y_reweighted, sample_weights


def train_with_reweighing(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sensitive_attr: str,
    privileged_value: int = 1,
    unprivileged_value: int = 0
):
    """
    Train a model with Reweighing intervention.

    Args:
        X_train: Training features
        y_train: Training labels
        sensitive_attr: Protected attribute name
        privileged_value: Value indicating privileged group
        unprivileged_value: Value indicating unprivileged group

    Returns:
        Trained model
    """
    privileged_groups = [{sensitive_attr: privileged_value}]
    unprivileged_groups = [{sensitive_attr: unprivileged_value}]

    # Apply reweighing
    X_rw, y_rw, weights = apply_reweighing(
        X_train, y_train, sensitive_attr,
        privileged_groups, unprivileged_groups
    )

    # Train model with sample weights
    model = LogisticRegression(max_iter=1000)
    model.fit(X_rw, y_rw, sample_weight=weights)

    return model
'''

        usage = '''# Example usage:
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your data
data = pd.read_csv('clinical_data.csv')
X = data.drop('outcome', axis=1)
y = data['outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Apply reweighing and train
model = train_with_reweighing(
    X_train, y_train,
    sensitive_attr='race',
    privileged_value=1,
    unprivileged_value=0
)

# Evaluate
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
'''

        return GeneratedCode(
            intervention_name="Reweighing",
            code=code,
            imports=imports,
            description="Reweighs training samples to remove bias before model training using AIF360",
            usage_example=usage,
            estimated_runtime="Fast (< 1 minute for most datasets)"
        )

    def _generate_resampling_code(self) -> GeneratedCode:
        """Generate code for Resampling intervention."""

        imports = [
            "import pandas as pd",
            "import numpy as np",
            "from imblearn.over_sampling import RandomOverSampler",
            "from imblearn.under_sampling import RandomUnderSampler",
            "from sklearn.utils import resample"
        ]

        code = '''def apply_fairness_resampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sensitive_attr: str,
    strategy: str = 'balance'
) -> tuple:
    """
    Resample training data to balance across sensitive attribute groups.

    Args:
        X_train: Training features
        y_train: Training labels
        sensitive_attr: Protected attribute name
        strategy: 'balance', 'oversample', or 'undersample'

    Returns:
        Tuple of (resampled_X, resampled_y)
    """
    # Combine for resampling
    data = X_train.copy()
    data['label'] = y_train
    data['sensitive'] = X_train[sensitive_attr]

    # Get group sizes
    groups = data.groupby(['sensitive', 'label'])
    group_sizes = groups.size()

    if strategy == 'balance':
        # Balance to the average group size
        target_size = int(group_sizes.mean())
    elif strategy == 'oversample':
        # Balance to the largest group
        target_size = int(group_sizes.max())
    else:  # undersample
        # Balance to the smallest group
        target_size = int(group_sizes.min())

    # Resample each group
    resampled_groups = []
    for (sens_val, label_val), group_df in groups:
        if len(group_df) < target_size:
            # Oversample
            resampled = resample(
                group_df,
                n_samples=target_size,
                replace=True,
                random_state=42
            )
        elif len(group_df) > target_size:
            # Undersample
            resampled = resample(
                group_df,
                n_samples=target_size,
                replace=False,
                random_state=42
            )
        else:
            resampled = group_df

        resampled_groups.append(resampled)

    # Combine all resampled groups
    data_resampled = pd.concat(resampled_groups, ignore_index=True)

    # Split back into X and y
    X_resampled = data_resampled.drop(['label', 'sensitive'], axis=1)
    y_resampled = data_resampled['label']

    return X_resampled, y_resampled
'''

        usage = '''# Example usage:
X_resampled, y_resampled = apply_fairness_resampling(
    X_train, y_train,
    sensitive_attr='race',
    strategy='balance'
)

# Train model on resampled data
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_resampled, y_resampled)
'''

        return GeneratedCode(
            intervention_name="Resampling",
            code=code,
            imports=imports,
            description="Balances training data through resampling to ensure fair representation",
            usage_example=usage,
            estimated_runtime="Fast (< 1 minute)"
        )

    def _generate_adversarial_code(self) -> GeneratedCode:
        """Generate code for Adversarial Debiasing."""

        imports = [
            "import pandas as pd",
            "import numpy as np",
            "from aif360.datasets import BinaryLabelDataset",
            "from aif360.algorithms.inprocessing import AdversarialDebiasing",
            "import tensorflow as tf"
        ]

        code = '''def train_adversarial_debiased_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sensitive_attr: str,
    privileged_groups: list,
    unprivileged_groups: list,
    adversary_loss_weight: float = 0.1,
    num_epochs: int = 50
):
    """
    Train model with adversarial debiasing.

    Args:
        X_train: Training features
        y_train: Training labels
        sensitive_attr: Protected attribute name
        privileged_groups: List of dicts defining privileged groups
        unprivileged_groups: List of dicts defining unprivileged groups
        adversary_loss_weight: Weight for adversarial loss
        num_epochs: Number of training epochs

    Returns:
        Trained adversarially debiased model
    """
    # Prepare dataset
    train_data = X_train.copy()
    train_data['label'] = y_train

    dataset = BinaryLabelDataset(
        df=train_data,
        label_names=['label'],
        protected_attribute_names=[sensitive_attr]
    )

    # Suppress TensorFlow warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    # Initialize and train adversarial debiasing
    sess = tf.compat.v1.Session()

    debiaser = AdversarialDebiasing(
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups,
        scope_name='debiaser',
        debias=True,
        adversary_loss_weight=adversary_loss_weight,
        num_epochs=num_epochs,
        batch_size=128,
        sess=sess
    )

    # Fit the model
    debiaser.fit(dataset)

    return debiaser

# NOTE: This requires TensorFlow 1.x or compatibility mode
# For production use, consider migrating to TensorFlow 2.x compatible alternatives
'''

        usage = '''# Example usage:
privileged_groups = [{'race': 1}]
unprivileged_groups = [{'race': 0}]

model = train_adversarial_debiased_model(
    X_train, y_train,
    sensitive_attr='race',
    privileged_groups=privileged_groups,
    unprivileged_groups=unprivileged_groups,
    adversary_loss_weight=0.1,
    num_epochs=50
)

# Make predictions
# dataset_test = BinaryLabelDataset(...)
# predictions = model.predict(dataset_test)
'''

        return GeneratedCode(
            intervention_name="Adversarial Debiasing",
            code=code,
            imports=imports,
            description="Uses adversarial learning to remove bias during model training",
            usage_example=usage,
            estimated_runtime="Moderate (5-15 minutes depending on dataset size)"
        )

    def _generate_calibrated_eo_code(self) -> GeneratedCode:
        """Generate code for Calibrated Equalized Odds."""

        imports = [
            "import pandas as pd",
            "import numpy as np",
            "from aif360.datasets import BinaryLabelDataset",
            "from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing"
        ]

        code = '''def apply_calibrated_equalized_odds(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sensitive_attr: str,
    privileged_groups: list,
    unprivileged_groups: list
):
    """
    Apply Calibrated Equalized Odds postprocessing to model predictions.

    Args:
        model: Trained base model
        X_train, y_train: Training data (used for calibration)
        X_test, y_test: Test data
        sensitive_attr: Protected attribute name
        privileged_groups: Privileged group definitions
        unprivileged_groups: Unprivileged group definitions

    Returns:
        Debiased predictions on test set
    """
    # Get base model predictions on training set (for calibration)
    y_train_pred = model.predict(X_train)

    # Prepare training dataset with predictions
    train_data = X_train.copy()
    train_data['label'] = y_train
    train_dataset = BinaryLabelDataset(
        df=train_data,
        label_names=['label'],
        protected_attribute_names=[sensitive_attr]
    )

    # Create dataset with predictions
    train_pred_dataset = train_dataset.copy()
    train_pred_dataset.labels = y_train_pred.reshape(-1, 1)

    # Initialize calibrated equalized odds
    cpp = CalibratedEqOddsPostprocessing(
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups,
        cost_constraint='weighted',
        seed=42
    )

    # Fit on training data
    cpp = cpp.fit(train_dataset, train_pred_dataset)

    # Apply to test set
    y_test_pred = model.predict(X_test)
    test_data = X_test.copy()
    test_data['label'] = y_test
    test_dataset = BinaryLabelDataset(
        df=test_data,
        label_names=['label'],
        protected_attribute_names=[sensitive_attr]
    )

    test_pred_dataset = test_dataset.copy()
    test_pred_dataset.labels = y_test_pred.reshape(-1, 1)

    # Transform predictions
    debiased_dataset = cpp.predict(test_pred_dataset)

    return debiased_dataset.labels.ravel()
'''

        usage = '''# Example usage:
from sklearn.linear_model import LogisticRegression

# Train base model
base_model = LogisticRegression(max_iter=1000)
base_model.fit(X_train, y_train)

# Apply calibrated equalized odds
privileged_groups = [{'race': 1}]
unprivileged_groups = [{'race': 0}]

y_pred_debiased = apply_calibrated_equalized_odds(
    base_model,
    X_train, y_train,
    X_test, y_test,
    sensitive_attr='race',
    privileged_groups=privileged_groups,
    unprivileged_groups=unprivileged_groups
)

# Evaluate
from sklearn.metrics import accuracy_score
print(f"Debiased Accuracy: {accuracy_score(y_test, y_pred_debiased):.3f}")
'''

        return GeneratedCode(
            intervention_name="Calibrated Equalized Odds",
            code=code,
            imports=imports,
            description="Postprocessing method to adjust predictions for equalized odds fairness",
            usage_example=usage,
            estimated_runtime="Fast (< 1 minute)"
        )

    def _generate_prejudice_remover_code(self) -> GeneratedCode:
        """Generate code for Prejudice Remover."""

        imports = [
            "import pandas as pd",
            "from aif360.datasets import BinaryLabelDataset",
            "from aif360.algorithms.inprocessing import PrejudiceRemover"
        ]

        code = '''def train_prejudice_remover(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sensitive_attr: str,
    eta: float = 1.0
):
    """
    Train model with prejudice remover regularization.

    Args:
        X_train: Training features
        y_train: Training labels
        sensitive_attr: Protected attribute name
        eta: Fairness penalty parameter (higher = more debiasing)

    Returns:
        Trained prejudice remover model
    """
    # Prepare dataset
    train_data = X_train.copy()
    train_data['label'] = y_train

    dataset = BinaryLabelDataset(
        df=train_data,
        label_names=['label'],
        protected_attribute_names=[sensitive_attr]
    )

    # Train prejudice remover
    model = PrejudiceRemover(
        eta=eta,
        sensitive_attr=sensitive_attr,
        class_attr='label'
    )

    model.fit(dataset)

    return model
'''

        usage = '''# Example usage:
model = train_prejudice_remover(
    X_train, y_train,
    sensitive_attr='race',
    eta=1.0  # Adjust for fairness-accuracy tradeoff
)

# Predict
test_data = X_test.copy()
test_data['label'] = y_test
test_dataset = BinaryLabelDataset(
    df=test_data,
    label_names=['label'],
    protected_attribute_names=['race']
)

predictions = model.predict(test_dataset)
'''

        return GeneratedCode(
            intervention_name="Prejudice Remover",
            code=code,
            imports=imports,
            description="Adds fairness regularization term during model training",
            usage_example=usage,
            estimated_runtime="Fast to Moderate (1-5 minutes)"
        )

    def _generate_generic_code(self, intervention_type: str) -> GeneratedCode:
        """Generate generic placeholder code for unknown interventions."""

        return GeneratedCode(
            intervention_name=intervention_type,
            code=f"# TODO: Implement {intervention_type}\n# Code generation for this intervention type is not yet implemented.",
            imports=["import pandas as pd", "import numpy as np"],
            description=f"Placeholder for {intervention_type} intervention",
            usage_example="# Implementation needed",
            estimated_runtime="Unknown"
        )


def generate_fix_code(
    intervention_type: str,
    model_code_snippet: Optional[str] = None,
    api_key: Optional[str] = None
) -> str:
    """
    Convenience function to generate intervention code.

    Args:
        intervention_type: Type of intervention
        model_code_snippet: Optional original model code
        api_key: Optional Gemini API key

    Returns:
        Generated Python code as string

    Example:
        >>> code = generate_fix_code("Reweighing")
        >>> print(code)
    """
    generator = CodeGenerator(api_key=api_key)
    result = generator.generate_fix_code(intervention_type, model_code_snippet)

    # Combine imports and code
    full_code = "\n".join(result.imports) + "\n\n" + result.code
    return full_code


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)

    generator = CodeGenerator()

    interventions = ["Reweighing", "Resampling", "Adversarial Debiasing", "Calibrated Equalized Odds"]

    for intervention in interventions:
        print("=" * 70)
        print(f"GENERATED CODE: {intervention}")
        print("=" * 70)

        result = generator.generate_fix_code(intervention)

        print("\nImports:")
        for imp in result.imports:
            print(f"  {imp}")

        print(f"\nDescription: {result.description}")
        print(f"Estimated Runtime: {result.estimated_runtime}")

        print("\nCode:")
        print(result.code[:500] + "..." if len(result.code) > 500 else result.code)
        print()
