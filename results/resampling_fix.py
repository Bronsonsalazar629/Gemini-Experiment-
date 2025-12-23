import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import resample

def apply_fairness_resampling(
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


# ======================================================================
# USAGE EXAMPLE
# ======================================================================
# Example usage:
X_resampled, y_resampled = apply_fairness_resampling(
    X_train, y_train,
    sensitive_attr='race',
    strategy='balance'
)

# Train model on resampled data
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_resampled, y_resampled)
