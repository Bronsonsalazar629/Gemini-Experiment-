
# AIF360 Reweighing (alias for Reweighing)
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

def apply_intervention(df: pd.DataFrame, sensitive_attr: str, outcome: str) -> tuple:
    """Apply AIF360 reweighing fairness intervention."""
    X = df.drop(columns=[outcome])
    y = df[outcome]

    dataset = BinaryLabelDataset(
        df=df,
        label_names=[outcome],
        protected_attribute_names=[sensitive_attr]
    )

    RW = Reweighing(
        unprivileged_groups=[{sensitive_attr: 0}],
        privileged_groups=[{sensitive_attr: 1}]
    )
    dataset_transformed = RW.fit_transform(dataset)
    sample_weights = dataset_transformed.instance_weights

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y, sample_weight=sample_weights)

    return model


# ======================================================================
# Validation Report
# Syntax Valid: True
# Security Safe: True
# Functional Tests Pass: True
# ======================================================================


# Usage: AIF360 Reweighing
import pandas as pd

df = pd.read_csv('data.csv')
model = apply_intervention(df, 'race_white', 'high_cost')
predictions = model.predict(df.drop(columns=['high_cost']))
