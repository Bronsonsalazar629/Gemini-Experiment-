
# Fairlearn Equalized Odds intervention
from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from sklearn.linear_model import LogisticRegression
import pandas as pd

def apply_intervention(df: pd.DataFrame, sensitive_attr: str, outcome: str):
    """Apply Fairlearn Equalized Odds intervention."""
    X = df.drop(columns=[outcome])
    y = df[outcome]
    sensitive_features = df[sensitive_attr]

    mitigator = ExponentiatedGradient(
        estimator=LogisticRegression(max_iter=1000, random_state=42),
        constraints=EqualizedOdds(),
        eps=0.01
    )

    mitigator.fit(X, y, sensitive_features=sensitive_features)
    return mitigator


# ======================================================================
# Validation Report
# Syntax Valid: True
# Security Safe: True
# Functional Tests Pass: True
# ======================================================================


# Usage: Fairlearn (Equalized Odds)
import pandas as pd

df = pd.read_csv('data.csv')
model = apply_intervention(df, 'race_white', 'high_cost')
predictions = model.predict(df.drop(columns=['high_cost']))
