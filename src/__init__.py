"""
Clinical Fairness Intervention System

A comprehensive toolkit for detecting and remediating bias in clinical ML models.
"""

__version__ = "1.0.0"
__author__ = "Bronson"

# Note: Selective imports to avoid circular dependencies
# Import main modules only when needed

__all__ = [
    "CausalAnalyzer",
    "BiasDetector",
    "InterventionEngine",
    "CodeGenerator",
]
