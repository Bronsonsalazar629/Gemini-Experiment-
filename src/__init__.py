"""
Clinical Fairness Intervention System

A comprehensive toolkit for detecting and remediating bias in clinical ML models.
"""

__version__ = "1.0.0"
__author__ = "Clinical Fairness Team"

from .causal_analysis import CausalAnalyzer, infer_causal_graph
from .bias_detection import BiasDetector, compute_fairness_metrics
from .intervention_engine import InterventionEngine, suggest_interventions
from .code_generator import CodeGenerator, generate_fix_code

__all__ = [
    "CausalAnalyzer",
    "infer_causal_graph",
    "BiasDetector",
    "compute_fairness_metrics",
    "InterventionEngine",
    "suggest_interventions",
    "CodeGenerator",
    "generate_fix_code",
]
