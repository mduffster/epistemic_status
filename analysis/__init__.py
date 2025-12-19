"""
Epistemic Probing Analysis Package

Modular analysis tools for understanding epistemic states in language models.
"""

from .loader import load_model_data, ModelData
from .core import basic_stats, failure_mode_analysis, analyze_prompt_features
from .probing import run_linear_probe, run_mlp_probe, layer_analysis, compare_positions, probe_with_controls, compare_linear_vs_mlp
from .entropy import entropy_analysis, entropy_vs_probe
from .calibration import confidence_calibration
from .effects import compute_effect_sizes, compute_roc_auc
from .comparison import cross_model_generalization, compare_models

__all__ = [
    'load_model_data',
    'ModelData',
    'basic_stats',
    'failure_mode_analysis',
    'analyze_prompt_features',
    'run_linear_probe',
    'run_mlp_probe',
    'layer_analysis',
    'compare_positions',
    'probe_with_controls',
    'compare_linear_vs_mlp',
    'entropy_analysis',
    'entropy_vs_probe',
    'confidence_calibration',
    'compute_effect_sizes',
    'compute_roc_auc',
    'cross_model_generalization',
    'compare_models',
]
