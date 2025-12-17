"""
Visualization functions for epistemic analysis.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .loader import ModelData


def plot_entropy_distributions(
    data: ModelData,
    save_path: Optional[str] = None,
    show: bool = True
):
    """Plot entropy distributions by category and correctness."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    df = data.df

    # By category
    ax = axes[0]
    categories = sorted(df['category'].unique())
    for cat in categories:
        cat_data = df[df['category'] == cat]['entropy']
        ax.hist(cat_data, bins=30, alpha=0.5, label=cat)
    ax.set_xlabel('Entropy')
    ax.set_ylabel('Count')
    ax.set_title(f'{data.name}: Entropy by Category')
    ax.legend()

    # By correctness
    ax = axes[1]
    for correct, label in [(True, 'Correct'), (False, 'Incorrect')]:
        cat_data = df[df['correct'] == correct]['entropy']
        ax.hist(cat_data, bins=30, alpha=0.5, label=label)
    ax.set_xlabel('Entropy')
    ax.set_ylabel('Count')
    ax.set_title(f'{data.name}: Entropy by Correctness')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_layer_analysis(
    layer_results: pd.DataFrame,
    model_name: str = "",
    metric: str = 'accuracy_mean',
    save_path: Optional[str] = None,
    show: bool = True
):
    """Plot layer-wise probe accuracy or AUC."""
    fig, ax = plt.subplots(figsize=(10, 5))

    if metric in layer_results.columns:
        y_col = metric
        y_label = 'Accuracy' if 'accuracy' in metric else 'AUC'
    elif 'auc' in layer_results.columns:
        y_col = 'auc'
        y_label = 'AUC'
    else:
        y_col = 'accuracy_mean'
        y_label = 'Accuracy'

    ax.plot(layer_results['layer'], layer_results[y_col], 'b-', marker='o', markersize=4)

    if 'accuracy_std' in layer_results.columns:
        ax.fill_between(
            layer_results['layer'],
            layer_results[y_col] - layer_results['accuracy_std'],
            layer_results[y_col] + layer_results['accuracy_std'],
            alpha=0.2
        )

    ax.set_xlabel('Layer')
    ax.set_ylabel(f'Probe {y_label}')
    ax.set_title(f'{model_name}: Correctness Probe by Layer')
    ax.axhline(y=0.5, color='r', linestyle='--', label='Chance', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_layer_comparison(
    results1: pd.DataFrame,
    results2: pd.DataFrame,
    name1: str,
    name2: str,
    metric: str = 'accuracy_mean',
    save_path: Optional[str] = None,
    show: bool = True
):
    """Plot layer-wise comparison between two models."""
    fig, ax = plt.subplots(figsize=(10, 5))

    y_col = metric if metric in results1.columns else 'accuracy_mean'
    y_label = 'Accuracy' if 'accuracy' in y_col else 'AUC'

    ax.plot(results1['layer'], results1[y_col], 'b-', marker='o', markersize=4, label=name1)
    ax.plot(results2['layer'], results2[y_col], 'r-', marker='s', markersize=4, label=name2)

    ax.set_xlabel('Layer')
    ax.set_ylabel(f'Probe {y_label}')
    ax.set_title(f'Layer-wise {y_label} Comparison')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_roc_curves(
    roc_results: Dict,
    model_name: str = "",
    save_path: Optional[str] = None,
    show: bool = True
):
    """Plot ROC curves for different predictors."""
    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {'entropy': 'blue', 'probe': 'green', 'best_layer': 'red'}
    labels = {
        'entropy': 'Entropy only',
        'probe': 'Full probe',
        'best_layer': 'Best layer'
    }

    for key in ['entropy', 'probe', 'best_layer']:
        if key in roc_results:
            res = roc_results[key]
            label = labels[key]
            if key == 'best_layer':
                label = f"{label} (L{res['layer']})"
            label = f"{label} (AUC={res['auc']:.3f})"

            ax.plot(res['fpr'], res['tpr'], color=colors[key], label=label, linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{model_name}: ROC Curves')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_calibration_curve(
    calibration_data: Dict,
    model_name: str = "",
    save_path: Optional[str] = None,
    show: bool = True
):
    """Plot confidence calibration curve."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Extract data
    bins = []
    accuracies = []
    counts = []
    expected = []

    bin_midpoints = {'1-3': 0.2, '4-5': 0.45, '6-7': 0.65, '8-9': 0.85, '10': 1.0}

    for bin_name, values in calibration_data.items():
        if values['count'] > 0 and not np.isnan(values['accuracy']):
            bins.append(bin_name)
            accuracies.append(values['accuracy'])
            counts.append(values['count'])
            expected.append(bin_midpoints.get(bin_name, 0.5))

    x = range(len(bins))

    # Plot bars
    bars = ax.bar(x, accuracies, alpha=0.7, label='Actual accuracy')

    # Plot expected (perfect calibration)
    ax.plot(x, expected, 'r--', marker='o', label='Perfect calibration', linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels(bins)
    ax.set_xlabel('Confidence Bin')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{model_name}: Confidence Calibration')
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'n={count}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_generalization_layers(
    gen_results: pd.DataFrame,
    train_name: str,
    test_name: str,
    save_path: Optional[str] = None,
    show: bool = True
):
    """Plot layer-wise generalization results."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(gen_results['layer'], gen_results['train_acc'], 'b-',
            marker='o', markersize=4, label=f'Train ({train_name})')
    ax.plot(gen_results['layer'], gen_results['test_acc'], 'r-',
            marker='s', markersize=4, label=f'Test ({test_name})')

    ax.fill_between(gen_results['layer'],
                    gen_results['train_acc'],
                    gen_results['test_acc'],
                    alpha=0.2, color='gray')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Cross-Model Generalization: {train_name} -> {test_name}')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
