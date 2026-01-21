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


# =============================================================================
# STEERING AND SVD FIGURES FOR PAPER
# =============================================================================

POLICY_CATEGORIES = ['confident_incorrect', 'nonsensical', 'ambiguous']
FACTUAL_CATEGORIES = ['confident_correct', 'uncertain_correct']
PAPER_FIGURES_DIR = 'paper/figures'


def plot_steering_by_category(
    steering_results: Dict,
    model_name: str = "",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot category movement along steering direction.

    Shows how much each category moves along the alignment direction,
    with policy vs factual distinction highlighted.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data
    projection_change = steering_results.get('projection_change', {})
    if not projection_change:
        print("No projection_change data in steering_results")
        return

    categories = list(projection_change.keys())
    changes = [projection_change[c]['mean_change'] for c in categories]

    # Color by policy vs factual
    colors = ['#d62728' if c in POLICY_CATEGORIES else '#1f77b4' for c in categories]

    # Sort by change magnitude
    sorted_idx = np.argsort(changes)[::-1]
    categories = [categories[i] for i in sorted_idx]
    changes = [changes[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]

    # Plot
    bars = ax.barh(range(len(categories)), changes, color=colors, alpha=0.8)

    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([c.replace('_', '\n') for c in categories])
    ax.set_xlabel('Movement along steering direction', fontsize=12)
    ax.set_title(f'{model_name}: Category Movement Along Alignment Direction', fontsize=14)

    # Add policy/factual legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', alpha=0.8, label='Policy categories'),
        Patch(facecolor='#1f77b4', alpha=0.8, label='Factual categories')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_svd_spectrum(
    lowrank_results: Dict,
    model_name: str = "",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot SVD singular value spectrum and cumulative variance.

    Two-panel figure showing:
    1. Singular value decay (log scale)
    2. Cumulative variance explained
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    singular_values = np.array(lowrank_results['singular_values'])
    variance_explained = np.array(lowrank_results['variance_explained'])
    effective_rank = lowrank_results['effective_rank']

    n_components = len(singular_values)
    x = np.arange(1, n_components + 1)

    # Panel 1: Singular value spectrum
    ax = axes[0]
    ax.semilogy(x, singular_values, 'b-', marker='o', markersize=3)
    ax.axvline(x=effective_rank, color='r', linestyle='--',
               label=f'Effective rank = {effective_rank}')
    ax.set_xlabel('Component', fontsize=12)
    ax.set_ylabel('Singular value (log scale)', fontsize=12)
    ax.set_title('Singular Value Spectrum', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Cumulative variance
    ax = axes[1]
    ax.plot(x, variance_explained * 100, 'b-', marker='o', markersize=3)
    ax.axhline(y=80, color='r', linestyle='--', label='80% variance')
    ax.axvline(x=effective_rank, color='r', linestyle=':', alpha=0.7)
    ax.fill_between(x[:effective_rank], 0, variance_explained[:effective_rank] * 100,
                    alpha=0.3, color='blue')
    ax.set_xlabel('Number of components', fontsize=12)
    ax.set_ylabel('Cumulative variance explained (%)', fontsize=12)
    ax.set_title('Low-Rank Structure of Alignment Changes', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)

    plt.suptitle(f'{model_name}', fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_steering_ratio_comparison(
    bootstrap_results: Dict[str, Dict],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot steering ratio comparison across models with bootstrap CIs.

    Args:
        bootstrap_results: Dict mapping model_name -> bootstrap_steering_ratio results
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(bootstrap_results.keys())
    n_models = len(models)

    ratios = [bootstrap_results[m]['ratio'] for m in models]
    ci_lows = [bootstrap_results[m]['ratio_ci'][0] for m in models]
    ci_highs = [bootstrap_results[m]['ratio_ci'][1] for m in models]

    # Error bars
    errors = [[r - l for r, l in zip(ratios, ci_lows)],
              [h - r for r, h in zip(ratios, ci_highs)]]

    y_pos = np.arange(n_models)

    # Color by training method (RLHF/DPO vs SFT-only)
    colors = []
    for m in models:
        if 'llama' in m.lower() or 'qwen' in m.lower():
            colors.append('#d62728')  # Red for RLHF/DPO
        else:
            colors.append('#1f77b4')  # Blue for SFT-only

    ax.barh(y_pos, ratios, xerr=errors, color=colors, alpha=0.8, capsize=5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('Policy/Factual Steering Ratio', fontsize=12)
    ax.set_title('Selective Steering: Policy Categories Move Further', fontsize=14)

    # Reference line at 1.0
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1.5, label='Equal movement')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', alpha=0.8, label='RLHF/DPO'),
        Patch(facecolor='#1f77b4', alpha=0.8, label='SFT-only'),
        plt.Line2D([0], [0], color='black', linestyle='--', label='Ratio = 1.0')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0.9, max(ci_highs) * 1.1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_probe_transfer_heatmap(
    transfer_results: Dict[str, Dict],
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot probe transfer accuracy as heatmap (factual vs policy rows).

    Args:
        transfer_results: Dict mapping model_name -> {'factual_acc': float, 'policy_acc': float}
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    models = list(transfer_results.keys())
    factual_accs = [transfer_results[m]['factual_acc'] for m in models]
    policy_accs = [transfer_results[m]['policy_acc'] for m in models]

    data = np.array([factual_accs, policy_accs])

    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0.3, vmax=1.0)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Factual', 'Policy'])

    # Add text annotations
    for i in range(2):
        for j in range(len(models)):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=11)

    ax.set_title('Probe Transfer Accuracy (Train Base → Test Instruct)', fontsize=14)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Transfer Accuracy', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_loading_ratio_by_category(
    lowrank_results: Dict,
    model_name: str = "",
    save_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot category loading magnitudes on top SVD components.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    by_category = lowrank_results.get('by_category', {})
    if not by_category:
        print("No by_category data in lowrank_results")
        return

    categories = list(by_category.keys())
    magnitudes = [by_category[c]['loading_magnitude'] for c in categories]

    # Color by policy vs factual
    colors = ['#d62728' if c in POLICY_CATEGORIES else '#1f77b4' for c in categories]

    # Sort by magnitude
    sorted_idx = np.argsort(magnitudes)[::-1]
    categories = [categories[i] for i in sorted_idx]
    magnitudes = [magnitudes[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]

    bars = ax.barh(range(len(categories)), magnitudes, color=colors, alpha=0.8)

    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([c.replace('_', '\n') for c in categories])
    ax.set_xlabel('Loading magnitude on top-10 SVD components', fontsize=12)
    ax.set_title(f'{model_name}: Category Loadings on Alignment Subspace', fontsize=14)

    # Add policy/factual legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', alpha=0.8, label='Policy categories'),
        Patch(facecolor='#1f77b4', alpha=0.8, label='Factual categories')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    ax.grid(True, alpha=0.3, axis='x')

    # Add summary stats
    policy_mean = np.mean([by_category[c]['loading_magnitude']
                          for c in POLICY_CATEGORIES if c in by_category])
    factual_mean = np.mean([by_category[c]['loading_magnitude']
                           for c in FACTUAL_CATEGORIES if c in by_category])
    ratio = policy_mean / factual_mean if factual_mean > 0 else 0

    ax.text(0.95, 0.05, f'Policy/Factual ratio: {ratio:.2f}x',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def generate_paper_figures(
    steering_results: Dict[str, Dict],
    lowrank_results: Dict[str, Dict],
    bootstrap_results: Dict[str, Dict],
    transfer_results: Dict[str, Dict],
    output_dir: str = PAPER_FIGURES_DIR,
    show: bool = False
):
    """
    Generate all figures for the paper.

    Args:
        steering_results: Dict mapping model_name -> steering_by_category results
        lowrank_results: Dict mapping model_name -> low_rank_analysis results
        bootstrap_results: Dict mapping model_name -> bootstrap_steering_ratio results
        transfer_results: Dict mapping model_name -> probe transfer results
        output_dir: Directory to save figures
        show: Whether to display figures
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nGenerating paper figures to {output_dir}/")
    print("=" * 50)

    # Figure 1: Steering ratio comparison across models
    if bootstrap_results:
        plot_steering_ratio_comparison(
            bootstrap_results,
            save_path=f"{output_dir}/fig1_steering_ratio.pdf",
            show=show
        )

    # Figure 2: SVD spectrum for representative model
    for model_name, results in lowrank_results.items():
        plot_svd_spectrum(
            results,
            model_name=model_name,
            save_path=f"{output_dir}/fig2_svd_{model_name.lower().replace(' ', '_')}.pdf",
            show=show
        )
        break  # Just one representative model

    # Figure 3: Category movement along steering direction
    for model_name, results in steering_results.items():
        plot_steering_by_category(
            results,
            model_name=model_name,
            save_path=f"{output_dir}/fig3_steering_{model_name.lower().replace(' ', '_')}.pdf",
            show=show
        )

    # Figure 4: Probe transfer heatmap
    if transfer_results:
        plot_probe_transfer_heatmap(
            transfer_results,
            save_path=f"{output_dir}/fig4_probe_transfer.pdf",
            show=show
        )

    # Figure 5: Loading ratios by category
    for model_name, results in lowrank_results.items():
        plot_loading_ratio_by_category(
            results,
            model_name=model_name,
            save_path=f"{output_dir}/fig5_loading_{model_name.lower().replace(' ', '_')}.pdf",
            show=show
        )

    print("\nFigure generation complete!")
