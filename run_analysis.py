#!/usr/bin/env python3
"""
Epistemic Probing Analysis CLI

Modular analysis runner for epistemic probing experiments.

Usage:
    python run_analysis.py --model qwen_base
    python run_analysis.py --model qwen_base --compare qwen_instruct
    python run_analysis.py --model qwen_base --analysis effects roc
    python run_analysis.py --model qwen_base --compare qwen_instruct --analysis generalization
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

# Output directories
PLOTS_DIR = Path(__file__).parent / "plots"
RESULTS_DIR = Path(__file__).parent / "results"

from analysis import (
    load_model_data,
    basic_stats,
    failure_mode_analysis,
    analyze_prompt_features,
    run_linear_probe,
    layer_analysis,
    compare_positions,
    probe_with_controls,
    entropy_analysis,
    entropy_vs_probe,
    confidence_calibration,
    compute_effect_sizes,
    compute_roc_auc,
    cross_model_generalization,
    compare_models,
)
from analysis.effects import compute_layer_auc
from analysis.comparison import bidirectional_generalization, layer_generalization
from analysis.entanglement import (
    probe_confidence_by_category,
    probe_confidence_layerwise,
    bootstrap_confidence_intervals,
    held_out_category_generalization,
    compare_base_instruct_entanglement,
    run_full_entanglement_analysis,
    activation_similarity_by_category,
    compare_activation_similarity,
)
from analysis.plotting import (
    plot_entropy_distributions,
    plot_layer_analysis,
    plot_layer_comparison,
    plot_roc_curves,
    plot_calibration_curve,
    plot_generalization_layers,
)


def save_results(results: dict, name: str):
    """Save results dictionary to JSON file."""
    RESULTS_DIR.mkdir(exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    def convert(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        if hasattr(obj, 'item'):
            return obj.item()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results_clean = convert(results)
    results_clean['_metadata'] = {
        'generated_at': datetime.now().isoformat(),
        'name': name
    }

    filepath = RESULTS_DIR / f"{name}.json"
    with open(filepath, 'w') as f:
        json.dump(results_clean, f, indent=2)
    print(f"💾 Saved results to {filepath}")


def run_core_analysis(data, save_plots=False):
    """Run core analysis suite."""
    basic_stats(data)
    entropy_analysis(data)
    failure_mode_analysis(data)
    analyze_prompt_features(data)

    print("\n" + "=" * 60)
    print("LINEAR PROBING")
    print("=" * 60)

    print("\n--- All categories, last position ---")
    run_linear_probe(data, target='correct', position='last')

    compare_positions(data, target='correct')

    print("\n--- Confident categories only ---")
    run_linear_probe(data, categories=['confident_correct', 'confident_incorrect'])

    print("\n--- Uncertain categories only ---")
    run_linear_probe(data, categories=['uncertain_correct', 'uncertain_incorrect'])

    if save_plots:
        plot_entropy_distributions(data, save_path=str(PLOTS_DIR / f"{data.name}_entropy.png"), show=False)


def run_controlled_probes(data):
    """Run controlled probing analysis."""
    print("\n" + "=" * 60)
    print("CONTROLLED PROBING")
    print("=" * 60)

    print("\n--- Excluding ambiguous and nonsensical ---")
    probe_with_controls(data, exclude_categories=['ambiguous', 'nonsensical'])

    print("\n--- Excluding first-person prompts ---")
    probe_with_controls(data, exclude_features=['has_first_person'])

    print("\n--- Core test: confident + uncertain, no first-person ---")
    probe_with_controls(
        data,
        exclude_categories=['ambiguous', 'nonsensical'],
        exclude_features=['has_first_person']
    )


def run_layer_analysis(data, save_plots=False):
    """Run layer-wise analysis."""
    layer_results = layer_analysis(data, target='correct', position='last')

    if save_plots:
        plot_layer_analysis(layer_results, data.name, save_path=str(PLOTS_DIR / f"{data.name}_layers.png"), show=False)

    return layer_results


def run_effects_analysis(data, save_plots=False):
    """Run effect size and ROC/AUC analysis."""
    compute_effect_sizes(data)
    roc_results = compute_roc_auc(data)

    if save_plots:
        plot_roc_curves(roc_results, data.name, save_path=str(PLOTS_DIR / f"{data.name}_roc.png"), show=False)

    return roc_results


def run_comparison_analysis(data1, data2, save_plots=False):
    """Run cross-model comparison analysis."""
    comparison = compare_models(data1, data2)

    # Layer-wise generalization
    gen_results = layer_generalization(data1, data2)

    if save_plots:
        plot_generalization_layers(
            gen_results, data1.name, data2.name,
            save_path=str(PLOTS_DIR / f"{data1.name}_to_{data2.name}_generalization.png"),
            show=False
        )

    return comparison, gen_results


def run_entanglement_analysis(data, n_bootstrap=50):
    """Run entanglement analysis suite."""
    print("\n" + "=" * 60)
    print("ENTANGLEMENT ANALYSIS")
    print("=" * 60)

    results = {}

    # Probe confidence by category
    results['probe_confidence'] = probe_confidence_by_category(data)

    # Bootstrap CIs for statistical rigor
    results['bootstrap_ci'] = bootstrap_confidence_intervals(data, n_bootstrap=n_bootstrap)

    # Held-out category generalization
    results['held_out'] = held_out_category_generalization(data)

    # Activation similarity
    results['activation_similarity'] = activation_similarity_by_category(data)

    save_results(results, f"{data.name}_entanglement")
    return results


def run_entanglement_comparison(data1, data2, n_bootstrap=50):
    """Compare entanglement between base and instruct models."""
    print("\n" + "=" * 60)
    print(f"ENTANGLEMENT COMPARISON: {data1.name} vs {data2.name}")
    print("=" * 60)

    results = {}
    results['error_rate_comparison'] = compare_base_instruct_entanglement(data1, data2, n_bootstrap=n_bootstrap)
    results['activation_similarity'] = compare_activation_similarity(data1, data2)

    save_results(results, f"{data1.name}_vs_{data2.name}_entanglement")
    return results


def main():
    parser = argparse.ArgumentParser(description="Epistemic Probing Analysis")
    parser.add_argument("--model", required=True, help="Model name (e.g., qwen_base)")
    parser.add_argument("--base_dir", default="activations", help="Base directory")
    parser.add_argument("--compare", help="Second model to compare")
    parser.add_argument(
        "--analysis",
        nargs='+',
        choices=['core', 'controlled', 'layers', 'entropy', 'calibration',
                 'effects', 'roc', 'generalization', 'entanglement', 'all'],
        default=['core'],
        help="Analysis types to run"
    )
    parser.add_argument("--save_plots", action="store_true", help="Save plots to files")

    args = parser.parse_args()

    # Expand 'all' to all analyses
    if 'all' in args.analysis:
        args.analysis = ['core', 'controlled', 'layers', 'entropy',
                         'calibration', 'effects', 'roc', 'entanglement']
        if args.compare:
            args.analysis.append('generalization')

    # Load primary model
    print(f"\n{'=' * 60}")
    print(f"LOADING: {args.model}")
    print(f"{'=' * 60}")

    data = load_model_data(args.model, args.base_dir)

    # Run requested analyses
    if 'core' in args.analysis:
        run_core_analysis(data, args.save_plots)

    if 'controlled' in args.analysis:
        run_controlled_probes(data)

    if 'layers' in args.analysis:
        layer_results = run_layer_analysis(data, args.save_plots)

    if 'entropy' in args.analysis:
        entropy_vs_probe(data)

    if 'calibration' in args.analysis:
        cal_results = confidence_calibration(data)
        if cal_results and args.save_plots:
            plot_calibration_curve(
                cal_results['calibration_curve'],
                data.name,
                save_path=str(PLOTS_DIR / f"{data.name}_calibration.png"),
                show=False
            )

    if 'effects' in args.analysis:
        compute_effect_sizes(data)

    if 'roc' in args.analysis:
        run_effects_analysis(data, args.save_plots)

    if 'entanglement' in args.analysis:
        run_entanglement_analysis(data)

    # Comparison with second model
    if args.compare:
        print(f"\n{'=' * 60}")
        print(f"LOADING: {args.compare}")
        print(f"{'=' * 60}")

        data2 = load_model_data(args.compare, args.base_dir)

        if 'generalization' in args.analysis:
            run_comparison_analysis(data, data2, args.save_plots)

        if 'entanglement' in args.analysis:
            run_entanglement_comparison(data, data2)

        # If layers analysis was run, also compare layers
        if 'layers' in args.analysis:
            layer_results2 = run_layer_analysis(data2, args.save_plots)
            if args.save_plots:
                plot_layer_comparison(
                    layer_results, layer_results2,
                    data.name, data2.name,
                    save_path=str(PLOTS_DIR / f"{data.name}_vs_{data2.name}_layers.png"),
                    show=False
                )


if __name__ == "__main__":
    main()
