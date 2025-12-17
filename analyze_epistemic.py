#!/usr/bin/env python3
"""
Epistemic Probing Analysis Script

Analyzes collected activations to understand:
1. Can we predict correctness from activations? ("knows what it knows")
2. How does this change from base to instruct models?
3. How accurately do instruct models report confidence?
4. What are the failure modes? (hallucination vs playing along vs autocomplete)

Usage:
    python analyze_epistemic.py --model qwen_base
    python analyze_epistemic.py --model qwen_base --compare qwen_instruct
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import evaluation function to re-evaluate responses
from utils import evaluate_response


class EpistemicAnalyzer:
    """Analyzes epistemic probing data."""

    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)
        self.metadata = None
        self.activations = {}
        self.df = None

    def load_data(self):
        """Load metadata and activations."""
        # Load metadata
        metadata_path = self.model_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        print(f"Loaded metadata for {self.metadata['model_config']['model_id']}")
        print(f"  Total prompts: {self.metadata['collection_info']['total_prompts']}")
        print(f"  Batches: {self.metadata['collection_info']['num_batches']}")

        # Convert to DataFrame for easier analysis
        self.df = pd.DataFrame(self.metadata['prompts'])

        # Re-evaluate responses with fixed evaluation logic
        print("  Re-evaluating responses with updated logic...")
        self.df['evaluation'] = self.df.apply(
            lambda row: evaluate_response(row['response'], row['correct_answer'], row['category']),
            axis=1
        )

        # Extract evaluation fields
        self.df['correct'] = self.df['evaluation'].apply(lambda x: x['contains_answer'])
        self.df['exact_match'] = self.df['evaluation'].apply(lambda x: x['exact_match'])
        self.df['acknowledged_unknown'] = self.df['evaluation'].apply(lambda x: x['acknowledged_unknown'])

        # Load activations
        self._load_activations()

        return self

    def _load_activations(self):
        """Load activation arrays from npz files."""
        batch_files = sorted(self.model_dir.glob("batch_*.npz"))

        # Initialize lists for each activation type
        act_types = ['resid_first', 'resid_middle', 'resid_last',
                     'mlp_first', 'mlp_middle', 'mlp_last']

        for act_type in act_types:
            self.activations[act_type] = []

        self.activations['entropy'] = []
        self.activations['prompt_indices'] = []

        for batch_file in batch_files:
            data = np.load(batch_file, allow_pickle=True)

            for act_type in act_types:
                if act_type in data:
                    self.activations[act_type].append(data[act_type])

            if 'entropy' in data:
                self.activations['entropy'].append(data['entropy'])
            if 'prompt_indices' in data:
                self.activations['prompt_indices'].append(data['prompt_indices'])

        # Concatenate all batches
        for act_type in act_types:
            if self.activations[act_type]:
                self.activations[act_type] = np.concatenate(self.activations[act_type], axis=0)
                print(f"  {act_type}: {self.activations[act_type].shape}")

        if self.activations['entropy']:
            self.activations['entropy'] = np.concatenate(self.activations['entropy'])
        if self.activations['prompt_indices']:
            self.activations['prompt_indices'] = np.concatenate(self.activations['prompt_indices'])

    def basic_stats(self):
        """Compute basic statistics about responses."""
        print("\n" + "="*60)
        print("BASIC STATISTICS")
        print("="*60)

        # Category breakdown
        print("\n--- Category Breakdown ---")
        category_stats = self.df.groupby('category').agg({
            'correct': ['sum', 'count', 'mean'],
            'entropy': 'mean',
            'acknowledged_unknown': 'sum'
        }).round(3)
        category_stats.columns = ['correct', 'total', 'accuracy', 'mean_entropy', 'acknowledged_unknown']
        print(category_stats)

        # Overall accuracy
        print(f"\nOverall accuracy: {self.df['correct'].mean():.3f}")
        print(f"Mean entropy: {self.df['entropy'].mean():.3f}")

        return category_stats

    def entropy_analysis(self):
        """Analyze entropy distributions by category and correctness."""
        print("\n" + "="*60)
        print("ENTROPY ANALYSIS")
        print("="*60)

        # Entropy by category
        print("\n--- Entropy by Category ---")
        for cat in self.df['category'].unique():
            cat_data = self.df[self.df['category'] == cat]
            print(f"{cat}:")
            print(f"  Mean: {cat_data['entropy'].mean():.3f}, Std: {cat_data['entropy'].std():.3f}")
            print(f"  Correct mean: {cat_data[cat_data['correct']]['entropy'].mean():.3f}")
            print(f"  Incorrect mean: {cat_data[~cat_data['correct']]['entropy'].mean():.3f}")

        # Entropy vs correctness overall
        print("\n--- Entropy vs Correctness (overall) ---")
        correct_entropy = self.df[self.df['correct']]['entropy'].mean()
        incorrect_entropy = self.df[~self.df['correct']]['entropy'].mean()
        print(f"Correct answers mean entropy: {correct_entropy:.3f}")
        print(f"Incorrect answers mean entropy: {incorrect_entropy:.3f}")

        return {
            'correct_entropy': correct_entropy,
            'incorrect_entropy': incorrect_entropy
        }

    def failure_mode_analysis(self):
        """Analyze failure modes in confident_incorrect category."""
        print("\n" + "="*60)
        print("FAILURE MODE ANALYSIS")
        print("="*60)

        # Focus on confident_incorrect (hallucinations on fictional entities)
        hallucinations = self.df[self.df['category'] == 'confident_incorrect']

        print(f"\nTotal hallucination prompts: {len(hallucinations)}")
        print(f"Acknowledged fictional: {hallucinations['acknowledged_unknown'].sum()}")
        print(f"Hallucinated confidently: {(~hallucinations['acknowledged_unknown']).sum()}")

        # Categorize failure modes by looking at responses
        failure_modes = defaultdict(list)

        for _, row in hallucinations.iterrows():
            response = row['response'].lower()
            prompt = row['prompt'].lower()

            # Check for various patterns
            if row['acknowledged_unknown']:
                failure_modes['acknowledged'].append(row)
            elif 'fictional' in response or "doesn't exist" in response or 'not real' in response:
                failure_modes['partial_acknowledgment'].append(row)
            elif any(x in prompt for x in ['president', 'king', 'emperor']):
                # Check if it seems like autocomplete confusion (like 51st -> 41st)
                failure_modes['autocomplete_confusion'].append(row)
            elif any(x in prompt for x in ['capital of', 'city of']):
                # Check if inventing plausible-sounding answer
                failure_modes['plausible_invention'].append(row)
            else:
                failure_modes['playing_along'].append(row)

        print("\n--- Failure Mode Breakdown ---")
        for mode, items in failure_modes.items():
            print(f"{mode}: {len(items)}")

        # Show examples of each mode
        print("\n--- Examples by Failure Mode ---")
        for mode, items in failure_modes.items():
            if items:
                print(f"\n{mode.upper()}:")
                for item in items[:3]:  # Show up to 3 examples
                    print(f"  Q: {item['prompt']}")
                    print(f"  A: {item['response'][:100]}...")
                    print(f"  Entropy: {item['entropy']:.2f}")

        return failure_modes

    def analyze_prompt_features(self):
        """
        Analyze prompt surface features that might confound probing.

        Key concern: Some categories have systematic prompt patterns
        (e.g., ambiguous questions often mention "I" or "my")
        """
        print("\n" + "="*60)
        print("PROMPT FEATURE ANALYSIS")
        print("="*60)

        # Define feature detectors
        def has_first_person(text):
            """Check if prompt refers to speaker (I, me, my)."""
            words = text.lower().split()
            return any(w in ['i', 'me', 'my', "i'm", "i've"] for w in words)

        def has_second_person(text):
            """Check if prompt refers to listener (you, your)."""
            words = text.lower().split()
            return any(w in ['you', 'your', "you're", "you've"] for w in words)

        def is_subjective(text):
            """Check for subjective/opinion markers."""
            markers = ['best', 'good', 'bad', 'should', 'better', 'worth', 'favorite']
            return any(m in text.lower() for m in markers)

        def has_temporal_deixis(text):
            """Check for time-relative terms."""
            markers = ['now', 'today', 'yesterday', 'tomorrow', 'current', 'recent']
            return any(m in text.lower() for m in markers)

        # Apply feature detectors
        self.df['has_first_person'] = self.df['prompt'].apply(has_first_person)
        self.df['has_second_person'] = self.df['prompt'].apply(has_second_person)
        self.df['is_subjective'] = self.df['prompt'].apply(is_subjective)
        self.df['has_temporal'] = self.df['prompt'].apply(has_temporal_deixis)

        # Feature distribution by category
        print("\n--- Feature Distribution by Category ---")
        features = ['has_first_person', 'has_second_person', 'is_subjective', 'has_temporal']

        feature_by_cat = self.df.groupby('category')[features].mean().round(3)
        print(feature_by_cat)

        # Check if features correlate with correctness
        print("\n--- Features vs Correctness ---")
        for feat in features:
            with_feat = self.df[self.df[feat]]['correct'].mean()
            without_feat = self.df[~self.df[feat]]['correct'].mean()
            n_with = self.df[feat].sum()
            print(f"{feat}:")
            print(f"  With feature ({n_with}): {with_feat:.3f} accuracy")
            print(f"  Without feature ({len(self.df) - n_with}): {without_feat:.3f} accuracy")

        return feature_by_cat

    def probe_with_controls(
        self,
        target: str = 'correct',
        position: str = 'last',
        exclude_categories: Optional[List[str]] = None,
        exclude_features: Optional[List[str]] = None
    ) -> Dict:
        """
        Run linear probe with controls for prompt features.

        Args:
            exclude_categories: Categories to exclude
            exclude_features: Exclude prompts with these features (e.g., 'has_first_person')
        """
        # Create mask
        mask = pd.Series([True] * len(self.df))

        if exclude_categories:
            mask &= ~self.df['category'].isin(exclude_categories)

        if exclude_features:
            for feat in exclude_features:
                if feat in self.df.columns:
                    mask &= ~self.df[feat]

        # Get indices
        indices = self.df[mask].index.tolist()

        print(f"\nControlled probe: {len(indices)} samples after filtering")

        # Get activations for these indices
        act_key = f'resid_{position}'
        X = self.activations[act_key][indices]
        X = X.reshape(X.shape[0], -1)  # Flatten layers

        y = self.df.loc[indices, 'correct'].values.astype(int)

        # Check class balance
        unique, counts = np.unique(y, return_counts=True)
        print(f"Class distribution: {dict(zip(unique, counts))}")

        if min(counts) < 5:
            print("Warning: Too few samples for reliable CV")
            return None

        # Standardize and run CV
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        n_folds = min(5, min(counts))
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')

        print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

        return {
            'n_samples': len(indices),
            'accuracy_mean': scores.mean(),
            'accuracy_std': scores.std()
        }

    def confidence_calibration_analysis(self):
        """
        Analyze confidence calibration for instruct models.
        Compare self-reported confidence to actual correctness.
        """
        if self.metadata['model_config']['variant'] != 'instruct':
            print("\nConfidence calibration only available for instruct models")
            return None

        print("\n" + "="*60)
        print("CONFIDENCE CALIBRATION ANALYSIS")
        print("="*60)

        # Filter to rows with confidence ratings
        has_confidence = self.df['confidence'].notna()
        conf_df = self.df[has_confidence].copy()

        if len(conf_df) == 0:
            print("No confidence ratings found")
            return None

        print(f"\nPrompts with confidence ratings: {len(conf_df)}/{len(self.df)}")

        # Basic stats
        print("\n--- Confidence Distribution ---")
        print(conf_df['confidence'].describe())

        # Confidence vs correctness
        print("\n--- Confidence by Correctness ---")
        correct_conf = conf_df[conf_df['correct']]['confidence'].mean()
        incorrect_conf = conf_df[~conf_df['correct']]['confidence'].mean()
        print(f"Correct answers mean confidence: {correct_conf:.2f}")
        print(f"Incorrect answers mean confidence: {incorrect_conf:.2f}")

        # Confidence by category
        print("\n--- Mean Confidence by Category ---")
        conf_by_cat = conf_df.groupby('category')['confidence'].mean().round(2)
        print(conf_by_cat)

        # Calibration: bin by confidence and check accuracy
        print("\n--- Calibration (binned by confidence) ---")
        conf_df['conf_bin'] = pd.cut(conf_df['confidence'], bins=[0, 3, 5, 7, 9, 10],
                                      labels=['1-3', '4-5', '6-7', '8-9', '10'])
        calibration = conf_df.groupby('conf_bin').agg({
            'correct': ['mean', 'count']
        }).round(3)
        calibration.columns = ['accuracy', 'count']
        print(calibration)

        # Over/under confidence analysis
        print("\n--- Confidence vs Entropy Correlation ---")
        corr = conf_df['confidence'].corr(conf_df['entropy'])
        print(f"Correlation (confidence, entropy): {corr:.3f}")
        print("(Negative means higher confidence = lower entropy, as expected)")

        return {
            'correct_mean_confidence': correct_conf,
            'incorrect_mean_confidence': incorrect_conf,
            'confidence_entropy_corr': corr,
            'calibration': calibration
        }

    def prepare_probe_data(
        self,
        target: str = 'correct',
        position: str = 'last',
        layer: Optional[int] = None,
        categories: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for linear probing.

        Args:
            target: What to predict ('correct', 'category', 'acknowledged_unknown')
            position: Token position ('first', 'middle', 'last')
            layer: Specific layer to use, or None for all layers concatenated
            categories: Subset of categories to include, or None for all

        Returns:
            X: Feature matrix
            y: Labels
        """
        # Get activations for specified position
        act_key = f'resid_{position}'
        X = self.activations[act_key]  # Shape: (n_samples, n_layers, hidden_dim)

        # Select specific layer or flatten all
        if layer is not None:
            X = X[:, layer, :]  # Shape: (n_samples, hidden_dim)
        else:
            # Flatten all layers
            X = X.reshape(X.shape[0], -1)  # Shape: (n_samples, n_layers * hidden_dim)

        # Filter by categories if specified
        if categories is not None:
            mask = self.df['category'].isin(categories).values
            X = X[mask]
            df_filtered = self.df[mask]
        else:
            df_filtered = self.df

        # Get labels
        if target == 'correct':
            y = df_filtered['correct'].values.astype(int)
        elif target == 'category':
            y = pd.Categorical(df_filtered['category']).codes
        elif target == 'acknowledged_unknown':
            y = df_filtered['acknowledged_unknown'].values.astype(int)
        else:
            raise ValueError(f"Unknown target: {target}")

        return X, y

    def run_linear_probe(
        self,
        target: str = 'correct',
        position: str = 'last',
        layer: Optional[int] = None,
        categories: Optional[List[str]] = None,
        n_folds: int = 5
    ) -> Dict:
        """
        Run linear probe with cross-validation.

        Returns:
            Dictionary with results including accuracy, std, and per-fold scores
        """
        X, y = self.prepare_probe_data(target, position, layer, categories)

        # Check for class balance
        unique, counts = np.unique(y, return_counts=True)
        print(f"\nClass distribution: {dict(zip(unique, counts))}")

        # Skip if only one class present
        if len(unique) < 2:
            print(f"Skipping: Only one class present in data")
            return None

        # Skip if too imbalanced
        if min(counts) < n_folds:
            print(f"Warning: Too few samples in minority class for {n_folds}-fold CV")
            n_folds = min(counts)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Run cross-validation
        clf = LogisticRegression(max_iter=1000, random_state=42)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')

        results = {
            'target': target,
            'position': position,
            'layer': layer,
            'categories': categories,
            'n_samples': len(y),
            'n_features': X.shape[1],
            'accuracy_mean': scores.mean(),
            'accuracy_std': scores.std(),
            'fold_scores': scores
        }

        print(f"Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

        return results

    def layer_analysis(
        self,
        target: str = 'correct',
        position: str = 'last',
        categories: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Run probes at each layer to see where information emerges."""
        n_layers = self.metadata['model_config']['n_layers']
        results = []

        print(f"\n--- Layer-wise Probe for {target} ---")
        for layer in range(n_layers):
            res = self.run_linear_probe(target, position, layer, categories, n_folds=5)
            res['layer'] = layer
            results.append(res)
            print(f"Layer {layer:2d}: {res['accuracy_mean']:.3f}")

        return pd.DataFrame(results)

    def compare_positions(
        self,
        target: str = 'correct',
        layer: Optional[int] = None,
        categories: Optional[List[str]] = None
    ) -> Dict:
        """Compare probe accuracy across token positions."""
        results = {}

        print(f"\n--- Position Comparison for {target} ---")
        for position in ['first', 'middle', 'last']:
            res = self.run_linear_probe(target, position, layer, categories)
            results[position] = res
            print(f"{position}: {res['accuracy_mean']:.3f} (+/- {res['accuracy_std']:.3f})")

        return results

    def plot_entropy_distributions(self, save_path: Optional[str] = None):
        """Plot entropy distributions by category and correctness."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # By category
        ax = axes[0]
        categories = self.df['category'].unique()
        for cat in sorted(categories):
            data = self.df[self.df['category'] == cat]['entropy']
            ax.hist(data, bins=30, alpha=0.5, label=cat)
        ax.set_xlabel('Entropy')
        ax.set_ylabel('Count')
        ax.set_title('Entropy Distribution by Category')
        ax.legend()

        # By correctness
        ax = axes[1]
        for correct, label in [(True, 'Correct'), (False, 'Incorrect')]:
            data = self.df[self.df['correct'] == correct]['entropy']
            ax.hist(data, bins=30, alpha=0.5, label=label)
        ax.set_xlabel('Entropy')
        ax.set_ylabel('Count')
        ax.set_title('Entropy Distribution by Correctness')
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        plt.show()

    def plot_layer_analysis(self, layer_results: pd.DataFrame, save_path: Optional[str] = None):
        """Plot layer-wise probe accuracy."""
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(layer_results['layer'], layer_results['accuracy_mean'], 'b-', marker='o')
        ax.fill_between(
            layer_results['layer'],
            layer_results['accuracy_mean'] - layer_results['accuracy_std'],
            layer_results['accuracy_mean'] + layer_results['accuracy_std'],
            alpha=0.2
        )

        ax.set_xlabel('Layer')
        ax.set_ylabel('Probe Accuracy')
        ax.set_title('Correctness Probe Accuracy by Layer')
        ax.axhline(y=0.5, color='r', linestyle='--', label='Chance')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")

        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze epistemic probing data")
    parser.add_argument("--model", required=True, help="Model directory name (e.g., qwen_base)")
    parser.add_argument("--base_dir", default="activations", help="Base directory for activations")
    parser.add_argument("--compare", help="Second model to compare (e.g., qwen_instruct)")
    parser.add_argument("--layer_analysis", action="store_true", help="Run layer-wise analysis")
    parser.add_argument("--save_plots", action="store_true", help="Save plots to files")

    args = parser.parse_args()

    # Load primary model
    model_dir = Path(args.base_dir) / args.model
    print(f"\n{'='*60}")
    print(f"ANALYZING: {args.model}")
    print(f"{'='*60}")

    analyzer = EpistemicAnalyzer(model_dir)
    analyzer.load_data()

    # Basic stats
    analyzer.basic_stats()

    # Entropy analysis
    analyzer.entropy_analysis()

    # Failure mode analysis
    analyzer.failure_mode_analysis()

    # Prompt feature analysis (for controlling confounds)
    analyzer.analyze_prompt_features()

    # Linear probing
    print("\n" + "="*60)
    print("LINEAR PROBING: Can we predict correctness from activations?")
    print("="*60)

    # Overall probe
    print("\n--- All categories, last position, all layers ---")
    analyzer.run_linear_probe(target='correct', position='last', layer=None)

    # Compare positions
    analyzer.compare_positions(target='correct')

    # Probe on different category subsets
    print("\n--- Probe on confident categories only ---")
    analyzer.run_linear_probe(
        target='correct',
        categories=['confident_correct', 'confident_incorrect']
    )

    print("\n--- Probe on uncertain categories only ---")
    analyzer.run_linear_probe(
        target='correct',
        categories=['uncertain_correct', 'uncertain_incorrect']
    )

    # Controlled probes (excluding confounded prompts)
    print("\n" + "="*60)
    print("CONTROLLED PROBING (excluding potential confounds)")
    print("="*60)

    print("\n--- Excluding ambiguous and nonsensical (edge cases) ---")
    analyzer.probe_with_controls(
        exclude_categories=['ambiguous', 'nonsensical']
    )

    print("\n--- Excluding first-person prompts ---")
    analyzer.probe_with_controls(
        exclude_features=['has_first_person']
    )

    print("\n--- Core test: confident + uncertain only, no first-person ---")
    analyzer.probe_with_controls(
        exclude_categories=['ambiguous', 'nonsensical'],
        exclude_features=['has_first_person']
    )

    # Confidence calibration (for instruct models)
    analyzer.confidence_calibration_analysis()

    # Layer analysis if requested
    if args.layer_analysis:
        layer_results = analyzer.layer_analysis(target='correct', position='last')

        if args.save_plots:
            analyzer.plot_layer_analysis(
                layer_results,
                save_path=f"{args.model}_layer_analysis.png"
            )

    # Plot entropy distributions
    if args.save_plots:
        analyzer.plot_entropy_distributions(
            save_path=f"{args.model}_entropy_distributions.png"
        )

    # Compare with second model if provided
    if args.compare:
        print(f"\n{'='*60}")
        print(f"COMPARING WITH: {args.compare}")
        print(f"{'='*60}")

        compare_dir = Path(args.base_dir) / args.compare
        if compare_dir.exists():
            analyzer2 = EpistemicAnalyzer(compare_dir)
            analyzer2.load_data()
            analyzer2.basic_stats()
            analyzer2.entropy_analysis()

            # Compare probe accuracy
            print("\n--- Probe comparison ---")
            print(f"\n{args.model}:")
            res1 = analyzer.run_linear_probe(target='correct', position='last')
            print(f"\n{args.compare}:")
            res2 = analyzer2.run_linear_probe(target='correct', position='last')
        else:
            print(f"Comparison model not found: {compare_dir}")


if __name__ == "__main__":
    main()
