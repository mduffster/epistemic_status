"""
Core statistics and analysis functions.
"""

from collections import defaultdict
from typing import Dict, List

import pandas as pd

from .loader import ModelData


def basic_stats(data: ModelData, print_output: bool = True) -> pd.DataFrame:
    """
    Compute basic statistics about responses.

    Returns:
        DataFrame with category-level statistics
    """
    if print_output:
        print("\n" + "=" * 60)
        print("BASIC STATISTICS")
        print("=" * 60)

    # Category breakdown
    category_stats = data.df.groupby('category').agg({
        'correct': ['sum', 'count', 'mean'],
        'entropy': 'mean',
        'acknowledged_unknown': 'sum'
    }).round(3)
    category_stats.columns = ['correct', 'total', 'accuracy', 'mean_entropy', 'acknowledged_unknown']

    if print_output:
        print("\n--- Category Breakdown ---")
        print(category_stats)
        print(f"\nOverall accuracy: {data.df['correct'].mean():.3f}")
        print(f"Mean entropy: {data.df['entropy'].mean():.3f}")

    return category_stats


def failure_mode_analysis(data: ModelData, print_output: bool = True) -> Dict[str, List]:
    """
    Analyze failure modes in confident_incorrect category.

    Only considers prompts where the model actually failed (got it wrong).
    Some models correctly identify fictional entities, so we don't count those.

    Returns:
        Dictionary mapping failure mode to list of examples
    """
    if print_output:
        print("\n" + "=" * 60)
        print("FAILURE MODE ANALYSIS")
        print("=" * 60)

    # Focus on confident_incorrect (hallucinations on fictional entities)
    # IMPORTANT: Only consider prompts where the model ACTUALLY failed
    all_hallucination_prompts = data.df[data.df['category'] == 'confident_incorrect']
    hallucinations = all_hallucination_prompts[~all_hallucination_prompts['correct']]

    if print_output:
        print(f"\nTotal prompts in confident_incorrect category: {len(all_hallucination_prompts)}")
        print(f"Model got correct (acknowledged fictional): {all_hallucination_prompts['correct'].sum()}")
        print(f"Model failed (hallucinated): {len(hallucinations)}")
        print(f"  - Of failures, acknowledged_unknown flag: {hallucinations['acknowledged_unknown'].sum()}")

    # Categorize failure modes
    failure_modes = defaultdict(list)

    for _, row in hallucinations.iterrows():
        response = row['response'].lower()
        prompt = row['prompt'].lower()

        if row['acknowledged_unknown']:
            failure_modes['acknowledged'].append(row)
        elif 'fictional' in response or "doesn't exist" in response or 'not real' in response:
            failure_modes['partial_acknowledgment'].append(row)
        elif any(x in prompt for x in ['president', 'king', 'emperor']):
            failure_modes['autocomplete_confusion'].append(row)
        elif any(x in prompt for x in ['capital of', 'city of']):
            failure_modes['plausible_invention'].append(row)
        else:
            failure_modes['playing_along'].append(row)

    if print_output:
        print("\n--- Failure Mode Breakdown ---")
        for mode, items in failure_modes.items():
            print(f"{mode}: {len(items)}")

        print("\n--- Examples by Failure Mode ---")
        for mode, items in failure_modes.items():
            if items:
                print(f"\n{mode.upper()}:")
                for item in items[:3]:
                    print(f"  Q: {item['prompt']}")
                    print(f"  A: {item['response'][:100]}...")
                    print(f"  Entropy: {item['entropy']:.2f}")

    return dict(failure_modes)


def analyze_prompt_features(data: ModelData, print_output: bool = True) -> pd.DataFrame:
    """
    Analyze prompt surface features that might confound probing.

    Returns:
        DataFrame with feature distribution by category
    """
    if print_output:
        print("\n" + "=" * 60)
        print("PROMPT FEATURE ANALYSIS")
        print("=" * 60)

    df = data.df.copy()

    # Define feature detectors
    def has_first_person(text):
        words = text.lower().split()
        return any(w in ['i', 'me', 'my', "i'm", "i've"] for w in words)

    def has_second_person(text):
        words = text.lower().split()
        return any(w in ['you', 'your', "you're", "you've"] for w in words)

    def is_subjective(text):
        markers = ['best', 'good', 'bad', 'should', 'better', 'worth', 'favorite']
        return any(m in text.lower() for m in markers)

    def has_temporal_deixis(text):
        markers = ['now', 'today', 'yesterday', 'tomorrow', 'current', 'recent']
        return any(m in text.lower() for m in markers)

    # Apply feature detectors
    df['has_first_person'] = df['prompt'].apply(has_first_person)
    df['has_second_person'] = df['prompt'].apply(has_second_person)
    df['is_subjective'] = df['prompt'].apply(is_subjective)
    df['has_temporal'] = df['prompt'].apply(has_temporal_deixis)

    # Update the main dataframe
    data.df['has_first_person'] = df['has_first_person']
    data.df['has_second_person'] = df['has_second_person']
    data.df['is_subjective'] = df['is_subjective']
    data.df['has_temporal'] = df['has_temporal']

    features = ['has_first_person', 'has_second_person', 'is_subjective', 'has_temporal']
    feature_by_cat = df.groupby('category')[features].mean().round(3)

    if print_output:
        print("\n--- Feature Distribution by Category ---")
        print(feature_by_cat)

        print("\n--- Features vs Correctness ---")
        for feat in features:
            with_feat = df[df[feat]]['correct'].mean()
            without_feat = df[~df[feat]]['correct'].mean()
            n_with = df[feat].sum()
            print(f"{feat}:")
            print(f"  With feature ({n_with}): {with_feat:.3f} accuracy")
            print(f"  Without feature ({len(df) - n_with}): {without_feat:.3f} accuracy")

    return feature_by_cat
