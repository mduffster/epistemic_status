"""
Confidence calibration analysis for instruct models.
"""

from typing import Dict, Optional

import pandas as pd

from .loader import ModelData


def confidence_calibration(data: ModelData, print_output: bool = True) -> Optional[Dict]:
    """
    Analyze confidence calibration for instruct models.

    Returns:
        Dictionary with calibration statistics or None if not instruct
    """
    if not data.is_instruct:
        if print_output:
            print("\nConfidence calibration only available for instruct models")
        return None

    if print_output:
        print("\n" + "=" * 60)
        print("CONFIDENCE CALIBRATION ANALYSIS")
        print("=" * 60)

    df = data.df

    # Filter to rows with confidence ratings
    has_confidence = df['confidence'].notna()
    conf_df = df[has_confidence].copy()

    if len(conf_df) == 0:
        if print_output:
            print("No confidence ratings found")
        return None

    if print_output:
        print(f"\nPrompts with confidence ratings: {len(conf_df)}/{len(df)}")

    results = {}

    # Basic stats
    results['distribution'] = {
        'count': len(conf_df),
        'mean': conf_df['confidence'].mean(),
        'std': conf_df['confidence'].std(),
        'median': conf_df['confidence'].median()
    }

    if print_output:
        print("\n--- Confidence Distribution ---")
        print(conf_df['confidence'].describe())

    # Confidence vs correctness
    correct_conf = conf_df[conf_df['correct']]['confidence'].mean()
    incorrect_conf = conf_df[~conf_df['correct']]['confidence'].mean()

    results['by_correctness'] = {
        'correct_mean': correct_conf,
        'incorrect_mean': incorrect_conf,
        'difference': correct_conf - incorrect_conf
    }

    if print_output:
        print("\n--- Confidence by Correctness ---")
        print(f"Correct answers mean confidence: {correct_conf:.2f}")
        print(f"Incorrect answers mean confidence: {incorrect_conf:.2f}")

    # Confidence by category
    conf_by_cat = conf_df.groupby('category')['confidence'].mean().round(2)
    results['by_category'] = conf_by_cat.to_dict()

    if print_output:
        print("\n--- Mean Confidence by Category ---")
        print(conf_by_cat)

    # Calibration: bin by confidence and check accuracy
    conf_df['conf_bin'] = pd.cut(
        conf_df['confidence'],
        bins=[0, 3, 5, 7, 9, 10],
        labels=['1-3', '4-5', '6-7', '8-9', '10']
    )
    calibration = conf_df.groupby('conf_bin', observed=False).agg({
        'correct': ['mean', 'count']
    }).round(3)
    calibration.columns = ['accuracy', 'count']

    results['calibration_curve'] = {
        bin_name: {'accuracy': row['accuracy'], 'count': row['count']}
        for bin_name, row in calibration.iterrows()
    }

    if print_output:
        print("\n--- Calibration (binned by confidence) ---")
        print(calibration)

    # Correlation analyses
    conf_entropy_corr = conf_df['confidence'].corr(conf_df['entropy'])
    conf_correct_corr = conf_df['confidence'].corr(conf_df['correct'].astype(float))

    results['correlations'] = {
        'confidence_entropy': conf_entropy_corr,
        'confidence_correct': conf_correct_corr
    }

    if print_output:
        print("\n--- Confidence vs Entropy Correlation ---")
        print(f"Correlation (confidence, entropy): {conf_entropy_corr:.3f}")
        print("(Negative means higher confidence = lower entropy, as expected)")

    # Expected calibration error (ECE)
    ece = 0
    total_samples = 0
    for bin_name, bin_data in calibration.iterrows():
        if bin_data['count'] > 0 and not pd.isna(bin_data['accuracy']):
            # Get expected accuracy from bin midpoint
            bin_ranges = {'1-3': 0.2, '4-5': 0.45, '6-7': 0.65, '8-9': 0.85, '10': 1.0}
            expected = bin_ranges.get(bin_name, 0.5)
            actual = bin_data['accuracy']
            ece += bin_data['count'] * abs(actual - expected)
            total_samples += bin_data['count']

    if total_samples > 0:
        ece /= total_samples
        results['ece'] = ece

        if print_output:
            print(f"\nExpected Calibration Error (ECE): {ece:.3f}")
            print("(Lower is better, 0 = perfectly calibrated)")

    return results
