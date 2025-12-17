"""
Data loading utilities for epistemic probing analysis.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import evaluate_response


@dataclass
class ModelData:
    """Container for loaded model data."""
    name: str
    df: pd.DataFrame
    activations: Dict[str, np.ndarray]
    metadata: dict

    @property
    def n_samples(self) -> int:
        return len(self.df)

    @property
    def n_layers(self) -> int:
        return self.metadata['model_config']['n_layers']

    @property
    def hidden_size(self) -> int:
        return self.metadata['model_config']['hidden_size']

    @property
    def model_id(self) -> str:
        return self.metadata['model_config']['model_id']

    @property
    def variant(self) -> str:
        return self.metadata['model_config']['variant']

    @property
    def is_instruct(self) -> bool:
        return self.variant == 'instruct'


def load_model_data(
    model_name: str,
    base_dir: str = "activations",
    re_evaluate: bool = True
) -> ModelData:
    """
    Load model data from disk.

    Args:
        model_name: Name of model directory (e.g., 'qwen_base')
        base_dir: Base directory for activations
        re_evaluate: Whether to re-evaluate responses with current logic

    Returns:
        ModelData object with df, activations, and metadata
    """
    model_dir = Path(base_dir) / model_name

    # Load metadata
    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    print(f"Loaded metadata for {metadata['model_config']['model_id']}")
    print(f"  Total prompts: {metadata['collection_info']['total_prompts']}")
    print(f"  Batches: {metadata['collection_info']['num_batches']}")

    # Convert to DataFrame
    df = pd.DataFrame(metadata['prompts'])

    # Re-evaluate responses with current evaluation logic
    if re_evaluate:
        print("  Re-evaluating responses with updated logic...")
        df['evaluation'] = df.apply(
            lambda row: evaluate_response(row['response'], row['correct_answer'], row['category']),
            axis=1
        )
        df['exact_match'] = df['evaluation'].apply(lambda x: x['exact_match'])
        df['acknowledged_unknown'] = df['evaluation'].apply(lambda x: x['acknowledged_unknown'])

        # Determine correctness based on category
        # For confident_incorrect (fictional entities): correct if model acknowledges it's fictional
        # For other categories: correct if response contains the answer
        def determine_correct(row):
            if row['category'] == 'confident_incorrect':
                return row['acknowledged_unknown']
            else:
                return row['evaluation']['contains_answer']

        df['correct'] = df.apply(determine_correct, axis=1)

    # Load activations
    activations = _load_activations(model_dir)

    return ModelData(
        name=model_name,
        df=df,
        activations=activations,
        metadata=metadata
    )


def _load_activations(model_dir: Path) -> Dict[str, np.ndarray]:
    """Load activation arrays from npz files."""
    batch_files = sorted(model_dir.glob("batch_*.npz"))

    act_types = ['resid_first', 'resid_middle', 'resid_last',
                 'mlp_first', 'mlp_middle', 'mlp_last']

    activations = {act_type: [] for act_type in act_types}
    activations['entropy'] = []
    activations['prompt_indices'] = []

    for batch_file in batch_files:
        data = np.load(batch_file, allow_pickle=True)

        for act_type in act_types:
            if act_type in data:
                activations[act_type].append(data[act_type])

        if 'entropy' in data:
            activations['entropy'].append(data['entropy'])
        if 'prompt_indices' in data:
            activations['prompt_indices'].append(data['prompt_indices'])

    # Concatenate all batches
    for act_type in act_types:
        if activations[act_type]:
            activations[act_type] = np.concatenate(activations[act_type], axis=0)
            print(f"  {act_type}: {activations[act_type].shape}")

    if activations['entropy']:
        activations['entropy'] = np.concatenate(activations['entropy'])
    if activations['prompt_indices']:
        activations['prompt_indices'] = np.concatenate(activations['prompt_indices'])

    return activations


def get_activation_matrix(
    data: ModelData,
    position: str = 'last',
    layer: Optional[int] = None,
    act_type: str = 'resid'
) -> np.ndarray:
    """
    Get activation matrix for probing.

    Args:
        data: ModelData object
        position: Token position ('first', 'middle', 'last')
        layer: Specific layer or None for all layers flattened
        act_type: 'resid' or 'mlp'

    Returns:
        Activation matrix (n_samples, n_features)
    """
    act_key = f'{act_type}_{position}'
    X = data.activations[act_key]

    if layer is not None:
        X = X[:, layer, :]
    else:
        X = X.reshape(X.shape[0], -1)

    return X
