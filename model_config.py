"""
Model configuration for epistemic probing experiments.
Defines model families, paths, and parameters.
"""

# Model family configurations
MODEL_FAMILIES = {
    "qwen": {
        "base": {
            "model_id": "Qwen/Qwen2.5-7B",
            "n_layers": 28,
            "hidden_size": 3584,
            "output_dir": "results_qwen"
        },
        "instruct": {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "n_layers": 28,
            "hidden_size": 3584,
            "output_dir": "results_qwen"
        }
    },
    "llama": {
        "base": {
            "model_id": "meta-llama/Llama-3.1-8B",
            "n_layers": 32,
            "hidden_size": 4096,
            "output_dir": "results_llama"
        },
        "instruct": {
            "model_id": "meta-llama/Llama-3.1-8B-Instruct",
            "n_layers": 32,
            "hidden_size": 4096,
            "output_dir": "results_llama"
        }
    },
    "mistral": {
        "base": {
            "model_id": "mistralai/Mistral-7B-v0.1",
            "n_layers": 32,
            "hidden_size": 4096,
            "output_dir": "results_mistral"
        },
        "instruct": {
            "model_id": "mistralai/Mistral-7B-Instruct-v0.1",
            "n_layers": 32,
            "hidden_size": 4096,
            "output_dir": "results_mistral"
        }
    },
    "yi": {
        "base": {
            "model_id": "01-ai/Yi-6B",
            "n_layers": 32,
            "hidden_size": 4096,
            "output_dir": "results_yi"
        },
        "instruct": {
            "model_id": "01-ai/Yi-6B-Chat",
            "n_layers": 32,
            "hidden_size": 4096,
            "output_dir": "results_yi"
        }
    }
}

# Default experiment parameters
DEFAULTS = {
    "seed": 42,
    "max_new_tokens": 30,  # Minimal tokens - answers are short (e.g., "Paris. Confidence: 8")
    "temperature": 0.0,     # Deterministic generation
    "checkpoint_interval": 50,  # Save every N prompts
    "dtype": "float32",     # float32 for MPS compatibility
}

# Token positions to extract activations from
# These will be computed dynamically based on sequence length
TOKEN_POSITIONS = {
    "first": 0,           # First content token (after any special tokens)
    "middle": "seq_len // 2",  # Middle of sequence
    "last": -1            # Last token before generation
}

# Activation hook names to extract
def get_activation_hooks(n_layers: int) -> list:
    """
    Get list of activation hook names for a model.
    
    Args:
        n_layers: Number of layers in the model
        
    Returns:
        List of hook names to capture
    """
    hooks = [
        "hook_embed",      # Input embeddings
        "ln_final.hook_normalized",  # Final layer norm output
    ]
    
    # Add per-layer hooks
    for layer in range(n_layers):
        hooks.extend([
            f"blocks.{layer}.attn.hook_pattern",  # Attention patterns [batch, head, seq, seq]
            f"blocks.{layer}.attn.hook_z",        # Attention output (pre-projection)
            f"blocks.{layer}.hook_resid_post",    # Residual stream after layer
            f"blocks.{layer}.mlp.hook_post",      # MLP output
        ])
    
    return hooks


def get_model_config(family: str, variant: str) -> dict:
    """
    Get configuration for a specific model.

    Args:
        family: Model family ('qwen', 'llama', 'mistral', 'yi')
        variant: Model variant ('base', 'instruct')
        
    Returns:
        Configuration dictionary
    """
    if family not in MODEL_FAMILIES:
        raise ValueError(f"Unknown model family: {family}. Choose from: {list(MODEL_FAMILIES.keys())}")
    
    if variant not in MODEL_FAMILIES[family]:
        raise ValueError(f"Unknown variant: {variant}. Choose from: {list(MODEL_FAMILIES[family].keys())}")
    
    config = MODEL_FAMILIES[family][variant].copy()
    config['family'] = family
    config['variant'] = variant
    config['hooks'] = get_activation_hooks(config['n_layers'])
    
    return config


def list_available_models() -> list:
    """List all available model configurations."""
    models = []
    for family, variants in MODEL_FAMILIES.items():
        for variant, config in variants.items():
            models.append({
                'family': family,
                'variant': variant,
                'model_id': config['model_id'],
                'n_layers': config['n_layers']
            })
    return models

