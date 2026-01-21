# Fine-Tuning Selectively Steers Epistemic Representations

We show that fine-tuning **selectively steers** policy representations more than factual ones. Across 8 models (4 families × base/instruct), policy categories—where fine-tuning trains epistemic behaviors like admitting ignorance—move 1.08–1.28× further along the alignment direction than factual categories requiring knowledge recall.

## Key Findings

### 1. Selective Steering

Policy categories move further along the alignment direction than factual categories:

| Model | Training | Policy Movement | Factual Movement | Ratio |
|-------|----------|-----------------|------------------|-------|
| Llama 3.1 | SFT + RS + DPO | 37.8 | 29.6 | **1.28×** |
| Yi 1.5 | SFT only | 24.6 | 20.2 | **1.22×** |
| Mistral | SFT only | 36.6 | 33.8 | **1.08×** |
| Qwen 2.5 | SFT + DPO + GRPO | 518.4 | 479.6 | **1.08×** |

All ratios significantly > 1.0 (p < 0.001 via bootstrap). Alignment changes concentrate in a low-rank subspace (14–19 dimensions capture 80% of variance).

### 2. Probe Transfer Confirms Selective Preservation

Training probes on base models and testing on instruct reveals asymmetric preservation:

| Model | Training | Factual Transfer | Policy Transfer | Gap |
|-------|----------|------------------|-----------------|-----|
| Qwen 2.5 | SFT + DPO + GRPO | 0.873 | 0.632 | +0.241 |
| Llama 3.1 | SFT + RS + DPO | 0.849 | 0.624 | +0.225 |
| Yi 1.5 | SFT only | 0.871 | 0.675 | +0.196 |
| Mistral | SFT only | 0.916 | 0.898 | +0.018 |

Preference-optimized models (Qwen, Llama) show clear asymmetry: factual representations transfer well (~0.85) while policy representations are reorganized (~0.63). SFT-only models show smaller gaps.

### 3. Training Method Profiles

Different training methods create distinct representational effects:

| Model | Training | Policy Convergence | Factual Convergence | Pattern |
|-------|----------|--------------------|---------------------|---------|
| Llama | SFT + RS + DPO | -16.3% | -11.2% | Broad compression |
| Qwen | SFT + DPO + GRPO | -3.4% | +3.4% | Selective (policy only) |
| Mistral | SFT only | +16.8% | +66.3% | Categorical divergence |
| Yi | SFT only | +7.6% | +7.0% | Uniform divergence |

- **RS + DPO** (Llama): Compresses entire representation space
- **DPO + GRPO** (Qwen): Selective—policy converges while factual diverges
- **SFT-only** (Mistral, Yi): Divergence, suggesting SFT may drive categorical structure

## Category Definitions

- **Policy categories** (`confident_incorrect`, `ambiguous`, `nonsensical`): Correct response requires trained behavior—admitting "I don't know," asking for clarification, recognizing category errors
- **Factual categories** (`confident_correct`, `uncertain_correct`): Correct response requires knowledge recall

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate the epistemic probing dataset
python gen_data.py

# Collect activations for a model
python collect_activations.py --family llama --variant base

# Run analysis
python run_analysis.py --model llama_base --analysis all
```

## Models Tested

| Family | Base | Instruct | Training Method |
|--------|------|----------|-----------------|
| Llama 3.1 | 8B | 8B-Instruct | SFT + RS + DPO |
| Qwen 2.5 | 7B | 7B-Instruct | SFT + DPO + GRPO |
| Mistral | 7B-v0.1 | 7B-Instruct-v0.1 | SFT only |
| Yi 1.5 | 6B | 6B-Chat | SFT only |

## Project Structure

```
epistemic_status/
├── gen_data.py              # Dataset generation
├── collect_activations.py   # Activation collection
├── run_analysis.py          # Analysis entry point
├── model_config.py          # Model definitions
├── utils.py                 # Evaluation, memory management
├── analysis/                # Analysis modules
│   ├── loader.py           # Data loading
│   ├── probing.py          # Linear probes
│   ├── steering.py         # Steering vector analysis
│   ├── comparison.py       # Cross-model analysis
│   └── statistics.py       # Significance testing
└── activations/            # Collected activation data
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (MPS or CUDA)
- TransformerLens
- scikit-learn, pandas, numpy

## Citation

If you use this work, please cite it.

## License

MIT License
