# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project investigates **epistemic states in language models** by probing internal activations. It collects model responses and hidden state activations across different epistemic categories (confident/uncertain, correct/incorrect) to analyze how models internally represent knowledge certainty.

## Key Commands

### Generate Dataset
```bash
python gen_data.py
```
Creates `epistemic_probing_dataset.csv` with ~600 prompts across 6 epistemic categories.

### Collect Activations
```bash
# Run with specific model family and variant
python collect_activations.py --family qwen --variant instruct
python collect_activations.py --family llama --variant base --start_idx 100

# Custom batch size
python collect_activations.py --family mistral --variant instruct --batch_size 25
```

Outputs to `activations/{family}_{variant}/` with:
- `batch_XX.npz` - Activation arrays (resid/mlp at first/middle/last token positions)
- `metadata.json` - Prompts, responses, confidence ratings, entropy, evaluation results

### Install Dependencies
```bash
pip install -r requirements.txt
```
Tested on macOS ARM64 with MPS. For CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

## Architecture

**Data Pipeline:**
1. `gen_data.py` - Generates prompts in 6 epistemic categories
2. `collect_activations.py` - Runs prompts through models, extracts activations
3. Output: NPZ files with activation tensors + JSON metadata

**Core Modules:**
- `model_config.py` - Model definitions (Qwen/Llama/Mistral/Yi, base/instruct variants), layer counts, hidden sizes
- `utils.py` - Memory management (MPS/CUDA cleanup), response evaluation, confidence extraction
- `collect_activations.py` - `EpistemicDataCollector` class handles batched collection with checkpointing

**Epistemic Categories (in dataset):**
- `confident_correct` - Clear factual questions (e.g., "What is 2+2?")
- `confident_incorrect` - Fictional entities models hallucinate on
- `uncertain_correct` - Obscure facts with lower confidence
- `uncertain_incorrect` - Misconceptions and contested claims
- `ambiguous` - Context-dependent questions
- `nonsensical` - Category error questions (e.g., "What color is jealousy?")

## Key Implementation Details

- Uses TransformerLens `HookedTransformer` for activation extraction
- Extracts residual stream and MLP outputs at first/middle/last token positions
- Instruct models receive confidence-eliciting prompts; base models get simple Q&A format
- Checkpointing supports resuming interrupted runs
- Memory-aware: monitors available RAM, triggers cleanup when low

## Analysis

### Run Analysis
```bash
# Core analysis (stats, probing, entropy)
python run_analysis.py --model qwen_base --analysis core

# Specific analyses
python run_analysis.py --model qwen_base --analysis effects roc
python run_analysis.py --model qwen_base --analysis layers

# Cross-model comparison
python run_analysis.py --model qwen_base --compare qwen_instruct --analysis generalization

# Full suite with plots
python run_analysis.py --model qwen_base --analysis all --save_plots
```

### Analysis Package Structure (`analysis/`)
- `loader.py` - Data loading, `ModelData` class
- `core.py` - Basic stats, failure mode analysis, prompt features
- `probing.py` - Linear probes, layer-wise analysis, controlled probes
- `entropy.py` - Entropy analysis, entropy vs probe comparison
- `calibration.py` - Confidence calibration (instruct models)
- `effects.py` - Effect sizes (Cohen's d), ROC/AUC curves
- `comparison.py` - Cross-model generalization, bidirectional transfer
- `plotting.py` - Visualization functions

### Key Analysis Features
- **Linear probing**: Predicts correctness from activations (~90% accuracy)
- **Effect sizes**: Cohen's d for activation differences between correct/incorrect
- **ROC/AUC**: Compares entropy-only vs probe-based prediction
- **Cross-model generalization**: Tests if probes transfer between base/instruct
- **Confidence calibration**: Compares self-reported confidence to actual correctness
- **Layer-wise analysis**: Shows where epistemic information emerges

### Evaluation Logic
- **confident_incorrect category**: Model is "correct" if it acknowledges the fictional entity doesn't exist
- **Other categories**: Model is "correct" if response contains the expected answer
- Failure mode analysis only shows prompts where the model actually failed

### Key Findings: Cross-Model Comparison

**Hallucination Detection (acknowledging fictional entities):**
| Model | Factual Acc | Hallucination Detection | Mean Entropy |
|-------|-------------|------------------------|--------------|
| Qwen base | 80.2% | 1.0% | 4.057 |
| Qwen instruct | 92.6% | **58.6%** | 0.646 |
| Mistral base | 89.7% | 6.1% | 2.803 |
| Mistral instruct | 89.7% | 28.3% | 1.891 |
| Yi base | 81.9% | 1.0% | 4.039 |

**Hidden Information (Probe AUC - Entropy AUC):**
| Model | Entropy AUC | Probe AUC | Hidden Info |
|-------|-------------|-----------|-------------|
| Qwen base | 0.764 | 0.942 | 17.9% |
| Qwen instruct | 0.641 | 0.931 | **29.0%** |
| Mistral base | 0.923 | 0.956 | 3.2% |
| Mistral instruct | 0.789 | 0.933 | 14.4% |
| Yi base | 0.845 | 0.926 | 8.1% |

### Key Insights

1. **Qwen hides more epistemic information** than Mistral (18-29% vs 3-14%)
2. **Instruct tuning increases hidden info** for both models, but more so for Qwen
3. **Qwen is 2x better at hallucination detection** (58.6% vs 28.3%) after instruct tuning
4. **Yi base tracks closer to Mistral** (8.1% hidden) than Qwen (17.9%), suggesting architecture may matter
5. **Two uncertainty strategies emerged**:
   - Mistral: Uncertainty leaks into entropy (implicit signal)
   - Qwen: Uncertainty expressed verbally but entropy stays confident (explicit signal)

### Alignment Implications

- **Entropy-based uncertainty estimation is model-dependent** - systems using logprobs work better with Mistral
- **RLHF can degrade output transparency** - Qwen's entropy became less informative after alignment
- **Internal epistemic state is recoverable** - linear probes achieve ~93% AUC across all models
- **Current RLHF doesn't prioritize entropy calibration** - the information exists internally but isn't surfaced
