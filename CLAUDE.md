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

**Full Results Table:**
| Model | Entropy AUC | Probe AUC | Overall Acc | Hall. Det | Hidden Info |
|-------|-------------|-----------|-------------|-----------|-------------|
| Qwen base | 0.764 | 0.946 | 34.5% | 1.0% | 18.3% |
| Qwen instruct | 0.641 | 0.946 | 52.6% | 58.6% | 30.4% |
| Mistral base | **0.923** | **0.970** | 39.6% | 6.1% | 4.7% |
| Mistral instruct | 0.789 | 0.945 | 44.3% | 28.3% | 15.6% |
| Yi base | 0.845 | 0.943 | 35.3% | 1.0% | 9.7% |
| Yi instruct | 0.695 | 0.930 | 40.9% | 19.2% | 23.5% |
| Llama base | **0.935** | 0.959 | 39.6% | 7.1% | **2.4%** |
| Llama instruct | 0.739 | 0.943 | 53.5% | **68.7%** | 20.5% |

**Architecture vs Training Data Analysis:**
| Model | Architecture | Training | Hidden Info (base) |
|-------|--------------|----------|-------------------|
| Llama 3.1 8B | LLaMA | English | **2.4%** |
| Mistral 7B | Custom | English | 4.7% |
| Yi 6B | LLaMA-derived | Chinese | 9.7% |
| Qwen 2.5 7B | Custom | Chinese | 18.3% |

### Key Insights

1. **Training data drives epistemic transparency, not architecture**:
   - Yi (LLaMA arch): 9.7% hidden info
   - Llama (LLaMA arch): 2.4% hidden info
   - Same architecture family, 4x difference - training data is the factor

2. **English-trained models have highly informative entropy** (0.92-0.94 AUC)
3. **Chinese-trained models hide more information** (0.76-0.85 entropy AUC, 10-18% hidden)

4. **Instruct tuning degrades entropy informativeness** across ALL models:
   - Qwen: +12.1% hidden info after instruct
   - Mistral: +10.9% hidden info after instruct
   - Yi: +13.8% hidden info after instruct
   - Llama: +18.1% hidden info after instruct

5. **Probe accuracy remains stable** (~0.93-0.97) regardless of instruct tuning - the information exists internally

6. **Hallucination detection improves with instruct tuning** but varies by model:
   - Llama: 7.1% → 68.7% (best)
   - Qwen: 1% → 58.6%
   - Mistral: 6.1% → 28.3%
   - Yi: 1% → 19.2%

### Alignment Implications

- **Entropy-based uncertainty estimation is model-dependent** - systems using logprobs work better with English-trained models (Llama, Mistral)
- **RLHF degrades output transparency** - entropy becomes less informative after alignment across all models tested
- **Internal epistemic state is always recoverable** - linear probes achieve ~95% AUC across all models
- **Current RLHF doesn't prioritize entropy calibration** - the information exists internally but isn't surfaced
- **Training data/RLHF origin may matter more than architecture** for uncertainty estimation strategies
