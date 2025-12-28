# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Fine-Tuning Entangles Epistemic Representations in Language Models**

This project shows that fine-tuning degrades the separability of epistemic states in language model activations. By probing hidden states across 8 models (4 families × base/instruct), we find that alignment training entangles **trained epistemic behaviors** (admitting ignorance, acknowledging ambiguity) with **genuine uncertainty**, making these internal states harder to distinguish despite improved behavioral performance. Critically, **RLHF/DPO roughly doubles the entanglement effect compared to SFT alone**.

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

# Entanglement analysis (probe confidence, bootstrap CIs, held-out generalization)
python run_analysis.py --model qwen_base --analysis entanglement

# Cross-model comparison
python run_analysis.py --model qwen_base --compare qwen_instruct --analysis generalization

# Entanglement comparison (base vs instruct)
python run_analysis.py --model qwen_base --compare qwen_instruct --analysis entanglement

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
- `entanglement.py` - Fine-tuning entanglement analysis (probe confidence, bootstrap CIs, held-out generalization)
- `plotting.py` - Visualization functions

### Key Analysis Features
- **Linear probing**: Predicts correctness from activations (~90% accuracy)
- **Effect sizes**: Cohen's d for activation differences between correct/incorrect
- **ROC/AUC**: Compares entropy-only vs probe-based prediction
- **Cross-model generalization**: Tests if probes transfer between base/instruct
- **Confidence calibration**: Compares self-reported confidence to actual correctness
- **Layer-wise analysis**: Shows where epistemic information emerges

### Evaluation Logic
- **confident_incorrect**: Model is "correct" if it acknowledges the fictional entity doesn't exist
- **nonsensical**: Model is "correct" if it recognizes the category error (e.g., "jealousy has no color")
- **ambiguous**: Model is "correct" if it asks for clarification or acknowledges multiple meanings
- **uncertain_incorrect**: Model is "correct" if it debunks the misconception
- **Other categories**: Model is "correct" if response contains the expected answer

### Key Findings: Cross-Model Comparison

**Full Results Table:**
| Model | Entropy AUC | Probe AUC | Overall Acc | Hall. Det | Hidden Info |
|-------|-------------|-----------|-------------|-----------|-------------|
| Qwen base | 0.788 | 0.935 | 41.8% | 1.0% | 14.6% |
| Qwen instruct | 0.553 | 0.760 | 75.9% | 58.6% | 20.7% |
| Mistral base | **0.930** | 0.946 | 41.4% | 6.1% | **1.6%** |
| Mistral instruct | 0.741 | 0.823 | 59.8% | 28.3% | 8.2% |
| Yi base | 0.825 | 0.956 | 39.6% | 1.0% | 13.1% |
| Yi instruct | 0.649 | 0.873 | 52.6% | 19.2% | 22.4% |
| Llama base | 0.914 | 0.943 | 45.3% | 7.1% | 3.0% |
| Llama instruct | 0.734 | 0.839 | **70.6%** | **68.7%** | 10.5% |

**Models and Training Methods:**
| Model | Architecture | Training Data | Instruct Training | Hidden Info (base) |
|-------|--------------|---------------|-------------------|-------------------|
| Llama 3.1 8B | LLaMA | English | SFT + RLHF + DPO | 3.0% |
| Mistral 7B | Custom | English | SFT only | **1.6%** |
| Yi 6B | LLaMA-derived | Chinese | SFT only | 13.1% |
| Qwen 2.5 7B | Custom | Chinese | SFT + DPO + GRPO | 14.6% |

This natural experiment allows comparing SFT-only vs RLHF/DPO effects.

### Key Insights

1. **Training data drives epistemic transparency, not architecture**:
   - Yi (LLaMA arch): 13.1% hidden info
   - Llama (LLaMA arch): 3.0% hidden info
   - Same architecture family, 4x difference - training data is the factor

2. **English-trained models have highly informative entropy** (0.91-0.93 AUC)
3. **Chinese-trained models hide more information** (0.79-0.83 entropy AUC, 13-15% hidden)

4. **Instruct tuning degrades entropy informativeness** across ALL models:
   - Qwen: 0.788 → 0.553 entropy AUC (+6.1% hidden info)
   - Mistral: 0.930 → 0.741 entropy AUC (+6.6% hidden info)
   - Yi: 0.825 → 0.649 entropy AUC (+9.3% hidden info)
   - Llama: 0.914 → 0.734 entropy AUC (+7.5% hidden info)

5. **Probe AUC drops for instruct models** - the corrected evaluation labels make correctness more nuanced

6. **Hallucination detection improves with instruct tuning** but varies by model:
   - Llama: 7.1% → 68.7% (best)
   - Qwen: 1% → 58.6%
   - Mistral: 6.1% → 28.3%
   - Yi: 1% → 19.2%

### Fine-Tuning Creates Representational Entanglement

We find that fine-tuning doesn't just change model outputs - it **entangles** internal representations for policy categories (where fine-tuning trains epistemic behaviors):

| Model | Training Method | Policy Δ | Factual Δ | Gap |
|-------|-----------------|----------|-----------|-----|
| Qwen | SFT + DPO + GRPO | +0.318 | -0.068 | **0.386** |
| Llama | SFT + RLHF + DPO | +0.286 | -0.071 | **0.357** |
| Mistral | SFT only | +0.247 | +0.092 | 0.155 |
| Yi | SFT only | +0.220 | +0.095 | 0.125 |

*Δ = change in probe error rate after instruct tuning. Policy categories: confident_incorrect, ambiguous, nonsensical (trained behaviors). Factual categories: confident_correct, uncertain_correct (knowledge recall).*

**Key finding**: RLHF/DPO roughly doubles the entanglement effect:
- **SFT-only models** (Mistral, Yi): Policy Δ ~+0.23, gap ~0.14
- **RLHF/DPO models** (Llama, Qwen): Policy Δ ~+0.30, gap ~0.37

**Probe transfer confirms the mechanism.** Training a probe on base and testing on instruct:

| Model | Training | Factual Transfer | Policy Transfer | Gap |
|-------|----------|------------------|-----------------|-----|
| Qwen | SFT + DPO + GRPO | 0.821 | 0.382 | **+0.44** |
| Llama | SFT + RLHF + DPO | 0.879 | 0.591 | **+0.29** |
| Mistral | SFT only | 0.252 | 0.650 | -0.40 |
| Yi | SFT only | 0.668 | 0.429 | +0.24 |

- **RLHF/DPO models**: Selective preservation - factual representations transfer well (~85%), policy representations warped (~49%)
- **SFT-only models**: Unpredictable restructuring - Mistral shows representational *inversion* (1-acc = 0.75), Yi shows uniform degradation

### Entanglement Analysis Functions

**`analysis/entanglement.py`:**
- `probe_confidence_by_category()` - Probe confidence (max probability) per category
- `probe_confidence_layerwise()` - Layer-wise probe confidence analysis
- `bootstrap_confidence_intervals()` - Statistical rigor for error rate comparisons
- `held_out_category_generalization()` - Train excluding one category, test on it
- `activation_similarity_by_category()` - Cosine similarity between category centroids
- `compare_activation_similarity()` - Track similarity changes after fine-tuning
- `compare_base_instruct_entanglement()` - Full base vs instruct comparison

**`analysis/comparison.py`:**
- `transfer_by_category()` - Probe transfer from base→instruct broken down by category

### Alignment Implications

- **Entropy-based uncertainty estimation is model-dependent** - systems using logprobs work better with English-trained models (Llama, Mistral)
- **Fine-tuning degrades output transparency** - entropy becomes less informative after alignment across all models tested
- **RLHF/DPO amplifies entanglement** - preference optimization roughly doubles the effect compared to SFT alone
- **Fine-tuning creates representational entanglement** - trained epistemic behaviors (admitting ignorance, acknowledging ambiguity) become entangled with genuine uncertainty
- **Internal epistemic state is recoverable** - linear probes achieve 0.76-0.96 AUC across models
- **Current fine-tuning doesn't prioritize entropy calibration** - the information exists internally but isn't surfaced
- **Training data origin may matter more than architecture** for uncertainty estimation strategies
