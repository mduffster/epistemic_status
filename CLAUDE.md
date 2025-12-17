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
- `model_config.py` - Model definitions (Qwen/Llama/Mistral, base/instruct variants), layer counts, hidden sizes
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

### Key Findings (Qwen 2.5-7B)
- **Probe accuracy**: Base 89.6%, Instruct 90.0%
- **Entropy signal**: Correct answers have lower entropy (base: 3.4 vs 4.4, instruct: 0.45 vs 0.79)
- **Probe vs entropy**: Probe adds ~20 percentage points over entropy-only prediction
- **Effect size**: Large effect (d=0.98) for entropy, max layer effect d=1.29 at layer 14
- **Cross-model transfer**: Base→Instruct transfers well (89.5%), Instruct→Base fails (34%)
- **Instruct calibration**: Poorly calibrated (reports 8-9 confidence but only 12% accurate)
