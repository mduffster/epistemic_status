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

## Analysis Script

```bash
# Basic analysis
python analyze_epistemic.py --model qwen_base

# With layer-wise probing
python analyze_epistemic.py --model qwen_base --layer_analysis

# Compare base vs instruct
python analyze_epistemic.py --model qwen_base --compare qwen_instruct
```

### Key Analysis Features
- **Linear probing**: Predicts correctness from activations (~90% accuracy on Qwen base)
- **Prompt feature controls**: Detects first-person, subjective, temporal markers to control confounds
- **Failure mode analysis**: Categorizes hallucinations (plausible invention, autocomplete confusion, playing along)
- **Confidence calibration**: For instruct models, compares self-reported confidence to actual correctness
- **Layer-wise analysis**: Shows where epistemic information emerges in the network

### Early Findings (Qwen 2.5-7B)
- Base model probe accuracy: 89.6% (model "knows what it knows" internally)
- Entropy signal: correct answers have lower entropy (3.4 vs 4.4)
- Layer progression: 79% at layer 0, peaks at 90% in layers 23-26
- Instruct model has 8x lower entropy than base (more deterministic)
- Instruct partially surfaces latent epistemic knowledge (sometimes refuses fictional entities)
