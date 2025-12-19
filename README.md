# RLHF Entangles Epistemic Representations in Language Models

We show that RLHF degrades the separability of epistemic states in language model activations. By probing hidden states across 8 models (4 families × base/instruct), we find that alignment training entangles **trained epistemic behaviors** (admitting ignorance, acknowledging ambiguity) with **genuine uncertainty**, making these internal states harder to distinguish despite improved behavioral performance.

**Context**: Prior work established that language models represent epistemic states internally ([Kadavath et al. 2022](https://arxiv.org/abs/2207.05221), [Azaria & Mitchell 2023](https://arxiv.org/abs/2304.13734)). We extend this by showing *how RLHF alters these representations* - specifically, that alignment creates targeted entanglement where it trains epistemic policy behaviors.

## Key Findings

### 1. Models Hide Epistemic Information

Linear probes on activations predict correctness better than output entropy alone. The gap reveals "hidden information" - uncertainty the model represents internally but doesn't surface:

| Model | Entropy AUC | Probe AUC | Hidden Info |
|-------|-------------|-----------|-------------|
| Mistral 7B base | **0.930** | 0.946 | **1.6%** |
| Llama 3.1 8B base | 0.914 | 0.943 | 3.0% |
| Yi 6B base | 0.825 | 0.956 | 13.1% |
| Qwen 2.5 7B base | 0.788 | 0.935 | 14.6% |

### 2. Training Origin Determines Transparency (Not Architecture)

Yi and Llama share the same architecture but differ 4x in hidden information:

| Model | Architecture | Training Data | Hidden Info |
|-------|--------------|---------------|-------------|
| Llama 3.1 8B | LLaMA | English | 3.0% |
| Yi 6B | LLaMA-derived | Chinese | 13.1% |

English-trained models (Llama, Mistral) have highly informative entropy. Chinese-trained models (Qwen, Yi) hide more - the model "knows" it's uncertain but doesn't signal it through logprobs.

### 3. RLHF Degrades Epistemic Transparency

Instruct tuning makes entropy *less* informative across all models:

| Model | Entropy (base) | Entropy (instruct) | Hidden (base) | Hidden (instruct) |
|-------|----------------|--------------------| --------------|-------------------|
| Llama | 0.914 | 0.734 | 3.0% | 10.5% |
| Mistral | 0.930 | 0.741 | 1.6% | 8.2% |
| Yi | 0.825 | 0.649 | 13.1% | 22.4% |
| Qwen | 0.788 | 0.553 | 14.6% | 20.7% |

The epistemic information exists internally but is increasingly hidden from the output distribution.

### 4. Entanglement Occurs Where RLHF Trains Policy Behaviors

The critical finding: representational degradation is *selective*. Probe error rates increase specifically for **policy categories** - where RLHF trains epistemic behaviors - while **factual categories** remain stable or improve:

| Model | Policy Δ | Factual Δ | Selective Gap |
|-------|----------|-----------|---------------|
| Qwen | +0.318 | -0.068 | **0.386** |
| Llama | +0.286 | -0.071 | **0.357** |
| Mistral | +0.247 | +0.092 | **0.155** |
| Yi | +0.220 | +0.095 | **0.125** |

*Δ = change in probe error rate after instruct tuning.*

**Why "policy" vs "factual"?**
- **Policy categories** (`confident_incorrect`, `ambiguous`, `nonsensical`): Correct response requires trained behavior - admitting "I don't know," asking for clarification, recognizing category errors. RLHF explicitly teaches these.
- **Factual categories** (`confident_correct`, `uncertain_correct`): Correct response requires recalling knowledge. RLHF doesn't specifically target these.

This suggests RLHF warps representational geometry specifically where it trains epistemic behaviors. Activation similarity analysis confirms the mechanism: `confident_incorrect` representations shift toward `uncertain_correct` after RLHF. The model learns to say "I don't know" by pushing those representations toward genuine uncertainty - entangling two epistemically distinct states.

### 5. The RLHF Paradox: Better Behavior, Worse Transparency

Despite internal entanglement, behavioral hallucination detection improves dramatically:

| Model | Base | Instruct |
|-------|------|----------|
| Llama | 7.1% | **68.7%** |
| Qwen | 1.0% | 58.6% |
| Mistral | 6.1% | 28.3% |
| Yi | 1.0% | 19.2% |

RLHF teaches models to *behave* as if they know what they don't know, while making internal representations *harder to interpret*.

## Implications for Alignment & Interpretability

1. **RLHF trades interpretability for behavior** - alignment achieves epistemic caution by warping internal representations, not by building distinct "I should acknowledge uncertainty" circuits
2. **Entanglement is targeted** - degradation occurs specifically where RLHF trains policy behaviors, suggesting interpretability researchers should focus on alignment-modified regions
3. **Entropy-based uncertainty is unreliable** - logprob-based uncertainty estimation works for some models but fails for others; internal probing may be necessary for robust uncertainty quantification
4. **Internal state remains recoverable** - linear probes achieve 0.76-0.96 AUC even after RLHF, suggesting interpretability tools could surface the hidden epistemic information that alignment obscures
5. **Calibration is not an alignment objective** - current RLHF prioritizes behavioral compliance over transparent uncertainty signaling

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

## Dataset

The dataset contains ~600 prompts across 6 epistemic categories, divided into **factual** (correct response = recall knowledge) and **policy** (correct response = trained epistemic behavior):

| Category | Type | Description | Correct Response |
|----------|------|-------------|------------------|
| `confident_correct` | Factual | Clear factual questions | Recall answer |
| `uncertain_correct` | Factual | Obscure but verifiable facts | Recall answer |
| `uncertain_incorrect` | Factual | Common misconceptions | Debunk myth |
| `confident_incorrect` | Policy | Fictional entities | Admit "I don't know" |
| `ambiguous` | Policy | Context-dependent questions | Acknowledge ambiguity |
| `nonsensical` | Policy | Category error questions | Recognize nonsense |

## Methodology

### Activation Collection
- Uses [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) to extract hidden states
- Captures residual stream and MLP outputs at first/middle/last token positions
- Stores response text, confidence ratings (instruct models), and token entropy

### Analysis
- **Linear probing**: Logistic regression on activations to predict correctness
- **ROC/AUC comparison**: Entropy-only vs probe-based prediction
- **Effect sizes**: Cohen's d for activation differences between correct/incorrect
- **Cross-model generalization**: Do probes transfer between base/instruct variants?
- **Entanglement analysis**: Probe confidence by category, held-out generalization, activation similarity

## Models Tested

| Family | Base | Instruct |
|--------|------|----------|
| Qwen 2.5 | 7B | 7B-Instruct |
| Mistral | 7B-v0.1 | 7B-Instruct-v0.1 |
| Yi | 6B | 6B-Chat |
| Llama 3.1 | 8B | 8B-Instruct |

## Project Structure

```
epistemic_status/
├── gen_data.py              # Dataset generation
├── collect_activations.py   # Activation collection pipeline
├── run_analysis.py          # Analysis entry point
├── cross_model_analysis.ipynb  # Cross-model comparison notebook
├── model_config.py          # Model definitions
├── utils.py                 # Evaluation, memory management
├── analysis/                # Analysis modules
│   ├── loader.py           # Data loading
│   ├── core.py             # Basic statistics
│   ├── probing.py          # Linear probes
│   ├── entropy.py          # Entropy analysis
│   ├── effects.py          # Effect sizes, ROC/AUC
│   ├── calibration.py      # Confidence calibration
│   ├── comparison.py       # Cross-model analysis
│   └── entanglement.py     # RLHF entanglement analysis
└── activations/            # Collected activation data
    ├── qwen_base/
    ├── qwen_instruct/
    └── ...
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (MPS or CUDA)
- TransformerLens
- scikit-learn, pandas, numpy

Tested on macOS ARM64 (M4) with MPS acceleration.

## Citation

If you use this work, please cite it.

## License

MIT License
