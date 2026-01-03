# Fine-Tuning Entangles Epistemic Representations in Language Models

We show that fine-tuning degrades the separability of epistemic states in language model activations. By probing hidden states across 8 models (4 families × base/instruct), we find that alignment training entangles **trained epistemic behaviors** (admitting ignorance, acknowledging ambiguity) with **genuine uncertainty**, making these internal states harder to distinguish despite improved behavioral performance.

**Context**: Prior work established that language models represent epistemic states internally ([Kadavath et al. 2022](https://arxiv.org/abs/2207.05221), [Azaria & Mitchell 2023](https://arxiv.org/abs/2304.13734)). We extend this by showing *how fine-tuning alters these representations* - specifically, that alignment creates targeted entanglement where it trains epistemic policy behaviors. Critically, we find that **RLHF/DPO roughly doubles the entanglement effect compared to SFT alone**.
    
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

### 3. Fine-Tuning Degrades Epistemic Transparency

Instruct tuning makes entropy *less* informative across all models:

| Model | Entropy (base) | Entropy (instruct) | Hidden (base) | Hidden (instruct) |
|-------|----------------|--------------------| --------------|-------------------|
| Llama | 0.914 | 0.734 | 3.0% | 10.5% |
| Mistral | 0.930 | 0.741 | 1.6% | 8.2% |
| Yi | 0.825 | 0.649 | 13.1% | 22.4% |
| Qwen | 0.788 | 0.553 | 14.6% | 20.7% |

The epistemic information exists internally but is increasingly hidden from the output distribution.

### 4. Entanglement Occurs Where Fine-Tuning Trains Policy Behaviors

The critical finding: representational degradation is *selective*. Probe error rates increase specifically for **policy categories** - where fine-tuning trains epistemic behaviors - while **factual categories** remain stable or improve:

| Model | Training Method | Policy Δ | Factual Δ | Selective Gap |
|-------|-----------------|----------|-----------|---------------|
| Qwen | SFT + DPO + GRPO | +0.318 | -0.068 | **0.386** |
| Llama | SFT + RLHF + DPO | +0.286 | -0.071 | **0.357** |
| Mistral | SFT only | +0.247 | +0.092 | 0.155 |
| Yi | SFT only | +0.220 | +0.095 | 0.125 |

*Δ = change in probe error rate after instruct tuning.*

**RLHF/DPO roughly doubles the entanglement effect:**
- **SFT-only models** (Mistral, Yi): Policy Δ ~+0.23, gap ~0.14
- **RLHF/DPO models** (Llama, Qwen): Policy Δ ~+0.30, gap ~0.37

**Probe transfer confirms the mechanism.** Training a probe on base and testing on instruct reveals how representations change:

| Model | Training | Factual Transfer | Policy Transfer | Gap |
|-------|----------|------------------|-----------------|-----|
| Qwen | SFT + DPO + GRPO | 0.821 | 0.382 | **+0.44** |
| Llama | SFT + RLHF + DPO | 0.879 | 0.591 | **+0.29** |
| Mistral | SFT only | 0.252 | 0.650 | -0.40 |
| Yi | SFT only | 0.668 | 0.429 | +0.24 |

RLHF/DPO models show **selective preservation**: factual representations transfer well (~85%) while policy representations are warped (~49%). The base model's "correct/incorrect" structure remains intact for factual questions but is disrupted for policy questions.

SFT-only models show **unpredictable restructuring**: Mistral's factual representations are actually *inverted* (1-accuracy = 0.75), while Yi shows uniform degradation. No consistent pattern.

**Why "policy" vs "factual"?**
- **Policy categories** (`confident_incorrect`, `ambiguous`, `nonsensical`): Correct response requires trained behavior - admitting "I don't know," asking for clarification, recognizing category errors. Fine-tuning explicitly teaches these.
- **Factual categories** (`confident_correct`, `uncertain_correct`): Correct response requires recalling knowledge. Fine-tuning doesn't specifically target these.

This suggests fine-tuning warps representational geometry specifically where it trains epistemic behaviors. The model learns to say "I don't know" through representational changes that entangle trained behaviors with genuine uncertainty states, making these epistemically distinct states harder to distinguish via linear probing.

#### Statistical Significance

Sample-level permutation tests confirm all entanglement effects are highly significant (p < 0.001). By comparing ~249 RLHF-category samples vs ~243 non-RLHF samples directly, we achieve proper statistical power:

| Model | Training | RLHF Δ | Non-RLHF Δ | Difference | 95% CI | Cohen's d |
|-------|----------|--------|------------|------------|--------|-----------|
| Qwen | SFT+DPO+GRPO | +0.211 | -0.116 | **+0.327** | [+0.26, +0.40] | 0.81 (large) |
| Yi | SFT only | +0.209 | -0.036 | **+0.244** | [+0.18, +0.31] | 0.73 (medium) |
| Llama | SFT+RLHF+DPO | +0.215 | -0.030 | **+0.245** | [+0.18, +0.31] | 0.65 (medium) |
| Mistral | SFT only | +0.210 | +0.062 | **+0.148** | [+0.08, +0.21] | 0.38 (small) |

All models show the same pattern: probe error increases significantly more for RLHF categories than non-RLHF categories. Effect sizes range from small (Mistral, d=0.38) to large (Qwen, d=0.81).

### 5. The Alignment Paradox: Better Behavior, Worse Transparency

Despite internal entanglement, behavioral hallucination detection improves dramatically:

| Model | Training | Base | Instruct |
|-------|----------|------|----------|
| Llama | SFT + RLHF + DPO | 7.1% | **68.7%** |
| Qwen | SFT + DPO + GRPO | 1.0% | 58.6% |
| Mistral | SFT only | 6.1% | 28.3% |
| Yi | SFT only | 1.0% | 19.2% |

Fine-tuning teaches models to *behave* as if they know what they don't know, while making internal representations *harder to interpret*. RLHF/DPO models show the largest behavioral gains but also the most entanglement.

## Implications for Alignment & Interpretability

1. **Fine-tuning trades interpretability for behavior** - alignment achieves epistemic caution by warping internal representations, not by building distinct "I should acknowledge uncertainty" circuits
2. **RLHF/DPO amplifies the effect** - preference optimization roughly doubles entanglement compared to SFT alone, suggesting the reward signal specifically targets epistemic behaviors
3. **Entanglement is targeted** - degradation occurs specifically where fine-tuning trains policy behaviors, suggesting interpretability researchers should focus on alignment-modified regions
4. **Entropy-based uncertainty is unreliable** - logprob-based uncertainty estimation works for some models but fails for others; internal probing may be necessary for robust uncertainty quantification
5. **Internal state remains recoverable** - linear probes achieve 0.76-0.96 AUC even after fine-tuning, suggesting interpretability tools could surface the hidden epistemic information that alignment obscures
6. **Calibration is not an alignment objective** - current fine-tuning prioritizes behavioral compliance over transparent uncertainty signaling

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
- **Significance testing**: Sample-level permutation tests with FDR correction for multiple comparisons

### Linear vs Non-Linear Probes

To verify that degradation isn't an artifact of linear probing, we compared linear probes to MLP classifiers (2 hidden layers, 256→128 units):

| Model | Linear | MLP | Diff |
|-------|--------|-----|------|
| qwen_base | 0.812 | 0.800 | -0.012 |
| qwen_instruct | 0.711 | 0.781 | +0.070 |
| llama_base | 0.869 | 0.883 | +0.014 |
| llama_instruct | 0.672 | 0.771 | +0.099 |
| mistral_base | 0.907 | 0.905 | -0.002 |
| mistral_instruct | 0.752 | 0.740 | -0.012 |
| yi_base | 0.825 | 0.839 | +0.014 |
| yi_instruct | 0.773 | 0.764 | -0.008 |

Base models are linearly encoded (MLP ≈ Linear). Qwen and Llama instruct show some non-linear structure (+7-10%), but even MLP probes don't recover base model performance, confirming genuine representational degradation.

## Models Tested

| Family | Base | Instruct | Training Method | Source |
|--------|------|----------|-----------------|--------|
| Llama 3.1 | 8B | 8B-Instruct | SFT + RLHF (PPO) + DPO | [Meta technical report](https://arxiv.org/abs/2407.21783) |
| Qwen 2.5 | 7B | 7B-Instruct | SFT + DPO + GRPO | [Alibaba documentation](https://qwenlm.github.io/blog/qwen2.5/) |
| Mistral | 7B-v0.1 | 7B-Instruct-v0.1 | SFT only | [Mistral announcement](https://mistral.ai/news/announcing-mistral-7b/) |
| Yi | 6B | 6B-Chat | SFT only | [01.AI documentation](https://01.ai/blog/yi-6b-chat) |

This natural experiment allows us to compare the effects of SFT alone vs SFT + preference optimization (RLHF/DPO).

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
│   ├── entanglement.py     # Fine-tuning entanglement analysis
│   └── statistics.py       # Significance testing, multiple comparison correction
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
