# Fine-Tuning Entangles Epistemic Representations in Language Models

We show that fine-tuning degrades the separability of epistemic states in language model activations. By probing hidden states across 8 models (4 families × base/instruct), we find that alignment training entangles **trained epistemic behaviors** (admitting ignorance, acknowledging ambiguity) with **genuine uncertainty**, making these internal states harder to distinguish despite improved behavioral performance.

**Context**: Prior work established that language models represent epistemic states internally ([Kadavath et al. 2022](https://arxiv.org/abs/2207.05221), [Azaria & Mitchell 2023](https://arxiv.org/abs/2304.13734)). These studies asked *whether* models have uncertainty representations. We extend this by asking *what happens to these representations during fine-tuning*.

Our contribution is empirical characterization:
- **Selective entanglement:** Degradation targets policy categories, not factual ones
- **RLHF/DPO amplification:** Preference optimization roughly doubles the effect vs SFT alone
- **The alignment paradox:** Behavioral improvements coexist with representational degradation

We do not claim these findings are surprising—it's plausible that training on epistemic outputs would affect epistemic representations. The contribution is *quantifying* the effect across training methods and identifying its selectivity.
    
## Key Findings

### 1. Models Hide Epistemic Information

Linear probes on activations predict output correctness better than output entropy alone. The gap reveals "hidden information"—uncertainty the model accurately represents internally but fails to surface:

| Model | Entropy AUC | Probe AUC | Hidden Info |
|-------|-------------|-----------|-------------|
| Mistral 7B base | **0.930** | 0.946 | **1.6%** |
| Llama 3.1 8B base | 0.914 | 0.943 | 3.0% |
| Yi 6B base | 0.825 | 0.956 | 13.1% |
| Qwen 2.5 7B base | 0.788 | 0.935 | 14.6% |

**Training origin matters more than architecture.** Yi and Llama share the same architecture but differ 4x in hidden information (13.1% vs 3.0%). English-trained models (Llama, Mistral) have highly informative entropy; Chinese-trained models (Qwen, Yi) hide more. The models "know" they're uncertain but the signal doesn't make it to logprobs.

### 2. Fine-Tuning Degrades Epistemic Transparency

Instruct tuning makes entropy *less* informative across all models, regardless of methodology:

| Model | Training | Entropy (base) | Probe (base) | Hidden (base) | Entropy (inst) | Probe (inst) | Hidden (inst) |
|-------|----------|----------------|--------------|---------------|----------------|--------------|---------------|
| Qwen | SFT+DPO+GRPO | 0.788 | 0.935 | 14.6% | 0.553 | 0.760 | 20.7% |
| Llama | SFT+RLHF+DPO | 0.914 | 0.943 | 3.0% | 0.734 | 0.839 | 10.5% |
| Mistral | SFT only | 0.930 | 0.946 | 1.6% | 0.741 | 0.823 | 8.2% |
| Yi | SFT only | 0.825 | 0.956 | 13.1% | 0.649 | 0.873 | 22.4% |

*Hidden = Probe AUC − Entropy AUC (information the model has internally but doesn't surface)*

### 3. Entanglement is Selective

The key finding: representational degradation is *selective*. Probe error rates increase specifically for **policy categories**, those where fine-tuning trains epistemic output behaviors, while **factual categories** remain relatively stable or improve:

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

**Probe transfer illustrates the mechanism.** Training a probe on base and testing on instruct reveals how representations change:

| Model | Training | Factual Transfer | Policy Transfer | Gap |
|-------|----------|------------------|-----------------|-----|
| Qwen | SFT + DPO + GRPO | 0.821 | 0.382 | **+0.44** |
| Llama | SFT + RLHF + DPO | 0.879 | 0.591 | **+0.29** |
| Mistral | SFT only | 0.252 | 0.650 | -0.40 |
| Yi | SFT only | 0.668 | 0.429 | +0.24 |

RLHF/DPO models show **selective preservation**: factual representations transfer well (~85%) while policy representations are warped (~49%). The base model's "correct/incorrect" structure remains intact for factual questions but is disrupted for policy questions.

SFT-only models show **unpredictable restructuring**: Mistral's factual representations are actually *inverted* (1-accuracy = 0.75), while Yi shows uniform degradation. We haven't uncovered a discernible pattern in this restructuring, though as mentioned non-transfer linear probes retain more information in **policy categories** for SFT-only models than they do in RLHF/DPO/GRPO models. 

**Why "policy" vs "factual"?**
- **Policy categories** (`confident_incorrect`, `ambiguous`, `nonsensical`): Correct response requires trained behavior like admitting "I don't know," asking for clarification, or recognizing category errors. Fine-tuning explicitly teaches these.
- **Factual categories** (`confident_correct`, `uncertain_correct`): Correct response requires recalling knowledge. Fine-tuning doesn't specifically target these.

This suggests fine-tuning warps representational geometry specifically where it trains epistemic output behaviors. The model learns to say "I don't know" through representational changes that entangle trained behaviors with genuine uncertainty states, making these epistemically distinct states harder to distinguish via linear probing.

#### Statistical Significance

Sample-level permutation tests confirm all entanglement effects are highly significant (p < 0.001). We compare ~249 fine-tuned-category samples vs ~243 non-fine-tuned samples directly:

| Model | Training | RLHF Δ | Non-RLHF Δ | Difference | 95% CI | Cohen's d |
|-------|----------|--------|------------|------------|--------|-----------|
| Qwen | SFT+DPO+GRPO | +0.211 | -0.116 | **+0.327** | [+0.26, +0.40] | 0.81 (large) |
| Yi | SFT only | +0.209 | -0.036 | **+0.244** | [+0.18, +0.31] | 0.73 (medium) |
| Llama | SFT+RLHF+DPO | +0.215 | -0.030 | **+0.245** | [+0.18, +0.31] | 0.65 (medium) |
| Mistral | SFT only | +0.210 | +0.062 | **+0.148** | [+0.08, +0.21] | 0.38 (small) |

All models show the same pattern: probe error increases significantly more for fine-tuned categories than non-fine-tuned categories. Effect sizes range from small (Mistral, d=0.38) to large (Qwen, d=0.81).

#### A Possible Mechanism

Why does fine-tuning cause selective entanglement? Recent work on D-STEER ([Gao et al. 2024](https://arxiv.org/abs/2512.11838)) established that DPO operates as a "low rank steering mechanism," modifying a narrow subspace of activations rather than broadly restructuring representations. The authors argue DPO teaches models "how to act aligned, not what to believe."

We tested whether this low-rank structure generalizes beyond DPO to other fine-tuning methods, and whether it explains the entanglement we observe.

#### Steering Vector Analysis

Using D-STEER's methodology, we extracted steering vectors (mean activation change from base→instruct) and performed SVD analysis across all four model families.

**Low-rank structure generalizes beyond DPO.** All fine-tuning methods—not just DPO—show alignment changes concentrated in a narrow subspace:

| Model | Training | Effective Rank (80% var) |
|-------|----------|-------------------------|
| Qwen | SFT+DPO+GRPO | 14 dimensions |
| Llama | SFT+RLHF+DPO | 19 dimensions |
| Mistral | SFT only | 18 dimensions |
| Yi | SFT only | 19 dimensions |

All models show low-rank structure (14-19 dimensions capture 80% of variance in a ~100k-dimensional space), consistent with D-STEER's "narrow subspace" finding.

**Category-specific steering reveals divergence.** We computed separate steering vectors for policy and factual categories. If fine-tuning affects them uniformly, these should be parallel. They're not:

| Model | Training | Policy↔Factual Similarity | Differential Effect (d) |
|-------|----------|--------------------------|------------------------|
| Qwen | SFT+DPO+GRPO | 0.88 | 2.47 (large) |
| Llama | SFT+RLHF+DPO | **0.76** | **3.15** (large) |
| Mistral | SFT only | 0.80 | 3.80 (large) |
| Yi | SFT only | 0.93 | 2.51 (large) |

Policy and factual categories move in *different directions* during fine-tuning (cosine similarities 0.76-0.93). The differential steering vector—the direction where they diverge—separates categories with large effect sizes (d > 2.4).

**Subcategory convergence reveals the entanglement mechanism.** We measured how centroid distances between categories change during fine-tuning. Policy subcategories (confident_incorrect, ambiguous, nonsensical) converge toward similar representation spaces:

| Model | Training | Policy Δ Distance | Factual Δ Distance | Pattern |
|-------|----------|-------------------|--------------------|--------------------|
| Llama | SFT+RLHF+DPO | **-41.0%** | -23.7% | Policy converges more |
| Yi | SFT only | **-16.8%** | -4.5% | Selective policy convergence |
| Qwen | SFT+DPO+GRPO | -17.2% | -21.3% | General convergence |
| Mistral | SFT only | +18.1% | +29.7% | Divergence (outlier) |

*Negative = centroids move closer together. Policy categories = confident_incorrect, ambiguous, nonsensical. Factual = confident_correct, uncertain_correct.*

**Interpretation.** Fine-tuning creates a shared "policy response" space where different trained epistemic behaviors overlap:
- **Llama (RLHF+DPO)** shows the strongest effect: policy centroids move 41% closer while factual only moves 24% closer
- **Yi (SFT)** shows selective convergence: policy converges 17%, factual barely moves (5%)
- **Mistral** is an outlier where everything diverges, though policy diverges less than factual

This is direct evidence of the entanglement mechanism: when models learn to admit ignorance (confident_incorrect), acknowledge ambiguity (ambiguous), and recognize nonsense (nonsensical), these epistemically distinct situations are being mapped to overlapping regions of representation space. The model learns a general "express uncertainty" response rather than maintaining distinct representations for each epistemic state.

### 4. The Alignment Paradox

Despite internal entanglement, behavioral performance improves dramatically—especially for policy categories:

| Model | Training | Factual (base) | Factual (inst) | Policy (base) | Policy (inst) |
|-------|----------|----------------|----------------|---------------|---------------|
| Qwen | SFT+DPO+GRPO | 81.1% | 93.8% | 5.2% | **62.7%** |
| Llama | SFT+RLHF+DPO | 90.9% | 94.7% | 4.8% | **55.8%** |
| Mistral | SFT only | 90.9% | 90.9% | 2.8% | 34.5% |
| Yi | SFT only | 83.1% | 88.1% | 1.6% | 27.3% |

*Behavioral accuracy = % of prompts answered correctly. Factual = confident_correct, uncertain_correct. Policy = confident_incorrect, ambiguous, nonsensical.*

Base models achieve near-zero policy accuracy—they don't admit ignorance, acknowledge ambiguity, or recognize nonsense. Fine-tuning fixes this behaviorally (+50-60pp for RLHF/DPO, +25-30pp for SFT-only), but the internal probe accuracy on policy categories *drops* (Section 3). Better behavior, worse transparency.

## Implications for Alignment & Interpretability

1. **Probe transfer as training monitoring** - Factual representations transfer well (~85%) from base→instruct for RLHF/DPO models. Probes trained on base activations could monitor what knowledge is preserved during fine-tuning; if transfer drops for factual categories, it may indicate unintended knowledge loss.

2. **Fine-tuning trades interpretability for behavior** - Alignment achieves epistemic caution by warping internal representations, not by building distinct "I should acknowledge uncertainty" circuits.

3. **RLHF/DPO amplifies the effect** - Preference optimization roughly doubles entanglement compared to SFT alone, consistent with D-STEER's finding that DPO modifies a narrow subspace.

4. **Entanglement is targeted** - Degradation occurs specifically where fine-tuning trains policy behaviors, suggesting interpretability researchers should focus on alignment-modified regions.

5. **Entropy-based uncertainty is unreliable** - Logprob-based uncertainty estimation works for some models but fails for others; internal probing may be necessary for robust uncertainty quantification.

6. **Internal state remains recoverable** - Linear probes achieve 0.76-0.96 AUC even after fine-tuning, suggesting interpretability tools could surface the hidden epistemic information that alignment obscures.

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
- **Steering vector analysis**: D-STEER-inspired SVD analysis, category-specific steering, ablation tests

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
│   ├── statistics.py       # Significance testing, multiple comparison correction
│   └── steering.py         # D-STEER-inspired steering vector analysis
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
