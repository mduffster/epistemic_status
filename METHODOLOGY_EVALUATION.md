# Methodology Evaluation & Novelty Assessment

## Part 1: Methodology Evaluation

### Strengths

| Component | Implementation | Assessment |
|-----------|----------------|------------|
| **Probe design** | Linear logistic regression, 5-fold stratified CV, StandardScaler per fold | ✓ Sound - standard practice, prevents leakage |
| **Activation extraction** | Residual stream + MLP at 3 positions, all layers | ✓ Comprehensive coverage |
| **Statistical testing** | Sample-level permutation tests (n≈300 per group), FDR correction | ✓ Strong - solves n=3 category limitation |
| **Cross-model validation** | 4 families × 2 variants = 8 models | ✓ Good - natural experiment (SFT vs RLHF/DPO) |
| **Effect size reporting** | Cohen's d, bootstrap CIs | ✓ Rigorous quantification |

### Methodological Gaps (Priority Ordered)

#### P0: Critical Issues

1. **Evaluation logic is phrase-based, not semantic**
   - Matches 150+ substring patterns like "doesn't exist", "fictional"
   - **No human validation exists** (confirmed via codebase search)
   - Risk: False positives (phrase appears incidentally) and false negatives (correct response, different wording)
   - Impact: If phrase matching has even 10% error rate, category-level differences could be artifacts
   - **To address**: Human annotation of 100 samples per category, measure inter-annotator agreement, compute precision/recall of phrase matching

2. **Causality vs correlation**
   - Shows: policy categories have higher probe error after RLHF
   - Doesn't prove: representation warping vs evaluation threshold change vs simple output shift
   - **To address**: Intervention study - fine-tune with explicit entropy preservation objective, see if entanglement still occurs

3. **Missing category: uncertain_incorrect**
   - Currently classified as "factual" but evaluation requires debunking (behavioral)
   - Could be policy-like (model must recognize misconception)
   - **To address**: Analyze uncertain_incorrect separately; may show intermediate entanglement

#### P1: Important Gaps

4. **Single entropy definition**
   - Only uses softmax entropy at last position
   - Doesn't capture token-by-token uncertainty during generation
   - **To address**: Compare multiple entropy definitions (mean across positions, top-k entropy, perplexity)

5. **No disentanglement of RLHF methods**
   - Qwen uses DPO+GRPO, Llama uses RLHF+DPO
   - Can't isolate which method causes entanglement
   - **To address**: Find models with only-DPO or only-PPO training; or analyze intermediate checkpoints

6. **Activation choice limitations**
   - No attention patterns or layer norm outputs
   - Flattening all layers mixes depth information
   - **To address**: Layer-wise analysis (already partially done), add attention probing

#### P2: Minor Gaps

7. **Dataset size (~100 per category)**
   - May not capture full distribution of epistemic situations
   - **To address**: Expand dataset with more diverse prompts

8. **Temperature=0 generation**
   - Deterministic output may not reflect real use
   - **To address**: Compare with temperature=0.7 generation

9. **TransformerLens vs official implementations**
   - Minor numerical differences possible
   - **To address**: Document version, run spot-checks

---

## Part 2: Literature Comparison

### What Existing Work Shows

| Topic | Key Finding | Source |
|-------|-------------|--------|
| **Epistemic probing works** | Linear probes achieve high accuracy on uncertainty classification | 2024-2025 UQ surveys |
| **D-STEER mechanism** | DPO is "low-rank steering" in narrow subspace, teaches behavior not belief | [Gao et al. 2024](https://arxiv.org/abs/2512.11838) |
| **Alignment causes overconfidence** | Aligned models show systematic overconfidence in calibration studies | 2025 80-model study |
| **Entropy degrades with alignment** | Alignment optimizes for confident outputs regardless of internal uncertainty | Multiple calibration papers |
| **Representation geometry** | Fine-tuning induces multimodal structure in later layers | ICML 2025 |

### What's NOVEL in This Project

| Finding | Why It's Novel |
|---------|----------------|
| **Selective entanglement** | Prior work shows alignment affects calibration broadly. This shows it's *category-specific* - policy categories degraded, factual preserved. Nobody has shown this selectivity. |
| **RLHF/DPO doubles the effect** | D-STEER shows DPO operates in narrow subspace. This quantifies that preference optimization causes 2x entanglement vs SFT alone (gap 0.36 vs 0.14). First empirical measurement. |
| **Probe transfer asymmetry** | Factual representations transfer ~85%, policy ~49%. Prior work hasn't characterized this asymmetric preservation. Suggests fine-tuning selectively restructures manifolds. |
| **Training data > architecture** | Yi vs Llama (same architecture, 4x difference in hidden info) shows training data dominates. Underexplored in representation geometry literature. |
| **The alignment paradox quantified** | Behavioral accuracy improves (5%→63% policy) while probe accuracy drops. Prior work notes calibration issues but doesn't connect to representation-level changes. |

### What's Confirmatory (Not Novel)

| Finding | Status |
|---------|--------|
| Probes achieve 76-96% accuracy | Confirms linear representability (known) |
| Entropy AUC degrades after alignment | Confirms overconfidence pattern (known) |
| RLHF affects internal representations | Known from D-STEER, but mechanism details novel |

---

## Part 3: Potential Further Analysis

### High-Value Additions

1. **Human validation study**
   - Annotate 100 samples per category
   - Compute precision/recall of phrase-based evaluation
   - This is the biggest validity threat; addressing it would significantly strengthen claims

2. **D-STEER steering vector analysis** ✓ COMPLETED
   - Extracted steering vectors à la D-STEER methodology
   - Confirmed low-rank structure (14-19 dimensions capture 80% variance)
   - Found policy and factual categories diverge during fine-tuning (d > 2.4)

3. **Subcategory convergence analysis** ✓ COMPLETED
   - Measured pairwise centroid distance changes during fine-tuning
   - **Key finding**: Policy subcategories (confident_incorrect, ambiguous, nonsensical) CONVERGE toward similar representation spaces:
     - Llama (RLHF/DPO): Policy centroids move **-41%** closer, factual only -24%
     - Yi (SFT): Policy centroids move **-17%** closer, factual only -5%
     - Qwen (DPO/GRPO): Both converge ~17-21%
     - Mistral (SFT): Outlier - everything diverges, but policy diverges less
   - This is direct evidence of entanglement: trained epistemic behaviors (hallucination acknowledgment, ambiguity recognition, nonsensical detection) are being mapped to overlapping regions of representation space

4. **Intervention experiment**
   - Fine-tune a base model with entropy preservation objective
   - See if entanglement still occurs
   - Would establish causality

5. **Expand uncertain_incorrect analysis**
   - Currently grouped with "factual" but requires behavioral response
   - May show intermediate entanglement pattern
   - Would refine the policy/factual distinction

### Medium-Value Additions

6. **Multiple entropy definitions**
   - Mean entropy, top-k entropy, perplexity
   - See if results hold across definitions

7. **Layer-wise entanglement analysis**
   - Which layers show the most entanglement?
   - Connect to D-STEER's "upper layer" finding

8. **Attention pattern probing**
   - Currently only residual/MLP
   - Attention may encode epistemic state differently

---

## Summary: Novelty Statement

**Core novel contribution:**

> Fine-tuning doesn't uniformly degrade epistemic representations—it *selectively entangles* categories where alignment trains specific behaviors (admitting ignorance, acknowledging ambiguity). RLHF/DPO roughly doubles this entanglement effect compared to SFT alone. The asymmetric probe transfer (85% factual vs 49% policy) suggests alignment preserves factual geometry while warping policy-relevant dimensions of the representation manifold.

**Subcategory convergence reveals the mechanism:**

> Policy subcategories (confident_incorrect, ambiguous, nonsensical) physically CONVERGE in representation space during fine-tuning. Llama (RLHF/DPO) shows the strongest effect: policy centroids move 41% closer while factual only moves 24% closer. Yi (SFT) shows selective convergence: policy -17%, factual only -5%. This is direct evidence that trained epistemic behaviors are being mapped to overlapping regions—the operational definition of entanglement.

**How this advances the field:**

1. **D-STEER shows DPO is "low-rank steering"** → We confirm low-rank structure (14-19 dims) and show *where* that steering causes damage (policy categories)
2. **Calibration studies show overconfidence** → We show this manifests as category-specific representation convergence
3. **Representation geometry work is abstract** → We provide concrete measurements: policy categories move 17-41% closer during fine-tuning
4. **Prior work asks "does alignment affect representations?"** → We show *which categories* converge and by *how much*

**Key differentiator:**

Prior work asks "does alignment affect representations?" We ask "*where*, *how much*, and *do categories converge?*"—and find policy categories (trained behaviors) converge while factual categories (knowledge recall) preserve separation.

---

## References

- [D-STEER: Preference Alignment Techniques Learn to Behave, not to Believe](https://arxiv.org/abs/2512.11838) - Gao et al. 2024
- [Kadavath et al. 2022](https://arxiv.org/abs/2207.05221) - Language models know what they know
- [Azaria & Mitchell 2023](https://arxiv.org/abs/2304.13734) - Internal state of LLMs
