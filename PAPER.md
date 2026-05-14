# Revisiting Pseudo-Rehearsal for Continual Learning with Pretrained Backbones

**Dexin Huang (dh3172)** — COMS4995W Applied Deep Learning — May 2026

## Abstract

Robins's "pseudo-rehearsal" (Connection Science, 1995) is a continual-learning idea that does not store data and does not train a generator: it feeds random input vectors through the old network, treats the resulting outputs as pseudo-targets, and mixes those pairs into training on the new task. The method mostly dropped out of the literature, replaced by generative replay (Shin et al., 2017) and model-inversion replay (Yin et al., 2020; Smith et al., 2021). I revisit it in the regime that now dominates continual learning — a frozen ImageNet-pretrained backbone with a small head trained sequentially — and test whether the backbone's natural-image prior makes the random pseudo-inputs informative again. The answer is no. On Split-CIFAR-100 with a frozen ViT-S/16 and a single 100-way linear head, the faithful Robins recipe lands within seed noise of plain sequential fine-tuning (0.103 vs 0.102 final accuracy over three seeds). A 2×2 ablation that swaps the input distribution (uniform random pixels vs STL-10 natural images) and the teacher source (per-task snapshot vs a joint-trained oracle head) gives the same null result in every cell. Diagnostics show that uniform random pixels through ViT-S collapse to a tight feature-space region whose dominant predicted class is exactly `argmax_c (W_c · μ_random + b_c)` from the joint head, but the failure persists even when the input distribution does not collapse and the teacher is perfect. An α sweep over 100× of the KD weight (0.5 to 50) confirms the result is not a tuning artifact: all KD-based methods plateau between 0.10 and 0.17, while class-balanced experience replay reaches 0.529 and joint training 0.788. The takeaway is that in single-linear-head Class-IL with seen-class masking, output-only regularization is structurally weak — only methods that inject actual old-class examples into the training batch can preserve old decision boundaries. Code: `github.com/Dexin-Huang/pseudo-rehearsal-pretrained`.

## 1. Introduction

A neural network trained sequentially on a stream of tasks tends to destroy what it learned on earlier tasks — McCloskey and Cohen (1989) called this catastrophic forgetting, and it remains the central problem of continual learning. Approaches fall into three broad families: regularization (constrain the parameters or outputs not to drift), rehearsal (keep some old examples around and replay them), and architectural (allocate new capacity per task). I revisit the oldest rehearsal-without-storage idea: Anthony Robins's *pseudo-rehearsal* (1995).

Robins's procedure is mechanically simple. After training on task `t`, freeze a copy of the network as a *teacher*. While training on task `t+1`, generate random input vectors `r`, run them through the teacher to get pseudo-targets `q = teacher(r)`, and mix `(r, q)` pairs into the new-task training batches. The student is now nudged to keep producing the teacher's outputs on randomly sampled inputs — a form of self-distillation that requires no stored data and no generative model. The idea predates GANs and VAEs and predates Shin et al.'s deep generative replay (DGR, 2017) by more than two decades.

Pseudo-rehearsal mostly dropped out of the literature. The conventional explanation is that random inputs to a small from-scratch MLP are off-manifold, so the teacher's response on them is uninformative about the old decision boundary. Modern continual learning instead uses learned generators (DGR, BiC, DER++), real-exemplar replay (iCaRL, DER), or model-inversion replay (DeepInversion, Always-Be-Dreaming). A separate recent line uses frozen ImageNet-pretrained backbones with small prompts or heads on top (L2P, DualPrompt, EASE) — there, the backbone's prior carries most of the representational work.

This paper asks a narrow empirical question: in the **frozen-pretrained-backbone** regime, does Robins's original recipe — uniform random pixels through the same backbone, snapshot teacher, KD onto a small student head — recover most of the benefit of methods that require real past data or a learned generator? If yes, pseudo-rehearsal is suddenly cheap and competitive again. If no, the conventional dismissal stands, and we should be able to say *why* with a sharper diagnosis than "random inputs are off-manifold."

My answer is that pseudo-rehearsal does not work, and the conventional dismissal undersells the failure mode. I run a controlled six-method comparison plus three diagnostic extensions on Split-CIFAR-100 with a frozen ViT-S/16 backbone (Section 3). The Robins recipe finishes at average accuracy 0.103, within seed noise of plain sequential fine-tuning (0.102) and well below class-balanced experience replay (0.529); a 2×2 ablation over input distribution and teacher quality, plus an α sweep over the KD weight, leaves the negative result untouched (Section 4). The mechanism check shows that ViT-S maps uniform random pixels to a tight feature-space region whose dominant predicted class is geometrically determined by `argmax_c (W_c · μ_random + b_c)` — predicted class 23, observed class 23, matching exactly. But the deeper finding is that even non-collapsing natural inputs and an oracle teacher do not rescue pseudo-rehearsal at any α tested (Section 5). In single-linear-head Class-IL with seen-class masking, output-only regularization is structurally weak; only injecting real old-class examples into the batch preserves old boundaries.

## 2. Background

**Pseudo-rehearsal (Robins, 1995).** At the end of task `t`, snapshot the network as a teacher `f_t`. While training on task `t+1`, in each step also sample random inputs `r` from some simple distribution (Robins used uniform pixel values), pass them through the teacher to get pseudo-targets `q = f_t(r)`, and add `(r, q)` to the batch with a loss term `KD(q, f_{t+1}(r))`. The original Connection Science paper used a small MLP and reported some benefit on toy sequential tasks; the approach was later extended to deep RL by Atkinson et al. (2018), but otherwise saw little modern follow-up.

**Generative and model-inversion replay.** The natural successor to Robins replaces random inputs with samples from a model. Shin et al. (2017) train a GAN per task as the generator; later work (Lesort et al., 2019; Cong et al., 2020) uses VAEs or normalizing flows. Yin et al. (2020) propose DeepInversion, which inverts classifier features to synthesize replay data; Smith et al. (2021) extend this to continual learning under the name Always-Be-Dreaming (ABD). These methods are "data-free" in the same sense Robins is, but they do substantial extra work — training a generator or running expensive inversion every task — to produce inputs that lie on or near the data manifold.

**Other regularization.** Elastic Weight Consolidation (EWC; Kirkpatrick et al., 2017) penalizes parameter drift weighted by the diagonal Fisher of the previous task; the online variant (Schwarz et al., 2018) accumulates Fisher across tasks. Learning without Forgetting (LwF; Li & Hoiem, 2017) distills the previous-task teacher onto the *current task's real features* — it is the closest comparator to pseudo-rehearsal, differing only in the input distribution used for KD. Experience replay simply stores a small buffer of past examples (iCaRL, DER); the class-balanced variant keeps a fixed number of examples per seen class.

**Pretrained-backbone CL.** Recent work (Wang et al., 2022, L2P; Wang et al., 2022, DualPrompt) attaches small prompts to a frozen ImageNet-pretrained ViT and trains those per task. The frozen backbone supplies the representation; the per-task adaptation is small. This regime is what makes the Robins revisit interesting now — the backbone might supply the "natural-image prior" that random pseudo-inputs need.

## 3. Method

### 3.1 Setting

**Dataset.** Split-CIFAR-100 in 10 tasks of 10 classes each. Class permutations are part of the seed, so each of the three seeds defines an independent grouping. Within each task, 5,000 train / 1,000 test images.

**Backbone.** `vit_small_patch16_224.augreg_in21k_ft_in1k` from `timm` (Wightman, 2019), pretrained on ImageNet-21k and fine-tuned on ImageNet-1k. CIFAR-100 images are upsampled to 224×224 with bilinear interpolation and ImageNet-normalized. The backbone is **frozen** throughout, and all features (CIFAR-100 train/test plus a 100,000-sample random-pixel pool and a 100,000-sample STL-10 unlabeled pool) are precomputed once. Each "training run" then operates on cached 384-dimensional CLS-token features, so a full 10-task continual run takes seconds.

**Head.** A single linear layer `Linear(384, 100)` with bias. About 38,500 parameters. The linear-head choice is deliberate: it keeps the comparison clean by removing head-capacity as a confound. Class-IL evaluation uses argmax over the full 100-way head, restricted to seen classes at the current point.

**Seen-class masking.** All cross-entropy and KD losses use additive `-inf` masks on unseen-class logits, so the model is never trained to predict away classes it has not seen as targets. Without this, sequential training would supply implicit negative gradients on future classes; with it, the comparison isolates the catastrophic-forgetting question rather than introducing a separate logit-suppression artifact.

### 3.2 Methods compared

All methods optimize the same linear head with AdamW (lr=1e-3, batch=256, 10 epochs per task) on cached features.

- **Sequential.** Default seen-class-masked CE on current-task data only. Lower bound.
- **Joint.** Same head, trained once on all 100 classes. Upper bound.
- **Class-balanced replay.** Buffer of 5 examples per seen class (up to 500 total). Each training batch is half current-task / half buffer at constant total batch size.
- **Online EWC.** Diagonal empirical Fisher computed in closed form for a masked linear-CE head as `F_W[c,j] = mean_i (p_i[c] - 1{c=y_i})^2 · x_i[j]^2`. Accumulated across tasks; one running θ\*.
- **Pseudo-rehearsal (Robins, faithful).** At the start of task `t+1`, snapshot the head as teacher `h_t`. Per step, sample `batch_size` random images (pixels uniform [0,1], ImageNet-normalized), pass through the frozen ViT (cached in advance), pass through the teacher, restrict to *previously-seen* classes (i.e. seen − current), and add `α · KD(q_old, p_student)` where KD is soft-label cross-entropy in teacher → student direction. Default α = 1.0.
- **LwF.** Same KD machinery as pseudo-rehearsal, but distill on *current-task real features* instead of random-pool features. The closest comparator: pseudo-rehearsal differs from LwF only in the choice of distillation inputs.

### 3.3 Extensions: 2×2 ablation and α sweep

The first sweep above (6 methods × 3 seeds = 18 runs) returns a clean null for pseudo-rehearsal. Two natural rescue stories follow, and I test each independently.

- **Input distribution.** Replace uniform random pixels with **STL-10 unlabeled** features through the same ViT (`pseudo_natural`). STL-10 is 100,000 real natural images, out-of-distribution for CIFAR-100. If pseudo-rehearsal needs natural-image structure in the inputs rather than random noise, this should help.
- **Teacher source.** Replace the snapshot teacher with the **joint-trained head** loaded from a checkpoint (`pseudo_oracle`). This is information leakage — the oracle has seen all 100 classes — and so is not a valid method, but it is a clean diagnostic: if pseudo-rehearsal fails because the snapshot teacher accumulates errors during CL, an oracle teacher should rescue it.
- **Both.** `pseudo_oracle_natural` combines both, giving the best-case for pseudo-rehearsal across the two factors.

A second sweep adds 3 new pseudo variants × 3 seeds = 9 runs.

To address the obvious tuning critique, a third sweep varies the KD weight `α ∈ {0.5, 1, 5, 10, 50}` on `pseudo`, `pseudo_oracle`, and `lwf` (45 runs). At sufficiently large α, the KD term dominates the new-task CE term and the student must mimic the teacher; for `pseudo_oracle` this should in principle approach joint accuracy. The question is how the curve looks in between.

### 3.4 Mechanism check

The diagnostic story for pseudo-rehearsal's failure should not rely on hand-waving about "off-manifold inputs." I make a sharp prediction: the dominant pseudo-target class on random inputs is determined by the geometry of the (random-feature mean, joint head) pair alone:

$$\hat{c} = \arg\max_c (W_c \cdot \mu_{\text{random}} + b_c)$$

If random features collapse tightly enough (small per-dimension std around the mean), per-sample variation is negligible and the head's response on the *mean* random feature predicts the dominant argmax. I compute `μ_random` over the 100K random-pool features and compare `ĉ` against the empirical top class on the full pool.

### 3.5 Diagnostic probes

To characterize *how* random pixels through ViT land in feature space, I report (i) L2 norm mean and std of real vs random features, (ii) nearest-neighbor cosine distance from a 5K random sample to a 5K real sample, (iii) predictive entropy of the joint head on real vs random features, and (iv) the top-class histogram of joint-head predictions over the 100K random pool.

## 4. Experiments and Results

### 4.1 Main comparison

Results on Split-CIFAR-100, mean ± std over three seeds:

| Method | Final ACC | FGT | BWT |
|---|---|---|---|
| Joint (upper bound) | **0.788 ± 0.000** | 0.000 | n/a |
| Class-balanced replay | **0.529 ± 0.004** | 0.450 | −0.450 |
| Online EWC | 0.192 ± 0.010 | 0.825 | −0.825 |
| LwF (α=1) | 0.111 ± 0.012 | 0.919 | −0.919 |
| Pseudo-natural | 0.106 ± 0.008 | 0.923 | −0.923 |
| Pseudo-oracle | 0.105 ± 0.005 | 0.925 | −0.925 |
| Pseudo (Robins, faithful) | 0.103 ± 0.005 | 0.927 | −0.927 |
| Sequential (lower bound) | 0.102 ± 0.004 | 0.928 | −0.928 |
| Pseudo-oracle-natural | 0.099 ± 0.003 | 0.932 | −0.932 |

The 2×2 ablation viewed as a table:

| | Random pixels | STL-10 natural |
|---|---|---|
| **CL teacher** | 0.103 | 0.106 |
| **Oracle teacher** | 0.105 | 0.099 |

All four pseudo cells overlap within seed noise. Neither switching the input distribution to natural images nor giving the method a perfect teacher rescues it. LwF lands in the same neighborhood. EWC offers a modest gain (it constrains the linear head's drift but cannot supply the missing per-class signal). Only experience replay clears the noise — at 0.529, it reaches more than 2/3 of the joint upper bound while storing 500 features total.

### 4.2 α sweep

Sweeping the KD weight `α` over two orders of magnitude (Figure 1):

| Method | α=0.5 | α=1.0 | α=5.0 | α=10.0 | α=50.0 |
|---|---|---|---|---|---|
| LwF | 0.110±0.011 | 0.111±0.012 | 0.114±0.013 | 0.117±0.013 | 0.164±0.014 |
| Pseudo | 0.103±0.004 | 0.103±0.005 | 0.107±0.007 | 0.114±0.010 | 0.144±0.017 |
| Pseudo-oracle | 0.103±0.005 | 0.105±0.005 | 0.111±0.009 | 0.118±0.013 | 0.142±0.018 |

All three KD methods trace nearly parallel monotonic curves, gaining only 0.04-0.05 over the default α=1.0 even at α=50. The oracle teacher provides no meaningful advantage over the snapshot teacher at any α tested. At α → ∞ the KD term would dominate CE and the student would converge to the teacher, recovering joint accuracy for `pseudo_oracle`; in the explored range the curve has not begun to bend toward that limit. The failure is therefore not a tuning issue.

### 4.3 Diagnostics

The joint head's behavior on the 100,000 random-pixel features (single seed, used as a fixed teacher for the diagnostic):

- Real-feature L2 norm: 49.52 ± 1.70. Random-feature L2 norm: 34.33 ± 1.77. Random features sit in a sphere ~31% smaller in radius.
- Mean nearest-neighbor cosine of random → real: 0.392 (p25 0.381, p75 0.401). Real CIFAR features are far from random-pixel features in direction.
- Predictive entropy on real features: 0.513 nats. On random features: 2.666 nats. Uniform-over-100 reference: 4.605. The teacher is more uncertain on random inputs but not uniformly so.
- Top-class histogram on random features: **class 23 takes 62.7% of all argmax predictions**, and the teacher predicts only **4 of 100 classes** anywhere across 100K random inputs. Distribution entropy over predicted classes: 0.664 nats.

### 4.4 Mechanism check

The geometric prediction holds exactly: `argmax_c (W_c · μ_random + b_c) = 23`, matching the empirical top class. The top-3 predicted classes from the mean feature are {23, 60, 76}; class 23 dominates. The per-dimension std of the random pool (averaged across 384 dims) is 0.305, confirming the random-feature distribution is tight enough that per-sample variation does not change the argmax. So the random-pixel collapse is fully determined by the joint head and the mean random feature.

### 4.5 Figures

`running_avg_accuracy.png` shows running mean accuracy `(1/(k+1)) Σ_{t≤k} A[k,t]` as a function of task index `k`. Replay separates from the pack immediately and decays gracefully; EWC tracks slightly above the bottom group; the four pseudo variants, LwF, and sequential are visually indistinguishable across all 10 tasks. `final_per_task.png` plots `A[T-1, t]` against `t`: replay maintains roughly uniform 40-60% accuracy across tasks, EWC partially preserves only the last 3-4 tasks, and every other method gets ~90% on task 9 and ~0% on everything before. `alpha_sweep.png` plots the α sweep on a log x-axis with replay (0.529) and joint (0.788) as horizontal references; the three KD curves crawl from 0.10 to 0.16 across two orders of magnitude.

## 5. Discussion

**The mechanism story is sharper than the conventional one.** The standard dismissal of Robins is "random inputs are off-manifold and the teacher is uninformative on them." The diagnostics confirm this — random pixels through ViT-S land in a tight, shrunken region of feature space, and the teacher classifies 62.7% of them as one dominant class. The mechanism check goes further: the collapse is *geometrically predictable* from the mean random feature and the head's first-layer weights. So Robins's recipe at the unit of analysis was always going to give the student degenerate pseudo-soft-targets — a near-constant signal pointing at one arbitrary class.

**But the conventional story isn't the whole story.** The 2×2 ablation shows that even *non-collapsing* natural inputs (STL-10 images, which are real natural images and produce non-degenerate teacher distributions) do not rescue pseudo-rehearsal. And even a *perfect oracle teacher* — the joint-trained classifier with full knowledge of all 100 classes — does not rescue it. Both factors combined still fail. The collapse explanation alone is therefore not the full diagnosis.

**What is the full diagnosis?** In single-linear-head Class-IL with seen-class masking, every training step on task `t` provides per-class CE gradients on 5,000 fresh examples per epoch, aggressively rewriting the head toward the current task's 10 classes. KD on previously-seen classes adds a counter-pressure on those old-class logits — but it acts only through the *outputs*, not the per-example supervision the new task gets. Even at α=50, the KD term cannot match the CE term's gradient mass without effectively shutting down new-task learning. This generalizes a known observation about LwF's weakness in Class-IL (a recency bias in the head); the extension here is that the weakness survives both an improved input distribution and a perfect teacher.

Replay works for the symmetric reason: it injects real per-example old-class supervision into the batch, on the same footing as new-task examples. EWC partially works because it constrains the head's weights from drifting *at all*, an approach that is independent of class-level signal. Output-only regularization, in this regime, sits in the worst of both worlds.

**Connection to data-free CL.** The finding sharpens what "data-free" CL needs. DeepInversion (Yin et al., 2020) and ABD (Smith et al., 2021) are both data-free in Robins's sense, but they do substantial extra work to produce inputs that lie on the data manifold and elicit class-discriminative responses from the teacher. The implicit assumption is that the *input distribution* is what was missing from Robins. This paper suggests that, at least in the linear-head Class-IL regime, the input distribution is not the bottleneck — the structural separation between output-only and example-level supervision is. Whether the same holds when the backbone is fine-tunable, or when the head has more capacity, is left to future work.

**Limitations.** The paper studies one backbone (ViT-S/16 ImageNet-21k → 1k), one dataset (Split-CIFAR-100), one continual setting (Class-IL, 10 tasks × 10 classes), one head (linear), and one feature-precomputation regime (frozen backbone). A CLIP or DINO backbone could place random features in a different feature-space region and change the diagnostic story even though, I conjecture, the structural finding would survive. A multi-layer MLP head or a fine-tunable backbone introduces representational degrees of freedom that EWC and LwF were originally designed to protect, and could plausibly shift the result. The α sweep stops at 50; in principle pseudo-oracle would approach joint at α → ∞, but only by abandoning new-task learning entirely. The replay number depends on storing 5 examples per class, which is small but not zero; a fair "no-storage" floor for pseudo-rehearsal is plain sequential FT, which is what it matches.

## 6. Conclusion

I asked whether a frozen ImageNet-pretrained backbone makes Robins's 1995 pseudo-rehearsal competitive again. The answer is no, even after testing three natural rescue stories: a natural-image input distribution, an oracle teacher, and a 100× sweep of the KD weight. The original failure mechanism (random-pixel collapse in feature space) is real and geometrically predictable, but it is not the deepest cause. In single-linear-head Class-IL with seen-class masking, output-only regularization — pseudo-rehearsal, LwF, EWC — is structurally weaker than methods that inject actual old-class supervision into the batch. The cheapest competitive method in the experiments stores 5 features per seen class.

## References

Atkinson, C., McCane, B., Szymanski, L., & Robins, A. (2018). Pseudo-rehearsal: Achieving deep reinforcement learning without catastrophic forgetting. *arXiv:1812.02464*.

Buzzega, P., Boschini, M., Porrello, A., Abati, D., & Calderara, S. (2020). Dark experience for general continual learning: a strong, simple baseline. *NeurIPS*.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. *ICLR*.

Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network. *NeurIPS Workshop*.

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13).

Li, Z., & Hoiem, D. (2017). Learning without forgetting. *IEEE TPAMI*, 40(12).

McCloskey, M., & Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation*, 24.

Robins, A. (1995). Catastrophic forgetting, rehearsal and pseudorehearsal. *Connection Science*, 7(2), 123-146.

Schwarz, J., Czarnecki, W., Luketina, J., et al. (2018). Progress & compress: a scalable framework for continual learning. *ICML*.

Shin, H., Lee, J. K., Kim, J., & Kim, J. (2017). Continual learning with deep generative replay. *NeurIPS*.

Smith, J., Hsu, Y.-C., Balloch, J., et al. (2021). Always be dreaming: A new approach for data-free class-incremental learning. *ICCV*.

Wang, Z., Zhang, Z., Lee, C.-Y., et al. (2022). Learning to prompt for continual learning. *CVPR*.

Wightman, R. (2019). PyTorch image models (timm). `github.com/huggingface/pytorch-image-models`.

Yin, H., Molchanov, P., Alvarez, J. M., et al. (2020). Dreaming to distill: Data-free knowledge transfer via DeepInversion. *CVPR*.
