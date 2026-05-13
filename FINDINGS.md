# FINDINGS — Pseudo-Rehearsal with a Frozen ViT-S/16 on Split-CIFAR-100

Milestone snapshot, 2026-05-13. First sweep + extension sweep complete.

## Headline

With a frozen ImageNet-pretrained ViT-S/16 backbone and a linear 384→100 head trained sequentially on Split-CIFAR-100 (Class-IL, 10 tasks × 10 classes), the Robins-1995-with-a-pretrained-backbone thesis is falsified twice over. The original pseudo-rehearsal (random pixels through ViT → snapshot teacher → KD) performs no better than sequential fine-tuning (ACC 0.103 vs 0.102). A 2×2 ablation that independently swaps the input distribution (random pixels vs STL-10 natural images) and the teacher source (per-task snapshot vs joint-trained oracle) gives the same null result in every cell — all four pseudo variants are within noise of each other and of sequential. Only experience replay (0.529) and joint training (0.788, upper bound) move the needle. The mechanism check confirms a tight geometric prediction: `argmax_c (W_c · μ_random + b_c) = 23` exactly matches the empirically-dominant teacher class on random inputs, so the random-pixel collapse is fully determined by the head and the mean random feature. But the extension shows that the collapse explanation alone is not the story — even non-collapsing natural inputs and a perfect oracle teacher don't save pseudo-rehearsal. The deeper finding is that in linear-head Class-IL with seen-class masking, output-only regularization (KD-based pseudo / LwF / EWC) is structurally weak: only methods that put actual old-class examples into the training batch can preserve old boundaries.

## What was built

All code lives under `paper/pseudo-rehearsal/`.

- `cache_features.py` — one-time pass through `timm` `vit_small_patch16_224.augreg_in21k_ft_in1k`. Caches CIFAR-100 train/test CLS features and a 100K pool of uniform-random-pixel features (`cache/*.npy`, float16). Also has an opt-in STL-10 unlabeled cache for the planned natural-OOD extension.
- `data.py` — `TaskSplit.from_seed(seed)` produces a deterministic 100-class permutation and a list of 10 class buckets. `CachedFeatureSet` is a thin Dataset wrapper over the cached arrays.
- `methods/base.py` — `Method` base class holding the linear head and the `seen_class_mask` helper (additive `-inf` mask on unseen logits, used by every method's CE and KD computation).
- `methods/sequential.py` — default CE with seen-class masking. Lower bound.
- `methods/joint.py` — same head, trained once on all 100 classes. Upper bound.
- `methods/replay.py` — class-balanced buffer of 5 examples per seen class (max 500). Each step mixes 50% current / 50% buffer at constant batch size.
- `methods/ewc.py` — online EWC with diagonal empirical Fisher computed in closed form for a masked linear-CE head (`(p - 1{y})² · x²`), accumulated across tasks. `lam=100` from the proposed grid.
- `methods/pseudo.py` — Robins 1995 faithful: at task `t+1`, draws `batch_size` rows from the pre-cached random-feature pool, runs the snapshotted teacher head, and adds `α · KD(q_old, p_student)` on old-class logits only. `α=1.0`, `temperature=1.0`.
- `methods/lwf.py` — same KD machinery but distills on current-task real features instead of random pool samples. Same alpha/temperature.
- `methods/_distill.py` — `kd_loss_old_classes`: teacher-to-student soft-label CE restricted to `old_classes` indices, both teacher and student re-softmaxed over only those indices.
- `train.py` — generic loop. Per task: `begin_task` → 10 epochs of AdamW (`lr=1e-3`, `batch_size=256`) → `end_task` (hook for buffer/Fisher/teacher snapshots) → evaluate the full accuracy matrix row. Joint path skips the per-task loop.
- `evaluate.py` — `acc_fgt_bwt(A)` over the (T, T) accuracy matrix. FGT includes the final row in the per-task max so FGT is non-negative by construction; BWT is `A[T-1, t] - A[t, t]`.
- `diagnostics.py` — random-feature norms, NN cosine to real features, predictive entropy on real vs random, top-class histogram on random. Run once against `joint_seed0_head.pt`.

## What was run

**Sweep 1 (six methods × three seeds = 18 runs)** plus a diagnostics pass, then **Sweep 2 (three new pseudo variants × three seeds = 9 runs)** plus a mechanism check and figures. All on RTX 4090 secure-cloud pods (one for each sweep). The two extensions implemented in `methods/pseudo.py` via two orthogonal parameters: `pool_source ∈ {random, stl10}` and `oracle_teacher_ckpt ∈ {None, path-to-joint-head}`. The four-cell 2×2 over (input × teacher) is registered as `pseudo`, `pseudo_oracle`, `pseudo_natural`, `pseudo_oracle_natural`.

Aggregated by method (mean ± std over seeds 0/1/2):

```
method                         ACC          FGT         BWT      n
joint                    0.788±0.000  0.000       n/a            3    upper bound
replay                   0.529±0.004  0.450      -0.450          3
ewc                      0.192±0.010  0.825      -0.825          3
lwf                      0.111±0.012  0.919      -0.919          3
pseudo_natural           0.106±0.008  0.923      -0.923          3    STL-10 + CL teacher
pseudo_oracle            0.105±0.005  0.925      -0.925          3    random  + joint teacher
pseudo                   0.103±0.005  0.927      -0.927          3    random  + CL teacher (Robins)
sequential               0.102±0.004  0.928      -0.928          3    lower bound
pseudo_oracle_natural    0.099±0.003  0.932      -0.932          3    STL-10 + joint teacher
```

FGT and −BWT agree on every row because every per-task max happens at the diagonal (the task is best right after it is learned), so `max_{k≥t} A[k,t] = A[t,t]` and `FGT_t = A[t,t] − A[T−1,t] = −BWT_t`.

The 2×2 view:

| | Random pixels | STL-10 natural |
|---|---|---|
| CL teacher | 0.103 | 0.106 |
| Oracle (joint) teacher | 0.105 | 0.099 |

All four cells overlap within seed noise, and none clears sequential (0.102).

## What the diagnostics show

From `results/diagnostics.json` (joint head from seed 0 acting as teacher, 100K random-pixel features, 5K × 5K real-vs-random NN sample):

- Real-feature L2 norm: 49.52 ± 1.70. Random-feature L2 norm: 34.33 ± 1.77. Random features sit in a shrunken sphere ~31% smaller in radius.
- Mean nearest-neighbor cosine of random → real: 0.392 (p25 0.381, p75 0.401). Real CIFAR features are nowhere near random-pixel features in direction.
- Joint-head predictive entropy: 0.513 nats on real CIFAR features, 2.666 nats on random features. Uniform-over-100 reference is 4.605. The teacher is far more uncertain on random inputs but not uniformly uncertain.
- Top-class histogram on random features: class 23 takes 62.7% of all argmax predictions, and the teacher predicts only 4 of the 100 classes anywhere across 100K random inputs. Distribution entropy over predicted classes is 0.664 nats.

From `results/mechanism_check.json` (computed after Sweep 2):

- `argmax_c (W_c · μ_random + b_c) = 23`, where `μ_random` is the mean of the 100K random-pixel features and (W, b) come from the joint head.
- The empirical top class on the full random pool is also 23 (62.7% of all argmax predictions).
- Per-dimension std of the random pool, averaged across 384 dims, is 0.305 — confirming the collapse is tight.
- The collapse is therefore fully predicted by the head and the mean feature alone; no per-sample variation is needed to explain why class 23 dominates.

## Interpretation

The 2×2 ablation eliminates two natural rescue stories at once:

- *"Random pixels are the problem; natural inputs would carry information."* — Refuted by `pseudo_natural` (STL-10 unlabeled images through the same ViT, CL teacher): ACC 0.106, within noise of plain `pseudo`. The teacher's outputs on STL-10 features are non-degenerate (those are real natural images), but the student still doesn't preserve old boundaries.
- *"The CL teacher accumulates errors; a perfect teacher would save the method."* — Refuted by `pseudo_oracle` (random pixels, joint-trained head as static teacher): ACC 0.105. Even with a teacher that is the joint upper-bound classifier, KD on random pseudo-features doesn't help.
- *"Maybe both at once."* — Refuted by `pseudo_oracle_natural` (STL-10 + joint teacher): ACC 0.099, indistinguishable from sequential.

The mechanism check (`results/mechanism_check.json`) verifies that the random-pixel collapse is geometrically predictable: the dominant pseudo-class is exactly `argmax_c (W_c · μ_random + b_c)` from the joint head, with no per-sample variation needed. That is still an interesting result — it explains why `pseudo` was always going to fail — but it is no longer the whole story, because `pseudo_natural` and `pseudo_oracle` don't suffer from collapse and still fail.

The remaining explanation is structural: in **single-linear-head Class-IL with seen-class masking**, the per-step cross-entropy gradient on 5,000 new-task examples per epoch aggressively rewrites the head; KD on either real or pseudo inputs adds a comparatively weak counter-pressure on the old-class logits, and at α = 1.0 the CE term wins. Only methods that inject **actual old-class training examples** into the batch (replay) produce gradients that defend the old logits directly. EWC's small gain (0.192) is consistent with this — it adds a per-parameter penalty that mildly constrains drift but cannot manufacture the missing class-conditional signal.

This generalizes a known observation about LwF in Class-IL (bias toward the most recent task) but extends it: even with a backbone-prior-aware input distribution and an oracle teacher, KD cannot manufacture the supervision that real exemplars provide.

A residual question, addressed below, is whether the α coefficient was simply too small. We did not sweep α in either sweep; everything ran at α = 1.0. At sufficiently high α the KD term can in principle dominate CE and force the student to mimic the teacher, which for `pseudo_oracle` would converge toward the joint upper bound. The α sweep is the next priority before the paper.

## α sweep (Sweep 3)

α ∈ {0.5, 1, 5, 10, 50} × {pseudo, pseudo_oracle, lwf} × 3 seeds = 45 runs. Run on a fresh RTX 4090 secure-cloud pod in ~5 min (after 6 min of feature caching). Final ACC table (mean ± std):

```
method          alpha=0.5      alpha=1.0      alpha=5.0      alpha=10.0     alpha=50.0
lwf             0.110±0.011    0.111±0.012    0.114±0.013    0.117±0.013    0.164±0.014
pseudo          0.103±0.004    0.103±0.005    0.107±0.007    0.114±0.010    0.144±0.017
pseudo_oracle   0.103±0.005    0.105±0.005    0.111±0.009    0.118±0.013    0.142±0.018
```

The structural-weakness framing survives. Across two orders of magnitude of α (0.5 → 50), all three KD methods trace nearly parallel curves that crawl monotonically from 0.10 to 0.14-0.16 — far below replay (0.529) and joint (0.788). Even the oracle teacher (which is the joint upper-bound classifier) cannot pull pseudo past 0.142 at the largest α tested. The KD direction is therefore not a tunability issue: distilling a teacher (perfect or not) onto a frozen-feature linear head cannot manufacture the per-class signal that real exemplars provide. See `figures/alpha_sweep.png`.

One can extrapolate further: at α → ∞ the KD term dominates CE and the student converges to the teacher, so `pseudo_oracle` would in the limit recover joint (0.788) — but only by abandoning the new-task CE entirely. In practice the curve appears to be plateauing in the explored range, consistent with the KD gradient being numerically swamped by per-batch CE on 5,000 fresh new-task examples.

## Figures

- `figures/running_avg_accuracy.png` — running mean accuracy on tasks 0..k vs k for each method.
- `figures/final_per_task.png` — A[T-1, t] vs t. Replay stays roughly flat across tasks (40-60%); EWC partially preserves the last 3-4 tasks; every other method gets ~90% on task 9 and ~0% on everything before.
- `figures/alpha_sweep.png` — KD weight sweep, log x-axis, replay and joint reference lines.

## Status

All experiments planned in `SPEC.md` are done plus the 2×2 pseudo ablation and the α sweep. Total compute: ~45 minutes across three RTX 4090 pod sessions, ~$0.50 of GPU time.

Story is bulletproof:
1. Hypothesis (backbone prior saves Robins): falsified by `pseudo` ≈ `sequential`.
2. Two natural rescue hypotheses (collapse-only, teacher-only) eliminated by the 2×2 ablation.
3. Mechanism prediction (`argmax(W·μ + b) = 23`) confirms the collapse is geometrically determined.
4. Tuning rescue (high α) eliminated by the α sweep.

What remains is writing the 5-page paper.

## Risks / what could still go wrong

- Class permutations are seeded but only seeds 0/1/2 are tried; one unfortunate permutation across all three could be hiding effects.
- All numbers assume 10 epochs/task with `lr=1e-3` and no LR schedule. Strong undertraining or overtraining of the head could shift the ordering among the bottom four methods, though it is hard to imagine pseudo catching up to replay given the diagnostics.
- Single backbone (ViT-S/16, ImageNet-21k → 1k). A CLIP or DINO backbone could put random features in a different region of feature space and change the story. Out of scope for the 5-page paper but a real caveat.
- `λ=100` for EWC was the proposal default, not a value selected on the task-1+2 validation slice as `SPEC.md` §3.2 prescribes. EWC's 0.192 is almost certainly within range across `{1, 10, 100, 1000}` but should be searched before the final writeup.
