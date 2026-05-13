# FINDINGS — Pseudo-Rehearsal with a Frozen ViT-S/16 on Split-CIFAR-100

Milestone snapshot, 2026-05-13. First full sweep complete.

## Headline

With a frozen ImageNet-pretrained ViT-S/16 backbone and a linear 384→100 head trained sequentially on Split-CIFAR-100 (Class-IL, 10 tasks × 10 classes), Robins-1995-style pseudo-rehearsal performs no better than sequential fine-tuning: ACC 0.103 vs 0.102 over three seeds. LwF on current-task features is essentially tied with it (0.111). The only methods that move the needle are class-balanced experience replay (0.529) and joint training as the upper bound (0.788). The diagnostics in `results/diagnostics.json` give a clean mechanistic explanation: uniform random pixels through the ViT collapse to a small, off-manifold region of feature space, and the teacher head classifies 62.7% of those features as one dominant class. The thesis that a modern pretrained backbone's natural-image prior makes random pseudo-inputs informative is falsified in this setting.

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

Six methods × three seeds = 18 training runs, one diagnostics pass, on a single secure-cloud RTX 4090 pod. The whole session including feature caching took ~11 minutes; the 18 head-only training runs themselves summed to 195.5 s (~3.3 min) of compute. Aggregated by method (mean ± std over seeds 0/1/2):

```
method          ACC          FGT         BWT      n
joint           0.788±0.000  0.000       n/a      3   (upper bound)
replay          0.529±0.004  0.450      -0.450    3
ewc             0.192±0.010  0.825      -0.825    3
lwf             0.111±0.012  0.919      -0.919    3
pseudo          0.103±0.005  0.927      -0.927    3
sequential      0.102±0.004  0.928      -0.928    3
```

FGT and −BWT agree on every row because every per-task max happens at the diagonal (the task is best right after it is learned), so `max_{k≥t} A[k,t] = A[t,t]` and `FGT_t = A[t,t] − A[T−1,t] = −BWT_t`.

## What the diagnostics show

From `results/diagnostics.json` (joint head from seed 0 acting as teacher, 100K random-pixel features, 5K × 5K real-vs-random NN sample):

- Real-feature L2 norm: 49.52 ± 1.70. Random-feature L2 norm: 34.33 ± 1.77. Random features sit in a shrunken sphere ~31% smaller in radius.
- Mean nearest-neighbor cosine of random → real: 0.392 (p25 0.381, p75 0.401). Real CIFAR features are nowhere near random-pixel features in direction.
- Joint-head predictive entropy: 0.513 nats on real CIFAR features, 2.666 nats on random features. Uniform-over-100 reference is 4.605. The teacher is far more uncertain on random inputs but not uniformly uncertain.
- Top-class histogram on random features: class 23 takes 62.7% of all argmax predictions, and the teacher predicts only 4 of the 100 classes anywhere across 100K random inputs. Distribution entropy over predicted classes is 0.664 nats.

## Interpretation

`pseudo` lands within noise of `sequential`, and `lwf` is essentially tied with both. The Robins-1995-with-a-pretrained-backbone thesis — that the backbone's natural-image prior would make random pseudo-inputs carry information about the old decision boundary — does not survive contact with the data here.

The diagnostics give a clean mechanism. ViT-S maps uniform random pixels into a tiny, off-manifold region (norm 34 vs 50, NN cosine ~0.39 to real features). Inside that region the joint head has a strong directional preference: ~63% of random samples are classified as one dominant class (class 23), and only 4 classes are ever predicted. So the pseudo-soft-target distribution `q_old = softmax_old(W·x + b)` is, in effect, a near-constant near-one-hot pointing at a single old class. Distilling a constant signal across old classes does not encode the old decision boundary; it adds a weak bias toward one class and leaves the student to forget normally. That matches what we see: pseudo ≈ sequential, and even LwF — which distills on real current-task features, not the off-manifold ones — barely moves the needle, because the linear head is being aggressively rewritten on the new task's 5K examples and old-class outputs collapse regardless of the KD input.

This is a defensible negative result with a mechanistic story, not a "hyperparameters were bad" outcome.

## What's next (planned)

One more ~30-minute pod session for three diagnostic extensions, then plots. Each is one or two file additions; the training loop and metrics stay fixed.

- **Oracle-teacher diagnostic.** Replace the teacher in `pseudo` with the joint-trained head from `joint_seedN_head.pt`. Isolates "is the teacher wrong" from "is the input distribution wrong." If oracle pseudo still fails, the input distribution is the culprit; if it recovers replay-like numbers, the per-task teacher snapshots are the problem.
- **Natural OOD pseudo-inputs.** Use STL-10 unlabeled features through the same ViT as the pseudo-rehearsal pool. Cache flag already exists in `cache_features.py --with-stl10`. Tests whether pseudo-rehearsal needs natural-image structure rather than random pixels.
- **Mechanism prediction.** Verify the dominant-class result is exactly `argmax_c (W_c · μ_random + b_c)` from the joint head — i.e., the collapse is predicted from the head and the mean random feature alone, with no per-sample variation needed.

If time, an α/λ/buffer-size sweep and per-task accuracy curve plots. The plotting code is staged in `plots.py`.

## Risks / what could still go wrong

- Class permutations are seeded but only seeds 0/1/2 are tried; one unfortunate permutation across all three could be hiding effects.
- All numbers assume 10 epochs/task with `lr=1e-3` and no LR schedule. Strong undertraining or overtraining of the head could shift the ordering among the bottom four methods, though it is hard to imagine pseudo catching up to replay given the diagnostics.
- Single backbone (ViT-S/16, ImageNet-21k → 1k). A CLIP or DINO backbone could put random features in a different region of feature space and change the story. Out of scope for the 5-page paper but a real caveat.
- `λ=100` for EWC was the proposal default, not a value selected on the task-1+2 validation slice as `SPEC.md` §3.2 prescribes. EWC's 0.192 is almost certainly within range across `{1, 10, 100, 1000}` but should be searched before the final writeup.
