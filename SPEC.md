# Paper Spec — Revisiting Pseudo-Rehearsal with Pretrained Backbones

**Author:** Dexin Huang (dh3172)
**Course:** COMS4995W Applied Deep Learning
**Due:** ~2026-05-15 (final exam day)
**Today:** 2026-05-13
**Status:** Revised after Codex review on 2026-05-13

---

## 1. Thesis (revised, narrower)

Robins's pseudo-rehearsal (Connection Science, 1995) generates random input vectors, runs them through the old network for pseudo-targets, and mixes those into new-task training — no stored data, no generative model. It dropped out of the literature because random inputs to a small from-scratch MLP are off-manifold and the pseudo-targets carry little information about the old decision boundary.

This paper asks a **narrow empirical question**: in the *frozen-pretrained-backbone* regime that dominates modern continual learning (L2P, DualPrompt, etc.), does Robins's original recipe — uniform random pixels → frozen pretrained backbone → old head → pseudo-soft-targets — recover most of the benefit of methods that require either real past data (experience replay) or current-task distillation (LwF)?

The stronger story — *"the backbone's natural-image prior makes random inputs informative"* — is a **hypothesis tested by the diagnostics in §3.7 and the LwF comparison in §3.6**, not asserted upfront.

**Research questions:**
1. **RQ1.** Does pseudo-rehearsal reduce forgetting on Split-CIFAR-100 vs sequential fine-tuning, with a frozen ImageNet-pretrained ViT-S/16 backbone?
2. **RQ2.** How does it compare to EWC, class-balanced experience replay, and LwF in the same setup?
3. **RQ3.** Do the diagnostics in §3.7 (random-feature norms, nearest-neighbor distance to real features, teacher entropy on random vs real features) support the claim that the backbone's prior makes random pseudo-inputs informative? Or are the gains explained more simply by output regularization?

---

## 2. Setting

### 2.1 Dataset

- **Split-CIFAR-100**, 10 tasks × 10 classes each.
- **Class permutation is part of the seed**: each seed defines an independent random permutation of the 100 classes into 10 groups. Same permutations used across all methods.
- Standard CIFAR-100 train/test split (50,000 / 10,000); within each task, 5,000 train / 1,000 test.
- **Class-IL**: single shared 100-way output head, no task ID at test time.
- No data augmentation in head training (features are precomputed and cached — see §4.1).

### 2.2 Backbone

- **ViT-S/16** pretrained on ImageNet-21k, fine-tuned on ImageNet-1k.
- Source: `timm` model `vit_small_patch16_224.augreg_in21k_ft_in1k` (~22M params).
- **Frozen** throughout.
- Input: CIFAR-100 images upsampled 32→224 (bilinear), ImageNet normalization.
- Feature: 384-dim CLS token.

### 2.3 Classifier head

- Single linear layer `Linear(384, 100)`. ~38.5K params.
- Bias included; no extra weight decay beyond what each CL method specifies.
- Rationale: linear head keeps the comparison clean. Differences across methods are about CL strategy, not head capacity. Stated as a deliberate scope choice.

---

## 3. Methods

All methods train the same 384→100 linear head sequentially across 10 tasks. Backbone stays frozen. **Cross-entropy and distillation losses are masked to seen classes only** — never train on unseen-class logits as implicit negatives (Codex flag).

### 3.1 Sequential (lower-bound baseline)

- Task `t`: CE over current task's 10 classes plus previously-seen classes only. Logits for not-yet-seen classes are masked out of the softmax.
- No regularization, no replay.

### 3.2 EWC (Kirkpatrick et al., 2017)

- After task `t`, compute diagonal Fisher `F_t` on task `t` data with the same seen-class masking.
- Online EWC: accumulate `Σ F_t · (θ − θ*_t)²`, single running estimate.
- `λ` selected by grid `{1, 10, 100, 1000}` on task-1+2 validation slice; single value used across all 10 tasks and 3 seeds.
- Noted as a weak baseline in this setup (~38K linear params; EWC's representation-preservation argument doesn't apply).

### 3.3 Class-balanced experience replay

- Buffer: **500 examples = 5 per class**, balanced (not reservoir — Codex flag).
- After each task, sample 5 examples per just-seen class and store features + labels.
- Each training batch: 50% new data, 50% sampled uniformly from buffer.

### 3.4 Pseudo-rehearsal (Robins 1995, faithful)

- After task `t`, snapshot head as **teacher** `h_t`.
- For task `t+1`:
  - Per training step, sample `N = batch_size` random images: pixels iid uniform [0, 1], ImageNet-normalized.
  - Random images → frozen backbone → pseudo-features → teacher `h_t` → pseudo-soft-targets `q_old`.
  - Loss = `CE_seen(new data) + α · KD(q_old, p_student)` where `p_student` is the student's softmax restricted to **previously-seen classes only** (mask current-task and unseen classes — Codex flag).
  - `KD` is soft-label CE: `−Σ q_old · log p_student` (teacher-to-student direction, not reverse KL).
- After task `t+1`, teacher becomes `h_{t+1}`.
- `α` selected from `{0.1, 0.5, 1.0, 5.0}` on task-1+2 validation slice.
- **No N-sweep ablation** (cut per Codex — not load-bearing for 5-page paper).

### 3.5 LwF — Learning without Forgetting (Li & Hoiem, 2017)

- Same as §3.4 but distill teacher on **current-task real features** instead of random-pixel features.
- This is the cleanest comparator for the pretrained-backbone-prior hypothesis: if §3.4 ≈ §3.5, the gain is from output regularization, not the backbone's prior. If §3.4 > §3.5, random pixels through ViT really do produce informative pseudo-targets that current data alone doesn't.
- Same `α` grid as §3.4.

### 3.6 Joint training (upper-bound baseline)

- Train the head on all 100 classes at once with standard CE.
- One run per seed (no continual aspect).
- Tells the reader how much accuracy is even available with this linear-head-on-frozen-features setup.

### 3.7 Diagnostic probes (mechanism check)

Computed once per seed, no training — fast, single-script:

- Real CIFAR feature norm (mean ± std) vs random-pixel feature norm.
- Nearest-neighbor cosine distance from random-pixel features to the real CIFAR train feature set.
- Teacher entropy on real old-task features vs random-pixel features.
- Teacher top-class histogram on random-pixel features (does the teacher collapse onto one class? uniformly distributed? near old-class distribution?).

If random features collapse to one region or the teacher's outputs are near-uniform, the strong thesis is dead — paper reframes as "pseudo-rehearsal works as output regularization, not via the prior."

---

## 4. Implementation

### 4.1 Feature caching

- Once: forward all 60K CIFAR-100 images through frozen ViT-S → save `(N, 384)` float16 tensor (~50 MB).
- Once: pre-compute a pool of 100K random-pixel features (~150 MB float16). Pseudo-rehearsal samples from this pool — removes the backbone from the training hot path.
- All training is then head-only on cached features. Each run is seconds, not minutes.

### 4.2 Hyperparameters (defaults, same across methods unless noted)

- Optimizer: AdamW, `lr=1e-3`, `weight_decay=0`, `betas=(0.9, 0.999)`.
- Batch size: 256.
- Epochs per task: 10.
- **3 seeds (0, 1, 2)** controlling: class permutation, dataloader order, buffer sampling, pseudo-sample draw. Same permutations across methods so paired comparisons are valid.
- Method-specific search: §3.2 `λ`, §3.4/§3.5 `α` — searched on a task-1+2 validation slice, then fixed.

### 4.3 Metrics

Maintain an **accuracy matrix** `A[k][t]` = test accuracy on task `t`'s classes after finishing task `k`, for `k, t ∈ {0..9}`, `t ≤ k` (only seen tasks). Evaluation uses the full 100-way head with argmax restricted to seen classes at that point.

After all 10 tasks:

- **Average Accuracy (ACC).** `(1/10) Σ_t A[9][t]`.
- **Forgetting (FGT).** `(1/9) Σ_{t<9} (max_{k≥t} A[k][t] − A[9][t])`.
- **Backward Transfer (BWT).** `(1/9) Σ_{t<9} (A[9][t] − A[t][t])`.

Report mean ± std over 3 seeds. Also report a per-task accuracy curve (mean over seeds).

### 4.4 Code layout

All work lives under `paper/pseudo-rehearsal/`. **Do not touch sibling directories** (`fem-wm/`, `forgotten-primitives/`) — they are abandoned earlier directions.

```
paper/pseudo-rehearsal/
  SPEC.md                  # this file
  README.md                # how to reproduce
  requirements.txt
  .gitignore
  cache_features.py        # one-time: CIFAR-100 + random-image features
  data.py                  # CachedFeatureDataset, task splitter, class-perm seed
  methods/
    base.py                # shared training loop interface
    sequential.py
    ewc.py
    replay.py
    pseudo.py
    lwf.py
    joint.py
  diagnostics.py           # §3.7 probes
  train.py                 # generic loop: takes a method config, runs 10 tasks
  evaluate.py              # accuracy matrix → ACC / FGT / BWT
  configs/
    sequential.yaml
    ewc.yaml
    replay.yaml
    pseudo.yaml
    lwf.yaml
    joint.yaml
  run_all.sh               # 6 methods × 3 seeds = 18 runs
  results/                 # JSON per run + aggregated CSV + plots
  cache/                   # gitignored: cached feature tensors
```

Plain PyTorch, no Avalanche dependency.

---

## 5. Paper structure (5 pages)

1. **Introduction** (~0.75 p) — catastrophic forgetting; Robins's pseudo-rehearsal; why it was dismissed; modern pretrained-backbone CL; the narrow empirical question.
2. **Background** (~0.75 p) — Robins 1995 explained; EWC; experience replay; LwF; generative replay in 2-3 sentences (not implemented; data-free model-inversion methods like ABD discussed briefly).
3. **Method** (~1.0 p) — exact pseudo-rehearsal procedure with the frozen backbone; explicit on random-pixel sampling, KD direction, seen-class masking.
4. **Experiments** (~1.75 p) — Split-CIFAR-100 setup, all six methods, hyperparameter selection, ACC/FGT/BWT table, per-task accuracy curve, **diagnostic table from §3.7**.
5. **Discussion & limitations** (~0.5 p) — what the diagnostics imply; frozen-backbone scope; Class-IL only; single dataset; head capacity.
6. **References** (~0.25 p).

---

## 6. Compute

- Real-feature cache: ~5 min on T4, ~2 min on A100.
- Random-feature pool (100K): ~10 min on T4.
- One training run (head-only, 10 tasks × 10 epochs): <1 min.
- All 18 runs + small grid searches: <1 hour.
- Hardware: RTX 2060 local can handle the cache (small backbone, batch through one pass). RunPod T4 for headroom and speed.

---

## 7. Realistic timeline (revised per Codex, ~34 hours)

| Hours | Block |
|---|---|
| 0-2 | Env setup, CIFAR-100 + ViT-S download, cache_features.py |
| 2-5 | data.py with class-permutation seed, sequential + joint baselines, accuracy matrix wiring |
| 5-8 | Class-balanced replay, EWC |
| 8-11 | Pseudo-rehearsal + LwF with seen-class-masked KD |
| 11-13 | Diagnostic probes (§3.7), one-seed smoke test all methods |
| 13-17 | Fix mask/metric bugs, small `λ`/`α` searches |
| 17-22 | Final 3-seed runs across all methods |
| 22-25 | Tables, forgetting curves, diagnostic table |
| 25-34 | 5-page writeup |

Tue 5/13 evening → Wed 5/14 evening: code + runs done. Thu 5/15: writeup + submit. Write while runs execute when possible.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Sequential FT forgets less than expected on linear head | §3.6 joint baseline reveals the actual sequential-vs-joint gap. If gap < 5pts, the playing field is too weak — adjust framing to "narrow-margin study." |
| EWC weak in linear-head regime | Acknowledge in §5 as scope-defining choice. Story angle: pseudo-rehearsal and LwF are the methods whose mechanism actually depends on the backbone. |
| Random-pixel features OOD junk | Exactly what §3.7 diagnostics test. If yes, reframe as a clean negative result with diagnostics. |
| Pseudo-rehearsal ≈ LwF | Then the gain is output regularization, not the prior. Honest result — paper says so. |
| Timeline slips | Cut LwF first (still defensible without it), then ablation, then EWC. Sequential + joint + replay + pseudo + diagnostics is the minimum-defensible-paper set. |
| 5/15 deadline ambiguous | `assignment.md` says "around 5/15" — confirm exact date on Ed if not posted. |

---

## 9. Out of scope (explicitly)

- Fine-tuning the backbone (changes the comparison; future work).
- Datasets beyond Split-CIFAR-100.
- Generative replay (DGR / ABD / DeepInversion) — discussed briefly in §2 background; not implemented.
- Task-IL with per-task heads — trivializes forgetting.
- N-sweep ablation over pseudo-sample count (cut per Codex).

---

## 10. Minimum defensible paper

If we run out of time, the cuts go in this order:
1. LwF (§3.5) — paper is still publishable comparing pseudo-rehearsal to sequential / joint / replay / EWC.
2. EWC (§3.2) — paper still works with sequential / joint / replay / pseudo + diagnostics.
3. Diagnostic probes (§3.7) — paper degrades to "applied comparison" without mechanism story.

Floor: sequential + joint + replay + pseudo, 3 seeds, ACC/FGT/BWT table, 5 pages.
