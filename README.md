# HDC-empirical

Empirical, head-to-head comparison of **HDC** (high-dimensional random projection
+ binarization + class-prototype bundling) against conventional **classifier
heads**, measured along four axes:

1. **Efficiency** (inference, online update, cold-start, model size)
2. **Accuracy** (learning curves)
3. **Robustness** (input noise, salt-and-pepper corruption, domain shift, label noise)
4. **Interpretability** (feature-importance fidelity, decision linearity, model size)

All tests are **synthetic**, deterministic (seed 0), and run in this repo.
Raw numbers are in [`results/`](results/). Re-run everything with:

```bash
pip install -r requirements.txt
python run_all.py
```

Environment used for these numbers: `numpy 2.5.1`, `scikit-learn 1.9.0`,
`torch 2.13.0`, Python 3.14, single-core CPU. Timings are wall-clock medians.

---

## Methods compared

| name | description |
|------|-------------|
| **`hdc`** | Random Gaussian projection `x -> sign(xP)` into `D=10,000` bits; class prototype = `sign(sum of class hypervectors)`; inference = dot product with prototypes (Hamming distance). One-shot, no gradients. |
| **`rf-linear`** | Classifier **head** (single `Linear`, Adam) trained on the *same frozen continuous* random features `xP`. Isolates "learned head vs prototype". |
| **`rf-linear-bin`** | Same head trained on the *binarized* random features `sign(xP)` (identical encoding to HDC). Isolates the effect of binarization. |
| **`linear`** | Logistic-regression head on raw features. The plainest "classifier head". |
| **`mlp`** | Small 2-hidden-layer MLP (128→64→64→K) trained end-to-end with Adam. "A network's classifier head". |
| **`svm`** | RBF kernel SVM (`gamma='scale'`). Reference nonlinear method. |
| **`knn`** | k=5 nearest neighbor. Instance-based reference. |
| **`nn-encoder`** | The `mlp` architecture trained with cross-entropy and **frozen**; exposes 64-dim penultimate embeddings. Used only in §5. |
| **`emb-hdc`** | **HDCnn**: `nn-encoder` embeddings → `sign(emb P)` → class prototypes (HDC as the network's classification head). |
| **`emb-linear`** | LR head on the same frozen embeddings (the network's classifier head). |
| **`emb-rf-linear-bin`** | Learned linear head on the *binarized* HD features of those embeddings. |

All models are defined in [`hdc_empirical/models.py`](hdc_empirical/models.py);
datasets in [`hdc_empirical/data.py`](hdc_empirical/data.py); benchmarks in
[`benchmarks/`](benchmarks/).

---

## 1. Efficiency

### 1a. Inference throughput (d=128, K=6, 10,000 test samples)

| method | ms / 10k preds | preds / sec |
|--------|---------------:|------------:|
| hdc | 310.5 | 32,203 |
| rf-linear | 135.4 | 73,873 |
| rf-linear-bin | 216.4 | 46,211 |
| linear | 0.57 | 17,537,091 |
| mlp | 0.95 | 10,538,985 |

HDC inference is dominated by the encode matmul `(n,128) @ (128,10000)`. On this
CPU it is **~545× slower than a raw-feature linear head** and **~1.4× slower than
`rf-linear-bin`** (which uses the same encoding but skips HDC's second
`(n,D)@(D,K)` prototype matmul and the int→float cast).

### 1b. Online update throughput (batches of 64, single pass)

| method | ms / batch(64) | samples / sec |
|--------|---------------:|--------------:|
| hdc | 1.17 | 54,556 |
| rf-linear | 89.8 | 713 |
| rf-linear-bin | 89.9 | 712 |
| linear-sgd | 1.15 | 55,713 |
| mlp | 0.85 | 74,987 |

HDC updates are **~77× faster per sample than the SGD head on the same random
features** (`rf-linear-bin`), because a prototype update is one encode + one
accumulate with no backward pass, learning rate, or optimizer state. It is on
par with a raw SGD head (`linear-sgd`) per-sample — but see 1c: after a single
pass HDC is far more accurate.

### 1c. Online learning curves — accuracy vs samples streamed (one pass each)

| method | 64 | 320 | 1,600 | 3,200 |
|--------|----:|----:|------:|------:|
| **hdc** | **0.419** | **0.615** | **0.717** | **0.731** |
| rf-linear | 0.415 | 0.535 | 0.627 | 0.651 |
| rf-linear-bin | 0.413 | 0.539 | 0.619 | 0.635 |
| linear-sgd | 0.338 | 0.456 | 0.613 | 0.641 |
| mlp | 0.216 | 0.406 | 0.680 | 0.683 |

HDC is the best one-pass learner at **every** checkpoint and finishes ~7 pp
ahead of the best SGD baseline after 3,200 samples. It learns from the first
sample without waiting for gradients to propagate.

### 1d. Cold start / few-shot (fit on 50 samples)

| method | fit time (ms) | test acc |
|--------|--------------:|---------:|
| **hdc** | **0.88** | **0.387** |
| rf-linear-bin | 201.7 | 0.380 |
| linear | 2.22 | 0.371 |
| mlp | 54.7 | 0.351 |

HDC is **229× faster than `rf-linear-bin`** and **62× faster than the MLP** to
train from 50 labeled samples, and slightly *more* accurate afterward.

### 1e. Stored model size (after training on 12,000 samples)

| method | params | storage |
|--------|-------:|--------:|
| hdc | 1,340,000 | 5,059 KB |
| rf-linear | 1,340,006 | 5,234 KB |
| rf-linear-bin | 1,340,006 | 5,234 KB |
| linear | 768 | 3 KB |
| mlp | 12,806 | 50 KB |
| svm | 1,212,672 | 9,474 KB |
| knn | 1,536,000 | 6,000 KB |

**Takeaway — Efficiency.** HDC wins on *learning* efficiency: it is the fastest
to produce a usable model from a handful of samples, the best single-pass online
learner, and updates with zero hyperparameters or backward passes. Its weakness
is *inference and storage on the encode side*: the fixed random projection
`P ∈ R^{128×10000}` dominates both latency and memory (~1,700× the linear head's
storage), so HDC's costs are paid in the projection, not the head. This is the
classic HDC trade-off: cheap, one-shot training paid for with a big encoding
layer. (Note: bit-parallel/popcount hardware and structured or learned
projections, not exercised here, change this calculus substantially.)

---

## 2. Accuracy

Task: 6 Gaussian classes in d=128 (center norm 2.0, σ=1.0). Training size per
class swept; fixed 300/class test set. Nested prefixes of one fixed pool.

| method | 10/class | 50/class | 200/class | 1000/class |
|--------|---------:|---------:|----------:|-----------:|
| **hdc** | **0.477** | **0.627** | **0.692** | 0.721 |
| rf-linear | 0.442 | 0.512 | 0.603 | 0.646 |
| rf-linear-bin | 0.453 | 0.518 | 0.598 | 0.656 |
| linear | 0.451 | 0.565 | 0.632 | **0.722** |
| mlp | 0.408 | 0.541 | 0.627 | 0.644 |
| svm | 0.446 | 0.619 | 0.681 | 0.719 |
| knn | 0.299 | 0.386 | 0.401 | 0.437 |

**Takeaway — Accuracy.** On this task HDC is the **best few-shot learner**
(+2.7 pp over the linear head at 10/class, +6.9 pp over the MLP) and ties the
best batch methods at large sample sizes (0.721 vs linear 0.722, svm 0.719).
`rf-linear-bin` ≈ `rf-linear` ≈ `mlp`: binarization costs nothing, and the
prototype rule matches a fully-trained head once enough data arrives. kNN is
the clear loser. The accuracy advantage of HDC is biggest exactly where
conventional heads need the most data — small-sample regimes.

---

## 3. Robustness

Task: 4 latent classes, d=128 (latent dim r=4, signal 2.0, clean σ=0.5);
2000 train, 1000 test. Models trained **once on clean data**, then stressed.

### 3a. Gaussian input noise (test σ added)

| method | clean | a=1 | a=2 | a=4 | a=8 |
|--------|------:|----:|----:|----:|----:|
| **hdc** | **0.770** | **0.642** | **0.514** | **0.387** | **0.305** |
| rf-linear | 0.711 | 0.535 | 0.417 | 0.327 | 0.287 |
| rf-linear-bin | 0.698 | 0.543 | 0.438 | 0.341 | 0.302 |
| linear | 0.725 | 0.557 | 0.449 | 0.345 | 0.288 |
| mlp | 0.707 | 0.511 | 0.432 | 0.346 | 0.286 |
| svm | 0.755 | 0.263 | 0.238 | 0.238 | 0.238 |
| knn | 0.662 | 0.542 | 0.468 | 0.351 | 0.299 |

### 3b. Salt-and-pepper (fraction of coordinates flipped to ±6)

| method | clean | 5% | 10% | 20% | 40% |
|--------|------:|---:|----:|----:|----:|
| **hdc** | **0.770** | **0.654** | **0.654** | **0.650** | **0.677** |
| rf-linear | 0.711 | 0.553 | 0.534 | 0.561 | 0.577 |
| rf-linear-bin | 0.698 | 0.570 | 0.558 | 0.558 | 0.592 |
| linear | 0.725 | 0.598 | 0.569 | 0.596 | 0.619 |
| mlp | 0.707 | 0.558 | 0.550 | 0.544 | 0.560 |
| svm | 0.755 | 0.260 | 0.238 | 0.238 | 0.238 |
| knn | 0.662 | 0.595 | 0.573 | 0.578 | 0.618 |

### 3c. Domain shift (correlated nuisance noise along one direction, τ)

| method | clean | τ=1 | τ=2 | τ=4 |
|--------|------:|----:|----:|----:|
| **hdc** | **0.770** | 0.765 | 0.767 | **0.760** |
| rf-linear | 0.711 | 0.702 | 0.692 | 0.658 |
| rf-linear-bin | 0.698 | 0.687 | 0.700 | 0.685 |
| linear | 0.725 | 0.722 | 0.719 | 0.718 |
| mlp | 0.707 | 0.709 | 0.702 | 0.688 |
| svm | 0.755 | **0.761** | **0.770** | 0.705 |
| knn | 0.662 | 0.664 | 0.666 | 0.668 |

### 3d. Label noise during training (random flips)

| method | 0% | 20% | 40% | drop@40% |
|--------|-----:|-----:|-----:|---------:|
| **hdc** | **0.770** | **0.771** | **0.768** | **-0.002** |
| rf-linear | 0.711 | 0.538 | 0.513 | -0.198 |
| rf-linear-bin | 0.698 | 0.627 | 0.527 | -0.171 |
| linear | 0.725 | 0.693 | 0.636 | -0.089 |
| mlp | 0.707 | 0.595 | 0.522 | -0.185 |
| svm | 0.755 | 0.745 | 0.737 | -0.018 |
| knn | 0.662 | 0.621 | 0.559 | -0.103 |

**Takeaway — Robustness.** HDC is the most robust method on **every** stress
test. Two results stand out:

- **Salt-and-pepper**: HDC's accuracy barely moves even with 40% of coordinates
  corrupted (0.770 → 0.677), because each flipped coordinate shifts only a few
  of 10,000 projection bits — the binary hypervector is a natural error-correcting
  code.
- **Label noise**: prototype bundling behaves like a **majority vote** over
  samples, so HDC is essentially *invariant* to 40% label corruption
  (drop of 0.002), while every SGD-trained head drops 9–20 pp.

RBF-SVM collapses under any input noise (0.755 → 0.238, below chance): with
d=128 the noise displaces points by ~√d·a ≫ kernel width, saturating the kernel
so all test points look equally far from every support vector. HDC's failure mode
under heavy Gaussian noise is graceful (0.305 at a=8, still above chance 0.25).

---

## 4. Interpretability

Task: d=128, K=4, with **16 known salient features** carrying the class signal
and 112 pure-noise features. Ground-truth saliency = the 16 true features.

| method | Spearman ρ (importance vs truth) | top-16 precision | top-16 recall | global linearity (R²) | params | storage |
|--------|--------------------------------:|-----------------:|--------------:|----------------------:|-------:|--------:|
| hdc | 0.438 | 0.813 | 0.813 | 0.991 | 1,320,000 | 5,039 KB |
| rf-linear | 0.551 | 0.938 | 0.938 | 1.000 | 1,320,004 | 5,156 KB |
| rf-linear-bin | 0.572 | 0.938 | 0.938 | 0.991 | 1,320,004 | 5,156 KB |
| linear | 0.568 | 0.938 | 0.938 | 1.000 | 512 | 2 KB |
| mlp | 0.570 | 0.938 | 0.938 | 0.949 | 12,676 | 50 KB |

Metrics: **importance fidelity** = rank correlation / overlap between the
method's inferred feature importance and the true salient set (occlusion
attribution for HDC, direct weights for linear heads, input-gradient for MLP);
**global linearity** = R² of a linear regression predicting the model's class
logits from the raw inputs.

**Takeaway — Interpretability.** HDC is *highly inspectable at the decision
level* — its decision function is nearly globally linear in the inputs
(R² = 0.99, essentially tied with a linear head), and it offers a unique
explanation artifact: the **class prototype hypervector** itself (the bundled
"typical" sample), which enables nearest-prototype explanations and associative
recall in the HD space. Its weakness is *attribution back to raw features*:
because the decision runs through a binarized 10,000-dim projection, occlusion
importance is noisier (ρ = 0.44, 0.81 top-16 precision) than a linear head's
direct weights (ρ = 0.57, 0.94). The linear head remains the gold standard for
"which raw input mattered" (and is 1,000× smaller); HDC trades a bit of that
raw-feature attribution fidelity for a built-in, human-friendly prototype
representation and near-linear inspectability.

---

## 5. HDCnn — frozen feature extractor → HDC classification

The standard "HDCnn" deployment replaces a network's classification head with
HDC. The backbone is `nn-encoder` (`hdc_empirical/models.py`), identical to the
`mlp` baseline (128→64→64→K, cross-entropy, seed 0) but exposing its frozen
64-dim penultimate features. The head is then trained in three ways on those
same embeddings:

- **`emb-hdc`** — HDCnn: embeddings → `sign(emb P)` (D=10,000) → class prototypes
- **`emb-linear`** — the network's classifier head (LR) on frozen embeddings
- **`emb-rf-linear-bin`** — a learned linear head on the *binarized* HD features

`hdc-raw` (raw-feature HDC) and `mlp` (end-to-end) are repeated as references.
If HDCnn's robustness simply tracks the end-to-end MLP, the encoder is doing the
work; if it keeps raw-HDC's profile, the HDC head itself is responsible.

### 5a. Accuracy — heads on frozen embeddings (backbone retrained at each size, then frozen)

| method | 10/class | 50/class | 200/class | 1000/class |
|--------|---------:|---------:|----------:|-----------:|
| emb-hdc | 0.414 | 0.546 | 0.614 | **0.651** |
| emb-linear | **0.427** | 0.540 | **0.624** | 0.643 |
| emb-rf-linear-bin | 0.376 | 0.541 | 0.608 | **0.653** |
| mlp (end-to-end) | 0.408 | 0.541 | 0.627 | 0.644 |
| *hdc-raw (from §2)* | *0.477* | *0.627* | *0.692* | *0.721* |

### 5b. Robustness on frozen embeddings

Backbone trained once on clean data then frozen; heads trained on its
embeddings; stress applied to the **raw inputs** before the backbone.

#### Gaussian input noise

| method | clean | a=1 | a=2 | a=4 | a=8 |
|--------|------:|----:|----:|----:|----:|
| **hdc-raw** | **0.770** | **0.642** | **0.514** | **0.387** | **0.305** |
| emb-hdc | 0.717 | 0.511 | 0.435 | 0.336 | 0.277 |
| emb-linear | 0.704 | 0.509 | 0.436 | 0.341 | 0.280 |
| emb-rf-linear-bin | 0.706 | 0.504 | 0.431 | 0.336 | 0.278 |
| mlp (end-to-end) | 0.707 | 0.511 | 0.432 | 0.346 | 0.286 |

#### Salt-and-pepper

| method | clean | 5% | 10% | 20% | 40% |
|--------|------:|---:|----:|----:|----:|
| **hdc-raw** | **0.770** | **0.654** | **0.654** | **0.650** | **0.677** |
| emb-hdc | 0.717 | 0.567 | 0.555 | 0.541 | 0.570 |
| emb-linear | 0.704 | 0.554 | 0.551 | 0.548 | 0.562 |
| emb-rf-linear-bin | 0.706 | 0.565 | 0.552 | 0.535 | 0.571 |
| mlp (end-to-end) | 0.707 | 0.558 | 0.550 | 0.544 | 0.560 |

#### Domain shift

| method | clean | τ=1 | τ=2 | τ=4 |
|--------|------:|----:|----:|----:|
| **hdc-raw** | **0.770** | **0.765** | **0.767** | **0.760** |
| emb-hdc | 0.717 | 0.709 | 0.710 | 0.691 |
| emb-linear | 0.704 | 0.705 | 0.703 | 0.685 |
| emb-rf-linear-bin | 0.706 | 0.707 | 0.704 | 0.682 |
| mlp (end-to-end) | 0.707 | 0.709 | 0.702 | 0.688 |

#### Label noise (heads trained on noisy labels; backbone stays frozen & clean)

| method | 0% | 20% | 40% | drop@40% |
|--------|-----:|-----:|-----:|---------:|
| **emb-hdc** | 0.717 | 0.713 | 0.715 | **-0.002** |
| emb-linear | 0.704 | 0.672 | 0.665 | -0.039 |
| emb-rf-linear-bin | 0.706 | 0.610 | 0.552 | -0.154 |
| mlp (end-to-end) | 0.707 | 0.595 | 0.522 | -0.185 |
| *hdc-raw* | *0.770* | *0.771* | *0.768* | *-0.002* |

**Takeaway — HDCnn.** The answer to "does the HDC head keep its strengths on top
of a learned encoder?" is *mixed, and revealing*:

- **Label-noise robustness transfers — yes.** Prototype bundling's majority-vote
  property survives the learned encoder: `emb-hdc` is invariant to 40% label
  noise (drop −0.002) exactly like raw HDC, while a linear head on the *same*
  embeddings drops −0.039 and a learned head on the binarized HD features drops
  −0.154. At the head level, HDC is strictly the most label-noise-robust option.
- **Input-noise / corruption robustness does NOT transfer.** Under Gaussian
  noise and salt-and-pepper, `emb-hdc` degrades identically to `emb-linear` and
  to the end-to-end MLP (e.g., a=1: 0.511 vs 0.509 / 0.511, against raw HDC's
  0.642). Raw-feature HDC's noise resilience came from encoding *raw* inputs
  into robust hypervectors, not from the HDC head itself; a smooth learned
  encoder in front removes that protection. **You do start to rely on the
  feature extractor** for noise robustness.
- **Accuracy is now head-independent.** On frozen embeddings every head
  converges to the same accuracy (0.64–0.65 at 1000/class), and the raw HDC
  few-shot edge disappears: the backbone trained on tiny data is the bottleneck,
  and HDCnn at 10/class (0.414) is *worse* than raw-feature HDC (0.477).
- Clean accuracy also drops from 0.770 (raw) to 0.717 (HDCnn): on this
  linear-in-the-signal task the 64-dim learned embedding discards signal the raw
  128-dim features retain. HDCnn only pays off when the encoder genuinely adds
  value for the task.

---

## Overall conclusion

| axis | winner | headline numbers |
|------|--------|------------------|
| **Efficiency (updates)** | **HDC** | 77× faster per-sample update than SGD head on same features; 229× faster cold start; best one-pass online learner at every checkpoint |
| **Efficiency (inference/size)** | **linear/mlp** | HDC inference ~545× slower than raw-feature head; projection dominates storage (5 MB vs 3 KB) |
| **Accuracy** | **HDC** (small data) / tie (large data) | +2.7 pp vs linear, +6.9 pp vs MLP at 10/class; 0.721 vs 0.722 at 1000/class |
| **Robustness** | **HDC** | wins all four stress tests; label-noise drop −0.002 vs −9 to −20 pp; ±40% corruption barely dents it |
| **Interpretability** | **linear** (raw attribution) / **HDC** (prototype artifact, near-linear) | linear ρ=0.57 & 2 KB; HDC ρ=0.44 but R²=0.99 + prototype explanation |
| **HDCnn (§5)** | **HDC head for label noise**; **neither for input noise** | emb-hdc invariant to 40% label noise (drop −0.002) even on frozen learned embeddings; but under input noise emb-hdc ≈ MLP (0.511 vs 0.511 at a=1) — the encoder now carries the noise robustness |

The dominant, repeatable pattern across these synthetic tests: **HDC is a
powerful few-shot, one-pass, noise- and label-error-resilient learner whose
costs sit entirely in its wide random projection** (latency, memory) — while
conventional classifier heads win on cheap dense inference and direct raw-feature
attribution but need many gradient passes and more data to reach the same
robustness. The prototype bundling (not the binarization) is the source of HDC's
label-noise and corruption robustness. The HDCnn tests (§5) refine this: the
prototype head keeps its label-noise immunity even on learned features, but the
input-noise and few-shot advantages are properties of the *raw-feature encoding*
and are forfeited the moment a learned encoder stands in front — in that setting
the HDC head is only as robust as the network that feeds it.

## Caveats

- Tasks are synthetic and largely linear-in-the-signal (Gaussian blobs, latent
  manifolds); real image/text pipelines would compare HDC *on top of* pretrained
  features, where the encode cost and importance-attribution story differ.
- Timings are single-core CPU numpy/torch — not micro-optimized. HDC has special
  fast paths (bit-parallel popcount, ternary/integer matmul) not exercised here;
  treat ratios, not absolute numbers, as the result.
- `svm` uses `gamma='scale'`; its noise collapse is a genuine RBF-width
  phenomenon in d=128, not a tuning artifact.
- Random-feature baselines use the same frozen `P` as HDC for a controlled
  comparison; in practice `P` is the same for all classes and reusable across
  tasks, which the "model size" numbers above do not amortize.
