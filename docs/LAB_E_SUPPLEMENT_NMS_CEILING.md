# Lab E 补充实验：NMS 与流水线收益上限

## Bonus Lab E — NMS and the ceiling of the async pipeline speedup

> **Student-contributed supplement** for `notebooks/05_Lab_E_AMD_CPU_GPU_Pipeline.ipynb`.
> **Status: extra experiment.** Its numbers are deliberately kept out of the
> official v2.0.1 results table. The sealed campaign under
> `validation/v2.0.1/campaigns/c20261108_1108/` was read only and was not modified.

---

## Summary / 摘要

- From the sealed `*_records.json`, the CPU share is **20–23% (serial) / 26% (pipeline)**,
  giving an Amdahl ceiling of **1.25–1.30×**. In this implementation `postprocess`
  runs on the GPU consumer thread and is therefore **not hidden**, so the real
  ceiling is only **1.16–1.25×** — and the pipeline already reaches **98%** of that floor.
- Explicit `nms=False` switches the head from one2many `(1,84,5880)` + external NMS to
  one2one `(1,300,6)` NMS-free. `post_ms` drops **0.6957 → 0.2536 ms (−63.5%)** and
  throughput rises **+2.70% serial / +1.93% pipeline**, but the **pipeline gain is
  unchanged (1.190× → 1.182×)**: optimising a non-hidden stage lowers absolute latency,
  it does not increase the overlap benefit.
- **Cost, measured under the course protocol**: mAP50-95 **0.639839 → 0.634109
  (Δ −0.005730)**, and **90/100 images differ** in per-detection parity
  (mean |Δconf| 0.04207, max 0.59725). This is **not** a loss-free optimisation.
- Cross-check: the baseline mAP50-95 of **0.639839** reproduces the published
  PyTorch-CUDA-640 value **0.6398**, confirming the measurement scope matches the course.

---

## 1. Motivation / 背景

Notebook 05 §5b asks *"when is a CPU+GPU pipeline NOT worth it? (hint: if the CPU
stage were ~0, what is the maximum gain?)"*. This supplement answers that question
with the sealed stage records, then tests a tempting but incorrect intuition:
*"if we make postprocess cheaper, the pipeline should get faster"*.

---

## 2. Setup / 实验设计

**One factor changed.** Both sides use the same official `yolo26l.pt`, the same 100
fixed validation images, and the unmodified `course.pipeline.run_serial()` /
`run_pipeline()` (queue_depth 4). The only difference is the warm-up `predict()` call:

| Side | Warm-up call | Effect |
|---|---|---|
| baseline | `predict(source=..., imgsz=640, device="cuda")` | `args.nms=None` → `AutoBackend.end2end=False` → one2many head → external NMS |
| nms_false | same + `nms=False` | `args.nms=False` → `end2end=True` → one2one head → NMS-free |

The module-level `Yolo26lStages` class is swapped at runtime; **no course source file
is edited**.

**Protocol.** 3 independent interleaved rounds (`AB / BA / AB`, matching
`BENCHMARK_PROTOCOL.md`), 100 images per mode per round, course `_prewarm()` over
all images, `cuda_sync()` around every timed forward, raw per-image records kept,
published values are **medians across rounds**. GPU busy = 0% before the run.
mAP uses the course `map5095` scope (fixed educational val subset of 120 images,
FP32, batch 8, imgsz 640, CUDA), once per variant as in `run_campaign.py`.

Because `nms=False` changes the inference head branch, this is explicitly labelled
an **extra experiment** under the honesty rules, separate from the official fair
comparison.

---

## 3. Results / 结果

### 3.1 CPU share and speedup ceilings (sealed campaign, read-only)

Sums over 100 images, ms. `Amdahl` = `1/(1 − CPU share)` assumes all CPU stages can
be hidden; `real` = `Σ(all) / max(Σ(load+pre), Σ(infer+post))` for the actual
implementation, where post stays on the consumer thread.

| file | load | pre | infer | post | CPU share | Amdahl | real | floor reached |
|---|---|---|---|---|---|---|---|---|
| serial 111119 | 174 | 48 | 1046 | 89 | 22.9% | 1.298× | 1.196× | 82.6% |
| serial 111253 | 182 | 52 | 1129 | 89 | 22.3% | 1.286× | 1.192× | 82.9% |
| serial 111427 | 181 | 58 | 1350 | 99 | 20.0% | 1.250× | 1.165× | 84.9% |
| async_pipeline 111119 | 194 | 103 | 1075 | 87 | 26.3% | 1.357× | 1.255× | **98.1%** |
| async_pipeline 111253 | 191 | 105 | 1076 | 93 | 26.5% | 1.361× | 1.253× | **98.0%** |
| async_pipeline 111427 | 192 | 100 | 1093 | 87 | 25.7% | 1.346× | 1.247× | **98.1%** |

The pipeline has essentially no headroom left under the current design.

### 3.2 What `nms=False` actually changes (single-image probe)

| | baseline `nms=None` | `nms=False` |
|---|---|---|
| `AutoBackend.end2end` | False | True |
| head | one2many | one2one (NMS-free) |
| raw prediction | `(1, 84, 5880)` | `(1, 300, 6)` |
| postprocess path | torchvision NMS | confidence filter only |
| postprocess (50 reps) | 0.5890 ms | 0.1825 ms |
| forward (30 reps) | 9.1967 ms | 9.3590 ms |

### 3.3 A/B result (median of 3 interleaved rounds, 100 images each)

| metric | baseline | `nms=False` | change |
|---|---|---|---|
| `post_ms` serial | 0.6957 ms | **0.2536 ms** | **−63.5%** |
| `post_ms` pipeline | 0.6897 ms | 0.2489 ms | −63.9% |
| serial throughput | 74.72 img/s | **76.73 img/s** | **+2.70%** |
| pipeline throughput | 88.88 img/s | **90.59 img/s** | **+1.93%** |
| **pipeline gain** | **1.190×** | **1.182×** | **unchanged** |
| CPU share (serial) | 21.85% | 19.14% | −2.7 pp |
| Amdahl ceiling (serial) | 1.280× | 1.237× | **decreases** |

Round-to-round serial totals: 1341.8 / 1335.4 / 1338.4 ms — stable, no baseline drift.

### 3.4 Accuracy cost / 精度代价

Course `map5095` scope, one measurement per variant:

| variant | mAP50-95 | mAP50 | mAP75 |
|---|---|---|---|
| baseline `nms=None` | **0.639839** | 0.796192 | 0.688182 |
| `nms=False` | **0.634109** | 0.791307 | 0.679421 |
| **Δ** | **−0.005730** | −0.004885 | −0.008761 |

Per-detection parity over the 100 benchmark images (same class, IoU ≥ 0.95,
|Δconf| ≤ 0.02): **10/100 identical, 90/100 differ**; 590 → 585 boxes,
46 images change box count (max +7); among 512 matched boxes mean IoU 0.98765,
mean |Δconf| 0.04207, max 0.59725.

> The sealed campaign's "100/100 bit-identical" parity still holds — that compares
> **serial vs pipeline** (same code path). This supplement compares **two head
> branches**, which are expected to differ.

---

## 4. Why / 结论

1. `postprocess` is executed in the GPU consumer thread in `run_pipeline()`
   (`t_gpu1 → t_gpu2`). Only `load + pre` is overlapped. Shrinking post therefore
   reduces absolute latency but cannot increase the overlap gain; it also lowers
   the CPU share, which *reduces* the theoretical ceiling (1.280× → 1.237×).
2. To raise the pipeline gain, optimise `load + pre` (the only hidden segment,
   16–20%) or move post into a worker pool so that `infer` becomes the sole bottleneck.
3. `nms=False` is **not** a free switch: it changes the head branch, so speed,
   ΔmAP and per-detection parity must all be reported together. At
   +2.7% throughput for ΔmAP −0.0057 with no pipeline-gain improvement, this
   supplement does **not** recommend changing the course default.

---

## 5. Two cross-check suggestions (documentation only, no data edits)

These are offered for maintainer judgement; **no sealed evidence file was touched**.

1. The Lab E **1.42×** in `README.md` / `README.en.md` / `INSTRUCTOR_GUIDE.md` is the
   genuine round-3 measurement, but that round's serial `infer` sum is 1350 ms vs
   1046 / 1129 ms in rounds 1–2, while pipeline stays at 1075 / 1076 / 1093 ms.
   The ratio therefore exceeds the 1.165–1.196× real ceiling because the baseline
   was slowed by shared-host load, not because the pipeline improved. A 3-round
   interleaved re-measurement on a quiet GPU gives **1.19×**. A footnote (or a
   median-of-rounds headline) would make this self-consistent.
2. `labE_pipeline_compare_*.json` describes the postprocess as *"same NMS-free
   postprocess"*. The probe shows the default path emits `(1,84,5880)` and calls
   torchvision NMS. The "same path" claim is correct; the "NMS-free" adjective is
   not. Suggested wording: `same postprocess path (one2many + external NMS)`.

Both fit naturally under the existing `Unreleased — v2.0.2 ... accuracy corrections`
changelog section, and double as a worked example of the course honesty rules:
*a flattering number that fails a ceiling check*.

---

## 6. Limits / 局限

- mAP measured once (course protocol); no confidence interval is claimed.
- No full formal campaign: CPU affinity pinning, `round_env` load/throttle capture
  and `verify_evidence.py` wiring are out of scope for a supplement.
- Shared host; results are claimed only for this machine class.
- Not a like-for-like performance-only comparison (head branch differs), hence the
  extra-experiment label.

---

## 7. Reproduce / 复现

```bash
source .venv/bin/activate
python scripts/run_labE_nms_supplement.py --probe
python scripts/run_labE_nms_supplement.py --full --rounds 3 --images 100 \
    --out <your/own/output>
python scripts/measure_labE_nms_map.py --out <your/own/output>
python scripts/analyze_labE_nms_supplement.py --out <your/own/output>
```

Scripts accept `LAB_REPO` so they can run from any directory on a shared machine;
`--out` keeps all outputs in the user's own space. Validation outputs go to
`<out>/val_runs`, never into the shared repo.

## 8. Evidence / 证据

`validation/v2.0.1/supplements/labE_nms_ceiling/` — 6 per-round records,
`ab_raw.json`, `det_parity.json`, `map_results.json`, `analysis_summary.json`,
`REPORT.txt`, `CAMPAIGN.json`, `SHA256SUMS` (course format, excludes itself).

```bash
cd validation/v2.0.1/supplements/labE_nms_ceiling
sha256sum -c SHA256SUMS
```

---

*Only new files were added: this document, three scripts, and one `supplements/`
evidence directory. No existing course file, notebook, weight, or result table is
modified.*
