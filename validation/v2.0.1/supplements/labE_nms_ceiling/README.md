# 补充实验证据包 / Bonus experiment evidence

**实验**：Lab E 补充实验 — 显式 `nms=False` 对 `post_ms` 与流水线吞吐的影响
**报告**：[`docs/LAB_E_SUPPLEMENT_NMS_CEILING.md`](../../../../docs/LAB_E_SUPPLEMENT_NMS_CEILING.md)
**复现**：`scripts/run_labE_nms_supplement.py` + `scripts/measure_labE_nms_map.py` + `scripts/analyze_labE_nms_supplement.py`

> ⚠️ 这是 **extra experiment（探索性补充实验）**，不属于正式 campaign `c20261108_1108` 的性能结论。
> 原因：`nms=False` 会切换模型 head 分支（one2many → one2one），检测结果发生变化；按
> `docs/BENCHMARK_PROTOCOL.md` 的诚实规则，它必须与正式公平对比分开标注，且必须报告代价。

## 文件清单 / Files

| 文件 | 内容 |
|---|---|
| `round1_baseline.json` … `round3_nms_false.json` | 6 份逐轮记录：每份含 `serial` 与 `async_pipeline` 的完整 `per_image`（100 张 × 四段耗时 + 检测框） |
| `ab_raw.json` | 3 轮统计摘要：每段均值/P50、总耗时、吞吐、CPU 占比、gain |
| `det_parity.json` | 跨变体逐框比对（100 张图） |
| `map_results.json` | 两变体的 mAP50-95 / mAP50 / mAP75 与 ΔmAP（课程 `map5095` 口径） |
| `analysis_summary.json` | 汇总中位数与上限计算 |
| `REPORT.txt` | 分析脚本原始输出（含原 campaign 6 文件的逐段求和表） |
| `CAMPAIGN.json` | 本补充实验的元信息（机器、模型、轮次、口径、已知局限） |
| `SHA256SUMS` | 除自身外所有文件的 SHA256 指纹（课程 `<sha256>␠␠<filename>` 格式） |
| `.gitattributes` | `* -text`：禁止 Git 对该目录做换行符转换，保证 Windows / Linux 上 `sha256sum -c` 都能通过 |

## 实验配置 / Setup

- 机器：AMD EPYC + Radeon PRO W7900D 48 GB (gfx1100)，实验前 `rocm-smi --showuse` = 0%
- 软件：torch `2.12.0+rocm7.14.1`，Ultralytics `8.4.208`，课程 `.venv`
- 模型：官方 `yolo26l.pt`（与正式 campaign 同一权重，未更换）
- 图片：课程固定 COCO val 教学子集前 100 张（与 Lab E 相同）
- 轮次：3 轮独立测量；顺序 `baseline→nms_false` / `nms_false→baseline` / `baseline→nms_false`（交错抵消漂移）
- 每轮：serial + async_pipeline 各 100 张；沿用课程 `_prewarm()` 全量预热、`cuda_sync()`、不改源码
- 发布值：3 轮**中位数**；所有原始逐图记录保留，无丢弃样本

## 关键结果 / Headline

| 指标（3 轮中位数） | baseline `nms=None` | `nms=False` |
|---|---|---|
| `post_ms`（serial） | 0.6957 ms | 0.2536 ms（−63.5%） |
| serial 吞吐 | 74.72 img/s | 76.73 img/s（+2.70%） |
| pipeline 吞吐 | 88.88 img/s | 90.59 img/s（+1.93%） |
| 流水线 gain | 1.190× | 1.182×（几乎不变） |
| mAP50-95 | 0.639839 | 0.634109（**Δ −0.005730**） |
| mAP50 / mAP75 | 0.796192 / 0.688182 | 0.791307 / 0.679421 |
| 跨变体逐框一致性 | — | 10/100 张完全一致，90/100 张有差异 |

**精度测量口径**：课程 `course.benchmark.map5095` 的设定（固定教学 val 子集 120 张、FP32、batch 8、imgsz 640、CUDA），
每变体测一次（与 `scripts/run_campaign.py` 一致）。baseline 的 **0.639839** 复现了课程公布的
PyTorch-CUDA-640 值 **0.6398**，可作为口径对齐的交叉验证。速度数字一律与 ΔmAP 并列报告。

## 核验 / Verify

```bash
cd validation/v2.0.1/supplements/labE_nms_ceiling
sha256sum -c SHA256SUMS
```

`SHA256SUMS` 采用与正式 campaign 相同的 `<sha256>␠␠<filename>` 格式，在全部文件写定之后生成，不包含自身。
目录内的 `.gitattributes` 用 `* -text` 关闭换行符自动转换，因此在 Windows 上 clone 后指纹依然有效。
