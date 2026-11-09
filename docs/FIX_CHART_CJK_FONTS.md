# Fix: hundreds of `findfont` / `Glyph missing` warnings from Lab chart cells
# 修复：Lab 图表单元格刷出成百上千行 findfont / Glyph missing 警告

**Affected / 受影响**：Lab A–E chart cells（`from course.charts import …` 的那几格）
**First reported / 首报**：2026-11-09，`01_Lab_A_CPU_vs_GPU_Baseline.ipynb` cell 10
`from course.charts import chart_cpu_vs_gpu` → `chart_cpu_vs_gpu(cpu_sum, gpu_sum)`

## Symptom / 现象

Chart cell appears to "explode with a long traceback"（看起来像一长串报错），e.g.:

```text
findfont: Font family 'Noto Sans CJK SC' not found.   (× dozens / 几十行)
findfont: Font family 'Noto Sans CJK SC' not found.
UserWarning: Glyph 21333 (单) missing from font(s) DejaVu Sans.
  fig.tight_layout(rect=(0, 0.045, 1, 1))
UserWarning: Glyph 24352 (张) missing from font(s) DejaVu Sans.
  ...
[chart] .../assets/slides/01_cpu_vs_gpu.png (+svg)    ← 最后一行其实是成功输出
```

## TL;DR / 结论

- **不是报错，任务没有失败**：最后一行 `[chart] … .png (+svg)` 说明 PNG/SVG 已
  正常导出。这只是 matplotlib 的字体缺失 **告警**（warning），不是异常。
- 根因 1：容器内没有中文字体（本机 `fc-list` 仅 8 个拉丁字体），而
  `course/charts.py` 的 `font.family` 里配置了中文 `Noto Sans CJK SC`，
  所有中文标签都会触发 `findfont` + `Glyph … missing` 告警。
- 根因 2（为什么 `apt install fonts-noto-cjk` 不够）：Debian/Ubuntu 的
  `fonts-noto-cjk` 只提供 **`.ttc` 集合文件**，matplotlib 的 fontManager
  **不支持索引 `.ttc`**（只认单 face 的 `.otf/.ttf`），装完仍然
  `Noto Sans CJK SC not found`。
- **必须重启 Jupyter kernel**（或重建字体缓存）：fontManager 是进程内单例，
  kernel 启动时已缓存旧字体列表，随后装多少字体它都看不见。

## Fix / 修复步骤（root，Radeon Cloud 容器）

1. Install the system CJK font package / 装系统中文字体包：

   ```bash
   sudo apt-get update && sudo apt-get install -y fonts-noto-cjk
   ```

2. Extract single-face OTFs — matplotlib can't read `.ttc` /
   抽取单 face OTF（matplotlib 读不了 `.ttc` 集合）；需要 `fonttools`
   （repo venv 已自带）：

   ```bash
   sudo mkdir -p /usr/local/share/fonts
   .venv/bin/python - <<'PY'
   from fontTools.ttLib import TTCollection
   jobs = [
       ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/local/share/fonts/NotoSansCJKsc-Regular.otf", "Noto Sans CJK SC"),
       ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/local/share/fonts/NotoSansCJKsc-Bold.otf",  "Noto Sans CJK SC Bold"),
   ]
   for src, dst, want in jobs:
       for f in TTCollection(src).fonts:
           if f["name"].getDebugName(4) == want:
               f.save(dst); print("saved", dst); break
   PY
   sudo fc-cache -f
   ```

   注意：按 **FullName（DebugName 4）精确匹配** 再落盘，否则循环会停在其他
   语言 face 上（实测默认落到了 `Noto Sans CJK JP Bold`）。

3. Clear matplotlib's font cache（the scan happens at first `pyplot` render）/
   清 matplotlib 字体缓存（扫描发生在首次 `pyplot` 渲染时）：

   ```bash
   rm -rf ~/.cache/matplotlib
   ```

4. **Restart the Jupyter kernel** (Kernel → Restart) and re-run Lab A cells
   1 → 10 / 重启 kernel 后重跑 Lab A 的 1→10 格。

## In-kernel alternative (no restart, no root) /
   无 root / 不重启 kernel 的替代方案

先在一格中运行（重建 fontManager，不落盘任何东西）：

```python
import matplotlib.font_manager as fm
fm.fontManager = fm._load_fontmanager(try_read_cache=False)  # 重建字体缓存
assert any("Noto Sans CJK SC" == f.name for f in fm.fontManager.ttflist)
```

无 root 时把步骤 2 的提取输出改到用户目录亦可（matplotlib 会扫描
`~/.local/share/fonts` 与 `~/.fonts`）：

```bash
mkdir -p ~/.local/share/fonts && fc-cache -f ~/.local/share/fonts
```

## Verify / 验证

不重跑基准也能验证（合成 stats 直接画图）：

```python
import sys; sys.path.insert(0, "..")
import copy
cpu_sum = {"stats"  : {"p50_ms": 181.1, "p95_ms": 188.0, "min_ms": 170.0,
                       "max_ms": 200.0, "mean_ms": 182.0, "n": 10},
           "run_id" : "labA_cpu_vs_gpu_pytorch_cpu_test",
           "meta"   : {"cpu": "EPYC 9334", "gpu": "W7900", "imgsz": 640,
                       "batch": 1, "precision": "fp32", "warmup": 3, "iters": 10},
           "evidence": {"csv": "x.csv", "csv_sha256": "0" * 12}}
gpu_sum = copy.deepcopy(cpu_sum)
gpu_sum["stats"] = {"p50_ms": 12.9, "p95_ms": 12.9, "min_ms": 12.5,
                    "max_ms": 14.0, "mean_ms": 13.0, "n": 10}

from course.charts import chart_cpu_vs_gpu
chart_cpu_vs_gpu(cpu_sum, gpu_sum, name="_font_selftest")
```

修复后应只输出一行 `[chart] …/_font_selftest.png (+svg)`，无任何
`findfont` / `Glyph` 输出；中文字样（延迟/吞吐/倍速框）正常显示。
自测产物用完删除即可：`rm assets/slides/_font_selftest.*`。

修复前后对比（同一段代码在同机生成）：

| | 修复前 | 修复后 |
|---|---|---|
| 输出行数 | ~600 行 warning | 1 行 `[chart] …` |
| 中文标签 | □□□ tofu 方框 | 正常渲染 |
| 结果文件 | ✅ 已保存 | ✅ 已保存 |

## Why this stays a warning, not an error / 为什么这只是告警

`charts.py` 在模块级 `plt.rcParams` 里声明了多 fallback 字体
（`["DejaVu Sans", "Noto Sans CJK SC"]`）。缺 CJK 字体时 matplotlib 逐 glyph
回退 DejaVu Sans 并逐 glyph 打印告警，**不会中断渲染**。所以 cell 看起来
"爆出一长串"，实际执行到最后成功保存。
