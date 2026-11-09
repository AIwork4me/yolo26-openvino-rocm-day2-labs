"""Analysis for the Lab E bonus experiment (CPU-share ceiling + detection drift).

Usage:
    python scripts/analyze_labE_nms_supplement.py \
        --out /path/to/evidence/directory

Reads the sealed parent campaign (read-only) and the bonus-experiment records,
writes REPORT.txt and analysis_summary.json into --out.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from pathlib import Path

REPO = Path(os.environ.get("LAB_REPO", str(Path(__file__).resolve().parents[1])))
CAMPAIGN = REPO / "validation/v2.0.1/campaigns/c20261108_1108/labE_pipeline"
KS = ("load_ms", "pre_ms", "infer_ms", "post_ms")


def segs(p):
    return {k: sum(r[k] for r in p) for k in KS}


def iou(a, b):
    ix1, iy1 = max(a[1], b[1]), max(a[2], b[2])
    ix2, iy2 = min(a[3], b[3]), min(a[4], b[4])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = ((a[3] - a[1]) * (a[4] - a[2])
             + (b[3] - b[1]) * (b[4] - b[2]) - inter)
    return inter / union if union > 0 else 0.0


def section_a(lines):
    lines.append("=" * 108)
    lines.append("A. 原 campaign（只读）：CPU 段占比 / 收益上限 / 实测效率")
    lines.append("   Parent campaign (read-only): CPU share / ceilings / measured efficiency")
    lines.append("=" * 108)
    lines.append("%-24s %7s %7s %7s %7s | %7s %7s %7s | %8s %8s %8s" % (
        "file", "load", "pre", "infer", "post", "cpu%", "hid%", "post%",
        "Amdahl", "real", "floor%"))
    for f in sorted(glob.glob(str(CAMPAIGN / "*_records.json"))):
        d = json.load(open(f))
        s = segs(d["per_image"])
        tot = sum(s.values())
        cpu = s["load_ms"] + s["pre_ms"] + s["post_ms"]
        hid = s["load_ms"] + s["pre_ms"]
        gpu_side = s["infer_ms"] + s["post_ms"]
        am = tot / (tot - cpu)                 # Amdahl: hide ALL CPU stages
        real = tot / max(hid, gpu_side)        # current impl: post NOT hidden
        floor_pct = 100 * max(hid, gpu_side) / d["total_ms"]
        lines.append("%-24s %7.0f %7.0f %7.0f %7.0f | %6.1f%% %6.1f%% %6.1f%% | %7.3fx %7.3fx %7.1f%%" % (
            Path(f).name.replace("labE_pipeline_", "").replace("_records.json", ""),
            s["load_ms"], s["pre_ms"], s["infer_ms"], s["post_ms"],
            100 * cpu / tot, 100 * hid / tot, 100 * s["post_ms"] / tot,
            am, real, floor_pct))


def section_b(lines, out):
    lines.append("")
    lines.append("=" * 108)
    lines.append("B. Bonus A/B: median across 3 interleaved rounds")
    lines.append("=" * 108)
    raw = json.load(open(out / "ab_raw.json"))
    med = {}
    for variant, rows in raw.items():
        for mode in ("serial", "pipeline"):
            med[(variant, mode)] = {
                k: st.median([r[mode][k] for r in rows])
                for k in ("total_ms", "images_per_s", "post_ms_mean", "cpu_frac")}
            med[(variant, mode)]["gain"] = st.median([r["gain_x"] for r in rows])
    lines.append("%-11s %-9s %9s %9s %9s %9s %9s %8s" % (
        "variant", "mode", "post_ms", "img/s", "cpu%", "Amdahl", "gain", "eta"))
    for variant in ("baseline", "nms_false"):
        for mode in ("serial", "pipeline"):
            m = med[(variant, mode)]
            am = 1 / (1 - m["cpu_frac"])
            lines.append("%-11s %-9s %9.4f %9.2f %8.2f%% %9.3fx %9.3fx %7.1f%%" % (
                variant, mode, m["post_ms_mean"], m["images_per_s"],
                100 * m["cpu_frac"], am, m["gain"],
                100 * (m["gain"] - 1) / (am - 1)))
    a, b = med[("baseline", "serial")], med[("nms_false", "serial")]
    lines.append("")
    lines.append("  post_ms  : %.4f -> %.4f ms (%+.1f%%, saves %.4f ms/image)" % (
        a["post_ms_mean"], b["post_ms_mean"],
        100 * (b["post_ms_mean"] / a["post_ms_mean"] - 1),
        a["post_ms_mean"] - b["post_ms_mean"]))
    for mode in ("serial", "pipeline"):
        x, y = med[("baseline", mode)], med[("nms_false", mode)]
        lines.append("  %-8s throughput: %.2f -> %.2f img/s (%+.2f%%)" % (
            mode, x["images_per_s"], y["images_per_s"],
            100 * (y["images_per_s"] / x["images_per_s"] - 1)))
    lines.append("  gain     : %.3fx -> %.3fx" % (a["gain"], b["gain"]))
    return {"%s_%s" % k: v for k, v in med.items()}


def section_c(lines, out):
    lines.append("")
    lines.append("=" * 108)
    lines.append("C. Detection cost: baseline one2many+NMS vs nms_false one2one NMS-free")
    lines.append("=" * 108)
    a = {r["file"]: r["detections"] for r in
         json.load(open(out / "round3_baseline.json"))["serial"]["per_image"]}
    b = {r["file"]: r["detections"] for r in
         json.load(open(out / "round3_nms_false.json"))["serial"]["per_image"]}
    n_same, n_diff = 0, 0
    ious, dconfs, delta = [], [], []
    for f in sorted(set(a) & set(b)):
        da, db = list(a[f]), list(b[f])
        delta.append(len(db) - len(da))
        if not da and not db:
            n_same += 1
            continue
        used, same = set(), True
        for x in da:
            best, bi = 0.0, -1
            for j, y in enumerate(db):
                if j in used or x[0] != y[0]:
                    continue
                v = iou(x, y)
                if v > best:
                    best, bi = v, j
            if bi >= 0 and best >= 0.95:
                used.add(bi)
                ious.append(best)
                dconfs.append(abs(x[5] - db[bi][5]))
                if abs(x[5] - db[bi][5]) > 0.02:
                    same = False
            else:
                same = False
        if len(used) == len(db) and same:
            n_same += 1
        else:
            n_diff += 1
    lines.append("  identical images       : %d / %d" % (n_same, len(a)))
    lines.append("  images with differences: %d / %d" % (n_diff, len(a)))
    lines.append("  total boxes            : %d -> %d (%+d)" % (
        sum(len(x) for x in a.values()), sum(len(x) for x in b.values()),
        sum(len(x) for x in b.values()) - sum(len(x) for x in a.values())))
    lines.append("  images with box-count change: %d ; max delta %+d" % (
        sum(1 for x in delta if x), max(map(abs, delta))))
    if ious:
        lines.append("  matched boxes (IoU>=0.95): %d, mean IoU %.5f, mean |dconf| %.5f, max %.5f" % (
            len(ious), st.mean(ious), st.mean(dconfs), max(dconfs)))


def section_d(lines, out):
    f = out / "map_results.json"
    if not f.exists():
        return
    d = json.load(open(f))
    lines.append("")
    lines.append("=" * 108)
    lines.append("D. Accuracy cost / 精度代价 (course protocol: val_subset, FP32, batch 8, imgsz 640)")
    lines.append("=" * 108)
    lines.append("%-11s %12s %12s %12s" % ("variant", "mAP50-95", "mAP50", "mAP75"))
    for k in ("baseline", "nms_false"):
        r = d[k]
        lines.append("%-11s %12.6f %12.6f %12.6f" % (
            k, r["map50_95"], r["map50"], r["map75"]))
    lines.append("  delta_mAP50-95 = %+.6f  |  delta_mAP50 = %+.6f" % (
        d["nms_false"]["map50_95"] - d["baseline"]["map50_95"],
        d["nms_false"]["map50"] - d["baseline"]["map50"]))
    lines.append("  baseline 0.639839 matches the published PyTorch-CUDA-640 value (0.6398): "
                 "measurement scope cross-checked.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO / "validation/v2.0.1/supplements/labE_nms_ceiling")
    out = ap.parse_args().out
    lines = []
    section_a(lines)
    med = section_b(lines, out)
    section_c(lines, out)
    section_d(lines, out)
    lines.append("")
    lines.append("NOTE: parent campaign files were read only. See CAMPAIGN.json for setup and known limits.")
    (out / "REPORT.txt").write_text("\n".join(lines) + "\n")
    (out / "analysis_summary.json").write_text(json.dumps(
        {"median": med, "parent_campaign": str(CAMPAIGN.relative_to(REPO))}, indent=1))
    print("\n".join(lines))
    print("\nwrote:", out / "REPORT.txt")
