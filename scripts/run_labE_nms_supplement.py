"""Lab E bonus experiment — does explicit nms=False improve the pipeline gain?

Course supplement for notebooks/05_Lab_E_AMD_CPU_GPU_Pipeline.ipynb.
See docs/LAB_E_SUPPLEMENT_NMS_CEILING.md for the full report.

This file is READ-ONLY with respect to the course module: it swaps the module
level Yolo26lStages class at runtime instead of editing course/pipeline.py.

Usage (from a checkout with the course venv active):
    python scripts/run_labE_nms_supplement.py --probe
    python scripts/run_labE_nms_supplement.py --full --rounds 3 --images 100 \
        --out /path/to/my/own/output

On a SHARED machine, always pass --out pointing to your own directory, e.g.
    --out /workspace/my_course/nms_ab
so that you never write into somebody else's evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path

# When installed as <repo>/scripts/... the parent of scripts/ is the repo.
# On a shared machine you can run this file from elsewhere with
# LAB_REPO=/path/to/checkout so that no course file ever needs editing.
REPO = Path(os.environ.get("LAB_REPO", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(REPO))

from course import pipeline as P  # noqa: E402
from course.benchmark import cuda_sync, load_bench_images, now_ms  # noqa: E402
from course.model import load_official_yolo26l  # noqa: E402

ORIG_STAGES = P.Yolo26lStages


class NmsFalseStages(ORIG_STAGES):
    """Identical to the shipped stages, but predict() gets explicit nms=False."""

    def __init__(self, imgsz: int = 640, warmup_image: Path | None = None):
        self.imgsz = imgsz
        self.model = load_official_yolo26l()
        warm = str(warmup_image or load_bench_images(1)[0])
        self.model.predict(source=warm, imgsz=imgsz, device="cuda",
                           verbose=False, nms=False)
        self.predictor = self.model.predictor


def use(variant: str) -> None:
    P.Yolo26lStages = NmsFalseStages if variant == "nms_false" else ORIG_STAGES


def describe(stages) -> dict:
    pr = stages.predictor
    be = getattr(pr, "model", None)
    raw = getattr(be, "model", None)
    head = getattr(getattr(raw, "model", None) and raw.model[-1], "end2end", None)
    return {"args.nms": getattr(pr.args, "nms", "<missing>"),
            "args.conf": pr.args.conf,
            "args.max_det": pr.args.max_det,
            "autobackend.end2end": getattr(be, "end2end", None),
            "head.end2end": head}


def probe() -> None:
    """Single-image check of what nms=False actually changes."""
    imgs = load_bench_images(3)
    for variant in ("baseline", "nms_false"):
        use(variant)
        s = P.Yolo26lStages(imgsz=640)
        print("\n[%s] %s" % (variant, json.dumps(describe(s), default=str)))
        p = imgs[0]
        im0, im = s.load(p), None
        im = s.preprocess(im0)
        preds = s.infer(im)
        cuda_sync()
        if isinstance(preds, dict):
            shapes = {k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
                      for k, v in preds.items()}
        elif isinstance(preds, (list, tuple)):
            shapes = [tuple(x.shape) if hasattr(x, "shape") else type(x).__name__
                      for x in preds]
        else:
            shapes = [tuple(preds.shape)]
        print("  preds:", shapes)
        t0 = now_ms()
        for _ in range(50):
            res = s.postprocess(preds, im, im0)[0]
        cuda_sync()
        print("  postprocess mean %.4f ms (50 reps)" % ((now_ms() - t0) / 50))
        print("  n_det:", len(res.boxes) if res.boxes is not None else 0)
        for _ in range(3):
            s.infer(im)
        cuda_sync()
        t0 = now_ms()
        for _ in range(30):
            s.infer(im)
        cuda_sync()
        print("  infer mean %.4f ms (30 reps)" % ((now_ms() - t0) / 30))


def stats(rec: dict) -> dict:
    p = rec["per_image"]
    out = {"mode": rec["mode"], "n_images": rec["n_images"],
           "total_ms": round(rec["total_ms"], 2),
           "images_per_s": round(rec["images_per_s"], 3)}
    for k in ("load_ms", "pre_ms", "infer_ms", "post_ms"):
        v = [r[k] for r in p]
        out[k + "_mean"] = round(st.mean(v), 4)
        out[k + "_p50"] = round(st.median(v), 4)
    out["post_ms_sum"] = round(sum(r["post_ms"] for r in p), 2)
    seg = {k: sum(r[k] for r in p) for k in ("load_ms", "pre_ms", "infer_ms", "post_ms")}
    out["seg_sum_ms"] = round(sum(seg.values()), 2)
    out["cpu_frac"] = round((seg["load_ms"] + seg["pre_ms"] + seg["post_ms"])
                            / sum(seg.values()), 4)
    out["n_det_total"] = sum(r["n_det"] for r in p)
    return out


def full(rounds: int, n_images: int, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    imgs = load_bench_images(n_images)
    results, det_maps = {}, {}
    order = [["baseline", "nms_false"], ["nms_false", "baseline"]]
    for r in range(rounds):
        for variant in order[r % 2]:  # interleave A/B and B/A to cancel drift
            use(variant)
            print("\n=== round %d · %s ===" % (r + 1, variant), flush=True)
            serial = P.run_serial(imgs, imgsz=640)
            pipe = P.run_pipeline(imgs, imgsz=640, queue_depth=4)
            ss, ps = stats(serial), stats(pipe)
            gain = ss["total_ms"] / ps["total_ms"]
            print("  serial  :", json.dumps(ss), flush=True)
            print("  pipeline:", json.dumps(ps), flush=True)
            print("  gain %.3fx" % gain, flush=True)
            results.setdefault(variant, []).append(
                {"round": r + 1, "serial": ss, "pipeline": ps,
                 "gain_x": round(gain, 4),
                 "parity_n_det_equal": ss["n_det_total"] == ps["n_det_total"]})
            det_maps[variant] = {x["file"]: x["detections"]
                                 for x in serial["per_image"]}
            (out / ("round%d_%s.json" % (r + 1, variant))).write_text(
                json.dumps({"serial": serial, "pipeline": pipe}, indent=1))
    (out / "ab_raw.json").write_text(json.dumps(results, indent=1))

    # Cost of the switch: per-detection parity across the two head branches.
    if "baseline" in det_maps and "nms_false" in det_maps:
        a, b = det_maps["baseline"], det_maps["nms_false"]
        diff = []
        for f in sorted(set(a) & set(b)):
            if len(a[f]) != len(b[f]):
                diff.append([f, len(a[f]), len(b[f])])
                continue
            for x, y in zip(a[f], b[f]):
                if (x[0] != y[0] or abs(x[5] - y[5]) > 0.02
                        or any(abs(u - v) > 1.0 for u, v in zip(x[1:5], y[1:5]))):
                    diff.append([f, "box/conf mismatch"])
                    break
        parity = {"images_compared": len(set(a) & set(b)),
                  "mismatch_count": len(diff), "examples": diff[:8]}
        (out / "det_parity.json").write_text(json.dumps(parity, indent=1))
        print("\ncross-variant parity:", json.dumps(parity), flush=True)
    print("\nwrote:", out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--images", type=int, default=100)
    ap.add_argument("--out", type=Path,
                    default=REPO / "validation/v2.0.1/supplements/labE_nms_ceiling")
    a = ap.parse_args()
    if a.probe:
        probe()
    elif a.full:
        full(a.rounds, a.images, a.out)
    else:
        ap.print_help()
