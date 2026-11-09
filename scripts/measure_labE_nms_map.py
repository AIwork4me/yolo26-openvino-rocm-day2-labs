"""Accuracy cost of the Lab E bonus experiment (mAP50-95, course protocol).

The course honesty rules require a speed result to be paired with its accuracy
cost. This script measures mAP for both nms variants on the fixed educational
val subset, using the same settings as `course.benchmark.map5095`
(imgsz 640, batch 8, cuda, FP32, official yolo26l.pt).

Usage:
    python scripts/measure_labE_nms_map.py --out /your/own/evidence/dir

On a shared machine ALWAYS pass --out to your own directory; validation outputs
are written under <out>/val_runs so nothing is written into the shared repo.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("LAB_REPO", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(REPO))

from course.model import load_official_yolo26l  # noqa: E402


def measure(variant: str, out: Path) -> dict:
    model = load_official_yolo26l()
    kwargs = {"nms": False} if variant == "nms_false" else {}
    t0 = time.perf_counter()
    m = model.val(
        data=str(REPO / "data" / "val_subset.yaml"),
        imgsz=640, batch=8, device="cuda", verbose=False,
        project=str(out / "val_runs"), name=variant, exist_ok=True,
        plots=False, save_json=False, **kwargs,
    )
    return {
        "variant": variant,
        "nms": False if variant == "nms_false" else None,
        "map50_95": round(float(m.box.map), 6),
        "map50": round(float(m.box.map50), 6),
        "map75": round(float(m.box.map75), 6),
        "wall_s": round(time.perf_counter() - t0, 2),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=REPO / "validation/v2.0.1/supplements/labE_nms_ceiling")
    out = ap.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for variant in ("baseline", "nms_false"):
        r = measure(variant, out)
        results[variant] = r
        print(json.dumps(r), flush=True)
    d = results["nms_false"]["map50_95"] - results["baseline"]["map50_95"]
    results["delta_map50_95"] = round(d, 6)
    print("delta_map50_95 = %+.6f" % d, flush=True)
    (out / "map_results.json").write_text(json.dumps(results, indent=1) + "\n")
    print("wrote:", out / "map_results.json", flush=True)
