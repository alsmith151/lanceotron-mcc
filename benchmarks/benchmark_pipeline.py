"""Compare application-cold/direct, staged and reusable-cache BigWig calling."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pybigtools

from mcc_peaks.api.adaptive import AdaptiveParams, call_adaptive_bigwigs, read_bed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bigwig", type=Path)
    parser.add_argument("--viewpoint-bed", type=Path)
    parser.add_argument("--viewpoint")
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--memory-budget-mb", type=float, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.bigwig) != bool(args.viewpoint_bed):
        parser.error("--bigwig and --viewpoint-bed must be provided together")
    if args.scratch_dir:
        args.scratch_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.scratch_dir, prefix="mcc-peaks-benchmark-") as tmp:
        tmp = Path(tmp)
        if args.bigwig:
            path = args.bigwig
            vp = read_bed(args.viewpoint_bed, args.viewpoint, require_single_group=True)
        else:
            path = tmp / "synthetic.bw"
            n, midpoint = 200_000, 100_000
            positions = np.arange(n)
            rng = np.random.default_rng(2026)
            signal = rng.poisson(1 + 15 * np.exp(-abs(positions - midpoint) / 20_000)).astype(float)
            signal[20_000:20_200] += 50
            signal[160_000:160_400] += 40
            writer = pybigtools.open(str(path), "w")
            writer.write(
                {"chr1": n}, (("chr1", i, i + 1, float(value)) for i, value in enumerate(signal))
            )
            vp = {"chr1": [(midpoint, midpoint + 100)]}
        params = AdaptiveParams(memory_budget_mb=args.memory_budget_mb)
        results, reference = [], None
        variants = [
            ("direct", None, None),
            ("staged", tmp / "stage", None),
            ("cache_cold", None, tmp / "cache"),
            ("cache_warm", None, tmp / "cache"),
        ]
        for name, scratch, cache in variants:
            calls, metrics = call_adaptive_bigwigs(
                [path], vp, params=params, scratch_dir=scratch, cache_dir=cache
            )
            if reference is None:
                reference = calls
            else:
                pd.testing.assert_frame_equal(reference, calls)
            results.append({"variant": name, "regions": len(calls), "metrics": metrics})
        output = {
            "synthetic": args.bigwig is None,
            "application_cache_comparison": True,
            "physical_filesystem_caches_cleared": False,
            "calls_identical": True,
            "parameters": params.model_dump(),
            "runs": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "calls_identical": True,
                    "runs": [
                        {
                            "variant": r["variant"],
                            "seconds": round(r["metrics"]["elapsed_seconds"], 4),
                            "read_calls": r["metrics"]["inputs"][0]["read_calls"],
                        }
                        for r in results
                    ],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
