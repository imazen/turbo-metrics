#!/usr/bin/env python3
"""Generate the JSONL chunk list for the zen-metrics sweep.

Each chunk is one (codec, image-subset) job. The worker picks them up
sequentially and emits one Pareto TSV per chunk. We split images into
small chunks so a worker that crashes mid-run loses only one chunk's
worth of work.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# 11-step quality grid covering the full range with extra density at
# the low-q end where production traffic + structural artifacts live.
# The brief's calibration discipline forbids "85/95 two-point" sweeps
# but tolerates a coarser-than-step-5 grid when total compute is
# bounded — which it is here ($9.50 hard cap, ~hours overnight).
# Step 10 is the floor; we hand-densify the perceptibility band.
Q_GRID = "5,15,25,35,45,55,65,75,85,95"

# Per-codec knob grids — small Cartesian products that exercise the
# axes most likely to shift Pareto behaviour. Kept small so total cell
# count stays inside the overnight budget.
KNOB_GRIDS = {
    "zenwebp": json.dumps({
        "method": [4, 6],
    }),
    "zenavif": json.dumps({
        "speed": [6, 8],
    }),
    "zenjxl": json.dumps({
        "effort": [3, 7],
    }),
}

# Metric set: CPU only. zensim + ssim2 + dssim are all relatively
# cheap (~10-50ms per cell on a 1MP image). butteraugli is dropped
# from this run because it dominates wall-clock at ~300ms/cell —
# we'll add it in a follow-up pass once the rest is durable.
METRICS = ["zensim", "ssim2", "dssim"]

CHUNK_SIZE = 50  # images per chunk

def main():
    sources_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/lilith/work/zentrain-corpus/mlp-tune-fast")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/tmp/chunks.jsonl")

    images = []
    for ext in ("png", "jpg", "jpeg"):
        images.extend(sources_root.rglob(f"*.{ext}"))
    images.sort()
    rels = [str(p.relative_to(sources_root)) for p in images]
    print(f"# {len(rels)} source images", file=sys.stderr)

    with out_path.open("w") as f:
        for codec, knob_grid in KNOB_GRIDS.items():
            for i in range(0, len(rels), CHUNK_SIZE):
                chunk_imgs = rels[i:i+CHUNK_SIZE]
                chunk_id = f"{codec}-{i//CHUNK_SIZE:03d}"
                spec = {
                    "codec": codec,
                    "chunk_id": chunk_id,
                    "q_grid": Q_GRID,
                    "knob_grid": knob_grid,
                    "metrics": METRICS,
                    "images": chunk_imgs,
                }
                f.write(json.dumps(spec))
                f.write("\n")

    total_chunks = sum(
        (len(rels) + CHUNK_SIZE - 1) // CHUNK_SIZE
        for _ in KNOB_GRIDS
    )
    print(f"# wrote {total_chunks} chunks to {out_path}", file=sys.stderr)
    print(f"# Q grid: {Q_GRID}", file=sys.stderr)
    print(f"# Knob grids:", file=sys.stderr)
    for c, k in KNOB_GRIDS.items():
        print(f"#   {c}: {k}", file=sys.stderr)
    print(f"# Metrics: {METRICS}", file=sys.stderr)

if __name__ == "__main__":
    main()
