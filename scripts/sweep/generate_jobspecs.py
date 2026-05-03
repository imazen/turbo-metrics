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

# 21-step quality grid covering the full 5..95 range. Denser at low-q
# than the typical "85, 95" two-point sweep (which is forbidden by the
# project's calibration discipline) and matched in coverage at high-q.
Q_GRID = ",".join(str(q) for q in range(5, 96, 5))

# Per-codec knob grids — small Cartesian products that exercise the
# axes most likely to shift Pareto behaviour. Kept small so total cell
# count stays inside the overnight budget.
KNOB_GRIDS = {
    "zenwebp": json.dumps({
        "method": [4, 6],
        "segments": [1, 4],
    }),
    "zenavif": json.dumps({
        "speed": [6, 8],
    }),
    "zenjxl": json.dumps({
        "effort": [3, 7],
    }),
}

# Conservative metric set: CPU-only. GPU metrics are still wired and
# will be added in a follow-up pass once the wgpu runtime is verified
# stable on whichever GPU sku the worker lands on.
METRICS = ["zensim", "ssim2", "butteraugli", "dssim"]

CHUNK_SIZE = 25  # images per chunk

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
