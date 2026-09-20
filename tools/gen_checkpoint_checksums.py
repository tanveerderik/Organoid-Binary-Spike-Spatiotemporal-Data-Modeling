#!/usr/bin/env python3
"""Write the checksum list `tools/fetch_checkpoints.py` verifies against.

The checkpoints are too large for git, so they are published as release assets
and fetched on demand. A download that is merely *present* is not enough: a
truncated or superseded file reproduces silently wrong numbers, which is the
failure mode this whole pipeline is built to prevent. So every published
checkpoint is pinned by SHA-256 here, and the fetcher refuses anything that
does not match.

Which files are published is decided the same way `tools/make_release.py`
decides it, and for the same reason: the manifest is generated from `main.py`'s
CKPTS dict, so it cannot drift from the code. Note that the Stage-1 pretrain is
marked `referenced` rather than `shipped` and is nonetheless REQUIRED --
`build_pipeline` raises without it -- so both statuses are included.

    python tools/gen_checkpoint_checksums.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "ckpts" / "CHECKPOINTS.md"
OUT = ROOT / "reports" / "checkpoint_downloads.json"

# Published even though the manifest calls them `referenced`: the eval and
# generation paths load them. Kept in step with make_release.ALSO_SHIP.
ALSO = ["spatial_bias_pretrain.pt", "context_prior.pkl",
        "token_adjacency_bank.pt", "null_baselines.pkl"]

# Read by training only, or too large to publish as a release asset.
EXCLUDE = {"vqvae_stage1_best.pt", "motif_prior_warmstart.pt",
           "motif_null_baselines.pkl"}


def published() -> list[str]:
    names = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or "**shipped**" not in line:
            continue
        m = re.match(r"\|\s*`([^`]+)`", line)
        if m:
            names.append(m.group(1))
    names += [n for n in ALSO if n not in names]
    return [n for n in names if n not in EXCLUDE]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    entries, missing, total = {}, [], 0
    for name in published():
        p = ROOT / "ckpts" / name
        if not p.is_file():
            missing.append(name)
            continue
        size = p.stat().st_size
        total += size
        entries[name] = {"sha256": sha256(p), "bytes": size}
        print(f"  {name:38s} {size/1e6:7.1f} MB  {entries[name]['sha256'][:16]}")

    if missing:
        print(f"\nnot found on disk: {', '.join(missing)}", file=sys.stderr)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"checkpoints": entries}, indent=2,
                              sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n{len(entries)} checkpoints, {total/1e6:.1f} MB "
          f"-> {OUT.relative_to(ROOT)}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
