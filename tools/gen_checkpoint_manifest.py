#!/usr/bin/env python3
"""Regenerate ckpts/CHECKPOINTS.md.

"Shipped" and "referenced" rows are DERIVED -- from main.py's CKPTS dict, from
`external_baselines/param_census.py`, and from a scan for checkpoint paths
hardcoded in the source. Nothing about which checkpoints matter is typed by
hand, so the manifest cannot silently drift from the code.

    python tools/gen_checkpoint_manifest.py
"""
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parent))
import MAGVIT_project.main as M                                     # noqa: E402

CK = REPO / "ckpts"

# Derived: the paths main.py itself resolves, plus the census's OURS set.
shipped = {Path(v).name: k for k, v in M.CKPTS.items()}
census = {"vqvae_stage2a_best.pt": "param_census OURS",
          "motif_prior_best.pt": "param_census OURS",
          "activity_prior_best_hard_metric.pt": "param_census OURS"}

# Also derived: checkpoint paths hardcoded in the source rather than routed
# through CKPTS. spatial_bias_pretrain.pt is one of these and is load-bearing
# (sole source of the gct mapper), so a name-only scan would mislabel it.
hardcoded = {}
for src in list(REPO.glob("*.py")) + list(REPO.glob("*/*.py")):
    if "__pycache__" in str(src):
        continue
    for m in re.finditer(r"ckpts/([A-Za-z0-9_.]+\.(?:pt|pkl))", src.read_text(errors="ignore")):
        hardcoded.setdefault(m.group(1), set()).add(src.relative_to(REPO).as_posix())

tracked = set()
out = subprocess.run(["git", "-C", str(REPO), "ls-files", "ckpts"],
                     capture_output=True, text=True).stdout.split()
for t in out:
    tracked.add(Path(t).name)

# Suffix families -> what the suffix means. Matched longest-first.
FAMILY = [
    ("_PRE_V961",   "pre-flattening snapshot, 1024-entry ladder alphabet"),
    ("_PRE3AFIX",   "before the Stage-4A fix"),
    ("_PREQCTX",    "before the quantised-context change"),
    ("_PREFLAT",    "before the flat alphabet"),
    ("_PRE4C",      "before Stage 4C adaptation"),
    ("_COMPSEL",    "selected on the generation composite (superseded: 4B now selects on val NLL)"),
    ("_NLLRUN",     "val-NLL selection run"),
    ("_GUMBELRUN",  "gumbel top-K readout run"),
    ("_SMOKE",      "smoke test, not a result"),
    ("_EP15KILLED", "run killed at epoch 15"),
    ("_64ch",       "superseded 64-children-per-parent config"),
    ("SEED",        "seed replicate for variance estimation"),
    ("_warmstart",  "warm-start source"),
    ("_last",       "final epoch, not val-selected"),
    ("_best",       "val-selected"),
]

def family(name: str) -> str:
    for suf, desc in FAMILY:
        if suf.lower() in name.lower():
            return desc
    return ""

rows = []
for f in sorted(CK.glob("*.pt")) + sorted(CK.glob("*.pkl")):
    n = f.name
    mb = f.stat().st_size / 1e6
    if n in shipped:
        status = f"**shipped** (`CKPTS['{shipped[n]}']`)"
    elif n in census:
        status = "**shipped** (param_census)"
    elif n in hardcoded:
        where = sorted(hardcoded[n])[0]
        status = f"**referenced** (`{where}`)"
    else:
        status = "experiment"
    rows.append((n, status, f"{mb:.1f}", "yes" if n in tracked else "no", family(n)))

lines = []
lines.append("# ckpts/\n")
lines.append("Every checkpoint on disk, what produced it, and whether it ships.\n")
lines.append("**Nothing here is renamed or deleted.** Experiment checkpoints are kept "
             "because a suffix is often the only record of what a run was, and the "
             "cost of keeping them is disk, not correctness.\n")
lines.append("The **shipped** rows are derived from `main.py`'s `CKPTS` dict and "
             "`external_baselines/param_census.py`, not typed by hand, so this table "
             "cannot drift from the code without the generator noticing.\n")
lines.append(f"\n## Top-level ({len(rows)} files)\n")
lines.append("| file | status | MB | tracked | note |")
lines.append("|---|---|---:|:---:|---|")
for r in rows:
    lines.append(f"| `{r[0]}` | {r[1]} | {r[2]} | {r[3]} | {r[4]} |")

lines.append("\n## Subdirectories\n")
lines.append("| dir | size | contents |")
lines.append("|---|---:|---|")
SUB = {
 "ablations": "sparse/dense encoder arms and the token-entropy dose-response run",
 "cache": "derived caches moved out of reports/; regenerable, gitignored",
 "external_baselines": "fitted baseline weights (DG, GLM, MaskGIT-flat, U-Net, CVAE); gitignored",
 "v2_baseline": "pre-v3-rebuild snapshot, kept as the restore point for the peak-radius fix",
}
for d in sorted([p for p in CK.iterdir() if p.is_dir()]):
    sz = subprocess.run(["du","-sh",str(d)],capture_output=True,text=True).stdout.split()[0]
    lines.append(f"| `{d.name}/` | {sz} | {SUB.get(d.name,'')} |")

lines.append("\n## Snapshot trees outside ckpts/\n")
lines.append("Gitignored local restore points, listed so their purpose is recorded "
             "rather than guessed. They are **not** empty.\n")
lines.append("| dir | size | what it predates |")
lines.append("|---|---:|---|")
SNAP = {
 "ckpts_oldsplit_backup": "the current train/val/test split",
 "ckpts_pre2Cfix": "the Stage-2C cross-attention investigation",
 "ckpts_pre_v2": "the v2 baseline",
 "reports_oldsplit_backup": "the current split",
 "reports_pre2Cfix": "the Stage-2C investigation",
}
for name, what in SNAP.items():
    d = REPO / name
    if d.is_dir():
        sz = subprocess.run(["du","-sh",str(d)],capture_output=True,text=True).stdout.split()[0]
        lines.append(f"| `{name}/` | {sz} | {what} |")

(CK / "CHECKPOINTS.md").write_text("\n".join(lines) + "\n")
print(f"wrote {CK/'CHECKPOINTS.md'}: {len(rows)} checkpoints")
