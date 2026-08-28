#!/usr/bin/env python3
"""Build a clean public copy of the repo in a sibling directory.

The working tree is never modified. This exists because the working tree
carries 2.4GB of checkpoints, ~190MB of reports and five snapshot trees that
are meaningful to us and noise to a reviewer -- but the code, the rendered
diagnostics and one checkpoint are the artifact.

    python tools/make_release.py                    # default: ../MAGVIT_release
    python tools/make_release.py --dest /tmp/rel
    python tools/make_release.py --no-checkpoint    # code only, ~1.7MB

What goes in:
  * every .py (excluding __pycache__)
  * README.md, analysis/README.md, external_baselines/README.md, LICENSE,
    .gitignore
  * the rendered diagnostics markdown under reports/external_baselines/
  * ckpts/CHECKPOINTS.md
  * ckpts/vqvae_stage2a_best.pt, unless --no-checkpoint

What stays out, and why RELEASE.md says so explicitly rather than leaving a
reviewer to wonder: every other checkpoint, every reports/*.json, the
snapshot trees, .log files, .bak_* files, __pycache__.

Anonymisation
-------------
ICLR is double-blind and identity revealed in the main text OR the
supplementary material is a desk reject, so every copied text file is passed
through `scrub()` on the way out: `@author` headers, the author's username, the
absolute working-directory prefix, the drive name, and any git remote URL. The
scrub runs on the destination copy; the working tree is never touched. After
writing, `--verify` (on by default) greps the export for the same patterns and
FAILS the build if any survive -- a scrubber that silently misses is worse than
no scrubber, because it is trusted.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DOCS = ["README.md", "LICENSE", ".gitignore",
        "analysis/README.md", "external_baselines/README.md",
        "reports/archive/README.md", "ckpts/CHECKPOINTS.md"]
DIAGNOSTICS_DIR = Path("reports/external_baselines")
CHECKPOINT = Path("ckpts/vqvae_stage2a_best.pt")

SKIP_DIR = {"__pycache__", ".git", ".pytest_cache", ".ipynb_checkpoints"}


# Identity that must not reach a reviewer. Each entry is (regex, replacement).
# The absolute-path rule runs before the bare-username rule so that a path like
# /media/derik/... collapses to one placeholder instead of two.
SCRUB = [
    (re.compile(r"^\s*@author:.*$", re.M), "@author: anonymised"),
    (re.compile(r"/media/[^\s\"\']*?/organoid_data"), "/PROJECT_ROOT"),
    (re.compile(r"/media/[A-Za-z0-9_.-]+/[A-Za-z0-9 _.-]+"), "/PROJECT_ROOT"),
    # Bare `/media/` with no resolvable path after it -- a prose mention in a
    # comment. Caught last so the two specific rules above win where they can.
    (re.compile(r"/media/"), "/PROJECT_ROOT/"),
    (re.compile(r"(?:git@|https://)github\.com[:/][^\s\"\')]+"), "ANONYMISED_REMOTE"),
    (re.compile(r"Seagate[ _]?Desktop[ _]?Drive", re.I), "PROJECT_ROOT"),
    # Case-insensitive: the LICENSE carries the name as `TanveerDerik`, and a
    # case-sensitive rule passed the build while leaving it in the export.
    (re.compile(r"\btanveerderik\b", re.I), "anonymised"),
    (re.compile(r"\bderik\b", re.I), "anonymised"),
]

# Only text is rewritten. A .pt is a tensor archive and a regex over it would
# corrupt the file while appearing to succeed.
TEXT_SUFFIX = {".py", ".md", ".txt", ".cfg", ".toml", ".yaml", ".yml",
               ".json", ".bib", ".tex", ".gitignore", ""}


def scrub(text: str) -> str:
    for pat, rep in SCRUB:
        text = pat.sub(rep, text)
    return text


def _copy(rel: Path, dest: Path) -> int:
    src = ROOT / rel
    if not src.is_file():
        return 0
    out = dest / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in TEXT_SUFFIX:
        try:
            out.write_text(scrub(src.read_text(encoding="utf-8")),
                           encoding="utf-8")
            shutil.copystat(src, out)
            return out.stat().st_size
        except UnicodeDecodeError:
            pass  # not text after all; fall through to a byte copy
    shutil.copy2(src, out)
    return src.stat().st_size


def verify_anonymous(dest: Path) -> list[str]:
    """Grep the finished export. Returns the offending "path:line" strings."""
    bad = []
    for f in sorted(dest.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in TEXT_SUFFIX:
            continue
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for i, line in enumerate(lines, 1):
            for pat, _ in SCRUB:
                if pat.search(line) and "anonymised" not in line.lower() \
                        and "PROJECT_ROOT" not in line \
                        and "ANONYMISED_REMOTE" not in line:
                    bad.append(f"{f.relative_to(dest)}:{i}: {line.strip()[:100]}")
                    break
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=str(ROOT.parent / "MAGVIT_release"))
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="omit the 18MB VQ-VAE checkpoint")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing destination")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the post-build anonymity check (not advised: "
                         "identity in a supplement is an ICLR desk reject)")
    a = ap.parse_args()

    dest = Path(a.dest).resolve()
    if dest == ROOT or ROOT in dest.parents:
        raise SystemExit(f"refusing to write inside the working repo: {dest}")
    if dest.exists():
        if not a.force:
            raise SystemExit(f"{dest} exists; pass --force to replace it")
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    total = n_py = 0
    for src in sorted(ROOT.rglob("*.py")):
        if any(p in SKIP_DIR for p in src.relative_to(ROOT).parts):
            continue
        total += _copy(src.relative_to(ROOT), dest)
        n_py += 1

    n_doc = 0
    for d in DOCS:
        s = _copy(Path(d), dest)
        total += s
        n_doc += bool(s)

    n_diag = 0
    for md in sorted((ROOT / DIAGNOSTICS_DIR).glob("*.md")):
        total += _copy(md.relative_to(ROOT), dest)
        n_diag += 1

    ck_mb = 0.0
    if not a.no_checkpoint:
        s = _copy(CHECKPOINT, dest)
        total += s
        ck_mb = s / 1e6
        if not s:
            print(f"WARNING: {CHECKPOINT} not found; released without it",
                  file=sys.stderr)

    (dest / "RELEASE.md").write_text(f"""# Release export

Built by `tools/make_release.py` from the working repository.

## Contents

| | |
|---|---|
| Python sources | {n_py} files |
| Documentation | {n_doc} files |
| Rendered diagnostics | {n_diag} markdown files under `reports/external_baselines/` |
| Checkpoints | {'none' if a.no_checkpoint else f'`{CHECKPOINT.as_posix()}` ({ck_mb:.1f} MB)'} |
| Total | {total / 1e6:.1f} MB |

## What was excluded, and how to regenerate it

**Checkpoints.** All except the Stage-2A VQ-VAE. `ckpts/CHECKPOINTS.md` is
included and lists every one of them, what produced it, and which ship, so the
omission is auditable. Regenerate by running the pipeline stages in
`main.py` (`TRAIN_STAGES`).

**`reports/*.json`.** Training and evaluation reports, regenerated by the
stage that writes them. The rendered `.md` diagnostics ARE included, because
those are the readable results.

**Snapshot trees** (`ckpts_oldsplit_backup/`, `ckpts_pre2Cfix/`,
`ckpts_pre_v2/`, `reports_oldsplit_backup/`, `reports_pre2Cfix/`). Local
restore points predating specific changes; listed in `ckpts/CHECKPOINTS.md`.

**`.log`, `.bak_*`, `__pycache__`.** Run output and editor backups.

## Note on scale

Parameter cost does not grow with the number of preparations: the pipeline
stores zero per-assay parameters, because `gct` is a seeded random code rather
than a stored table. That is a *scalability* claim, not a transfer claim -- a
new preparation still requires training exposure.
""")

    print(f"wrote {dest}")
    print(f"  {n_py} python, {n_doc} docs, {n_diag} diagnostics"
          f"{'' if a.no_checkpoint else ', 1 checkpoint'}")
    print(f"  {total / 1e6:.1f} MB total")

    if a.no_verify:
        print("  anonymity NOT verified (--no-verify)")
        return 0
    bad = verify_anonymous(dest)
    if bad:
        print(f"\nANONYMITY CHECK FAILED: {len(bad)} line(s) still identify "
              f"the authors.", file=sys.stderr)
        for b in bad[:20]:
            print(f"  {b}", file=sys.stderr)
        if len(bad) > 20:
            print(f"  ... and {len(bad) - 20} more", file=sys.stderr)
        return 1
    print("  anonymity verified: no author name, absolute path or remote URL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
