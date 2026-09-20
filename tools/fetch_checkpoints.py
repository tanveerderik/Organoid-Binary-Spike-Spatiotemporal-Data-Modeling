#!/usr/bin/env python3
"""Download the learned weights the paper's numbers were produced with.

The checkpoints are ~92MB and are published as release assets rather than
committed: binaries in git history are permanent and this repository has
already been shrunk once because of them.

    python tools/fetch_checkpoints.py              # fetch what is missing
    python tools/fetch_checkpoints.py --list       # show, download nothing
    python tools/fetch_checkpoints.py --check      # verify what is on disk
    python tools/fetch_checkpoints.py --force      # re-download everything
    python tools/fetch_checkpoints.py --base-url URL/   # a mirror

Every file is pinned by SHA-256 in `reports/checkpoint_downloads.json`. A
checkpoint that downloads but does not match is deleted rather than kept: a
truncated or superseded file would otherwise reproduce silently wrong numbers,
and a wrong number that looks right is the failure this repository is built to
prevent. Files already on disk are hashed and skipped when they match, so
re-running is cheap and safe.

`--check` is the one to run before reporting a reproduction mismatch. It
answers whether the weights are the published ones before anyone starts
debugging the code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "ckpts"
INDEX = ROOT / "reports" / "checkpoint_downloads.json"

DEFAULT_BASE = ("https://github.com/tanveerderik/"
                "Organoid-Binary-Spike-Spatiotemporal-Data-Modeling/"
                "releases/download/checkpoints-v1/")


# The anonymous supplement is scrubbed of remote URLs, so this constant does
# not survive into it. That export ships the checkpoints directly, making the
# fetcher unnecessary there; say so rather than printing a mangled address.
ANONYMIZED = "ANONYMISED" in DEFAULT_BASE


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_index() -> dict:
    if not INDEX.is_file():
        sys.exit(f"{INDEX.relative_to(ROOT)} is missing. "
                 "Run: python tools/gen_checkpoint_checksums.py")
    return json.loads(INDEX.read_text(encoding="utf-8"))["checkpoints"]


def download(url: str, dest: Path, expect: str, size: int) -> bool:
    """Fetch to a temporary name, verify, then move into place."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "fetch_checkpoints"})
        with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as fh:
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if size:
                    pct = 100.0 * done / size
                    print(f"\r  {dest.name:38s} {pct:5.1f}%", end="", flush=True)
        print(f"\r  {dest.name:38s} ", end="")
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as exc:
        tmp.unlink(missing_ok=True)
        print(f"FAILED ({exc})")
        return False

    got = sha256(tmp)
    if got != expect:
        tmp.unlink(missing_ok=True)
        print(f"CHECKSUM MISMATCH\n     expected {expect}\n     got      {got}")
        return False
    tmp.replace(dest)
    print("ok")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default=DEFAULT_BASE,
                    help="where the assets live; must end with '/'")
    ap.add_argument("--list", action="store_true", help="show, download nothing")
    ap.add_argument("--check", action="store_true",
                    help="verify what is on disk, download nothing")
    ap.add_argument("--force", action="store_true", help="re-download everything")
    a = ap.parse_args()

    index = load_index()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    if ANONYMIZED and a.base_url == DEFAULT_BASE and not (a.list or a.check):
        sys.exit("This is the anonymized export: the checkpoints are already "
                 "in ckpts/, and the download URL is withheld for review. "
                 "Use --check to verify them, or --base-url to point "
                 "elsewhere.")

    if a.list:
        total = sum(m["bytes"] for m in index.values())
        for name, meta in sorted(index.items()):
            here = (CKPT_DIR / name).is_file()
            print(f"  {'present' if here else 'missing':8s} "
                  f"{meta['bytes']/1e6:7.1f} MB  {name}")
        print(f"\n{len(index)} checkpoints, {total/1e6:.1f} MB total")
        print("source: withheld in the anonymized export; the files are"
              " already in ckpts/" if ANONYMIZED and a.base_url == DEFAULT_BASE
              else f"source: {a.base_url}")
        return 0

    if a.check:
        bad, absent = [], []
        for name, meta in sorted(index.items()):
            p = CKPT_DIR / name
            if not p.is_file():
                absent.append(name)
            elif sha256(p) != meta["sha256"]:
                bad.append(name)
        for name in absent:
            print(f"  missing   {name}")
        for name in bad:
            print(f"  MISMATCH  {name}")
        ok = len(index) - len(bad) - len(absent)
        print(f"\n{ok}/{len(index)} verified"
              + (f", {len(absent)} missing" if absent else "")
              + (f", {len(bad)} MISMATCHED" if bad else ""))
        return 1 if (bad or absent) else 0

    if not a.base_url.endswith("/"):
        a.base_url += "/"

    fetched = skipped = failed = 0
    for name, meta in sorted(index.items()):
        dest = CKPT_DIR / name
        if dest.is_file() and not a.force:
            if sha256(dest) == meta["sha256"]:
                skipped += 1
                continue
            print(f"  {name:38s} on disk but does not match; re-fetching")
        if download(a.base_url + name, dest, meta["sha256"], meta["bytes"]):
            fetched += 1
        else:
            failed += 1

    print(f"\n{fetched} fetched, {skipped} already present, {failed} failed")
    if failed:
        print("Assets not published yet? Point --base-url at wherever they live.",
              file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
