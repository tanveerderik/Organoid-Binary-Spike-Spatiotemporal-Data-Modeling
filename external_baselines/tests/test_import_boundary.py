#!/usr/bin/env python3
"""Enforce the one-directional dependency between the pipeline and baselines/.

Rules, in order of how much damage a violation does:

  1. NO main-pipeline module may import `external_baselines`. The shipped results must
     not be able to change because a baseline changed. This is the one that
     actually protects the paper.
  2. Only the two files in MAIN_IMPORT_ALLOWED may import `main`. Every other baseline
     module reaches the pipeline through the shared harness, so a refactor of
     `main.py` breaks one import site instead of five.
  3. Every file under external_baselines/ must parse. A bulk rename here once produced
     `STAGE4B-refine_...`, which `import` did not catch because the module was
     imported lazily -- so parse them all explicitly.

Run standalone (no pytest needed):

    python external_baselines/tests/test_import_boundary.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINES = ROOT / "external_baselines"

PIPELINE_DIRS = ("model", "training", "inference", "utils", "analysis", "visualization")
PIPELINE_FILES = ("main.py", "dataset.py")

# The sanctioned bridges. Two, and the split is deliberate: `data.py` bridges
# the SPLIT and the loaders, `pipeline.py` bridges CONSTRUCTION of the shipped
# model for `pipeline_reference`. Anything else routes through them.
MAIN_IMPORT_ALLOWED = {
    BASELINES / "common" / "data.py",
    BASELINES / "common" / "pipeline.py",
}


def _imported_names(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.add(node.module)
            if node.level:                      # relative import
                out.add("." * node.level + (node.module or ""))
    return out


def _pipeline_files() -> list[Path]:
    files = [ROOT / f for f in PIPELINE_FILES]
    for d in PIPELINE_DIRS:
        files += sorted((ROOT / d).rglob("*.py"))
    return [f for f in files if f.exists() and "__pycache__" not in f.parts]


def _baseline_files() -> list[Path]:
    return [f for f in sorted(BASELINES.rglob("*.py")) if "__pycache__" not in f.parts]


def main() -> int:
    failures: list[str] = []
    checks = 0

    # -- rule 3 first: everything must parse, including the pipeline -------
    trees: dict[Path, ast.AST] = {}
    for f in _pipeline_files() + _baseline_files():
        try:
            trees[f] = ast.parse(f.read_text(), filename=str(f))
            checks += 1
        except SyntaxError as e:
            failures.append(f"SYNTAX  {f.relative_to(ROOT)}:{e.lineno}: {e.msg}")

    # -- rule 1: pipeline must not import baselines ------------------------
    for f in _pipeline_files():
        if f not in trees:
            continue
        for name in _imported_names(trees[f]):
            # Only absolute references to the top-level package count.
            # `training/__init__.py` does `from .baselines import ...`, which is
            # the INTERNAL nulls module `training/baselines.py` -- a different
            # thing, and the reason this package is called external_baselines.
            if name.startswith("."):
                continue
            if name.split(".")[0] == "external_baselines" or ".external_baselines" in name:
                failures.append(
                    f"BOUNDARY  {f.relative_to(ROOT)} imports {name!r} -- the main "
                    f"pipeline must never depend on external_baselines/"
                )
        checks += 1

    # -- rule 2: only common/data.py may import main -----------------------
    for f in _baseline_files():
        if f not in trees or f in MAIN_IMPORT_ALLOWED:
            continue
        for name in _imported_names(trees[f]):
            head = name.split(".")
            is_main = name == "main" or head[:1] == ["main"] or name.endswith(".main")
            if is_main and "MAGVIT_project" not in name.replace(".main", ""):
                pass
            if name in ("main", "MAGVIT_project.main"):
                failures.append(
                    f"BRIDGE  {f.relative_to(ROOT)} imports {name!r} -- route it "
                    f"through external_baselines/common/data.py instead"
                )
        checks += 1

    print(f"checked {checks} files")
    if failures:
        print(f"\n{len(failures)} violation(s):\n")
        for v in failures:
            print("  " + v)
        return 1
    print("import boundary intact")
    return 0


if __name__ == "__main__":
    sys.exit(main())
