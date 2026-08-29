#!/usr/bin/env python3
"""Extract the shipped pipeline's training hyperparameters into JSON.

The paper needs a conventional hyperparameter table, and the standing rule is
that no number is typed by hand. Most settings live in module-level constants
in `main.py`, which can simply be imported; the rest are literal keyword
arguments inside the `run_stage*` functions, which cannot. Those are read out
of the source with `ast`.

Reading them rather than copying them has one property worth the extra code: if
a call is renamed or an argument moves, this script raises instead of silently
reporting a stale value. A hand-maintained table would not.

Only the shipped stages appear. Rejected designs (Stage 4B-refine, the
cross-attention branch, the continuous adapter) are excluded -- the table
describes what the paper ships, and the negative results have their own
appendix section.

    python tools/extract_hparams.py

Writes reports/hyperparameters.json, which tools/make_paper_tables.py renders.
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

import MAGVIT_project.main as M                                # noqa: E402

MAIN = ROOT / "main.py"
STAGE3 = ROOT / "training" / "stage3_lct.py"
OUT = ROOT / "reports" / "hyperparameters.json"


class MissingSetting(RuntimeError):
    """A setting the table promises could not be found in the source."""


def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _function(path: Path, name: str) -> ast.FunctionDef:
    for node in ast.walk(_module(path)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise MissingSetting(f"{path.name}: no function named {name}")


def _callee_name(node: ast.Call) -> str:
    """`torch.optim.AdamW(...)` -> 'AdamW'; `fit_vqvae(...)` -> 'fit_vqvae'."""
    f = node.func
    while isinstance(f, ast.Attribute):
        f = f.attr if isinstance(f.attr, str) else f.value
        if isinstance(f, str):
            return f
    return f.id if isinstance(f, ast.Name) else ""


def _literal(node: ast.AST) -> Any:
    """Resolve an argument to a value.

    Three forms occur in these call sites: a bare literal (`lr=3e-4`), a
    module-level constant (`epochs=STAGE4A_EPOCHS`), and a constant wrapped in
    a cast (`lr=float(STAGE4C_LR)`). Anything else is deliberately not
    resolved -- guessing at an expression is how a table goes quietly wrong.
    """
    if isinstance(node, ast.Call) and _callee_name(node) in {"int", "float", "str", "tuple"}:
        return _literal(node.args[0])
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        base = _literal(node.value)
        if isinstance(base, dict):
            return base[node.slice.value]
        raise MissingSetting(f"cannot subscript {ast.dump(node.value)[:60]}")
    if isinstance(node, ast.Name):
        if not hasattr(M, node.id):
            raise MissingSetting(f"main.py has no constant {node.id}")
        return getattr(M, node.id)
    if isinstance(node, ast.Attribute):
        raise MissingSetting(f"runtime attribute {node.attr} is not a constant")
    return ast.literal_eval(node)


def kwargs_of(path: Path, func: str, callee: str) -> dict[str, Any]:
    """Keyword arguments of the first `callee(...)` inside `func`.

    Values that cannot be resolved statically are dropped rather than guessed;
    every key the table actually reads is asserted present by `_require`.
    """
    for node in ast.walk(_function(path, func)):
        if isinstance(node, ast.Call) and _callee_name(node) == callee:
            out: dict[str, Any] = {}
            for kw in node.keywords:
                if kw.arg is None:
                    continue
                try:
                    out[kw.arg] = _literal(kw.value)
                except (MissingSetting, ValueError):
                    continue
            return out
    raise MissingSetting(f"{path.name}:{func}: no call to {callee}")


def stage4b_lr() -> float:
    """Stage 4B's learning rate, which the call site reaches indirectly.

    `run_stage4b` builds `config = dict(STAGE4B_HYPERPARAMETERS)`, updates it
    with runtime shapes, and passes `lr=float(config["lr"])`. The AST cannot
    resolve a local dict, so the value is taken from the constant -- but only
    after checking that the local really is that constant and that nothing
    overwrites `lr` in between. If either assumption breaks, this raises rather
    than reporting a number the code no longer uses.
    """
    fn = _function(MAIN, "run_stage4b")
    seeded = any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "config" for t in n.targets)
        and isinstance(n.value, ast.Call) and _callee_name(n.value) == "dict"
        and isinstance(n.value.args[0], ast.Name)
        and n.value.args[0].id == "STAGE4B_HYPERPARAMETERS"
        for n in ast.walk(fn)
    )
    if not seeded:
        raise MissingSetting("run_stage4b no longer seeds config from "
                             "STAGE4B_HYPERPARAMETERS")
    for n in ast.walk(fn):
        if (isinstance(n, ast.Call) and _callee_name(n) == "update"
                and n.args and isinstance(n.args[0], ast.Dict)):
            keys = [k.value for k in n.args[0].keys if isinstance(k, ast.Constant)]
            if "lr" in keys:
                raise MissingSetting("run_stage4b overwrites config['lr']; the "
                                     "table would report the wrong rate")
    return float(M.STAGE4B_HYPERPARAMETERS["lr"])


def _require(d: dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise MissingSetting(f"{where}: expected keyword {key!r}, found "
                             f"{sorted(d)}")
    return d[key]


def _defaults_of(path: Path, func: str) -> dict[str, Any]:
    """Default values of a function's keyword-only parameters.

    Stage 3 takes its patience from its own signature default rather than from
    a constant in `main.py`, so the signature is the source of truth for it.
    """
    fn = _function(path, func)
    args = fn.args
    out: dict[str, Any] = {}
    for name, default in zip(args.kwonlyargs, args.kw_defaults):
        if default is not None:
            try:
                out[name.arg] = _literal(default)
            except (MissingSetting, ValueError):
                continue
    for name, default in zip(args.args[len(args.args) - len(args.defaults):],
                             args.defaults):
        try:
            out[name.arg] = _literal(default)
        except (MissingSetting, ValueError):
            continue
    return out


def stages() -> list[dict[str, Any]]:
    s1_opt = kwargs_of(MAIN, "run_stage1_gct_pretrain", "AdamW")
    s1_fit = kwargs_of(MAIN, "run_stage1_gct_pretrain", "fit_spatial_prior_pretrain")

    s2_opt = kwargs_of(MAIN, "run_stage2a", "make_optimizer")
    s2_sch = kwargs_of(MAIN, "run_stage2a", "CosineAnnealingLR")
    s2_fit = kwargs_of(MAIN, "run_stage2a", "fit_vqvae")

    s3_sig = _defaults_of(STAGE3, "run_stage3_lct")
    s3_opt = kwargs_of(STAGE3, "run_stage3_lct", "AdamW")
    s3_sch = kwargs_of(STAGE3, "run_stage3_lct", "CosineAnnealingLR")

    s4a_opt = kwargs_of(MAIN, "run_stage4a", "AdamW")
    s4a_fit = kwargs_of(MAIN, "run_stage4a", "train_motif_prior_mgit")

    s4b_opt = kwargs_of(MAIN, "run_stage4b", "AdamW")
    s4b_fit = kwargs_of(MAIN, "run_stage4b", "train_maskgit_activity_prior")
    s4b = dict(M.STAGE4B_MASKGIT_HYPERPARAMETERS)

    s4c_opt = kwargs_of(MAIN, "run_stage4c", "AdamW")
    s4c_fit = kwargs_of(MAIN, "run_stage4c", "train_motif_prior_mgit")

    # Stage 4B reads its learning rate from STAGE4B_HYPERPARAMETERS and its
    # schedule from STAGE4B_MASKGIT_HYPERPARAMETERS. The two dicts are merged
    # at the call site, so the table has to follow the same merge or it will
    # report the wrong learning rate.
    s4b_lr = stage4b_lr()

    return [
        {
            "stage": "1",
            "what": "Global-code mapper pretrain",
            "trained": "global embedder + spatial-map head",
            "optimizer": "AdamW",
            "lr": _require(s1_opt, "lr", "stage 1"),
            "weight_decay": _require(s1_opt, "weight_decay", "stage 1"),
            "scheduler": "none",
            "warmup_epochs": 0,
            "epochs": _require(s1_fit, "epochs", "stage 1"),
            "grad_clip": None,
            "early_stop_patience": _require(s1_fit, "early_stop_patience", "stage 1"),
            "select_on": "spatial-map loss",
            "loss_terms": {"lambda_sep": _require(s1_fit, "lambda_sep", "stage 1")},
        },
        {
            "stage": "2A",
            "what": "Tokeniser (3-level residual VQ-VAE)",
            "trained": "encoder, decoder, EMA codebooks",
            "optimizer": "AdamW",
            "lr": _require(s2_opt, "lr", "stage 2A"),
            "weight_decay": _require(s2_opt, "weight_decay", "stage 2A"),
            "scheduler": "cosine",
            "eta_min": _require(s2_sch, "eta_min", "stage 2A"),
            "warmup_epochs": 0,
            "epochs": int(M.STAGE2_EPOCHS),
            "grad_clip": None,
            "early_stop_patience": _require(s2_fit, "early_stop_patience", "stage 2A"),
            "select_on": "val exact AUPRC",
            "loss_terms": {
                "lambda_ctx": float(M.STAGE2_LAMBDA_CTX_BALANCED
                                    if M.STAGE2_BALANCED_CTX
                                    else M.STAGE2_LAMBDA_CTX),
                "lambda_ctx_field": _require(s2_fit, "lambda_ctx_field", "stage 2A"),
                "lambda_code_norm": _require(s2_fit, "lambda_code_norm", "stage 2A"),
                "pos_weight": f"{_require(s2_fit, 'pos_weight_start', 'stage 2A'):.0f}"
                              f" to {_require(s2_fit, 'pos_weight_end', 'stage 2A'):.0f}"
                              f" over {_require(s2_fit, 'pos_decay_epochs', 'stage 2A')}"
                              " epochs",
            },
        },
        {
            "stage": "3",
            "what": "Local-code mapper (texton multinomial)",
            "trained": "lct trunk + heads (tokeniser frozen)",
            "optimizer": "AdamW",
            "lr": _require(s3_opt, "lr", "stage 3"),
            "weight_decay": _require(s3_opt, "weight_decay", "stage 3"),
            "scheduler": "cosine",
            "eta_min": _require(s3_sch, "eta_min", "stage 3"),
            "warmup_epochs": 0,
            "epochs": int(M.STAGE3_EPOCHS),
            "grad_clip": None,
            "early_stop_patience": _require(s3_sig, "patience", "stage 3"),
            "select_on": "val multinomial NLL",
            "loss_terms": {
                "n_textons": int(M.STAGE3_NUM_TEXTONS),
                "texton_basis": str(M.STAGE3_TEXTON_BASIS),
                "batches_per_epoch": int(M.STAGE3_BATCHES_PER_EPOCH),
            },
        },
        {
            "stage": "4A",
            "what": "Motif prior (MaskGIT over V=961)",
            "trained": "motif prior (tokeniser and mappers frozen)",
            "optimizer": "AdamW",
            "lr": _require(s4a_opt, "lr", "stage 4A"),
            "weight_decay": _require(s4a_opt, "weight_decay", "stage 4A"),
            "scheduler": "warmup + cosine",
            "warmup_epochs": int(M.STAGE4A_WARMUP_EPOCHS),
            "epochs": int(M.STAGE4A_EPOCHS),
            "grad_clip": _require(s4a_fit, "grad_clip", "stage 4A"),
            "early_stop_patience": _require(s4a_fit, "early_stop_patience", "stage 4A"),
            "select_on": f"val {str(M.STAGE4A_SELECT_ON).upper()}",
            "loss_terms": {
                "$(z_1,\\alpha)$ weights": list(_require(s4a_fit, "loss_weights", "stage 4A")),
                "lambda_z1_neighbor_ce": _require(s4a_fit, "lambda_z1_neighbor_ce", "stage 4A"),
                "lambda_ctx": _require(s4a_fit, "lambda_ctx", "stage 4A"),
                "lambda_adj": _require(s4a_fit, "lambda_adj", "stage 4A"),
                "lambda_spatial": _require(s4a_fit, "lambda_spatial", "stage 4A"),
                "full_mask_prob": _require(s4a_fit, "full_mask_prob", "stage 4A"),
            },
        },
        {
            "stage": "4B",
            "what": "Activity prior (where, plus count head)",
            "trained": "activity prior",
            "optimizer": "AdamW",
            "lr": s4b_lr,
            "weight_decay": _require(s4b_opt, "weight_decay", "stage 4B"),
            "scheduler": "warmup + cosine",
            "warmup_epochs": int(s4b["warmup_epochs"]),
            "epochs": int(s4b["epochs"]),
            "grad_clip": _require(s4b_fit, "grad_clip", "stage 4B"),
            "early_stop_patience": int(s4b["early_stop_patience"]),
            "select_on": f"val {str(s4b['select_on']).upper()}",
            "loss_terms": {
                "lambda_bce": float(s4b["lambda_bce"]),
                "pos_weight": float(s4b["pos_weight"]),
                "lambda_count": float(s4b["lambda_count"]),
                "lambda_spatial": float(s4b["lambda_spatial"]),
                "random_mask_prob": float(s4b["random_mask_prob"]),
            },
        },
        {
            "stage": "4C",
            "what": "Adaptation of the motif prior to emitted maps",
            "trained": "motif prior (activity prior frozen)",
            "optimizer": "AdamW",
            "lr": _require(s4c_opt, "lr", "stage 4C"),
            "weight_decay": _require(s4c_opt, "weight_decay", "stage 4C"),
            "scheduler": "warmup + cosine",
            "warmup_epochs": int(M.STAGE4C_WARMUP_EPOCHS),
            "epochs": int(M.STAGE4C_EPOCHS),
            "grad_clip": _require(s4c_fit, "grad_clip", "stage 4C"),
            "early_stop_patience": int(M.STAGE4C_EARLY_STOP_PATIENCE),
            "select_on": "val MRR",
            "loss_terms": {
                "emitted-map ramp": f"0 to {M.STAGE4C_MAX_P} over "
                                    f"{M.STAGE4C_RAMP_EPOCHS} epochs",
                "readout": str(M.STAGE4C_READOUT),
                "soft_field": bool(M.STAGE4C_SOFT_FIELD),
            },
        },
    ]


def main() -> int:
    payload = {
        "shared": {
            "batch_size": int(M.batch_size),
            "grad_accum_steps": int(M.grad_accum_steps),
            "effective_batch": int(M.batch_size) * int(M.grad_accum_steps),
            "prior_dropout": 0.1,
            "precision": "CUDA autocast, FP16",
            # Every stage inherits the process seed; only Stage 3 sets its own,
            # and Stage 4C's seed sweep is reported separately in the seed
            # appendix rather than here.
            "seed": int(_defaults_of(STAGE3, "run_stage3_lct")["seed"]),
        },
        "stages": stages(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT}  ({len(payload['stages'])} stages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
