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
STAGE4B = ROOT / "training" / "stage4_activity.py"
TRAIN_VQVAE = ROOT / "training" / "train_vqvae.py"
TRAIN_SPATIAL = ROOT / "training" / "train_spatial_prior.py"
VQVAE = ROOT / "model" / "vqvae.py"
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
            "what": "Tokenizer (3-level residual VQ-VAE)",
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
            "trained": "lct trunk + heads (tokenizer frozen)",
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
            "trained": "motif prior (tokenizer and mappers frozen)",
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
                # loss_weights[1] is not listed: the alpha head it weighted
                # was removed, and model/prior.py reads element 0 only. It is
                # recorded in the `disabled` block instead.
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


# ---------------------------------------------------------------------------
# The objective decomposition.
#
# The paper's Method names each loss term and says what it constrains; the
# appendix has to be able to list every term with its coefficient, or the
# pointer the Method makes is a dead end. The COEFFICIENTS are read from source
# exactly as the table's are. The one-line descriptions are prose and live here
# in the generator, which is version-controlled -- the standing rule is that no
# NUMBER is typed by hand, not that no word is.
#
# Resolution order for Stage 2A matters and is not obvious: a keyword at the
# `fit_vqvae` call site wins; failing that, `common_fit_kwargs`, which is
# splatted into the same call; failing that, the signature default. Several
# terms differ between those layers -- `lambda_ctx` defaults to 1e-3 and is
# called at 2.8, `lambda_code_norm` defaults to 0.0 and is called at 0.1 -- so
# reading the wrong layer silently reports a coefficient the run never used.
# ---------------------------------------------------------------------------


def calls_of(path: Path, func: str, callee: str) -> list[dict[str, Any]]:
    """Every `callee(...)` inside `func`, not just the first.

    Stage 1 calls `support_map_loss` twice with different weights, once for the
    token arm and once for the pixel arm.
    """
    out = []
    for node in ast.walk(_function(path, func)):
        if isinstance(node, ast.Call) and _callee_name(node) == callee:
            kw: dict[str, Any] = {}
            for k in node.keywords:
                if k.arg is None:
                    continue
                try:
                    kw[k.arg] = _literal(k.value)
                except (MissingSetting, ValueError):
                    continue
            out.append(kw)
    if not out:
        raise MissingSetting(f"{path.name}:{func}: no call to {callee}")
    return out


def _stage2a_coeffs() -> dict[str, Any]:
    """Stage 2A's coefficients, resolved through the three layers above."""
    call = kwargs_of(MAIN, "run_stage2a", "fit_vqvae")
    common = kwargs_of(MAIN, "common_fit_kwargs", "dict")
    sig = _defaults_of(TRAIN_VQVAE, "fit_vqvae")

    splatted = any(
        isinstance(n, ast.Call) and _callee_name(n) == "fit_vqvae"
        and any(k.arg is None and isinstance(k.value, ast.Call)
                and _callee_name(k.value) == "common_fit_kwargs"
                for k in n.keywords)
        for n in ast.walk(_function(MAIN, "run_stage2a")))
    if not splatted:
        raise MissingSetting("run_stage2a no longer splats common_fit_kwargs "
                             "into fit_vqvae; the middle layer is stale")

    def pick(name: str) -> Any:
        for layer in (call, common, sig):
            if name in layer:
                return layer[name]
        raise MissingSetting(f"stage 2A: no value anywhere for {name}")

    out = {n: pick(n) for n in (
        "lambda_vq", "lambda_isi", "lambda_ctx", "lambda_ctx_field",
        "lambda_code_norm", "lambda_sp_token", "lambda_sp_pixel",
        "lambda_blank", "lambda_blank_sep", "lambda_enc_var",
        "lambda_tok_entropy")}
    # lambda_ctx is the one coefficient no layer reports correctly. The call
    # site passes `lambda_ctx=STAGE2_LAMBDA_CTX`, whose module value is 0.1 --
    # but `run_stage2a` rebinds that global to STAGE2_LAMBDA_CTX_BALANCED
    # (2.8) through `globals()[...]` before the call, so the static read is
    # 28x low. Follow the rebind, and assert it is still there rather than
    # assuming it.
    rebound = any(
        isinstance(n, ast.Call) and _callee_name(n) == "globals"
        for n in ast.walk(_function(MAIN, "run_stage2a")))
    if M.STAGE2_BALANCED_CTX and not rebound:
        raise MissingSetting(
            "STAGE2_BALANCED_CTX is on but run_stage2a no longer rebinds "
            "STAGE2_LAMBDA_CTX; the table would report the unbalanced 0.1")
    out["lambda_ctx"] = float(M.STAGE2_LAMBDA_CTX_BALANCED
                              if M.STAGE2_BALANCED_CTX else M.STAGE2_LAMBDA_CTX)
    vq = _defaults_of(VQVAE, "__init__")
    out["vq_beta"] = vq["vq_beta"]
    out["usage_loss_weight"] = vq["usage_loss_weight"]
    out["vq_decay"] = vq["vq_decay"]
    out["pos_weight"] = (f"{call['pos_weight_start']:.0f} to "
                         f"{call['pos_weight_end']:.0f} over "
                         f"{call['pos_decay_epochs']} epochs")
    return out


def _term(name: str, coeff: Any, constrains: str) -> dict[str, Any]:
    return {"term": name, "coeff": coeff, "constrains": constrains}


def objectives() -> list[dict[str, Any]]:
    s1_fit = kwargs_of(MAIN, "run_stage1_gct_pretrain", "fit_spatial_prior_pretrain")
    s1_sup = calls_of(TRAIN_SPATIAL, "fit_spatial_prior_pretrain", "support_map_loss")
    s1_sep = calls_of(TRAIN_SPATIAL, "fit_spatial_prior_pretrain",
                      "spatial_support_separation_loss")[0]
    s1_sig = _defaults_of(TRAIN_SPATIAL, "fit_spatial_prior_pretrain")
    if len(s1_sup) < 2:
        raise MissingSetting("stage 1: expected a token arm and a pixel arm")

    c2 = _stage2a_coeffs()
    s4a = kwargs_of(MAIN, "run_stage4a", "train_motif_prior_mgit")
    s4b = dict(M.STAGE4B_MASKGIT_HYPERPARAMETERS)
    s4b_sig = _defaults_of(STAGE4B, "train_maskgit_activity_prior")

    return [
        {
            "stage": "1",
            "symbol": "\\mathcal{L}_{\\mathrm{global}}",
            "terms": [
                _term("support, token grid",
                      f"outside {s1_sup[0]['outside_w']}, mass {s1_sup[0]['mass_w']}",
                      "cover the recording's active sites, penalize mass "
                      "outside them, match the total"),
                _term("support, full array",
                      f"outside {s1_sup[1]['outside_w']}, mass {s1_sup[1]['mass_w']}",
                      "the same at electrode resolution; ramped in as the "
                      "token arm decays"),
                _term("separation",
                      _require(s1_fit, "lambda_sep", "stage 1"),
                      "repel the predicted maps of recordings whose true maps "
                      f"already differ, at margin {s1_sep['margin']}"),
                _term("adjacency",
                      _require(s1_sig, "lambda_adj", "stage 1 signature"),
                      "match the recording's same-site short-gap rates"),
            ],
        },
        {
            "stage": "2A",
            "symbol": "\\mathcal{L}_{\\mathrm{tok}}",
            "terms": [
                _term("tolerant spike reconstruction", c2["pos_weight"],
                      "exact BCE at a decaying positive weight, plus "
                      "max-pooled hit, peak-margin and multi-count terms"),
                _term("vector quantization", c2["lambda_vq"],
                      f"commitment at $\\beta = {c2['vq_beta']}$ plus a usage-entropy "
                      f"term at {c2['usage_loss_weight']} against a detached "
                      f"codebook; codebooks are EMA at decay {c2['vq_decay']}"),
                _term("context", c2["lambda_ctx"],
                      "the nine local moments recomputed from the decoder's "
                      "own logits must match the clip's true descriptor"),
                _term("context field", c2["lambda_ctx_field"],
                      "the same nine moments per patch rather than per clip"),
                _term("spatial, token grid", c2["lambda_sp_token"],
                      "no mass where the recording's support map forbids it"),
                _term("spatial, full array", c2["lambda_sp_pixel"],
                      "the same at electrode resolution, ramped in late"),
                _term("short-gap rate", c2["lambda_isi"],
                      "same-site inter-spike gaps must not exceed or fall "
                      "short of the recording's measured rates"),
                _term("code norm", c2["lambda_code_norm"],
                      "a soft ceiling on encoder output norms, so codes do "
                      "not drift out of the EMA's reach"),
                _term("blank hinge", c2["lambda_blank"],
                      "empty patches must not produce spike logits"),
                _term("blank/active separation", c2["lambda_blank_sep"],
                      "the decoder's blank and content embeddings stay apart"),
                _term("encoder isotropy", c2["lambda_enc_var"],
                      "a variance floor and an off-diagonal correlation "
                      "penalty, against encoder collapse"),
            ],
        },
        {
            "stage": "3",
            "symbol": "\\mathcal{L}_{\\mathrm{local}}",
            "terms": [
                _term("texton multinomial NLL", 1.0,
                      "the predicted distribution over the "
                      f"{int(M.STAGE3_NUM_TEXTONS)}-texton basis must match "
                      "the clip's soft texton histogram"),
                _term("flat-code multinomial NLL", 1.0,
                      "the same against the $V{=}961$ alphabet"),
                _term("regional summary", 1.0,
                      "squared error on per-code mean time and per-frame "
                      "active fraction"),
            ],
        },
        {
            "stage": "4A / 4C",
            "symbol": "\\mathcal{L}_{M}",
            "terms": [
                _term("categorical cross-entropy", 1.0,
                      "the correct motif at each masked active cell"),
                _term("neighborhood CE",
                      _require(s4a, "lambda_z1_neighbor_ce", "stage 4A"),
                      "partial credit over the five codebook-nearest "
                      "alternatives at temperature "
                      f"{_require(s4a, 'z1_neighbor_tau', 'stage 4A')}, so a "
                      "near miss in alphabet geometry is not a total miss"),
                _term("expected code distance",
                      _require(s4a, "lambda_z1_distance", "stage 4A"),
                      "the whole predicted distribution is pulled toward the "
                      "target's neighborhood, not just its mode"),
                _term("context", _require(s4a, "lambda_ctx", "stage 4A"),
                      "soft-decode the motif logits through the frozen "
                      "tokenizer; the result's nine moments must match the clip"),
                _term("context field",
                      _require(s4a, "lambda_ctx_field", "stage 4A"),
                      "the same moment match evaluated per patch, so a clip "
                      "cannot be right on average and wrong everywhere"),
                _term("adjacency", _require(s4a, "lambda_adj", "stage 4A"),
                      "the decoded volume's short-gap rates must match the "
                      "recording's"),
                _term("spatial", _require(s4a, "lambda_spatial", "stage 4A"),
                      "no decoded mass outside the recording's support map"),
            ],
        },
        {
            "stage": "4B",
            "symbol": "\\mathcal{L}_{A}",
            "terms": [
                _term("per-cell BCE", float(s4b["lambda_bce"]),
                      "active against blank at each token cell, at positive "
                      f"weight {float(s4b['pos_weight'])} -- deliberately not "
                      "the class-balancing ratio, which over-produces"),
                _term("count", float(s4b["lambda_count"]),
                      "one categorical over the clip's total active count, "
                      "with an ordinal neighbor term at "
                      f"{_require(s4b_sig, 'lambda_count_neighbor', 'stage 4B')} "
                      "and an expected-distance term at "
                      f"{_require(s4b_sig, 'lambda_count_distance', 'stage 4B')} "
                      "so a near-miss count is scored as one"),
                _term("temporal co-activation", float(s4b["lambda_adj_t"]),
                      "the predicted field's co-activation rate along time "
                      "must match the batch's"),
                _term("spatial co-activation", float(s4b["lambda_adj_s"]),
                      "the same co-activation match taken across the array "
                      "instead of along time"),
                _term("spatial support", float(s4b["lambda_spatial"]),
                      "no predicted activity in columns the true map leaves "
                      "empty"),
            ],
        },
    ]


def disabled() -> list[dict[str, str]]:
    """Terms present in the code and inactive in the shipped configuration.

    A reader who finds these in the release should not have to work out whether
    they ran. Each is asserted to still be off, so this list cannot go stale in
    the direction that matters.
    """
    sig_vq = _defaults_of(TRAIN_VQVAE, "fit_vqvae")
    s4a = kwargs_of(MAIN, "run_stage4a", "train_motif_prior_mgit")
    out = []

    tok_ent = _stage2a_coeffs()["lambda_tok_entropy"]
    if float(tok_ent) != 0.0:
        raise MissingSetting(f"lambda_tok_entropy is now {tok_ent}, not 0")
    out.append({"where": "Stage 2A", "what": "token-profile entropy",
                "why": "the within-token blur experiment; reported as a "
                       "failure and not shipped"})

    if float(sig_vq.get("lambda_tok_entropy", 0.0)) != 0.0:
        raise MissingSetting("fit_vqvae's own default is no longer 0")

    lw = list(_require(s4a, "loss_weights", "stage 4A"))
    if len(lw) < 2:
        raise MissingSetting("loss_weights is no longer a pair; the dead "
                             "second element may have been cleaned up")
    out.append({"where": "Stage 4A/4C",
                "what": f"the second entry of loss_weights ({lw[1]})",
                "why": "the alpha head it weighted was removed; the motif "
                       "prior reads loss_weights[0] only"})
    out.append({"where": "Stage 2A", "what": "false-positive weighting",
                "why": "tolerant_spike_loss is called with fp_weight 0, so "
                       "the hallucination penalty is inactive"})
    out.append({"where": "Stage 2A", "what": "latent masking, CFG dropout",
                "why": "both off in the shipped run"})
    return out


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
        "objectives": objectives(),
        "disabled": disabled(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    n_terms = sum(len(o["terms"]) for o in payload["objectives"])
    print(f"wrote {OUT}  ({len(payload['stages'])} stages, {n_terms} loss terms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
