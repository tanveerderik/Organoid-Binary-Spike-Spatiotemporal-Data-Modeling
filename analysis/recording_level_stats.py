#!/usr/bin/env python3
"""Re-test the headline comparisons with the recording, not the clip, as the unit.

Every paired test in reports/external_baselines treats each test clip as an
independent observation. The 279 test clips are 9 per recording, drawn with
replacement from that recording's burst windows, and recordings 10, 11 and 13
are byte-identical copies of 4, 3 and 6. This script recovers which recording
each clip came from, folds the duplicates together, and re-runs:

  family A  ours vs every learned arm          (as cross_model_tests.py)
  family B  static per-assay site map vs arm   (as derived_stats.py)

each on voxel AP and site AP, for all four tasks, at three levels: the original
clip-level Wilcoxon, a Wilcoxon on per-recording mean differences (n = 28), and
a two-level bootstrap CI (recordings within tissue, then clips within recording).

No model, data or GPU is needed: everything is read from the stored per-clip
results.

## How clip -> recording labels are recovered

There is no recording id stored per clip, so the mapping is reconstructed from
how the test set is built and then checked against independent evidence:

1. The test loader is a `DeterministicSubset` over an `NpzBurstDataset` with
   `balance_mode="equal"` and a quota of 9, iterated with `shuffle=False`
   (dataset.py `make_loaders_for_assays`). `_pick_index` maps item i to assay
   block `i // 9` over the dataset's assay list, which is the sorted assay ids.
   So on `recon` (no clips dropped) clip i belongs to `sorted_ids[i // 9]`.
2. Other tasks drop clips with no spike in the hole. `task_eval_nulls.json`
   stores the clip key (a hash that includes the assay index) for every kept
   clip; each task's key list is aligned to the recon key list as an ordered
   subsequence, which carries the label across.
3. Check A: identical keys must never span two recordings, and every recon
   block of 9 must be internally consistent with the keys.
4. Check B: the per-clip AP stored for the dumped clips (dumps/*.npz hold the
   decoded field of the first clip of assays 0 and 1 per task) is recomputed
   from the field and must match the stored per-clip value at the index the
   mapping predicts, and nowhere else in the list.

If a check fails the script stops without writing results.

    python analysis/recording_level_stats.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.stats import (DUPLICATE_OF, bh, hierarchical_bootstrap, holm,  # noqa: E402
                         merge_duplicates, paired_clip_test,
                         paired_recording_test)

DIR = ROOT / "reports" / "external_baselines"
OUT_JSON = ROOT / "reports" / "recording_level_stats.json"
OUT_MD = ROOT / "reports" / "recording_level_stats.md"
TASKS = ("recon", "causal", "noncausal", "spatial")
FAMILIES = (("per_clip", "voxel AP"), ("per_clip_site", "site AP"))
QUOTA = 9
ARM_KEY = "model_mc"
# (key, label, candidate payloads in order of preference) -- same files the
# diagnostics tables read (diagnose_table.py prefers the calibrated MaskGIT run).
ARMS = [
    ("pipeline", "Ours (4C+soft)", ["task_eval_pipeline.json"]),
    ("maskgit_flat", "MaskGIT-flat", ["task_eval_maskgit_flat_cal.json",
                                      "task_eval_maskgit_flat.json"]),
    ("unet3d", "3D U-Net (det.)", ["task_eval_unet3d.json"]),
    ("unet3d_seed2", "3D U-Net seed 2", ["task_eval_unet3d_seed2.json"]),
    ("cvae3d", "3D CVAE", ["task_eval_cvae3d.json"]),
    ("dg", "Dich. Gaussian (ref)", ["task_eval_dg.json"]),
    ("glm", "Coupled GLM (ref)", ["task_eval_glm.json"]),
]
LEARNED = {"maskgit_flat", "unet3d", "unet3d_seed2", "cvae3d"}


class LabelRecoveryError(RuntimeError):
    pass


# --------------------------------------------------------------- labels ----
def _provenance():
    d = json.loads((ROOT / "reports" / "data_provenance.json").read_text())
    rows = [r for r in d["used"] if r.get("used_in_pipeline", True)]
    ids = sorted(int(r["assay_idx"]) for r in rows)
    tissue = {int(r["assay_idx"]): ("slice" if "hippocampal" in r["preparation"]
                                     else "organoid") for r in rows}
    return ids, tissue


def recover_labels(nulls: dict, ids: list) -> dict:
    """Assay label per kept clip, per task. Raises if any consistency check fails."""
    rk = [int(k) for k in nulls["tasks"]["recon"]["clip_keys"]]
    if len(rk) != QUOTA * len(ids):
        raise LabelRecoveryError(
            f"recon has {len(rk)} clips, expected {QUOTA} x {len(ids)}")
    recon_lab = np.asarray([ids[i // QUOTA] for i in range(len(rk))])
    owner = {}
    for k, a in zip(rk, recon_lab):
        if owner.setdefault(k, a) != a:
            raise LabelRecoveryError(f"key {k} appears in two recordings")
    labels = {"recon": recon_lab}
    for t in TASKS[1:]:
        tk = [int(k) for k in nulls["tasks"][t]["clip_keys"]]
        out, j = [], 0
        for k in tk:
            while j < len(rk) and rk[j] != k:
                j += 1
            if j == len(rk):
                raise LabelRecoveryError(f"{t}: keys are not a subsequence of recon")
            out.append(recon_lab[j])
            j += 1
        labels[t] = np.asarray(out)
    return labels


def _clip_ap(score, target, seed):
    """Copy of task_eval.clip_ap (numpy only)."""
    s = np.asarray(score, np.float32).reshape(-1)
    t = (np.asarray(target).reshape(-1) > 0.5).astype(np.float64)
    n_pos = float(t.sum())
    if n_pos < 1.0:
        return float("nan")
    perm = np.random.default_rng(seed).permutation(s.size)
    s, t = s[perm], t[perm]
    order = np.argsort(-s, kind="stable")
    t = t[order]
    prec = np.cumsum(t) / np.arange(1, t.size + 1)
    return float((prec * t).sum() / n_pos)


def verify_with_dumps(labels: dict, nulls: dict, payloads: dict) -> list:
    """Check B: recompute AP from dumped fields and locate the clip they came from.

    A dumped clip is matched against the stored per-clip lists by AP alone, so a
    single field can coincide with an unrelated clip. Where the dump also holds
    the oracle field, both APs must match the SAME stored clip; that joint match
    is decisive and is required to carry the predicted recording label. Dumps
    with only one field are recorded as supporting evidence when unambiguous.
    """
    checks = []
    dumps = {"pipeline": "pipeline.npz", "unet3d": "unet3d.npz",
             "cvae3d": "cvae3d.npz"}
    decisive = 0
    for key, fn in dumps.items():
        f = DIR / "dumps" / fn
        if not f.is_file() or key not in payloads:
            continue
        z = np.load(f)
        shape = tuple(int(v) for v in z["_shape"])
        seed0 = int(payloads[key]["seed"])
        for t in TASKS:
            stored = {"field": np.asarray(
                          payloads[key]["tasks"][t]["per_clip"][ARM_KEY], float),
                      "field_oracle": np.asarray(
                          payloads[key]["tasks"][t]["per_clip"]["oracle"], float)}
            keys = [int(k) for k in nulls["tasks"][t]["clip_keys"]]
            tags = sorted({n.split("/")[1] for n in z.files
                           if n.startswith(t + "/") and n.endswith("/field")})
            for tag in tags:
                a = int(z[f"{t}/{tag}/assay"][0])
                real = np.zeros(shape, bool)
                idx = z[f"{t}/{tag}/real"].astype(np.int64)
                real[tuple(idx.T)] = True
                roi = np.unpackbits(z[f"{t}/{tag}/roi_packed"])[:np.prod(shape)]
                roi = roi.reshape(shape).astype(bool)
                if real[roi].sum() < 1:
                    continue        # this clip was dropped from the task
                tgt = real[roi]
                kinds = [k for k in ("field", "field_oracle")
                         if f"{t}/{tag}/{k}" in z.files]
                vals = {k: z[f"{t}/{tag}/{k}"].astype(np.float32)[roi] for k in kinds}
                # ties are rare in a continuous field, so AP barely depends on
                # the tie-break seed: screen with one seed, then recompute exactly
                screen = np.zeros(len(keys))
                for k in kinds:
                    screen += np.abs(stored[k] - _clip_ap(vals[k], tgt, seed0))
                cand = np.where(np.nan_to_num(screen, nan=np.inf) < 5e-3 * len(kinds))[0]
                err = {}
                for i in cand:
                    s = seed0 + keys[i] % 100003
                    err[int(i)] = float(sum(abs(stored[k][i] - _clip_ap(vals[k], tgt, s))
                                            for k in kinds))
                ranked = sorted(err, key=err.get)
                best = ranked[0] if ranked else -1
                runner_up = float(err[ranked[1]]) if len(ranked) > 1 else float("inf")
                lab_best = int(labels[t][best]) if best >= 0 else None
                joint = len(kinds) == 2
                # float16 storage costs up to ~1e-4 of AP per field
                tight = best >= 0 and err[best] < 5e-4 * len(kinds)
                separated = runner_up > 10 * err.get(best, np.inf)
                status = ("match" if tight and separated and lab_best == a
                          else "ambiguous" if tight and not separated and lab_best == a
                          else "mismatch")
                checks.append({"arm": key, "task": t, "tag": tag, "assay": a,
                               "fields": kinds, "matched_index": best,
                               "label_at_match": lab_best,
                               "abs_err": float(err.get(best, np.inf)),
                               "runner_up_err": runner_up, "status": status})
                if status == "mismatch" or (joint and status != "match"):
                    raise LabelRecoveryError(
                        f"{key}/{t}/{tag}: dumped clip of assay {a} best matches "
                        f"index {best} (label {lab_best}, err "
                        f"{err.get(best, np.inf):.1e}, runner-up {runner_up:.1e}, "
                        f"fields {kinds})")
                decisive += int(joint and status == "match")
    if decisive == 0:
        raise LabelRecoveryError("no decisive (two-field) dump match available")
    return checks


# ---------------------------------------------------------------- tests ----
def _load_payloads(nulls: dict, suffix: str = ""):
    """First candidate payload per arm that scored the canonical clip set.

    The calibrated MaskGIT-flat payload is a 12-batch run (48 clips), so it is
    passed over in favour of the 70-batch run, which aligns with the nulls."""
    out, src, passed = {}, {}, {}
    n_ref = len(nulls["tasks"]["recon"]["per_clip"]["marginal"])
    for key, _, cands in ARMS:
        for fn in cands:
            if suffix:
                fn = fn.replace(".json", f"{suffix}.json")
            f = DIR / fn
            if not f.is_file():
                continue
            d = json.loads(f.read_text())
            if len(d["tasks"]["recon"]["per_clip"]["marginal"]) != n_ref:
                passed.setdefault(key, []).append(
                    f"{fn}: {d.get('batches')} batches, not the canonical clip set")
                continue
            out[key], src[key] = d, fn
            break
    return out, src, passed


def _aligned(payload, nulls, t, fam):
    """Same pairing check the original scripts use: the model-free marginal arm
    stored in each payload must equal the canonical one bitwise."""
    cur = payload["tasks"][t][fam]
    base = np.asarray(nulls["tasks"][t][fam]["marginal"], float)
    chk = np.asarray(cur["marginal"], float)
    return chk.shape == base.shape and np.array_equal(chk, base, equal_nan=True)


def run(n_boot: int = 10000, suffix: str = "") -> dict:
    ids, tissue = _provenance()
    nulls = json.loads((DIR / "task_eval_nulls.json").read_text())
    payloads, src, passed = _load_payloads(nulls, suffix)
    labels = recover_labels(nulls, ids)
    # The dumps were written by the full-clip runs; for a suffixed run the
    # label mapping is the same (same loader, same clips), so it is verified
    # against the shipped payloads and reused.
    if suffix:
        checks = verify_with_dumps(labels, nulls, _load_payloads(nulls)[0])
    else:
        checks = verify_with_dumps(labels, nulls, payloads)
    merged = {t: merge_duplicates(labels[t]) for t in TASKS}
    strata = {a: tissue[a] for a in ids}

    rows, skipped = [], []
    for fam, famlab in FAMILIES:
        for t in TASKS:
            lab = merged[t]
            ref = np.asarray(payloads["pipeline"]["tasks"][t][fam][ARM_KEY], float)
            lookup = np.asarray(nulls["tasks"][t][fam]["marginal"], float)
            for key, label, _ in ARMS:
                if key not in payloads:
                    continue
                if not _aligned(payloads[key], nulls, t, fam):
                    skipped.append({"family": famlab, "task": t, "arm": label,
                                    "reason": "marginal arm differs from canonical; "
                                              "clips not aligned"})
                    continue
                arm = np.asarray(payloads[key]["tasks"][t][fam][ARM_KEY], float)
                pairs = []
                if key in LEARNED:
                    pairs.append(("A: ours vs arm", ref, arm))
                pairs.append(("B: site map vs arm", lookup, arm))
                for fam_ab, a, b in pairs:
                    clip = paired_clip_test(a, b)
                    rec = paired_recording_test(a, b, lab)
                    boot = hierarchical_bootstrap(a - b, lab, strata,
                                                  n_boot=n_boot, seed=20261107)
                    rows.append({"comparison": fam_ab, "metric": famlab,
                                 "task": t, "arm": label,
                                 "first_mean": float(np.nanmean(a)),
                                 "arm_mean": float(np.nanmean(b)),
                                 "clip": clip, "recording": rec,
                                 "bootstrap": boot})
    for level in ("clip", "recording"):
        ps = [r[level]["p"] for r in rows]
        for r, q, h in zip(rows, bh(ps), holm(ps)):
            r[level]["q_bh"] = float(q)
            r[level]["p_holm"] = float(h)
            r[level]["verdict"] = (
                "first" if q < 0.05 and r[level]["median_delta"] > 0
                else "arm" if q < 0.05 and r[level]["median_delta"] < 0
                else "ns")
    for r in rows:
        r["verdict_changed"] = r["clip"]["verdict"] != r["recording"]["verdict"]

    n_rec = len(np.unique(merged["recon"]))
    return {
        "payload_suffix": suffix,
        "unit": "recording (duplicates merged)",
        "n_recordings": int(n_rec),
        "n_recording_ids_before_merge": len(ids),
        "duplicates_merged": {str(k): v for k, v in DUPLICATE_OF.items()},
        "payloads": src,
        "payloads_passed_over": passed,
        "arm_key": ARM_KEY,
        "label_recovery": {
            "method": "recon index // 9 over sorted assay ids; other tasks via "
                      "clip_keys subsequence alignment",
            "dump_checks": checks},
        "bootstrap": {"levels": "recordings within tissue, then clips within "
                                "recording; equal weight per recording",
                      "n_boot": n_boot, "seed": 20261107},
        "multiplicity": f"BH and Holm over all {len(rows)} tests jointly, per level",
        "rows": rows,
        "skipped": skipped,
    }


def write_md(res: dict) -> str:
    L = ["# Recording-level re-analysis of the paired comparisons", "",
         f"Unit: {res['unit']}; n = {res['n_recordings']} recordings "
         f"({res['n_recording_ids_before_merge']} ids before merging "
         f"duplicates {res['duplicates_merged']}). Arm field: `{res['arm_key']}`.",
         f"Multiplicity: {res['multiplicity']}. Bootstrap: "
         f"{res['bootstrap']['levels']}, {res['bootstrap']['n_boot']} draws.", "",
         f"Clip-to-recording mapping verified on "
         f"{len(res['label_recovery']['dump_checks'])} dumped clips "
         f"(model and oracle fields recomputed from the dumps matched one stored "
         f"clip carrying the predicted recording; "
         f"{sum(c['status'] == 'match' for c in res['label_recovery']['dump_checks'])} "
         f"decisive matches, 0 mismatches).",
         "", "`first` = ours (A) or the site map (B) is better; `arm` = the "
         "other arm is better; `ns` = q >= 0.05. Δ = first − arm, mean of "
         "recording means, with its 95% bootstrap CI.", ""]
    for comp in ("A: ours vs arm", "B: site map vs arm"):
        L += [f"## {comp}", "",
              "| metric | task | arm | first | arm | Δ (95% CI) | rec. won | "
              "clip q | clip verdict | rec. q | rec. verdict | changed |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in res["rows"]:
            if r["comparison"] != comp:
                continue
            b = r["bootstrap"]
            L.append(
                f"| {r['metric']} | {r['task']} | {r['arm']} | "
                f"{r['first_mean']:.4f} | {r['arm_mean']:.4f} | "
                f"{b['estimate']:+.4f} ({b['ci_low']:+.4f}, {b['ci_high']:+.4f}) | "
                f"{100 * r['recording']['recordings_won_frac']:.0f}% | "
                f"{r['clip']['q_bh']:.1e} | {r['clip']['verdict']} | "
                f"{r['recording']['q_bh']:.1e} | {r['recording']['verdict']} | "
                f"{'**yes**' if r['verdict_changed'] else ''} |")
        L.append("")
    tally = {}
    for r in res["rows"]:
        for lv in ("clip", "recording"):
            k = (r["comparison"], lv, r[lv]["verdict"])
            tally[k] = tally.get(k, 0) + 1
    L += ["## Tally", "", "| comparison | level | first | arm | ns |",
          "|---|---|---|---|---|"]
    for comp in ("A: ours vs arm", "B: site map vs arm"):
        for lv in ("clip", "recording"):
            L.append(f"| {comp} | {lv} | {tally.get((comp, lv, 'first'), 0)} | "
                     f"{tally.get((comp, lv, 'arm'), 0)} | "
                     f"{tally.get((comp, lv, 'ns'), 0)} |")
    if res["skipped"]:
        L += ["", "## Skipped", ""] + [
            f"- {s['family']} {s['task']} {s['arm']}: {s['reason']}"
            for s in res["skipped"]]
    return "\n".join(L) + "\n"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--suffix", default="",
                    help="read task_eval_<model><suffix>.json, e.g. _ctxvisible, "
                         "and write recording_level_stats<suffix>.{json,md}")
    a = ap.parse_args()
    global OUT_JSON, OUT_MD
    if a.suffix:
        OUT_JSON = OUT_JSON.with_name(OUT_JSON.stem + a.suffix + ".json")
        OUT_MD = OUT_MD.with_name(OUT_MD.stem + a.suffix + ".md")
    try:
        res = run(a.n_boot, a.suffix)
    except LabelRecoveryError as e:
        print(f"label recovery failed, nothing written: {e}", file=sys.stderr)
        return 1
    OUT_JSON.write_text(json.dumps(res, indent=2) + "\n")
    OUT_MD.write_text(write_md(res))
    print(write_md(res))
    print(f"wrote {OUT_JSON.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
