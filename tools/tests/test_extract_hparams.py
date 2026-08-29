"""The hyperparameter table reads settings out of source, so test the reader.

A table transcribed by hand goes stale silently. This one is extracted, which
converts staleness into an exception -- but only if the extractor actually
fails when a call site moves. That property is what these tests pin: a renamed
function, a missing keyword, or a Stage 4B config that no longer derives from
the constant the table quotes must all raise rather than return a plausible
number.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2].parent))
from MAGVIT_project.tools import extract_hparams as EH  # noqa: E402


@pytest.fixture(scope="module")
def hparams():
    """The payload as `main()` would write it, without touching the file."""
    return {"stages": EH.stages(), "objectives": EH.objectives(),
            "disabled": EH.disabled()}


def test_every_shipped_stage_is_present():
    stages = {s["stage"] for s in EH.stages()}
    assert stages == {"1", "2A", "3", "4A", "4B", "4C"}


def test_rejected_designs_are_absent():
    """4B-refine is a rejected design and must not appear as a shipped stage."""
    stages = {s["stage"] for s in EH.stages()}
    assert not any("refine" in s.lower() for s in stages)


def test_values_match_the_imported_constants():
    by_stage = {s["stage"]: s for s in EH.stages()}
    assert by_stage["4A"]["epochs"] == int(EH.M.STAGE4A_EPOCHS)
    assert by_stage["4A"]["warmup_epochs"] == int(EH.M.STAGE4A_WARMUP_EPOCHS)
    assert by_stage["4C"]["lr"] == pytest.approx(float(EH.M.STAGE4C_LR))
    assert by_stage["2A"]["epochs"] == int(EH.M.STAGE2_EPOCHS)
    assert by_stage["3"]["epochs"] == int(EH.M.STAGE3_EPOCHS)


def test_stage4b_lr_comes_from_the_dict_the_call_site_uses():
    """4B passes `lr=float(config["lr"])` from a local dict.

    The table can only quote the constant if the local really is seeded from
    it, which is what `stage4b_lr` checks before returning.
    """
    assert EH.stage4b_lr() == pytest.approx(
        float(EH.M.STAGE4B_HYPERPARAMETERS["lr"]))


def test_missing_function_raises():
    with pytest.raises(EH.MissingSetting):
        EH.kwargs_of(EH.MAIN, "run_stage_that_does_not_exist", "AdamW")


def test_missing_call_raises():
    with pytest.raises(EH.MissingSetting):
        EH.kwargs_of(EH.MAIN, "run_stage4a", "SomeOptimiserWeNeverUse")


def test_missing_keyword_raises():
    with pytest.raises(EH.MissingSetting):
        EH._require({"weight_decay": 0.01}, "lr", "test")


def test_grad_clip_is_reported_as_absent_not_as_zero():
    """The tokeniser stages clip nothing.

    `0.0` would read as "clipped at zero", which is a different and much
    stranger setting than "not clipped". None renders as a dash.
    """
    by_stage = {s["stage"]: s for s in EH.stages()}
    assert by_stage["2A"]["grad_clip"] is None
    assert by_stage["4A"]["grad_clip"] == 1.0


# ---------------------------------------------------------------------------
# The objective decomposition. Two coefficients here cannot be read off the
# obvious source line, and both were wrong in an earlier hand-written pass.
# ---------------------------------------------------------------------------

def _objectives(payload):
    return {o["stage"]: {t["term"]: t["coeff"] for t in o["terms"]}
            for o in payload["objectives"]}


def test_stage2a_context_weight_follows_the_runtime_rebind(hparams):
    """lambda_ctx is 2.8, not the 0.1 the call site statically reads.

    `run_stage2a` rebinds the module global through `globals()[...]` before
    calling `fit_vqvae`, so a static read of `STAGE2_LAMBDA_CTX` is 28x low.
    If this ever reports 0.1, the extractor stopped following the rebind.
    """
    assert _objectives(hparams)["2A"]["context"] == 2.8


def test_stage4b_lambdas_come_from_the_maskgit_dict(hparams):
    """Stage 4B has two config dicts and only one of them is read by the loss.

    STAGE4B_HYPERPARAMETERS carries lambda_count=1.0, lambda_adj_t=0.0 and
    lambda_adj_s=0.0, none of which the run uses; the loss reads
    STAGE4B_MASKGIT_HYPERPARAMETERS. Quoting the first dict would report a
    count weight 15x too high and claim the co-activation terms were off.
    """
    b = _objectives(hparams)["4B"]
    assert b["count"] == 0.065, "lambda_count came from the wrong dict"
    assert b["temporal co-activation"] == 0.1
    assert b["spatial co-activation"] == 0.1


def test_activity_bce_is_not_class_balanced(hparams):
    """pos_weight is 1.0 by design; the Method's failure-mode argument
    depends on it, so a change here invalidates that paragraph."""
    terms = {t["term"]: t for o in hparams["objectives"] if o["stage"] == "4B"
             for t in o["terms"]}
    assert "positive weight 1.0" in terms["per-cell BCE"]["constrains"]


def test_every_objective_term_has_a_coefficient_and_a_purpose(hparams):
    for o in hparams["objectives"]:
        assert o["terms"], o["stage"]
        for t in o["terms"]:
            assert t["coeff"] not in (None, ""), (o["stage"], t["term"])
            assert len(t["constrains"]) > 20, (o["stage"], t["term"])


def test_disabled_terms_are_declared(hparams):
    """A reader finding these in the release should not have to work out
    whether they ran."""
    what = " ".join(x["what"] for x in hparams["disabled"])
    assert "token-profile entropy" in what
    assert "loss_weights" in what


def test_stage_table_does_not_advertise_the_dead_alpha_weight(hparams):
    """model/prior.py reads loss_weights[0] only; the second element weighted
    a head that no longer exists."""
    for st in hparams["stages"]:
        assert not any("alpha" in k for k in st["loss_terms"]), st["stage"]
