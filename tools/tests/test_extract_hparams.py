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
