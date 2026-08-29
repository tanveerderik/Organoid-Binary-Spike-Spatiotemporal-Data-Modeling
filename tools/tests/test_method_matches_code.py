"""The Method's stated dimensions must be the code's dimensions.

Three claims in the submitted draft were wrong, and all three were wrong in the
same way: a number or a data path was described from an earlier version of the
model and never re-read. Prose is the one part of the manuscript no generator
writes, so it is the one part that can drift silently.

These tests read the constants out of `main.py` and `dataset.py` with `ast` --
no import, so no torch -- and fail if the manuscript disagrees.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
METHOD = ROOT / "paper" / "sections" / "04_method.tex"
LIMITS = ROOT / "paper" / "sections" / "06_limitations.tex"
APPENDIX = ROOT / "paper" / "sections" / "99_appendix.tex"


def _module_const(path: Path, name: str):
    """A module-level `name = <literal>` assignment, resolved."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{path.name}: no module-level constant {name}")


def _kwonly_default(path: Path, func: str, param: str):
    """A keyword parameter's default from a function signature."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func:
            a = node.args
            pairs = list(zip(a.kwonlyargs, a.kw_defaults))
            pairs += list(zip(a.args[len(a.args) - len(a.defaults):], a.defaults))
            for arg, default in pairs:
                if arg.arg == param and default is not None:
                    return ast.literal_eval(default)
    raise AssertionError(f"{path.name}:{func}: no default for {param}")


def _flat(text: str) -> str:
    """Whitespace-normalised text.

    LaTeX sources wrap at column 79, so a line break falls in the middle of
    almost every phrase worth matching. The first version of this file asserted
    against raw text and one test passed only because the phrase it was
    forbidding happened to straddle a newline.
    """
    return " ".join(text.split())


@pytest.fixture(scope="module")
def method() -> str:
    # paper/ is gitignored, so a checkout without the manuscript would make
    # every assertion below vacuous. Fail loudly instead.
    assert METHOD.is_file(), f"manuscript missing: {METHOD}"
    return _flat(METHOD.read_text())


def test_global_code_dim_is_the_dataset_codebook_width(method):
    """The per-recording code is `dim_assay_for_emb`-dimensional.

    The draft said 32, which is `global_emb_dim` -- the Stage-1 mapper's
    OUTPUT. The code the mapper consumes is built in dataset.py at
    `global_ctx_dim`, wired from main.py's `dim_assay_for_emb`.
    """
    n = _module_const(ROOT / "main.py", "dim_assay_for_emb")
    assert f"${n}$-dimensional" in method, (
        f"Method does not state the global code as ${n}$-dimensional")
    assert "$32$-dimensional $\\pm1$" not in method, (
        "the 32-dimensional claim is back; 32 is global_emb_dim, not the code")


def test_dataset_codebook_width_is_wired_from_main(method):
    """Guard the wiring the previous test depends on.

    `build_dataloaders(dim_assays=dim_assay_for_emb)` -> `NpzBurstDataset(
    global_ctx_dim=dim_assays)`. If that link is ever broken, the signature
    default in dataset.py (32) would silently become the real width and the
    previous test would be asserting the wrong number.
    """
    src = (ROOT / "dataset.py").read_text()
    assert "global_ctx_dim=dim_assays" in src, (
        "dataset.py no longer takes its codebook width from dim_assays")
    assert "dim_assays=dim_assay_for_emb" in (ROOT / "main.py").read_text(), (
        "main.py no longer passes dim_assay_for_emb as dim_assays")


def test_the_two_priors_are_not_claimed_to_share_one_mapper(method):
    """Only the motif prior reads the code through the frozen Stage-1 mapper.

    The activity prior is constructed with `global_dim=dim_assay_for_emb` and
    `local_dim=9` and projects the RAW code with its own Linear
    (model/prior.py, MaskGITActivityPrior.forward). Claiming a single shared
    mapper overstates how contained a descriptor swap would be.
    """
    assert "Both priors read the recording only through" not in method
    for path in (LIMITS, APPENDIX):
        t = _flat(path.read_text())
        assert "exactly one frozen mapper" not in t, path.name
        assert "the single frozen mapper" not in t, path.name


def test_activity_prior_still_takes_the_raw_code(method):
    """The asymmetry the Method now describes must still be true of the code."""
    src = (ROOT / "main.py").read_text()
    assert "global_dim=dim_assay_for_emb" in src, (
        "the activity prior no longer takes the raw code; Method 4.3's "
        "asymmetry paragraph is now wrong")
    assert "local_dim=9" in src


def test_regions_are_described_as_roi_occupancy_not_as_a_count_target(method):
    """The 16 regions are input context, not a coarse count prediction.

    `MaskGITActivityPrior` builds `roi_occupancy_proj(occupancy) +
    roi_region_embed` as memory tokens; `maskgit_activity_loss` has no regional
    term and the count head is one categorical over the whole ROI.
    """
    assert "how-much-activity is predicted at a coarser scale" not in method
    assert "16 regions" in method or "$16$ regions" in method
    prior = (ROOT / "model" / "prior.py").read_text()
    assert "roi_occupancy_proj" in prior, (
        "the ROI-occupancy path is gone; Method 4.4 needs re-reading")
    assert "region_head" not in prior.split("def maskgit_activity_loss")[-1], (
        "a regional head appeared in the activity loss; Method 4.4 is now wrong")


def test_paper_does_not_present_the_tokeniser_as_convolution_free(method):
    """The tokeniser is a hybrid: ConvStem3D in, transformer middle,
    PatchRenderer3D (ConvTranspose3d + Conv3d) out.

    An earlier draft of Section 4.1 argued for attention "rather than
    convolution", which reads as though the pipeline contains none. The
    missing ablation is of the transformer CORE against a convolutional one,
    and the paper has to say so or a reader of the release will catch it.
    """
    base = (ROOT / "model" / "base.py").read_text()
    assert "class ConvStem3D" in base
    assert "ConvTranspose3d" in base
    vq = (ROOT / "model" / "vqvae.py").read_text()
    assert "ConvStem3D" in vq, "the stem left the tokeniser; 4.1 needs re-reading"
    assert "convolutional stem" in method, (
        "Section 4.1 does not say the tokeniser has a convolutional stem")
    limits = _flat(LIMITS.read_text())
    assert "all-convolutional core" in limits or "convolutional core" in limits, (
        "Limitations still implies no convolution is present at all")
