"""One visual system for all four figures.

Colour is assigned by the JOB each mark does, not one hue per arm. The paper
makes a two-way comparison -- our prior against the matched flat tokenizer --
and everything else on the page is context: two convolutional arms that are
direct-supervision references, two per-recording lookup tables that are
memorisation ceilings, and a site-map null that beats all of them. Giving six
arms six saturated hues would render that argument as a six-way competition,
which is exactly what the results do not show.

So there are two categorical hues and everything else is a subordinate
register: open markers in secondary ink, identity carried by marker shape plus
a direct label rather than by colour alone.

PALETTE VALIDATION. Run from the dataviz skill directory:

    node scripts/validate_palette.js "#2a78d6,#eb6834,#4a3aa7" \
         --mode light --pairs all

    [PASS] chroma floor        all 3 >= 0.1
    [PASS] CVD separation      worst all-pairs violet<->blue dE 13.0 (deutan)
    [PASS] normal-vision floor worst all-pairs violet<->blue dE 16.3
    [PASS] contrast vs surface all 3 >= 3:1

The main pair separates at dE 33.6 normal / 24.7 protan. Two earlier candidates
were rejected by that script rather than by eye: neutral greys as categorical
fills failed the chroma floor and the normal-vision floor (dE 14.2), and a red
null rule sat at dE 7.1 against the orange series -- indistinguishable.

These figures are printed in a paper, so only the light surface is defined; the
skill's dark-mode requirement is a property of charts rendered on screen in a
themeable page, which a PDF figure is not.
"""
from __future__ import annotations

import matplotlib

TEXT_W = 5.5   # ICLR single-column body width, inches

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8880"

PALETTE = {
    # Categorical -- the comparison the paper actually makes.
    "pipeline":     "#2a78d6",   # slot 1, blue
    "maskgit_flat": "#eb6834",   # slot 2, orange
    # Reference register. Not categorical: identity is shape + direct label.
    "unet3d":       INK_2,
    "cvae3d":       INK_2,
    "dg":           INK_MUTED,
    "glm":          INK_MUTED,
    # The lookup null every learned arm loses to. Its own hue because it is the
    # figure's most important reference, and it must not be confusable with
    # either series: violet clears both at dE >= 16.3 normal vision.
    "null":         "#4a3aa7",
    "null_unseen":  INK_MUTED,
    "real":         INK,
}

# ---------------------------------------------------------------------------
# Voxel panels (appendix figures F5/F6). These draw a CONTINUOUS field with
# three CATEGORICAL marks on top, which is the opposite budget from the charts
# above: the field is the reference and the marks are the content. So the field
# gets no chroma at all -- a grey ramp truncated well short of white -- and the
# three marks get fully saturated, maximally separated hues.
#
# An earlier draft used magma for the field and the violet null hue for the
# third mark; violet on near-black was invisible. White was legible but, being
# the highest-luminance thing on the page, made the error class the most
# salient mark in rows where errors outnumber hits, and it disappeared against
# the page in the legend. Blue / red / yellow separate by hue AND by luminance,
# so red-vs-yellow survives protan and deutan, where a blue/orange/green triple
# would not.
#
# Every mark is stroked in INK. The field peaks exactly where the spikes are,
# so the brightest ground sits under the densest marks; an outline is what
# makes a glyph readable at any field value rather than only on a dark one.
FIELD_RAMP = ["#000000", "#b0b0b0"]
VOXEL = {"hit": "#4da3ff", "missed": "#f4442e", "hallucinated": "#ffd400"}
VOXEL_MARKER = {"hit": "o", "missed": "X", "hallucinated": "P"}
OBSERVED_WASH = "#5d6b7d"    # cool, so it reads apart from the grey field

# Reference row of the voxel strip (free generation, which observes nothing).
# The strip is drawn on the PAGE, not on a dark panel, and every other row is
# ink, so this has to separate from black on white. PALETTE["null"] is the
# right role but the wrong value here: at #4a3aa7 it reads as just another
# black line. Crimson clears black on white and sits far enough from the
# panels' `missed` red (#f4442e) that the figure carries no confusable pair.
VOXEL_REFERENCE = "#c2185b"

# Marker shape carries identity wherever colour is subordinate.
MARKER = {"pipeline": "o", "maskgit_flat": "s", "unet3d": "^", "cvae3d": "v",
          "dg": "D", "glm": "P", "null": "|", "null_unseen": "|"}

# Display order and labels, as the paper argues: subject, peer, then the
# reference classes. The dagger and the ref mark travel with the label.
ARMS = [("pipeline", "Ours"), ("maskgit_flat", "MaskGIT-flat"),
        ("unet3d", "3D U-Net$^\\dagger$"), ("cvae3d", "3D CVAE")]

# Keys as cross_model_tests.json / generation_families.json spell them.
JSON_ARM = {"maskgit_flat": "3D placeholder"}   # overridden per file below
CROSS_ARM = {"maskgit_flat": "MaskGIT-flat", "unet3d": "3D U-Net (det.)",
             "cvae3d": "3D CVAE"}
FAM_ARM = {"pipeline": "Ours (4C+soft)", "maskgit_flat": "MaskGIT-flat",
           "unet3d": "3D U-Net (det.)†", "cvae3d": "3D CVAE",
           "dg": "Dich. Gaussian", "glm": "Coupled GLM"}

TASKS = [("recon", "free gen."), ("causal", "causal"),
         ("noncausal", "noncausal"), ("spatial", "spatial")]


def apply_style() -> None:
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.color": "#d9d8d2",
        "grid.alpha": 0.9,
        "grid.linewidth": 0.4,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "legend.frameon": False,
    })
