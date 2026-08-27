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
