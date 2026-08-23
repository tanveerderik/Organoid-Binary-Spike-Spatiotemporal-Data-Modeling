#!/usr/bin/env python3
"""Does the token-entropy term do what it claims, and can it collapse a patch?

The stated worry when this was proposed: an entropy penalty is minimised by a
patch that decodes to nothing, so the term could quietly turn active patches
blank. These check that it cannot -- and that its other guards hold.

    python ablations/tests/test_token_entropy.py
"""
import sys
from pathlib import Path
import torch

R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R)); sys.path.insert(0, str(R.parent))
from MAGVIT_project.utils.losses import token_profile_entropy_loss as L

P = 60
ok = fail = 0


def check(name, cond, extra=""):
    global ok, fail
    if cond:
        ok += 1; print(f"  [PASS] {name}")
    else:
        fail += 1; print(f"  [FAIL] {name}  {extra}")


def patch(idx, hi=6.0, lo=-9.0):
    v = torch.full((1, 1, P), lo)
    for i in idx:
        v[0, 0, i] = hi
    return v


# 1. scale invariance -- the crux of the collapse question
tgt = patch([3, 17], hi=1.0, lo=0.0)
lg = patch([3, 17])
a = L(lg, tgt)
b = L(lg - 12.0, tgt)                       # every logit shifted far down
c = L(lg + 12.0, tgt)                       # ... and far up
check("EXACT shift invariance: turning the patch down cannot lower the loss",
      torch.allclose(a, b, atol=1e-6) and torch.allclose(a, c, atol=1e-6),
      f"{a.item():.3e} / down {b.item():.3e} / up {c.item():.3e}")

# 2. an all-blank patch is NOT rewarded: uniform is the worst case
uni = torch.full((1, 1, P), -9.0)           # every voxel identical => uniform q
sharp = patch([3, 17])
check("uniform (all-blank) patch scores WORSE than a concentrated one",
      L(uni, tgt).item() > L(sharp, tgt).item(),
      f"uniform {L(uni,tgt).item():.4f} vs sharp {L(sharp,tgt).item():.4f}")

# 3. hinge: matching the truth's concentration gives exactly zero
# A perfectly concentrated patch still carries residual entropy from its P-k
# background voxels; with a 15-nat logit gap that is ~1e-4, not exactly 0.
check("loss is ~0 when the model matches the true support",
      L(sharp, tgt).item() < 1e-3, f"{L(sharp,tgt).item():.3e}")

# 4. no pressure past the target: a delta where truth has 2 spikes is also 0,
#    i.e. the term never *asks* for over-concentration
delta = patch([3])
check("over-concentrated patch is not penalised, but is not rewarded either",
      L(delta, tgt).item() < 1e-3 and L(sharp, tgt).item() < 1e-3)

# 5. entropy is stationary at the uniform distribution, so a flat patch gets an
#    almost-zero symmetric gradient. This is a REAL limitation, not a bug: the
#    term cannot break symmetry and must be ramped in after reconstruction has
#    put structure in the patch. Asserted so the property stays documented.
lg2 = torch.full((1, 1, P), 0.0, requires_grad=True)
t2 = patch([10], hi=1.0, lo=0.0)
L(lg2, t2).backward()
g = lg2.grad[0, 0]
check("flat patch -> vanishing, symmetric gradient (ramp in late)",
      g.abs().max().item() < 1e-6 and (g.max() - g.min()).abs().item() < 1e-12,
      f"grad range {g.min():.3e}..{g.max():.3e}")

# 5b. once structure exists, the gradient sharpens it rather than dimming it
lg3 = torch.full((1, 1, P), -9.0); lg3[0, 0, :8] = -2.0
lg3.requires_grad_(True)
L(lg3, patch([0], hi=1.0, lo=0.0)).backward()
g3 = lg3.grad[0, 0]
check("with structure present, the gradient is not all one sign (it reshapes)",
      bool((g3 > 0).any()) and bool((g3 < 0).any()),
      f"grad range {g3.min():.3e}..{g3.max():.3e}")

# 6. blank patches are excluded
blank_t = torch.zeros(1, 1, P)
check("patch with no true spike contributes nothing",
      L(torch.randn(1, 1, P), blank_t).item() == 0.0)

# 7. min_true_count gate
one_t = patch([5], hi=1.0, lo=0.0)
check("min_true_count=2 skips single-spike patches",
      L(torch.randn(1, 1, P), one_t, min_true_count=2).item() == 0.0)

# 8. reported parts are consistent
_, parts = L(uni, tgt, return_parts=True)
import math
check("h_model of a uniform patch equals log(P)",
      abs(parts["h_model"] - math.log(P)) < 1e-3,
      f"{parts['h_model']:.4f} vs {math.log(P):.4f}")
check("h_true of a 2-spike patch equals log(2)",
      abs(parts["h_true"] - math.log(2)) < 1e-6)

# 9. a genuinely blurred patch IS penalised
blur = torch.full((1, 1, P), -9.0); blur[0, 0, :20] = 0.0
check("blurred patch (20 voxels lit) is penalised against a 2-spike truth",
      L(blur, tgt).item() > 0.5, f"{L(blur,tgt).item():.4f}")

print(f"\n{ok}/{ok+fail} passed")
raise SystemExit(1 if fail else 0)
