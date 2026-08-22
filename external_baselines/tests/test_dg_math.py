#!/usr/bin/env python3
"""Verify the DG inversion and the FFT field sampler against ground truth.

A statistical baseline that is quietly wrong is worse than no baseline: it
produces plausible numbers that a reviewer cannot catch either. Both pieces here
have closed-form or Monte-Carlo checks, so there is no excuse for trusting them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import norm

sys.path.insert(0, "/media/derik/Seagate Desktop Drive/organoid_data")
from MAGVIT_project.external_baselines.dichotomized_gaussian.model import (
    DichotomizedGaussian, _joint_exceedance, _joint_exceedance_vec,
    gaussian_corr_for_joint, gaussian_corr_heterogeneous)

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAIL.append(name)


def main() -> int:
    rng = np.random.default_rng(0)

    # -- 1. independence limit -----------------------------------------
    print("\nDG joint-exceedance")
    for p in (1e-4, 1e-3, 1e-2):
        th = float(norm.isf(p))
        got = _joint_exceedance(th, 0.0)
        check(f"lam=0 gives p^2 at p={p:g}",
              abs(got - p * p) / (p * p) < 1e-6, f"{got:.3e} vs {p*p:.3e}")

    # -- 2. perfect-correlation limit ----------------------------------
    for p in (1e-4, 1e-2):
        th = float(norm.isf(p))
        got = _joint_exceedance(th, 0.999999)
        check(f"lam->1 gives p at p={p:g}",
              abs(got - p) / p < 0.05, f"{got:.3e} vs {p:.3e}")

    # -- 3. Monte-Carlo agreement --------------------------------------
    print("\nMonte-Carlo vs quadrature")
    for p, lam in ((1e-2, 0.5), (1e-2, 0.9), (1e-3, 0.8)):
        th = float(norm.isf(p))
        n = 4_000_000
        x = rng.standard_normal(n)
        y = lam * x + np.sqrt(1 - lam ** 2) * rng.standard_normal(n)
        emp = float(((x > th) & (y > th)).mean())
        quad = _joint_exceedance(th, lam)
        rel = abs(emp - quad) / max(quad, 1e-12)
        check(f"p={p:g} lam={lam}", rel < 0.05, f"mc={emp:.3e} quad={quad:.3e} rel={rel:.3f}")

    # -- 4. inversion round-trip ---------------------------------------
    print("\ninversion round-trip")
    for p in (1.5e-4, 1e-3, 1e-2):
        th = float(norm.isf(p))
        for lam in (0.1, 0.35, 0.7, 0.95):
            pj = _joint_exceedance(th, lam)
            rec = gaussian_corr_for_joint(p, pj)
            check(f"p={p:g} lam={lam}", abs(rec - lam) < 2e-3, f"recovered {rec:.5f}")

    # -- 5. below independence clamps to 0 ------------------------------
    check("p_joint below independence -> 0",
          gaussian_corr_for_joint(1e-3, 1e-7) == 0.0)

    # -- 5b. heterogeneous inversion ------------------------------------
    print("\nheterogeneity-aware inversion")
    # degenerate case: all thresholds equal -> must match the scalar routine
    for p, lam in ((1.5e-4, 0.4), (1e-3, 0.75)):
        th = float(norm.isf(p))
        a = np.array([th, th]); w = np.array([1.0, 3.0])
        vec = _joint_exceedance_vec(a, a, w, lam)
        sca = _joint_exceedance(th, lam)
        check(f"vec==scalar when homogeneous p={p:g} lam={lam}",
              abs(vec - sca) / sca < 2e-3, f"{vec:.4e} vs {sca:.4e}")
        rec = gaussian_corr_heterogeneous(a, a, w, sca)
        check(f"heterogeneous round-trip p={p:g} lam={lam}",
              abs(rec - lam) < 5e-3, f"recovered {rec:.5f}")

    # the point of the correction: heterogeneous marginals raise the pooled
    # joint rate at lam=0, so a homogeneous inversion invents coupling
    th_het = norm.isf(np.array([3e-3, 3e-5]))
    w_het = np.array([1.0, 1.0])
    p_bar = float(np.mean([3e-3, 3e-5]))
    pj0 = _joint_exceedance_vec(th_het, th_het, w_het, 0.0)
    check("heterogeneous independence exceeds p_bar^2",
          pj0 > p_bar ** 2 * 1.5, f"{pj0:.3e} vs {p_bar**2:.3e}")
    check("heterogeneous inversion returns 0 at independence",
          gaussian_corr_heterogeneous(th_het, th_het, w_het, pj0) == 0.0)
    # The correct answer here is exactly 0 -- these voxels are independent.
    # A homogeneous inversion cannot see that and reports coupling. The size of
    # the error grows with the spread of the site map; this two-site toy spans
    # only 100x, while the real map spans several orders of magnitude.
    naive = gaussian_corr_for_joint(p_bar, pj0)
    check("homogeneous inversion INVENTS coupling where there is none",
          naive > 0.02, f"naive lam={naive:.3f} vs correct 0.0")

    # -- 6. FFT sampler reproduces the requested kernel -----------------
    print("\nFFT stationary field")
    dg = DichotomizedGaussian(t_lags=8, s_radius=6)
    dg.rho_t = np.array([1.0, .80, .62, .48, .36, .26, .18, .12, .08])
    dg.rho_s = np.array([1.0, .70, .48, .32, .20, .12, .06])
    T, H, W = 48, 40, 40
    g = torch.Generator().manual_seed(7)
    field = dg._sample_field(64, (T, H, W), "cpu", g)

    check("field is unit variance",
          abs(float(field.std()) - 1.0) < 0.05, f"std={float(field.std()):.4f}")
    check("field is zero mean",
          abs(float(field.mean())) < 0.02, f"mean={float(field.mean()):.4f}")

    f = field - field.mean()
    var = float((f * f).mean())
    for k in (1, 2, 4):
        emp = float((f[:, k:] * f[:, :-k]).mean()) / var
        want = dg.rho_t[k]
        check(f"temporal rho at dt={k}", abs(emp - want) < 0.06,
              f"emp={emp:.3f} want={want:.3f}")
    for d in (1, 2, 4):
        emp = float((torch.roll(f, d, dims=2) * f).mean()) / var
        want = dg.rho_s[d]
        check(f"spatial rho at dh={d}", abs(emp - want) < 0.06,
              f"emp={emp:.3f} want={want:.3f}")

    # -- 7. end-to-end: threshold the field, recover the binary rate ----
    print("\nthresholded field")
    p = 1.5e-4
    th = float(norm.isf(p))
    binv = (field > th).float()
    check("thresholded rate matches target",
          abs(float(binv.mean()) - p) / p < 0.35,
          f"got {float(binv.mean()):.3e} want {p:.3e}")

    print(f"\n{'ALL PASS' if not FAIL else str(len(FAIL)) + ' FAILURES: ' + ', '.join(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
