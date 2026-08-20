"""Stage 3: train the standalone local-context (lct) mapper.

Multinomial reformulation, not R^2 on a histogram
-------------------------------------------------
Scoring a per-clip code histogram with R^2 is capped. Write the target as
y = lambda + eps, lambda the clip's true code-usage rate and eps multinomial
noise from drawing n ~ 86 active tokens:

    R2_max = Var(lambda) / (Var(lambda) + Var(eps)) ~= n*CV^2 / (V + n*CV^2)

At n=86: V=32 -> 0.73, V=961 -> 0.082. More epochs shrink the *estimator*
variance and approach R2_max; they cannot move R2_max, which is set by n and V.
The flat histogram looked hopeless because of the ceiling, not the signal.

A multinomial likelihood has no such denominator: each clip contributes n real
samples from the predicted distribution, so a 961-way alphabet is exactly as
estimable as a 32-way one. The score is dNLL against the marginal, in
nats/token -- which is also what the MaskGIT prior optimises, so the trunk is
objective-matched to its consumer.

Belief update: the marginal baseline accumulates online across every batch of
every epoch (Dirichlet(1) posterior mean), not from one collection pass. That
is the aggregation many epochs genuinely buys.

Arms lct(9) / gct(g_emb) / both share one frozen-encoder pass, so unique
contribution (both - gct) is measured exactly and for free. Only the lct arm's
trunk is the deliverable; the heads are discarded.

Textons are clustered on the z1 basis, NOT the flat codebook: descriptor space
and decoder space are different jobs. z1 carries reusable motif identity;
z2/z3 carry exactness detail lct cannot predict. Measured z1 > z12 > flat,
monotone, all 10+ sd.

Promoted from the scratch script that produced the shipped checkpoint; only the
I/O paths and the entry point changed.
"""
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

TAU = 1.0
L2, L3 = 8, 4


class Arm(nn.Module):
    """Shared trunk + one head per target family. Only the trunk survives."""

    def __init__(self, in_dim, n_flat, n_tex, n_reg, hidden=256, latent=32):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, latent),
        )
        self.hf = nn.Linear(latent, n_flat)
        self.ht = nn.Linear(latent, n_tex)
        self.hr = nn.Linear(latent, n_reg)

    def forward(self, z):
        h = self.trunk(z)
        return self.hf(h), self.ht(h), self.hr(h)


def _nll(logits, cnt):
    q = cnt / cnt.sum(1, keepdim=True).clamp_min(1e-9)
    return -(q * F.log_softmax(logits, 1)).sum(1).mean()


def _nll_marg(m, cnt):
    q = cnt / cnt.sum(1, keepdim=True).clamp_min(1e-9)
    return -(q * m.log()).sum(1).mean()


def _r2(p, t):
    ss = (t - p).pow(2).sum(0)
    tt = (t - t.mean(0, keepdim=True)).pow(2).sum(0).clamp_min(1e-9)
    return float((1 - ss / tt).mean())


def run_stage3_lct(
    model,
    train_loader,
    test_loader,
    device,
    *,
    batch_to_device,
    flat_codebook_path,
    out_path,
    report_path,
    epochs: int = 300,
    batches_per_epoch: int = 120,
    patience: int = 40,
    n_textons: int = 128,
    tex_basis: str = "z1",
    tex_batches: int = 40,
    test_batches: int = 80,
    seed: int = 0,
):
    print("\n" + "=" * 80)
    print("STAGE 3: local-context mapper (multinomial, belief-updated marginals)")
    print("=" * 80, flush=True)

    torch.manual_seed(seed)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    b0 = next(iter(train_loader))
    _, _, T0, H0, W0 = b0["x"].shape
    pt, ph, pw = map(int, model.patch_size)
    Tt, Hh, Ww = T0 // pt, H0 // ph, W0 // pw
    K1 = int(model.vq.num_codes_per_level[0])
    blank = int(getattr(model.vq, "blank_code", -1))

    # The trunk was trained on RAW lct: local_ctx_mean/scale were never fitted
    # (0 / 1). They are carried explicitly so the contract is visible rather
    # than accidental. Changing them requires retraining the trunk.
    LM = torch.zeros(9, device=device)
    LS = torch.ones(9, device=device)

    fc = torch.load(str(flat_codebook_path), map_location=device)
    EMB = fc["embed"].float().to(device)
    MERGE = fc["merge_map"].long().to(device)
    n_flat = EMB.shape[0]
    VF = n_flat
    PROV = fc["provenance"].long().to(device)

    t1 = model.vq.tree_embeds[0].detach().float() * float(model.vq.level_scales[0])
    t2 = model.vq.tree_embeds[1].detach().float() * float(model.vq.level_scales[1])
    basis = {"z1": t1[PROV[:, 0]]}
    basis["z12"] = basis["z1"] + t2[PROV[:, 0], PROV[:, 1]]
    basis["flat"] = EMB
    TEX_EMB = basis[tex_basis]
    EPAD = torch.cat(
        [TEX_EMB, TEX_EMB.mean(0, keepdim=True),
         torch.zeros(1, TEX_EMB.shape[1], device=device)], 0
    )
    n_nominal = int(model.vq.num_codes_per_level[0]) * L2 * L3

    print(
        f"grid {Tt}x{Hh}x{Ww} | K1={K1} | flat V={n_flat} | lct 9 "
        f"| gct_in {model.global_ctx_in_dim} | textons {n_textons} on {tex_basis}",
        flush=True,
    )

    @torch.no_grad()
    def flat_ids(codes_b):
        a, b, c = codes_b[..., 0], codes_b[..., 1], codes_b[..., 2]
        m = a != blank
        nom = (a.clamp_min(0) * L2 + b.clamp_min(0)) * L3 + c.clamp_min(0)
        f = MERGE[nom.clamp(0, n_nominal - 1)]
        return f, m

    @torch.no_grad()
    def descriptors(f, m):
        fi = f.clone()
        fi[~m] = VF
        g = EPAD[fi]
        return torch.cat(
            [g,
             0.5 * (torch.roll(g, 1, 0) + torch.roll(g, -1, 0)),
             0.5 * (torch.roll(g, 1, 1) + torch.roll(g, -1, 1)),
             0.5 * (torch.roll(g, 1, 2) + torch.roll(g, -1, 2))], -1
        )[m]

    @torch.no_grad()
    def encode(x, g, l):
        return model(
            x, global_ctx=g, local_ctx=l,
            predict_mask_spec=[{"type": "recon"}] * x.shape[0],
        )["codes"].long()

    # ---- texton vocabulary
    print("building texton vocabulary...", flush=True)
    D = []
    with torch.no_grad():
        for i, b in enumerate(train_loader):
            x, g, l, _, _ = batch_to_device(b, device)
            cd = encode(x, g, l)
            for bi in range(x.shape[0]):
                f, m = flat_ids(cd[bi].view(Tt, Hh, Ww, -1))
                if int(m.sum()) >= 10:
                    D.append(descriptors(f, m))
            if i + 1 >= tex_batches:
                break
    D = torch.cat(D)
    print(f"  {D.shape[0]} active neighbourhoods, dim {D.shape[1]}", flush=True)
    C = D[torch.randperm(D.shape[0])[:n_textons]].clone()
    for _ in range(50):
        asg = torch.cdist(D, C).argmin(1)
        for k in range(n_textons):
            sel = asg == k
            if sel.any():
                C[k] = D[sel].mean(0)
    print(f"  kmeans done, {n_textons} textons", flush=True)
    del D

    @torch.no_grad()
    def batch_targets(x, g, l):
        B = x.shape[0]
        cd = encode(x, g, l)
        gemb = model.global_embedder(g.float())
        CF, CT, RG, keep = [], [], [], []
        for bi in range(B):
            cb = cd[bi].view(Tt, Hh, Ww, -1)
            f, m = flat_ids(cb)
            if int(m.sum()) < 10:
                continue
            CF.append(torch.bincount(f[m], minlength=VF).float())
            d = descriptors(f, m)
            CT.append(torch.softmax(-torch.cdist(d, C) / TAU, 1).sum(0))
            zb = cb[..., 0]
            tix = torch.arange(Tt, device=device).float()[:, None, None].expand_as(zb).float()
            tc = torch.full((K1,), 0.5, device=device)
            for cc in torch.unique(zb[m]):
                sel = m & (zb == cc)
                if sel.any():
                    tc[int(cc)] = tix[sel].mean() / max(Tt - 1, 1)
            RG.append(torch.cat([tc, m.view(Tt, -1).float().mean(1)]))
            keep.append(bi)
        if not keep:
            return (None,) * 5
        lz = (l[keep].float() - LM) / LS
        return lz, gemb[keep].float(), torch.stack(CF), torch.stack(CT), torch.stack(RG)

    GD = int(model.global_emb_dim)
    NREG = K1 + Tt
    arms = nn.ModuleDict({
        "lct": Arm(9, VF, n_textons, NREG),
        "gct": Arm(GD, VF, n_textons, NREG),
        "both": Arm(9 + GD, VF, n_textons, NREG),
    }).to(device)

    def feed(k, lz, ge):
        return lz if k == "lct" else ge if k == "gct" else torch.cat([lz, ge], 1)

    opt = torch.optim.AdamW(arms.parameters(), lr=1e-3, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    # ---- fixed held-out set
    print("building fixed held-out set...", flush=True)
    EL, EG, EF, ET, ER = [], [], [], [], []
    with torch.no_grad():
        for i, b in enumerate(test_loader):
            x, g, l, _, _ = batch_to_device(b, device)
            lz, ge, cf, ct, rg = batch_targets(x, g, l)
            if lz is None:
                continue
            EL.append(lz); EG.append(ge); EF.append(cf); ET.append(ct); ER.append(rg)
            if i + 1 >= test_batches:
                break
    EL, EG, EF, ET, ER = [torch.cat(t) for t in (EL, EG, EF, ET, ER)]
    RMU, RSD = ER.mean(0, keepdim=True), ER.std(0, keepdim=True).clamp_min(1e-6)
    ERn = (ER - RMU) / RSD
    print(
        f"held-out {len(EL)} clips | mean active {float(EF.sum(1).mean()):.1f}",
        flush=True,
    )

    # ---- belief-updated marginals, Dirichlet(1)
    cnt_f = torch.ones(VF, device=device)
    cnt_t = torch.ones(n_textons, device=device)

    best = -9e9
    bad = 0
    log = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        arms.train()
        tot = nb = 0
        for i, b in enumerate(train_loader):
            x, g, l, _, _ = batch_to_device(b, device)
            lz, ge, cf, ct, rg = batch_targets(x, g, l)
            if lz is None:
                continue
            cnt_f += cf.sum(0)
            cnt_t += ct.sum(0)          # belief update, every batch of every epoch
            rgn = (rg - RMU) / RSD
            opt.zero_grad()
            loss = 0.0
            for k in arms:
                lf, lt, lr = arms[k](feed(k, lz, ge))
                loss = loss + _nll(lf, cf) + _nll(lt, ct) + (lr - rgn).pow(2).mean()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
            if i + 1 >= batches_per_epoch:
                break
        sch.step()
        arms.eval()

        mf = cnt_f / cnt_f.sum()
        mt = cnt_t / cnt_t.sum()
        bf, bt_ = float(_nll_marg(mf, EF)), float(_nll_marg(mt, ET))
        row = {"epoch": ep, "train": tot / max(nb, 1), "base_flat": bf, "base_tex": bt_}
        with torch.no_grad():
            for k in arms:
                lf, lt, lr = arms[k](feed(k, EL, EG))
                row[k] = {
                    "dnll_flat": bf - float(_nll(lf, EF)),
                    "dnll_tex": bt_ - float(_nll(lt, ET)),
                    "r2_tcent": _r2(lr[:, :K1], ERn[:, :K1]),
                    "r2_tmarg": _r2(lr[:, K1:], ERn[:, K1:]),
                }
        log.append(row)
        score = (
            row["lct"]["dnll_flat"] + row["lct"]["dnll_tex"]
            + 0.1 * (row["lct"]["r2_tcent"] + row["lct"]["r2_tmarg"])
        )
        if score > best + 1e-5:
            best = score
            bad = 0
            torch.save(
                {
                    "arms": arms.state_dict(),
                    "textons": C.cpu(), "tex_basis": tex_basis, "ntex": n_textons,
                    "RMU": RMU.cpu(), "RSD": RSD.cpu(),
                    "cnt_flat": cnt_f.cpu(), "cnt_tex": cnt_t.cpu(),
                    "epoch": ep, "row": row,
                    "local_ctx_mean": LM.cpu(), "local_ctx_scale": LS.cpu(),
                },
                str(out_path),
            )
        else:
            bad += 1
        if ep % 5 == 0 or ep == 1:
            print(
                f"ep {ep:3d} [{(time.time() - t0) / 60:5.1f}m] loss {row['train']:.3f} "
                f"| base_flat {bf:.3f} "
                + " | ".join(
                    f"{k} dF{row[k]['dnll_flat']:+.4f} dT{row[k]['dnll_tex']:+.4f}"
                    for k in ("lct", "gct", "both")
                ),
                flush=True,
            )
            json.dump(log, open(str(report_path), "w"), indent=2)
        if bad >= patience:
            print(f"early stop at {ep}", flush=True)
            break

    json.dump(log, open(str(report_path), "w"), indent=2)
    print(f"saved {out_path} and {report_path}")
    return log
