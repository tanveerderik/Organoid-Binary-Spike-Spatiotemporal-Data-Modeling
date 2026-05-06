#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:03:10 2026

@author: derik
"""

import os
import numpy as np
import torch
import imageio.v2 as iio
import json
from matplotlib import cm
from ..utils.recon import compute_activity_ctx
from ..utils.metrics import f1_from_bool



def save_volume_as_mp4_imageio(volume_hw_t, filename, fps=30, to_rgb=True):
    """
    Save a video from a volume of shape (H, W, T) using imageio (ffmpeg).
    to_rgb=True writes 3-channel frames for broad player compatibility.
    """
    vol = np.asarray(volume_hw_t)
    H, W, T = vol.shape
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    with iio.get_writer(filename, fps=fps, codec='libx264', macro_block_size=None) as writer:
        for t in range(T):
            frame = vol[:, :, t]
            if to_rgb:
                frame = np.repeat(frame[..., None], 3, axis=2)  # (H,W,3)
            writer.append_data(frame)
            


def temporal_maxpool_np(volume_hw_t: np.ndarray, pool_t: int) -> np.ndarray:
    """
    Temporal MAX-pooling along the last axis (keeps binary structure).
    volume_hw_t: (H, W, T) numpy array
    pool_t: int >= 1
    """
    if pool_t is None or pool_t <= 1:
        return volume_hw_t
    H, W, T = volume_hw_t.shape
    T_trunc = (T // pool_t) * pool_t
    if T_trunc == 0:
        raise ValueError("pool_t is larger than the number of frames.")
    v = volume_hw_t[:, :, :T_trunc]
    v = v.reshape(H, W, T_trunc // pool_t, pool_t).max(axis=-1)  # (H,W,T//pool_t)
    return v



# ---------- Heatmap + spike overlay helpers ----------


def _colormap_rgb(img01: np.ndarray, cmap_name: str = "magma") -> np.ndarray:
    """
    Map a 2D array in [0,1] to an RGB image uint8 via a matplotlib colormap.
    """
    arr = np.clip(img01.astype(np.float32), 0.0, 1.0)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(arr)  # (H,W,4) float in [0,1]
    rgb  = (rgba[..., :3] * 255.0 + 0.5).astype(np.uint8)
    return rgb

def _binary_dilate(mask: np.ndarray, r: int = 1) -> np.ndarray:
    """
    Simple square structuring-element dilation (radius r) with numpy only.
    """
    if r <= 0:
        return mask
    H, W = mask.shape
    out = np.zeros((H, W), dtype=bool)
    pad = r
    m = np.pad(mask.astype(bool), pad, mode="constant", constant_values=False)
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            out |= m[pad+dy:pad+dy+H, pad+dx:pad+dx+W]
    return out

def _overlay_spikes(rgb: np.ndarray, spikes: np.ndarray, color=(0,255,255), r: int = 1, alpha: float = 1.0) -> np.ndarray:
    """
    Paint (dilated) spike pixels onto an RGB image. rgb: (H,W,3) uint8; spikes: (H,W) bool.
    """
    H, W, _ = rgb.shape
    dil = _binary_dilate(spikes, r=r)
    over = rgb.copy()
    cy, cx = np.nonzero(dil)
    if cy.size:
        # solid paint (alpha can be <1 if you want blending)
        if alpha >= 1.0:
            over[cy, cx] = np.array(color, dtype=np.uint8)
        else:
            col = np.array(color, dtype=np.float32)
            over[cy, cx] = (alpha * col + (1.0 - alpha) * over[cy, cx].astype(np.float32)).astype(np.uint8)
    return over


def save_heatmap_videos(
    prob_hw_t: np.ndarray,              # (H,W,T) float in [0,1]
    spikes_hw_t: np.ndarray,            # (H,W,T) bool
    out_path_heatmap: str,
    out_path_overlay: str,
    fps: int = 30,
    cmap_name: str = "magma",
    spike_color=(0, 255, 255),          # cyan
    spike_radius: int = 1,
    spike_alpha: float = 1.0,
):
    """
    Write two MP4s: heatmap only and heatmap + spikes overlay.
    """
    H, W, T = prob_hw_t.shape
    os.makedirs(os.path.dirname(out_path_heatmap) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_path_overlay) or ".", exist_ok=True)

    with iio.get_writer(out_path_heatmap, fps=fps, codec="libx264", macro_block_size=None) as wh, \
         iio.get_writer(out_path_overlay, fps=fps, codec="libx264", macro_block_size=None) as wo:
        for t in range(T):
            heat_rgb = _colormap_rgb(prob_hw_t[..., t], cmap_name=cmap_name)          # (H,W,3) uint8
            over_rgb = _overlay_spikes(heat_rgb, spikes_hw_t[..., t], color=spike_color,
                                       r=spike_radius, alpha=spike_alpha)
            wh.append_data(heat_rgb)
            wo.append_data(over_rgb)
            
            

def make_three_panel_from_two(vol_ref_hw_t: np.ndarray, vol_bin_hw_t: np.ndarray) -> np.ndarray:
    """
    Build a 1Ã—3 grid from TWO inputs:
      panel 1: vol_ref (original)
      panel 2: vol_bin (binary prediction)
      panel 3: contrast panel = scaled (bin - ref) for quick visual diffs
    All inputs: (H, W, T). Returns (H, 3W, T) uint8.
    """
    assert vol_ref_hw_t.ndim == 3 and vol_bin_hw_t.ndim == 3
    H1, W1, T1 = vol_ref_hw_t.shape
    H2, W2, T2 = vol_bin_hw_t.shape
    assert H1 == H2 and W1 == W2, "All volumes must share same H,W"
    T = min(T1, T2)
    v0 = vol_ref_hw_t[:, :, :T]
    v1 = vol_bin_hw_t[:, :, :T]
    # v2: visual difference (scaled)
    v2 = np.clip((v1.astype(np.float32) - v0.astype(np.float32) + 255.0) * 0.45, 0.0, 255.0).astype(np.uint8)
    return np.concatenate([v0, v1, v2], axis=1)




# ---------- Spatial bias map helpers ----------

def upsample_spatial_bias_to_hw(
    bias_tok: np.ndarray,
    patch_h: int,
    patch_w: int,
    out_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Convert token-grid spatial bias (h_tok, w_tok) into image-space (H, W)
    by repeating each token cell over its patch footprint.
    """
    import numpy as np

    bias_hw = np.repeat(np.repeat(bias_tok, patch_h, axis=0), patch_w, axis=1)

    if out_hw is not None:
        H, W = out_hw
        bias_hw = bias_hw[:H, :W]

    return bias_hw.astype(np.float32)


def find_bias_peaks(
    bias_hw: np.ndarray,
    top_k: int = 5,
    min_distance: int = 6,
) -> list[dict]:
    """
    Very simple non-max style peak picker for positive peaks.
    Returns a list of dicts with y, x, value.

    This avoids marking many neighboring pixels from the same hotspot.
    """
    import numpy as np

    arr = np.asarray(bias_hw, dtype=np.float32)
    H, W = arr.shape

    flat_idx = np.argsort(arr.ravel())[::-1]
    peaks = []

    for idx in flat_idx:
        y, x = np.unravel_index(idx, arr.shape)
        val = float(arr[y, x])

        # only keep positive peaks
        if val <= 0:
            break

        too_close = False
        for p in peaks:
            dy = y - p["y"]
            dx = x - p["x"]
            if (dy * dy + dx * dx) <= (min_distance * min_distance):
                too_close = True
                break

        if too_close:
            continue

        peaks.append({
            "y": int(y),
            "x": int(x),
            "value": val,
        })

        if len(peaks) >= top_k:
            break

    return peaks

def save_spatial_bias_heatmap_png(
    bias_hw: np.ndarray,
    out_path: str,
    peaks: list[dict] | None = None,
    title: str = "Spatial bias",
):
    """
    Save a heatmap PNG for the spatial bias.
    Uses a zero-centered diverging colormap.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    arr = np.asarray(bias_hw, dtype=np.float32)
    vmax = float(np.max(np.abs(arr)))
    vmax = max(vmax, 1e-8)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(arr, cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Bias value")

    if peaks is not None:
        for i, p in enumerate(peaks):
            x = p["x"]
            y = p["y"]
            ax.scatter(x, y, s=40, marker="x", c="black", linewidths=1.2)
            ax.text(
                x + 1,
                y + 1,
                f"{i+1}",
                color="black",
                fontsize=8,
                ha="left",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6),
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    
    
    
# ---------- Video maker function ----------


@torch.no_grad()
def make_model_videos_vqvae(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    out_root: str = "viz_out_vqvae",
    max_samples: int | None = None,
    fps: int = 30,
    pool_t: int | None = 1,
    thr: float | None = None,
    cmap_name: str = "viridis",
    save_spatial_bias_npy: bool = True,
    save_spatial_bias_png: bool = True,
    spatial_bias_top_k: int = 5,
):
    """
    For each sample (expects B=1, C=1), saves:
      - reference.mp4
      - recon_prob.mp4
      - recon_bin.mp4
      - heatmap_prob.mp4
      - heatmap_prob+pred.mp4
      - grid_1x3.mp4  [ original | bin | contrast(bin - original) ]
      - spatial_bias_heatmap.png          (if available)
      - spatial_bias_hw.npy               (if available and enabled)
      - results.json
      - arrays.npz
    """

    model.eval()
    device = next(model.parameters()).device

    if thr is None:
        thr = float(getattr(model, "best_thr", 0.5))
        print("Using threshold:", thr)

    os.makedirs(out_root, exist_ok=True)
    count = 0

    def _get_ctx(batch, key, fallback=None):
        val = batch.get(key, None)
        if val is None and fallback is not None:
            val = batch.get(fallback, None)

        if isinstance(val, (list, tuple)) and len(val) > 0:
            val = val[0]

        if isinstance(val, torch.Tensor):
            return val.to(device=device, dtype=torch.float32, non_blocking=True)

        return None

    for k, batch in enumerate(loader):
        if (max_samples is not None) and (count >= max_samples):
            break

        x = batch["x"].to(device, non_blocking=True)  # (B=1,1,T,H,W)

        B, C, *_ = x.shape
        assert B == 1 and C == 1, "Expect B=1, C=1 for this visualizer."

        gct = _get_ctx(batch, "global_ctx")
        lct = _get_ctx(batch, "local_ctx")

        assay_name = batch.get("assay_name", "assay")
        if isinstance(assay_name, (list, tuple)) and len(assay_name) > 0:
            assay_name = assay_name[0]
        assay_name = str(assay_name).replace(os.sep, "_")

        pT, pH, pW = model.patch_size
        _, _, T, H, W = x.shape
        assert T % pT == 0 and H % pH == 0 and W % pW == 0, \
            f"Input {(T, H, W)} not divisible by patch {(pT, pH, pW)}"

        out = model(
            x,
            global_ctx=gct,
            local_ctx=lct,
            roi_hw=batch.get("roi_hw", None),
            pad_hw=batch.get("pad_hw", None),
        )
        grid = out["grid"]
        
        # with torch.no_grad():
        #     blank_mask = out["blank_mask"].detach().bool()
        #     thr_debug = float(thr)
        
        #     raw_blank_logits = out["pred_patches_raw"][blank_mask]
        #     final_blank_logits = out["pred_patches"][blank_mask]
        
        #     raw_blank_prob = torch.sigmoid(raw_blank_logits)
        #     final_blank_prob = torch.sigmoid(final_blank_logits)
        
        #     print("\n[BLANK DEBUG]")
        #     print("thr =", thr_debug)
        #     print("blank patch frac =", blank_mask.float().mean().item())
        
        #     print("RAW blank max logit =", raw_blank_logits.max().item())
        #     print("RAW blank max prob  =", raw_blank_prob.max().item())
        #     print("RAW blank voxels > thr =", (raw_blank_prob > thr_debug).sum().item())
        #     print("RAW blank patches any > thr =",
        #           (raw_blank_prob.amax(dim=-1) > thr_debug).sum().item(),
        #           "/", raw_blank_prob.shape[0])
        
        #     print("FINAL blank max logit =", final_blank_logits.max().item())
        #     print("FINAL blank max prob  =", final_blank_prob.max().item())
        #     print("FINAL blank voxels > thr =", (final_blank_prob > thr_debug).sum().item())
        #     print("FINAL blank patches any > thr =",
        #           (final_blank_prob.amax(dim=-1) > thr_debug).sum().item(),
        #           "/", final_blank_prob.shape[0])
        
        #     patch_score = 0.25 * torch.logsumexp(raw_blank_logits / 0.25, dim=1)
        #     print("RAW patch_score max =", patch_score.max().item())
        #     print("RAW patch_score > -6 =", (patch_score > -6.0).sum().item())
        
        
        logits = out["logits_vol"]
        prob = torch.sigmoid(logits)
        
        # logits_from_patches = model.unpatchify(out["pred_patches"], grid)
        # logits_direct = out["logits_vol"]
        
        # print("unpatchify mismatch:",
        #       (logits_from_patches - logits_direct).abs().max().item())

        # This should already be in HW token-space or token-space 2D map from your model output
        sp_bias_hw = out.get("assay_spatial_pix2d_support", None)

        x_unp = x
        prob_unp = prob

        # numpy as (H,W,T)
        ref_vol = x_unp[0, 0].float().cpu().numpy().transpose(1, 2, 0)
        recon_prob = prob_unp[0, 0].float().cpu().numpy().transpose(1, 2, 0)

        # display-only temporal pooling
        ref_vol = temporal_maxpool_np(ref_vol, pool_t)
        recon_prob = temporal_maxpool_np(recon_prob, pool_t)

        # basic views
        ref_u8 = (ref_vol > 0.5).astype(np.uint8) * 255
        prob_u8 = (np.clip(recon_prob, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        bin_u8 = (recon_prob >= thr).astype(np.uint8) * 255
        pred_bin_bool = (recon_prob >= thr)
        
        # voxel-space blank patch mask: 1 where the full 16x16x16 target patch is blank
        blank_patch_u8 = out["blank_mask"].float().unsqueeze(-1).expand_as(out["pred_patches"])
        blank_vox = model.unpatchify(blank_patch_u8, grid)  # (B,1,T,H,W)
        
        blank_vol = blank_vox[0, 0].float().cpu().numpy().transpose(1, 2, 0)
        blank_vol = temporal_maxpool_np(blank_vol, pool_t)
        
        active_u8 = (blank_vol < 0.5).astype(np.uint8) * 255
        

        global_ctx = None
        local_ctx_list = None

        if isinstance(lct, torch.Tensor):
            local_ctx_list = lct[0].detach().cpu().numpy().tolist()

        if isinstance(batch.get("global_ctx", None), torch.Tensor):
            global_ctx = batch["global_ctx"][0].detach().cpu().numpy().tolist()

        out_dir = os.path.join(out_root, assay_name, f"sample_{k:04d}")
        os.makedirs(out_dir, exist_ok=True)

        # -----------------------------
        # Save videos
        # -----------------------------
        ref_path = os.path.join(out_dir, "reference.mp4")
        prob_path = os.path.join(out_dir, "recon_prob.mp4")
        bin_path = os.path.join(out_dir, "recon_bin.mp4")
        grid_path = os.path.join(out_dir, "grid_1x3.mp4")
        heatmap_path = os.path.join(out_dir, "heatmap_prob.mp4")
        overlay_path = os.path.join(out_dir, "heatmap_prob+pred.mp4")
        active_mask_path = os.path.join(out_dir, "active_patch_mask.mp4")

        save_volume_as_mp4_imageio(ref_u8, ref_path, fps=fps)
        save_volume_as_mp4_imageio(prob_u8, prob_path, fps=fps)
        save_volume_as_mp4_imageio(bin_u8, bin_path, fps=fps)
        save_volume_as_mp4_imageio(active_u8, active_mask_path, fps=fps)

        grid_1x3_u8 = make_three_panel_from_two(ref_u8, bin_u8)
        save_volume_as_mp4_imageio(grid_1x3_u8, grid_path, fps=fps)

        save_heatmap_videos(
            recon_prob.astype(np.float32),
            pred_bin_bool.astype(bool),
            out_path_heatmap=heatmap_path,
            out_path_overlay=overlay_path,
            fps=fps,
            cmap_name=cmap_name,
            spike_color=(255, 0, 0),
            spike_radius=1,
            spike_alpha=1.0,
        )

        # -----------------------------
        # Metrics and metadata
        # -----------------------------
        ref_bin_bool = (ref_u8 > 0)
        pred_bin_bool = (recon_prob >= thr)

        f1 = f1_from_bool(ref_bin_bool, pred_bin_bool)

        ctx_ref_on_window = compute_activity_ctx(
            ref_bin_bool.transpose(2, 0, 1).astype(np.uint8)
        )
        ctx_pred_on_window = compute_activity_ctx(
            pred_bin_bool.transpose(2, 0, 1).astype(np.uint8)
        )

        assay_id = None
        if "assay_idx" in batch:
            if isinstance(batch["assay_idx"], torch.Tensor):
                assay_id = int(batch["assay_idx"].flatten()[0].item())
            elif isinstance(batch["assay_idx"], (list, tuple)):
                assay_id = int(batch["assay_idx"][0])
            else:
                assay_id = int(batch["assay_idx"])
        elif "global_ctx_ids" in batch and isinstance(batch["global_ctx_ids"], torch.Tensor):
            assay_id = int(batch["global_ctx_ids"][0, 0].item())

        mode = None
        mv = batch.get("mode", None)
        if isinstance(mv, str):
            mode = mv
        elif isinstance(mv, (list, tuple)) and len(mv) > 0:
            mode = str(mv[0])
        elif isinstance(mv, torch.Tensor):
            try:
                mode = str(mv.flatten()[0].item())
            except Exception:
                mode = str(mv.detach().cpu().tolist())

        print("Assay ID:", assay_id, ", Mode:", mode)

        # -----------------------------
        # Save numeric arrays
        # -----------------------------
        npz_name = "arrays.npz"
        npz_path = os.path.join(out_dir, npz_name)
        np.savez_compressed(
            npz_path,
            ref_u8=ref_u8.astype(np.uint8),
            prob_u8=prob_u8.astype(np.uint8),
            pred_u8=bin_u8.astype(np.uint8),
        )

        # -----------------------------
        # Spatial bias save
        # -----------------------------
        spatial_bias_block = None

        if sp_bias_hw is not None:
            # expected shape after selecting batch item -> (h_tok, w_tok)
            sp_bias_hw = sp_bias_hw[0].detach().float().cpu().numpy()
            peaks = find_bias_peaks(sp_bias_hw, top_k=spatial_bias_top_k)

            spatial_bias_block = {
                "abs_mean": float(np.mean(np.abs(sp_bias_hw))),
                "std": float(np.std(sp_bias_hw)),
                "top_peaks": peaks,
            }

            if save_spatial_bias_npy:
                sb_npy_name = "spatial_bias_hw.npy"
                np.save(
                    os.path.join(out_dir, sb_npy_name),
                    sp_bias_hw.astype(np.float32),
                )
                spatial_bias_block["hw_npy"] = sb_npy_name

            if save_spatial_bias_png:
                sb_png_name = "spatial_bias_heatmap.png"
                save_spatial_bias_heatmap_png(
                    sp_bias_hw,
                    os.path.join(out_dir, sb_png_name),
                    peaks=peaks,
                    title="Spatial bias",
                )
                spatial_bias_block["heatmap_png"] = sb_png_name

        # -----------------------------
        # Write JSON report
        # -----------------------------
        results_path = os.path.join(out_dir, "results.json")
        report = {
            "assay_name": assay_name,
            "assay_id": assay_id,
            "mode": mode,
            "threshold": float(thr),
            "global_ctx": global_ctx,
            "local_ctx": local_ctx_list,
            "ctx_ref_on_window": [float(x) for x in ctx_ref_on_window.tolist()],
            "ctx_pred_on_window": [float(x) for x in ctx_pred_on_window.tolist()],
            "f1_volume": float(f1),
            "shapes": {
                "ref_u8": list(ref_u8.shape),
                "prob_u8": list(prob_u8.shape),
                "pred_u8": list(bin_u8.shape),
            },
            "videos": {
                "reference": os.path.basename(ref_path),
                "recon_prob": os.path.basename(prob_path),
                "recon_bin": os.path.basename(bin_path),
                "grid_1x3": os.path.basename(grid_path),
                "heatmap_prob": os.path.basename(heatmap_path),
                "heatmap_overlay": os.path.basename(overlay_path),
            },
            "arrays_npz": npz_name,
        }

        if spatial_bias_block is not None:
            report["spatial_bias"] = spatial_bias_block

        with open(results_path, "w") as f:
            json.dump(report, f, indent=2)

        print("Saved:", out_dir)
        count += 1