#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 12:34:36 2025

@author: derik
"""

#%%
import os
import numpy as np
from typing import Any, Dict, Iterable, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from .utils.io import npz_load_volume, ensure_thw, load_thw_raw_only
from .utils.transforms import temporal_pool_max, random_spatial_crop, pick_temporal_span
from .utils.padding import crop_time_to_multiple, pad_hw_symmetric_to_multiple
from .utils.recon import compute_activity_ctx
from .utils.embed import fourier_embed_2d
# =========================
# Helpers (standalone)
# =========================


def _safe_rel_id(path: str, src_root: str | None = None) -> str:
    """Filesystem-safe id derived from the source path, relative to src_root if provided."""
    ap = os.path.abspath(path)

    if src_root is not None:
        sr = os.path.abspath(src_root)
        try:
            rel = os.path.relpath(ap, sr)
            # If path is outside src_root, rel starts with ".." -> fall back to basename
            if rel.startswith(".."):
                rel = os.path.basename(ap)
        except Exception:
            rel = os.path.basename(ap)
    else:
        # fallback: basename only (avoids /media nesting but may collide if filenames repeat)
        rel = os.path.basename(ap)

    rel = rel.replace(":", "")          # windows safety
    rel = rel.lstrip("/\\")             # no leading separators
    rel = rel.replace("\\", "/")        # normalize
    return rel

def _cache_paths_for_source(
    cache_dir: str,
    src_path: str,
    axis_order: str,
    temporal_pool: int | None,
    cache_mode: str,
    src_root: str | None = None,
) -> tuple[str, str | None, str]:
    """Deterministic cache paths derived from the original file path.

    Includes axis_order + temporal_pool in the filename so changing these
    doesn't accidentally reuse an incompatible cache.

    Returns: (data_path, shape_path_or_None, stamp_path)
    """
    rel = _safe_rel_id(src_path, src_root=src_root)
    rel_noext = os.path.splitext(rel)[0]
    base = os.path.join(cache_dir, rel_noext + f".axis{axis_order}.pool{temporal_pool}")

    if cache_mode == "packbits":
        data_path = base + ".pb.npy"
        shape_path = base + ".shape.npy"
    elif cache_mode == "uint8":
        data_path = base + ".thw.npy"
        shape_path = None
    else:
        raise ValueError(f"Unknown cache_mode={cache_mode!r} (use 'packbits' or 'uint8').")

    # stamp stores source size + mtime_ns so we can detect stale cache
    stamp_path = base + ".stamp.npy"
    return data_path, shape_path, stamp_path


def _stamp_matches(src_path: str, stamp_path: str) -> bool:
    """Check whether an existing cache stamp matches the current source file."""
    try:
        if not os.path.exists(stamp_path):
            return False
        st = os.stat(src_path)
        stamp = np.load(stamp_path, allow_pickle=False)
        stamp = np.asarray(stamp).astype(np.int64)
        if stamp.shape != (2,):
            return False
        size, mtime_ns = int(stamp[0]), int(stamp[1])
        return (size == int(st.st_size)) and (mtime_ns == int(st.st_mtime_ns))
    except Exception:
        return False


def _write_stamp_atomic(src_path: str, stamp_path: str) -> None:
    """Write a 2-int stamp [size, mtime_ns] atomically."""
    st = os.stat(src_path)
    os.makedirs(os.path.dirname(stamp_path), exist_ok=True)
    tmp = stamp_path + f".tmp.{os.getpid()}.npy"
    np.save(tmp, np.array([int(st.st_size), int(st.st_mtime_ns)], dtype=np.int64), allow_pickle=False)
    os.replace(tmp, stamp_path)


def _load_cached_thw(
    cache_dir: str,
    src_path: str,
    axis_order: str,
    temporal_pool: int | None,
    cache_mode: str,
    src_root: str | None = None
) -> np.ndarray | None:
    data_path, shape_path, stamp_path = _cache_paths_for_source(
        cache_dir, src_path, axis_order, temporal_pool, cache_mode, src_root=src_root
    )
    if not os.path.exists(data_path):
        return None
    # stale cache protection
    if not _stamp_matches(src_path, stamp_path):
        return None

    if cache_mode == "uint8":
        arr = np.load(data_path, allow_pickle=False, mmap_mode="r")
        return np.asarray(arr, dtype=np.uint8)

    # packbits
    if shape_path is None or (not os.path.exists(shape_path)):
        return None
    shape = np.load(shape_path, allow_pickle=False)
    shape = tuple(int(x) for x in shape.tolist())  # (T,H,W)
    packed = np.load(data_path, allow_pickle=False, mmap_mode="r")
    packed = np.asarray(packed, dtype=np.uint8)

    n = shape[0] * shape[1] * shape[2]
    bits = np.unpackbits(packed)[:n]
    return bits.reshape(shape).astype(np.uint8)


def _save_cached_thw(
    cache_dir: str,
    src_path: str,
    axis_order: str,
    temporal_pool: int | None,
    thw: np.ndarray,
    cache_mode: str,
    src_root: str | None = None
) -> None:
    data_path, shape_path, stamp_path = _cache_paths_for_source(
        cache_dir, src_path, axis_order, temporal_pool, cache_mode, src_root=src_root
    )
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    # Write atomically to avoid partial files if a worker dies mid-write.
    # IMPORTANT: temp names MUST end with .npy (np.save appends otherwise).
    tmp_data = data_path + f".tmp.{os.getpid()}.npy"
    tmp_shape = (shape_path + f".tmp.{os.getpid()}.npy") if shape_path else None

    if cache_mode == "uint8":
        np.save(tmp_data, thw.astype(np.uint8), allow_pickle=False)
        os.replace(tmp_data, data_path)
        _write_stamp_atomic(src_path, stamp_path)
        return

    # packbits
    flat = thw.astype(np.uint8, copy=False).reshape(-1)
    packed = np.packbits(flat)
    np.save(tmp_data, packed, allow_pickle=False)
    np.save(tmp_shape, np.array(thw.shape, dtype=np.int32), allow_pickle=False)
    os.replace(tmp_data, data_path)
    os.replace(tmp_shape, shape_path)
    _write_stamp_atomic(src_path, stamp_path)
    
    

def _enforce_cache_limit(cache_dir: str, max_gb: float) -> None:
    if max_gb is None or max_gb <= 0:
        return
    max_bytes = int(max_gb * (1024 ** 3))
    
    files = []
    total = 0
    for root, _, fns in os.walk(cache_dir):
        for fn in fns:
            p = os.path.join(root, fn)
            if not os.path.isfile(p):
                continue
            st = os.stat(p)
            files.append((st.st_mtime, p, st.st_size))
            total += st.st_size

    if total <= max_bytes:
        return

    files.sort()  # oldest first
    for _, p, sz in files:
        try:
            os.remove(p)
            total -= sz
        except Exception:
            pass
        if total <= max_bytes:
            break


def _to_chw_thw(thw: np.ndarray) -> torch.Tensor:
    """
    Convert (T,H,W) uint8 -> (1,T,H,W) float32 in {0,1}
    """
    t = torch.from_numpy(thw.astype(np.float32))
    t = t.unsqueeze(0)  # (1,T,H,W)
    return t


def _sample_task(task_keys: List[str], task_cum: np.ndarray, rng: np.random.Generator) -> str:
    r = rng.random()
    i = int(np.searchsorted(task_cum, r, side="right"))
    return task_keys[min(i, len(task_keys) - 1)]



def burst_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate for spike volumes + contexts + prior masking metadata.

    Expects per-sample dict to contain at least:
      x:           (1,T,H,W) float
      local_ctx:   (D,) float          [optional]
      global_ctx:  (A,) float          [optional]
      task_id:     int
      mask_spec:   dict
      prefix_frames: int|None          [optional]

    Returns:
      x:            (B,1,T,H,W)
      local_ctx:    (B,D)             if present
      global_ctx:   (B,A)             if present
      task_id:      (B,) long         if present
      assay_idx:  (B,) long         if present
      mask_spec:    list[dict]
      prefix_frames:(B,) long with -1 where None (if present)

      Meta fields (python lists when present):
        assay_name, mode, path,
        t0, T_full, crop_len,
        roi_t_pool, roi_hw, pad_hw, temporal_pool,
        prefix_tokens (deprecated compat)
    """
    if not batch:
        raise ValueError("burst_collate got an empty batch")

    out: Dict[str, Any] = {}

    # ---------------- volumes ----------------
    xs = [b["x"] for b in batch]

    def _ensure_1thw(t: torch.Tensor) -> torch.Tensor:
        # Normalize per-sample volume to (1,T,H,W)
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"Expected tensor for x, got {type(t)}")

        if t.dim() == 5:
            # legacy: (1,1,T,H,W) stored per sample
            if t.size(0) != 1:
                raise ValueError(f"Per-sample x should not include batch dim; got {tuple(t.shape)}")
            t = t.squeeze(0)  # -> (1,T,H,W) or (1,1,T,H,W)

        if t.dim() == 4 and t.size(0) == 1:
            return t
        if t.dim() == 3:
            return t.unsqueeze(0)

        raise ValueError(f"Expected x as (T,H,W) or (1,T,H,W) or (1,1,T,H,W); got {tuple(t.shape)}")

    xs = [_ensure_1thw(t) for t in xs]

    # sanity: all shapes must match for stacking
    x0 = xs[0].shape
    for i, x in enumerate(xs):
        if x.shape != x0:
            raise ValueError(f"x shape mismatch at idx {i}: expected {x0}, got {tuple(x.shape)}")

    out["x"] = torch.stack(xs, dim=0)  # (B,1,T,H,W)

    # ---------------- contexts ----------------
    if "local_ctx" in batch[0]:
        out["local_ctx"] = torch.stack(
            [b["local_ctx"].to(dtype=torch.float32) for b in batch], dim=0
        )

    if "global_ctx" in batch[0]:
        out["global_ctx"] = torch.stack(
            [b["global_ctx"].to(dtype=torch.float32) for b in batch], dim=0
        )

    # ---------------- ids ----------------
    if "task_id" in batch[0]:
        out["task_id"] = torch.tensor([int(b["task_id"]) for b in batch], dtype=torch.long)

    if "assay_idx" in batch[0]:
        out["assay_idx"] = torch.tensor([int(b["assay_idx"]) for b in batch], dtype=torch.long)

    # ---------------- masking metadata for prior ----------------
    out["mask_spec"] = [b.get("mask_spec", None) for b in batch]

    if "prefix_frames" in batch[0]:
        out["prefix_frames"] = torch.tensor(
            [(-1 if b.get("prefix_frames", None) is None else int(b["prefix_frames"])) for b in batch],
            dtype=torch.long
        )

    # ---------------- meta fields kept as python lists ----------------
    meta_keys = [
        "assay_name", "mode", "path", "t0", "T_full", "crop_len",
        "prefix_tokens",
        "roi_t_pool",
        "roi_hw",         # (y0,x0,y1,x1) pre-pad coords in full pooled frame
        "pad_hw",         # (top,bottom,left,right), applied after crop
        "full_hw",        # (H_full, W_full)
        "img_hw",         # (H_img, W_img) after crop+pad
        "temporal_pool",
    ]
    for k in meta_keys:
        if k in batch[0]:
            out[k] = [b.get(k, None) for b in batch]

    return out


# =========================
# Dataset
# =========================





class NpzBurstDataset(Dataset):

    def __init__(
        self,
        assay_indices: Iterable[int],
        assay_dict: Dict[int, Dict[str, Any]],
        *,
        axis_order: str = "HWT",
        dtype: torch.dtype = torch.float32,
        transform: Optional[Any] = None,

        # Data shaping
        temporal_crop: Optional[int] = None,         # crop length in raw, random cropping will be done on max(1,floor(temporal_crop/temporal_pool))
        temporal_pool: Optional[int] = None,         # max-pool factor along time (None or >1)
        spatial_crop: Optional[Tuple[int, int]] = None,
        seed: int = 0,

        # Balancing
        balance_mode: str = "equal",                 # "none"|"equal"|"weighted"
        per_assay_quota: Optional[int] = None,       # for "equal"
        total_quota: Optional[int] = None,           # for "weighted"

        # Task mix & AR tokenization
        task_probs: Optional[Dict[str, float]] = None,  # {"recon": 0.25, "causal": 0.25, "noncausal": 0.25, "spatial": 0.25}
        patch_size: Optional[Tuple[int,int,int]] = None,   # (pt, ph, pw), e.g. (16,16,16)

        # Local context computation control
        use_activity_ctx: bool = True,

        # Global context mapping
        task_id_map: Optional[Dict[str, int]] = None,  # default {"recon": 0, "causal": 1, "noncausal": 2, "spatial": 3}
        n_assays: int = 1000,
        global_ctx_dim: int = 32,
        
        # ---- optional removable cache ----
        cache_dir: Optional[str] = None,
        cache_mode: str = "packbits",         # "packbits" (small) or "uint8" (faster, bigger)
        cache_max_gb: Optional[float] = None, # hard disk cap for cache
        cache_write_prob: float = 1.0,        # 1.0 = cache everything; <1 caches a subset
        cache_src_root: Optional[str] = None,   # if None, auto = commonpath of all sample paths
    ):
        super().__init__()
        # --- store config ---
        self.assay_indices = list(assay_indices)
        self.assay_dict = dict(assay_dict)
        self.axis_order = axis_order.upper()
        assert self.axis_order in {"HWT", "THW"}
        self.dtype = dtype
        self.transform = transform

        self.temporal_crop = temporal_crop
        # Normalize: None -> 1 (no pooling), otherwise int >= 1
        self.temporal_pool = 1 if temporal_pool is None else int(temporal_pool)
        if self.temporal_pool < 1:
            raise ValueError(f"temporal_pool must be >= 1 or None, got {temporal_pool!r}")
        self.spatial_crop = spatial_crop
        self.n_assays = n_assays
        
        # ---- Fixed assay codebook (categorical, non-smooth) ----
        self.global_ctx_dim = global_ctx_dim  # <-- you can change to 16/64 if needed
        
        g = torch.Generator()
        g.manual_seed(0)
        
        codebook = torch.randint(
            0, 2,
            (self.n_assays, self.global_ctx_dim),
            generator=g
        ).float()
        
        codebook = codebook * 2.0 - 1.0  # {0,1} -> {-1,+1}
        codebook = torch.nn.functional.normalize(codebook, dim=-1)
        
        self.assay_codebook = codebook  # (n_assays, global_ctx_dim)
        
        self.cache_dir = cache_dir
        self.cache_mode = cache_mode
        self.cache_max_gb = cache_max_gb
        self.cache_write_prob = float(cache_write_prob)

        self.balance_mode = balance_mode.lower()
        assert self.balance_mode in {"none", "equal", "weighted"}

        self.rng = np.random.default_rng(seed)

        # Tasks
        if task_probs is None:
            task_probs = {"recon": 0.25, "causal": 0.25, "noncausal": 0.25, "spatial": 0.25}
        # sanitize / normalize
        keys = list(task_probs.keys())
        vals = [max(0.0, float(task_probs[k])) for k in keys]
        s = sum(vals)
        if s <= 0:
            keys = ["recon"]; vals = [1.0]; s = 1.0
        vals = [v / s for v in vals]
        self._task_keys = keys
        self._task_cum = np.cumsum(vals)        
        
        self.patch_size = patch_size
        if self.patch_size is not None:
            if len(self.patch_size) != 3:
                raise ValueError(f"patch_size must be a 3-tuple (pt,ph,pw), got {self.patch_size}")
            self.pt, self.ph, self.pw = (int(self.patch_size[0]), int(self.patch_size[1]), int(self.patch_size[2]))
            if self.pt < 1 or self.ph < 1 or self.pw < 1:
                raise ValueError(f"patch_size values must be >=1, got {self.patch_size}")
        else:
            self.pt = self.ph = self.pw = None

        if task_id_map is None:
            task_id_map = {"recon": 0, "causal": 1, "noncausal": 2, "spatial": 3}
        self.task_id_map = dict(task_id_map)

        # --- build flat sample list ---
        self.samples: List[Tuple[int, str, str]] = []
        self.assay_to_indices: Dict[int, List[int]] = {}
        self.assays: List[int] = []
        for aidx in self.assay_indices:
            info = self.assay_dict[aidx]
            assay_name = info.get("assay_name", f"assay_{aidx}")
            files = list(info.get("files", []))
            for p in files:
                self.samples.append((aidx, assay_name, p))
            if aidx not in self.assay_to_indices:
                self.assay_to_indices[aidx] = []
        for i, (aidx, _, _) in enumerate(self.samples):
            self.assay_to_indices.setdefault(aidx, []).append(i)
            
        # --- cache root for relative paths (prevents /media/... nesting) ---
        self.cache_src_root = None
        if self.cache_dir:
            if cache_src_root is not None:
                self.cache_src_root = os.path.abspath(cache_src_root)
            else:
                # auto: common root of all sample paths
                all_paths = [p for (_, _, p) in self.samples]
                if len(all_paths) > 0:
                    self.cache_src_root = os.path.commonpath([os.path.abspath(p) for p in all_paths]) 
                    
                
        self.assays = list({a for (a, _, _) in self.samples} | set(self.assay_indices))
        self.num_assays = len(self.assays)

        # --- balancing plan ---
        if self.balance_mode == "none":
            # length equals number of files
            pass
        elif self.balance_mode == "equal":
            counts = [len(self.assay_to_indices.get(a, [])) for a in self.assays]
            default_quota = max(counts) if len(counts) > 0 else 0
            self.per_assay_quota = int(per_assay_quota) if per_assay_quota is not None else int(default_quota)
        else:  # "weighted"
            assert (total_quota is not None) and (total_quota > 0), \
                "Provide total_quota for balance_mode='weighted'"
            self.total_quota = int(total_quota)
            sizes = np.array([len(self.assay_to_indices.get(a, [])) for a in self.assays], dtype=np.float64)
            with np.errstate(divide='ignore'):
                inv = np.where(sizes > 0, 1.0 / sizes, 0.0)
            if inv.sum() > 0:
                self.assay_probs = inv / inv.sum()
            else:
                self.assay_probs = np.ones_like(inv) / max(1, len(inv))

        # Context knobs
        self.use_activity_ctx = bool(use_activity_ctx)

    # --------------- length ----------------
    def __len__(self) -> int:
        if self.balance_mode == "none":
            return len(self.samples)
        elif self.balance_mode == "equal":
            return self.per_assay_quota * max(1, self.num_assays)
        else:  # weighted
            return self.total_quota

    # --------------- index sampler (balancing) ----------------
    def _pick_index(self, i: int) -> int:
        if self.balance_mode == "none":
            return i
        elif self.balance_mode == "equal":
            # block of per_assay_quota per assay
            a_idx = i // self.per_assay_quota
            a_idx = int(np.clip(a_idx, 0, self.num_assays - 1))
            assay_id = self.assays[a_idx]
            pool = self.assay_to_indices.get(assay_id, [])
            if not pool:
                # fallback: random from all
                return int(self.rng.integers(0, max(1, len(self.samples))))
            return int(pool[int(self.rng.integers(0, len(pool)))])
        else:  # weighted
            # choose an assay by inverse-size probability, then a file within it
            assay_id = int(self.rng.choice(self.assays, p=self.assay_probs))
            pool = self.assay_to_indices.get(assay_id, [])
            if not pool:
                return int(self.rng.integers(0, max(1, len(self.samples))))
            return int(pool[int(self.rng.integers(0, len(pool)))])


    def _sample_large_spatial_box(self, H, W,
                                  area_min=0.25, area_max=0.60,
                                  aspect_min=0.5, aspect_max=2.0):
        a = float(self.rng.uniform(area_min, area_max))
        A = max(1.0, a * H * W)
    
        r = float(self.rng.uniform(aspect_min, aspect_max))  # h/w
        bh = int(round(np.sqrt(A * r)))
        bw = int(round(np.sqrt(A / r)))
    
        bh = int(np.clip(bh, 1, H))
        bw = int(np.clip(bw, 1, W))
    
        y0 = int(self.rng.integers(0, max(1, H - bh + 1)))
        x0 = int(self.rng.integers(0, max(1, W - bw + 1)))
        return (y0, y0 + bh, x0, x0 + bw)

    # --------------- main item ----------------
    def __getitem__(self, i: int) -> Dict[str, Any]:
        # ---------------- choose sample ----------------
        sidx = self._pick_index(i)
        assay_idx, assay_name, path = self.samples[sidx]
    
        # choose a task for this sample
        mode = _sample_task(self._task_keys, self._task_cum, self.rng)
        task_id = int(self.task_id_map.get(mode, 0))
    
        # ---------------- load base THW (full recording, pooled) ----------------
        pool = int(self.temporal_pool)  # always >= 1
    
        thw_full = None
    
        # try cache first (even pool==1)
        if self.cache_dir:
            thw_full = _load_cached_thw(
                self.cache_dir,
                src_path=path,
                axis_order=self.axis_order,
                temporal_pool=pool,
                cache_mode=self.cache_mode,
                src_root=self.cache_src_root,
            )
           
        
        if thw_full is None:
            thw_raw = load_thw_raw_only(path=path, axis_order=self.axis_order)  # (T_raw,H,W)
            thw_full = temporal_pool_max(thw_raw, pool) if pool > 1 else thw_raw
        
            if self.cache_dir and (self.rng.random() < float(getattr(self, "cache_write_prob", 1.0))):
                try:
                    _save_cached_thw(
                        self.cache_dir,
                        src_path=path,
                        axis_order=self.axis_order,
                        temporal_pool=pool,
                        thw=thw_full,
                        cache_mode=self.cache_mode,
                        src_root=self.cache_src_root,
                    )
                except Exception as e:
                    print("CACHE SAVE ERROR:", path, e)
        
                if self.cache_max_gb and self.cache_max_gb > 0 and self.rng.random() < 0.01:
                    try:
                        _enforce_cache_limit(self.cache_dir, self.cache_max_gb)
                    except Exception as e:
                        print("CACHE MEMORY LIMIT REACHED:", e)
        

    
        # pooled full shape
        T_pool_full, H_full, W_full = thw_full.shape
        
        
    
        # ---------------- random temporal crop (AFTER cache) ----------------
        # defaults mean "use full span"
        t0_pool, t1_pool = 0, T_pool_full
        t0_raw = 0  # in raw units (approx aligned to pool stride)
    
        thw_span = thw_full  # start from full pooled
        
    
        if self.temporal_crop is not None and self.temporal_crop > 0:
            # temporal_crop is specified in RAW units -> convert to pooled units
            crop_len_pool = max(1, int(self.temporal_crop // pool))
            t0_pool, t1_pool = pick_temporal_span(T_pool_full, crop_len_pool, self.rng)
            thw_span = thw_full[t0_pool:t1_pool]  # (T_pool_span,H,W)
            
            t0_raw = int(t0_pool * pool)
    
        # ---------------- random spatial crop (optional, metadata only for now) ----------------
        # roi in PRE-PAD coords, relative to the (H_full,W_full) frame (since we crop from pooled full)
        roi_hw = (0, 0, H_full, W_full)
    
        if self.spatial_crop is not None:
            thw_span, roi_hw = random_spatial_crop(thw_span, self.spatial_crop, self.rng)  # returns (thw, roi)
    
        # ---------------- enforce patch-size compatibility ----------------
        pad_info = (0, 0, 0, 0)  # (pad_top, pad_bottom, pad_left, pad_right)
    
        if self.patch_size is not None:
            # 1) Temporal: end-crop to multiple of pt
            thw_span = crop_time_to_multiple(thw_span, self.pt)
            
    
            # 2) Spatial: symmetric pad to multiples of (ph,pw)
            thw_span, pad_info = pad_hw_symmetric_to_multiple(thw_span, self.ph, self.pw)
    
        Tc, H2, W2 = thw_span.shape  # AFTER pad/crop

    
        # Safety: some masking modes require at least 2 frames
        if Tc < 2:
            mode = "recon"
            task_id = int(self.task_id_map.get(mode, 0))
    
        # ---------------- x=y always ----------------
        x_thw = thw_span
    
        # ---------------- mask spec for the PRIOR (metadata only) ----------------
        mask_spec = {"type": "recon"}
        prefix_frames = None
    
        if mode == "causal":
            min_frac, max_frac = 0.25, 0.75
            pf = int(round(Tc * float(self.rng.uniform(min_frac, max_frac))))
            pf = int(np.clip(pf, 1, Tc - 1))
            prefix_frames = int(pf)
            mask_spec = {"type": "causal", "prefix_frames": int(prefix_frames)}
    
        elif mode == "noncausal":
            frac = 0.30
            L = max(1, int(round(frac * Tc)))
            start = int(self.rng.integers(0, max(1, Tc - L + 1)))
            end = int(start + L)
            mask_spec = {"type": "noncausal", "time_spans": [(int(start), int(end))]}
    
        elif mode == "spatial":
            y0, y1, x0, x1 = self._sample_large_spatial_box(H2, W2)
            mask_spec = {"type": "spatial", "spatial_box": (int(y0), int(y1), int(x0), int(x1))}
    
        # ---------------- tensors ----------------
        x = _to_chw_thw(x_thw)  # (1,T,H,W) float32 {0,1}
    
        # ---------------- local contexts ----------------
        if self.use_activity_ctx:
            local_ctx_np = compute_activity_ctx(thw_span).astype(np.float32, copy=False)
        else:
            local_ctx_np = np.zeros(9, dtype=np.float32)
        local_ctx = torch.from_numpy(local_ctx_np)
    
        # ---------------- global contexts ----------------
        # assay_embed = fourier_embed_2d(assay_idx, self.n_assays)
        # global_ctx = torch.tensor(assay_embed, dtype=torch.float32)
        
        # global_ctx = torch.nn.functional.one_hot(
        #     torch.tensor(assay_idx, dtype=torch.long),
        #     num_classes=self.n_assays
        # ).float()
        
        global_ctx = self.assay_codebook[assay_idx].clone().to(torch.float32)
    
        # ---------------- output ----------------
        out = {
            "x": x.to(self.dtype),
    
            "local_ctx": local_ctx,
            "global_ctx": global_ctx,
    
            # prior masking metadata
            "mask_spec": mask_spec,
            "prefix_frames": prefix_frames,
            "prefix_tokens": None,  # keep compat
    
            # provenance / debug
            "assay_idx": assay_idx,
            "assay_name": assay_name,
            "task_id": task_id,
            "mode": mode,
            "path": path,
    
            # time + shape meta
            "t0": int(t0_raw),                 # raw start (aligned to pool stride)
            "T_full": int(T_pool_full),        # pooled full recording length
            "crop_len": int(Tc),               # returned T AFTER pt-crop
            
            "pad_hw": pad_info,                # (top,bottom,left,right), applied AFTER crop
            "roi_t_pool": (int(t0_pool), int(t1_pool)),
            "roi_hw": tuple(int(v) for v in roi_hw),     # (y0,x0,y1,x1) in PRE-PAD full-frame coords
            "full_hw": (int(H_full), int(W_full)),       # full pooled frame size before spatial crop
            "img_hw": (int(H2), int(W2)),                # final sample spatial size after crop+pad
            "temporal_pool": int(pool),
        }
    
        if self.transform is not None:
            out = self.transform(out)
        return out



def split_indices(n: int, val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert 0.0 <= val_frac < 1 and 0.0 <= test_frac < 1 and val_frac + test_frac < 1
    rng = np.random.default_rng(seed)
    idxs = np.arange(n)
    rng.shuffle(idxs)
    n_test = int(round(n * test_frac))
    n_val  = int(round(n * val_frac))
    n_train = max(0, n - n_val - n_test)
    return idxs[:n_train], idxs[n_train:n_train + n_val], idxs[n_train + n_val:]



def make_loaders_for_assays(
    assay_indices,
    assay_dict,
    axis_order: str = "HWT",
    batch_size: int = 1,
    num_workers: int = 0,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    pin_memory: bool = True,
    shuffle_train: bool = True,
    *,
    # ---- data shaping (passed to dataset) ----
    temporal_crop: Optional[int] = None,
    temporal_pool: Optional[int] = None,
    spatial_crop: Optional[Tuple[int, int]] = None,

    # ---- balancing knobs (passed to dataset) ----
    balance_mode: str = "equal",                # "none" | "equal" | "weighted"
    per_assay_quota: Optional[int] = 500,       # used when balance_mode == "equal"
    total_quota: Optional[int] = None,          # used when balance_mode == "weighted"
    n_assays: Optional[int] = 1000,
    dim_assays: Optional[int] = 32,
    # ---- loader tweaks ----
    drop_last: bool = False,
    collate_fn=None,                             # defaults to burst_collate if None
    
    # ---- optional removable cache ----
    cache_dir: Optional[str] = None,
    cache_mode: str = "packbits",
    cache_max_gb: Optional[float] = None,
    cache_write_prob: float = 1.0,

    # ---- EVERYTHING ELSE goes straight to NpzBurstDataset ----
    **ds_kwargs: Any,                            # e.g. task_probs, patch_size, use_activity_ctx, etc.
):
    """
    Builds train/val/test DataLoaders.

    Examples of extras via ds_kwargs:
      task_probs={"recon": 0.25, "causal": 0.25, "noncausal": 0.25, "spatial": 0.25}, patch_size=(16,16,16),
      use_activity_ctx=True, task_id_map={"recon": 0, "causal": 1, "noncausal": 2, "spatial": 3}
    """
    # Default to the mode-safe collate if not supplied
    if collate_fn is None:
        collate_fn = burst_collate

    # Construct the base dataset (length reflects chosen balance_mode)
    base_ds = NpzBurstDataset(
        assay_indices=assay_indices,
        assay_dict=assay_dict,
        axis_order=axis_order,

        # balancing
        balance_mode=balance_mode,
        per_assay_quota=per_assay_quota,
        total_quota=total_quota,

        # shaping
        temporal_crop=temporal_crop,
        temporal_pool=temporal_pool,
        spatial_crop=spatial_crop,
        
        cache_dir=cache_dir,
        cache_mode=cache_mode,
        cache_max_gb=cache_max_gb,
        cache_write_prob=cache_write_prob,
        
        n_assays=n_assays,
        global_ctx_dim=dim_assays,

        seed=seed,
        **ds_kwargs,   # <— forward extras (task_probs, patch_size, etc.)
    )

    # Split indices over the *dataset length* (not number of files)
    tr_idx, va_idx, te_idx = split_indices(len(base_ds), val_frac, test_frac, seed)

    ds_train = Subset(base_ds, tr_idx)
    ds_val   = Subset(base_ds, va_idx) if len(va_idx) else None
    ds_test  = Subset(base_ds, te_idx) if len(te_idx) else None

    # persistent_workers only valid if num_workers>0 and shuffle semantics ok
    persistent_ok = bool(num_workers and num_workers > 0)
    
    
    dl_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=persistent_ok,
        drop_last=drop_last,
    )
    if num_workers and num_workers > 0:
        dl_kwargs["prefetch_factor"] = 1
    
    loader_train = DataLoader(ds_train, shuffle=shuffle_train, **dl_kwargs)
    loader_val   = DataLoader(ds_val,   shuffle=False, **{**dl_kwargs, "drop_last": False}) if ds_val is not None else None
    loader_test  = DataLoader(ds_test,  shuffle=False, **{**dl_kwargs, "drop_last": False}) if ds_test is not None else None
        

    meta = {
        "num_items_total": len(base_ds),
        "splits": {"train": len(tr_idx), "val": len(va_idx), "test": len(te_idx)},
        "balance_mode": balance_mode,
        "per_assay_quota": per_assay_quota,
        "total_quota": total_quota,
        "temporal_crop": temporal_crop,
        "temporal_pool": temporal_pool,
        "spatial_crop": spatial_crop,
        "dataset_kwargs": dict(ds_kwargs),  # for logging/repro
    }
    return loader_train, loader_val, loader_test, meta
