"""
PyTorch fallback for flex_gemm.ops.grid_sample.grid_sample_3d.

The original flex_gemm kernel scatters sparse voxel features into a dense
volume then samples at arbitrary 3D query positions via trilinear
interpolation. This pure-PyTorch implementation does the same in two steps:

  1. Scatter (feats, coords) → dense tensor (B, C, D, H, W)
  2. Sample at query grid positions with torch.nn.functional.grid_sample

Performance is lower than the fused CUDA kernel, but the results are
numerically equivalent for inference.
"""

import torch
import torch.nn.functional as F


def grid_sample_3d(
    feats: torch.Tensor,
    coords: torch.Tensor,
    shape=None,
    grid: torch.Tensor = None,
    mode: str = "trilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Scatter sparse features into a dense volume and sample at query positions.

    This function is called with two different conventions in the TRELLIS.2
    codebase:

    Convention A — positional args (mesh/base.py):
        grid_sample_3d(feats, coords_with_batch, voxel_shape, grid, mode=...)
        where voxel_shape is the 3rd positional arg and grid is the 4th.

    Convention B — keyword args (pipelines/trellis2_texturing.py):
        grid_sample_3d(feats, coords, shape=..., grid=..., mode=...)

    Both conventions are handled transparently via the signature below.

    Args:
        feats:  (N, C) sparse feature values at occupied voxels.
        coords: (N, 4) integer voxel indices [batch, d0, d1, d2].
                Alternatively (N, 3) [d0, d1, d2] (batch=0 assumed).
        shape:  Volume shape hint.  Accepted formats:
                  (B, C, D, H, W) – full dense shape (texturing convention)
                  (D, H, W)       – spatial dims only
                If omitted, spatial dims are inferred from coords.
        grid:   (B, M, 3) query positions in voxel-index space, i.e. each
                coordinate ranges over [0, spatial_dim].
        mode:   'trilinear' (default) or 'nearest'.
        align_corners: Passed to F.grid_sample (default False so that
                coordinate 0 aligns with the left edge of voxel 0 and
                coordinate D aligns with the right edge of voxel D-1).

    Returns:
        (B, M, C) sampled feature vectors when *grid* is provided,
        or (B, C, D, H, W) dense volume when *grid* is None.
    """
    device = feats.device
    dtype = feats.dtype
    C = feats.shape[1]

    # ------------------------------------------------------------------ #
    # Decode coords
    # ------------------------------------------------------------------ #
    if coords.shape[-1] == 4:
        batch_idx = coords[:, 0].long()
        spatial_idx = coords[:, 1:].long()   # (N, 3)  [d0, d1, d2]
    else:
        batch_idx = torch.zeros(coords.shape[0], dtype=torch.long, device=device)
        spatial_idx = coords.long()

    B = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 1

    # ------------------------------------------------------------------ #
    # Determine spatial dimensions
    # ------------------------------------------------------------------ #
    if shape is not None:
        s = list(shape)
        if len(s) == 5:       # (B, C, D, H, W)
            D, H, W = int(s[2]), int(s[3]), int(s[4])
        elif len(s) == 3:     # (D, H, W)
            D, H, W = int(s[0]), int(s[1]), int(s[2])
        else:
            D = int(spatial_idx[:, 0].max().item()) + 1
            H = int(spatial_idx[:, 1].max().item()) + 1
            W = int(spatial_idx[:, 2].max().item()) + 1
    else:
        D = int(spatial_idx[:, 0].max().item()) + 1
        H = int(spatial_idx[:, 1].max().item()) + 1
        W = int(spatial_idx[:, 2].max().item()) + 1

    # ------------------------------------------------------------------ #
    # Scatter into dense volume  (B, C, D, H, W)
    # ------------------------------------------------------------------ #
    dense = torch.zeros(B, C, D, H, W, device=device, dtype=dtype)
    if feats.numel() > 0:
        d0 = spatial_idx[:, 0].clamp(0, D - 1)
        d1 = spatial_idx[:, 1].clamp(0, H - 1)
        d2 = spatial_idx[:, 2].clamp(0, W - 1)
        dense[batch_idx, :, d0, d1, d2] = feats

    if grid is None:
        return dense

    # ------------------------------------------------------------------ #
    # Sample at query positions
    # ------------------------------------------------------------------ #
    # grid: (B, M, 3) in voxel-index space [0, D/H/W]
    # F.grid_sample expects coordinates in [-1, 1] and uses (x, y, z)
    # convention where x->W, y->H, z->D (opposite of our dim ordering).
    #
    # With align_corners=False:
    #   -1 = left edge of voxel 0
    #   +1 = right edge of voxel (dim-1)
    #   formula: norm = coord / dim * 2 - 1
    M = grid.shape[1]
    g = grid.float()

    gx = g[..., 2] / W * 2.0 - 1.0  # W-axis  → x
    gy = g[..., 1] / H * 2.0 - 1.0  # H-axis  → y
    gz = g[..., 0] / D * 2.0 - 1.0  # D-axis  → z

    # Reshape to (B, 1, 1, M, 3) for 5D grid_sample
    norm_grid = torch.stack([gx, gy, gz], dim=-1).unsqueeze(1).unsqueeze(1)

    gs_mode = "bilinear" if mode in ("trilinear", "bilinear") else "nearest"
    sampled = F.grid_sample(
        dense,
        norm_grid,
        mode=gs_mode,
        padding_mode="zeros",
        align_corners=align_corners,
    )  # (B, C, 1, 1, M)

    # → (B, M, C)
    return sampled.squeeze(2).squeeze(2).permute(0, 2, 1).contiguous()
