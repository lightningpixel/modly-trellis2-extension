from typing import Union

import torch

from ._helpers import batched

__all__ = [
    "perspective",
    "perspective_from_fov_xy",
    "intrinsics_from_focal_center",
    "intrinsics_from_fov_xy",
    "view_look_at",
    "extrinsics_look_at",
    "intrinsics_to_perspective",
    "extrinsics_to_view",
]


@batched(0, 0, 0, 0)
def perspective(
    fov_y: Union[float, torch.Tensor],
    aspect: Union[float, torch.Tensor],
    near: Union[float, torch.Tensor],
    far: Union[float, torch.Tensor],
) -> torch.Tensor:
    n = fov_y.shape[0]
    ret = torch.zeros((n, 4, 4), dtype=fov_y.dtype, device=fov_y.device)
    ret[:, 0, 0] = 1.0 / (torch.tan(fov_y / 2) * aspect)
    ret[:, 1, 1] = 1.0 / torch.tan(fov_y / 2)
    ret[:, 2, 2] = (near + far) / (near - far)
    ret[:, 2, 3] = 2.0 * near * far / (near - far)
    ret[:, 3, 2] = -1.0
    return ret


def perspective_from_fov_xy(
    fov_x: Union[float, torch.Tensor],
    fov_y: Union[float, torch.Tensor],
    near: Union[float, torch.Tensor],
    far: Union[float, torch.Tensor],
) -> torch.Tensor:
    aspect = torch.tan(fov_x / 2) / torch.tan(fov_y / 2)
    return perspective(fov_y, aspect, near, far)


@batched(0, 0, 0, 0)
def intrinsics_from_focal_center(
    fx: Union[float, torch.Tensor],
    fy: Union[float, torch.Tensor],
    cx: Union[float, torch.Tensor],
    cy: Union[float, torch.Tensor],
) -> torch.Tensor:
    n = fx.shape[0]
    ret = torch.zeros((n, 3, 3), dtype=fx.dtype, device=fx.device)
    zeros = torch.zeros(n, dtype=fx.dtype, device=fx.device)
    ones = torch.ones(n, dtype=fx.dtype, device=fx.device)
    ret = torch.stack([fx, zeros, cx, zeros, fy, cy, zeros, zeros, ones], dim=-1).unflatten(-1, (3, 3))
    return ret


def intrinsics_from_fov_xy(fov_x: Union[float, torch.Tensor], fov_y: Union[float, torch.Tensor]) -> torch.Tensor:
    focal_x = 0.5 / torch.tan(fov_x / 2)
    focal_y = 0.5 / torch.tan(fov_y / 2)
    return intrinsics_from_focal_center(focal_x, focal_y, 0.5, 0.5)


@batched(1, 1, 1)
def view_look_at(eye: torch.Tensor, look_at: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    n = eye.shape[0]
    z = eye - look_at
    x = torch.cross(up, z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    z = z / z.norm(dim=-1, keepdim=True)
    r = torch.stack([x, y, z], dim=-2)
    t = -torch.matmul(r, eye[..., None])
    ret = torch.zeros((n, 4, 4), dtype=eye.dtype, device=eye.device)
    ret[:, :3, :3] = r
    ret[:, :3, 3] = t[:, :, 0]
    ret[:, 3, 3] = 1.0
    return ret


@batched(1, 1, 1)
def extrinsics_look_at(eye: torch.Tensor, look_at: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    n = eye.shape[0]
    z = look_at - eye
    x = torch.cross(-up, z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    z = z / z.norm(dim=-1, keepdim=True)
    r = torch.stack([x, y, z], dim=-2)
    t = -torch.matmul(r, eye[..., None])
    ret = torch.zeros((n, 4, 4), dtype=eye.dtype, device=eye.device)
    ret[:, :3, :3] = r
    ret[:, :3, 3] = t[:, :, 0]
    ret[:, 3, 3] = 1.0
    return ret


@batched(2, 0, 0)
def intrinsics_to_perspective(
    intrinsics: torch.Tensor,
    near: Union[float, torch.Tensor],
    far: Union[float, torch.Tensor],
) -> torch.Tensor:
    n = intrinsics.shape[0]
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 1, 1]
    cx, cy = intrinsics[:, 0, 2], intrinsics[:, 1, 2]
    ret = torch.zeros((n, 4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[:, 0, 0] = 2 * fx
    ret[:, 1, 1] = 2 * fy
    ret[:, 0, 2] = -2 * cx + 1
    ret[:, 1, 2] = 2 * cy - 1
    ret[:, 2, 2] = (near + far) / (near - far)
    ret[:, 2, 3] = 2.0 * near * far / (near - far)
    ret[:, 3, 2] = -1.0
    return ret


@batched(2)
def extrinsics_to_view(extrinsics: torch.Tensor) -> torch.Tensor:
    return extrinsics * torch.tensor([1, -1, -1, 1], dtype=extrinsics.dtype, device=extrinsics.device)[:, None]
