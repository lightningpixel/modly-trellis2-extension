from typing import Literal, Union

import nvdiffrast.torch as dr
import torch

__all__ = ["RastContext", "rasterize_triangle_faces"]


class RastContext:
    def __init__(self, nvd_ctx: Union[dr.RasterizeCudaContext, dr.RasterizeGLContext] = None, *, backend: Literal["cuda", "gl"] = "gl", device: Union[str, torch.device] = None):
        if nvd_ctx is not None:
            self.nvd_ctx = nvd_ctx
            return
        if backend == "gl":
            self.nvd_ctx = dr.RasterizeGLContext(device=device)
        elif backend == "cuda":
            self.nvd_ctx = dr.RasterizeCudaContext(device=device)
        else:
            raise ValueError(f"Unknown backend: {backend}")


def rasterize_triangle_faces(
    ctx: RastContext,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    width: int,
    height: int,
    attr: torch.Tensor = None,
    uv: torch.Tensor = None,
    texture: torch.Tensor = None,
    model: torch.Tensor = None,
    view: torch.Tensor = None,
    projection: torch.Tensor = None,
    antialiasing: Union[bool, list[int]] = True,
    diff_attrs: Union[None, list[int], str] = None,
):
    assert vertices.ndim == 3
    assert faces.ndim == 2
    if vertices.shape[-1] == 2:
        vertices = torch.cat([vertices, torch.zeros_like(vertices[..., :1]), torch.ones_like(vertices[..., :1])], dim=-1)
    elif vertices.shape[-1] == 3:
        vertices = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
    elif vertices.shape[-1] != 4:
        raise ValueError(f"Wrong shape of vertices: {vertices.shape}")
    mvp = projection if projection is not None else torch.eye(4).to(vertices)
    if view is not None:
        mvp = mvp @ view
    if model is not None:
        mvp = mvp @ model
    pos_clip = vertices @ mvp.transpose(-1, -2)
    faces = faces.contiguous()
    if attr is not None:
        attr = attr.contiguous()
    rast_out, rast_db = dr.rasterize(ctx.nvd_ctx, pos_clip, faces, resolution=[height, width], grad_db=True)
    face_id = rast_out[..., 3].flip(1)
    depth = rast_out[..., 2].flip(1)
    mask = (face_id > 0).float()
    depth = (depth * 0.5 + 0.5) * mask + (1.0 - mask)
    ret = {"depth": depth, "mask": mask, "face_id": face_id}
    if attr is not None:
        image, image_dr = dr.interpolate(attr, rast_out, faces, rast_db, diff_attrs=diff_attrs)
        if antialiasing is True:
            image = dr.antialias(image, rast_out, pos_clip, faces)
        elif isinstance(antialiasing, list):
            aa_image = dr.antialias(image[..., antialiasing], rast_out, pos_clip, faces)
            image[..., antialiasing] = aa_image
        ret["image"] = image.flip(1).permute(0, 3, 1, 2)
        if diff_attrs is not None:
            ret["image_dr"] = image_dr.flip(1).permute(0, 3, 1, 2)
    if uv is not None:
        uv_map, uv_map_dr = dr.interpolate(uv, rast_out, faces, rast_db, diff_attrs="all")
        ret["uv"] = uv_map
        ret["uv_dr"] = uv_map_dr
        if texture is not None:
            ret["texture"] = dr.texture(ctx.nvd_ctx, uv_map, uv_map_dr).flip(1).permute(0, 3, 1, 2)
    return ret
