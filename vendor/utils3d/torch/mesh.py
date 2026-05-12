from typing import List, Optional, Tuple, Union

import torch

__all__ = [
    "triangulate",
    "compute_edges",
    "compute_connected_components",
    "compute_edge_connected_components",
    "compute_dual_graph",
    "remove_unreferenced_vertices",
]


def _group(
    values: torch.Tensor,
    required_group_size: Optional[int] = None,
    return_values: bool = False,
) -> Tuple[Union[List[torch.Tensor], torch.Tensor], Optional[torch.Tensor]]:
    sorted_values, indices = torch.sort(values)
    nondupe = torch.cat([torch.tensor([True], dtype=torch.bool, device=values.device), sorted_values[1:] != sorted_values[:-1]])
    nondupe_indices = torch.cumsum(nondupe, dim=0) - 1
    counts = torch.bincount(nondupe_indices)
    if required_group_size is None:
        groups = torch.split(indices, counts.tolist())
        if return_values:
            return groups, sorted_values[nondupe]
        return groups
    counts = counts[nondupe_indices]
    groups = indices[counts == required_group_size].reshape(-1, required_group_size)
    if return_values:
        return groups, sorted_values[nondupe][counts[nondupe] == required_group_size]
    return groups


def triangulate(faces: torch.Tensor, vertices: torch.Tensor = None, backslash: bool = None) -> torch.Tensor:
    if faces.shape[-1] == 3:
        return faces
    p = faces.shape[-1]
    if vertices is not None:
        assert faces.shape[-1] == 4, "now only support quad mesh"
        if backslash is None:
            faces_idx = faces.long()
            backslash = torch.norm(vertices[faces_idx[..., 0]] - vertices[faces_idx[..., 2]], p=2, dim=-1) < torch.norm(vertices[faces_idx[..., 1]] - vertices[faces_idx[..., 3]], p=2, dim=-1)
    if backslash is None:
        loop_indice = torch.stack([torch.zeros(p - 2, dtype=int), torch.arange(1, p - 1, dtype=int), torch.arange(2, p, dtype=int)], axis=1)
        return faces[:, loop_indice].reshape(-1, 3)
    assert faces.shape[-1] == 4, "now only support quad mesh"
    if isinstance(backslash, bool):
        if backslash:
            return faces[:, [0, 1, 2, 0, 2, 3]].reshape(-1, 3)
        return faces[:, [0, 1, 3, 3, 1, 2]].reshape(-1, 3)
    return torch.where(backslash[:, None], faces[:, [0, 1, 2, 0, 2, 3]], faces[:, [0, 1, 3, 3, 1, 2]]).reshape(-1, 3)


def compute_edges(faces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = faces.shape[0]
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
    edges = torch.sort(edges, dim=1).values
    edges, inv_map, counts = torch.unique(edges, return_inverse=True, return_counts=True, dim=0)
    face2edge = inv_map.view(3, t).T
    return edges, face2edge, counts


def compute_connected_components(faces: torch.Tensor, edges: torch.Tensor = None, face2edge: torch.Tensor = None) -> List[torch.Tensor]:
    t = faces.shape[0]
    if edges is None or face2edge is None:
        edges, face2edge, _ = compute_edges(faces)
    e = edges.shape[0]
    labels = torch.arange(t, dtype=torch.int32, device=faces.device)
    while True:
        edge_labels = torch.scatter_reduce(torch.zeros(e, dtype=torch.int32, device=faces.device), 0, face2edge.flatten().long(), labels.view(-1, 1).expand(-1, 3).flatten(), reduce="amin", include_self=False)
        new_labels = torch.min(edge_labels[face2edge], dim=-1).values
        if torch.equal(labels, new_labels):
            break
        labels = new_labels
    return _group(labels)


def compute_edge_connected_components(edges: torch.Tensor) -> List[torch.Tensor]:
    e = edges.shape[0]
    verts, edges = torch.unique(edges.flatten(), return_inverse=True)
    edges = edges.view(-1, 2)
    v = verts.shape[0]
    labels = torch.arange(e, dtype=torch.int32, device=edges.device)
    while True:
        vertex_labels = torch.scatter_reduce(torch.zeros(v, dtype=torch.int32, device=edges.device), 0, edges.flatten().long(), labels.view(-1, 1).expand(-1, 2).flatten(), reduce="amin", include_self=False)
        new_labels = torch.min(vertex_labels[edges], dim=-1).values
        if torch.equal(labels, new_labels):
            break
        labels = new_labels
    return _group(labels)


def compute_dual_graph(face2edge: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    all_edge_indices = face2edge.flatten()
    dual_edges, dual_edge2edge = _group(all_edge_indices, required_group_size=2, return_values=True)
    dual_edges = dual_edges // face2edge.shape[1]
    return dual_edges, dual_edge2edge


def remove_unreferenced_vertices(faces: torch.Tensor, *vertice_attrs, return_indices: bool = False) -> Tuple[torch.Tensor, ...]:
    p = faces.shape[-1]
    fewer_indices, inv_map = torch.unique(faces, return_inverse=True)
    faces = inv_map.to(torch.int32).reshape(-1, p)
    ret = [faces]
    for attr in vertice_attrs:
        ret.append(attr[fewer_indices])
    if return_indices:
        ret.append(fewer_indices)
    return tuple(ret)
