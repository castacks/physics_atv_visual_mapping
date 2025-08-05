import torch

from ros_torch_converter.datatypes.pointcloud import FeaturePointCloudTorch

from physics_atv_visual_mapping.feature_key_list import FeatureKeyList
from physics_atv_visual_mapping.localmapping.metadata import LocalMapperMetadata
from physics_atv_visual_mapping.localmapping.voxel.voxel_localmapper import VoxelGrid

"""
check sub-voxel stats to make sure that everything works as expected
"""

if __name__ == '__main__':
    device = 'cuda'

    metadata = LocalMapperMetadata(
        origin=[-10., -10., -10.],
        length=[20., 20., 20.],
        resolution=[0.1, 0.1, 0.1],
        device=device
    )

    # make pts that fall in voxel centers
    xs = metadata.origin[0] + torch.arange(metadata.N[0], device=device) * metadata.resolution[0] + metadata.resolution[0] / 2
    ys = metadata.origin[1] + torch.arange(metadata.N[1], device=device) * metadata.resolution[1] + metadata.resolution[1] / 2

    # make pts that fall in voxel centers
    xs = metadata.origin[0] + torch.arange(2*metadata.N[0], device=device) * 0.5 * metadata.resolution[0] + metadata.resolution[0] / 4
    ys = metadata.origin[1] + torch.arange(2*metadata.N[1], device=device) * 0.5 * metadata.resolution[1] + metadata.resolution[1] / 4

    xs, ys = torch.meshgrid(xs, ys, indexing='ij')

    #random sine terrain
    zs = torch.sin(xs * 0.1) + torch.cos(ys * 0.1) - 2.

    pts = torch.stack([xs.flatten(), ys.flatten(), zs.flatten()], dim=-1).to(device)

    rand_idxs = torch.randperm(pts.shape[0])[:1000]
    feats = torch.zeros((rand_idxs.shape[0], 4), device=device)
    feat_mask = torch.zeros(pts.shape[0], dtype=torch.bool, device=device)
    feat_mask[rand_idxs] = True
    fks = FeatureKeyList(
        label=[f"feat_{i}" for i in range(feats.shape[-1])],
        metainfo=["aaa"] * feats.shape[-1]
    )

    feat_pc = FeaturePointCloudTorch.from_torch(
        pts=pts,
        features=feats,
        mask=feat_mask,
        feature_keys=fks,
    )

    voxel_grid = VoxelGrid.from_feature_pc(feat_pc, metadata=metadata)