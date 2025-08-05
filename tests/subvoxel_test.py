import pytest

import torch
torch.manual_seed(37)

from ros_torch_converter.datatypes.pointcloud import FeaturePointCloudTorch

from physics_atv_visual_mapping.feature_key_list import FeatureKeyList
from physics_atv_visual_mapping.localmapping.metadata import LocalMapperMetadata
from physics_atv_visual_mapping.localmapping.bev.bev_localmapper import BEVGrid
from physics_atv_visual_mapping.localmapping.voxel.voxel_localmapper import VoxelGrid, VoxelLocalMapper
from physics_atv_visual_mapping.terrain_estimation.processing_blocks.elevation_stats import ElevationStats

"""
check sub-voxel stats to make sure that everything works as expected
"""
@pytest.fixture(scope="module")
def insert_data():
    origin = torch.tensor([-5., -5., -5.])
    length = torch.tensor([10., 10., 10.])
    res = torch.tensor([0.05, 0.05, 0.05])

    metadata = LocalMapperMetadata(
        origin = origin,
        length = length,
        resolution = res
    )

    xs = metadata.origin[0] + torch.arange(metadata.N[0]) * metadata.resolution[0] + metadata.resolution[0]/2.
    ys = metadata.origin[1] + torch.arange(metadata.N[1]) * metadata.resolution[1] + metadata.resolution[1]/2.

    xs, ys = torch.meshgrid(xs, ys, indexing='ij')
    zs = (xs * 0.78).cos() + (ys * 1.29).sin()

    pts = torch.stack([xs, ys, zs], dim=-1).reshape(-1, 3)
    fks = FeatureKeyList(label=["aaa"], metainfo=["aaa"])

    voxel_localmapper = VoxelLocalMapper(
        metadata = metadata,
        feature_keys = fks,
        ema = 1.0
    )

    pos = torch.zeros(7)
    pos[-1] = 1.
    voxel_localmapper.update_pose(pos)

    pcs = []

    grid_idxs = []

    #add jitter to verify min/max/etc
    for i in range(10):
        sig = (torch.rand(*pts.shape)-0.5) * metadata.resolution * 0.9
        sig[:, 2] *= 0 #cant noise z bc not axis-aligned
        pts_new = pts.clone() + sig

        pc = FeaturePointCloudTorch.from_torch(
            pts=pts_new,
            features=torch.zeros(0, 1),
            mask=torch.zeros(pts.shape[0], dtype=torch.bool),
            feature_keys=fks
        )

        voxel_localmapper.add_feature_pc(pos, pc)
        pcs.append(pc)

    return {
        'pos': pos,
        'metadata': metadata,
        'pcs': pcs,
        'localmapper': voxel_localmapper
    }


@pytest.fixture(scope="module")
def elevation_data():
    origin = torch.tensor([-5., -5., -5.])
    length = torch.tensor([10., 10., 10.])
    res = torch.tensor([0.05, 0.05, 0.05])

    metadata = LocalMapperMetadata(
        origin = origin,
        length = length,
        resolution = res
    )

    xs = metadata.origin[0] + torch.arange(metadata.N[0]) * metadata.resolution[0] + metadata.resolution[0]/2.
    ys = metadata.origin[1] + torch.arange(metadata.N[1]) * metadata.resolution[1] + metadata.resolution[1]/2.

    xs, ys = torch.meshgrid(xs, ys, indexing='ij')

    z1s = (xs * 0.78).cos() + (ys * 1.29).sin()
    z2s = -z1s

    pts1 = torch.stack([xs, ys, z1s], dim=-1).reshape(-1, 3)
    pts2 = torch.stack([xs, ys, z2s], dim=-1).reshape(-1, 3)
    fks = FeatureKeyList(label=["aaa"], metainfo=["aaa"])

    pc1 = FeaturePointCloudTorch.from_torch(
        pts=pts1,
        features=torch.zeros(0, 1),
        mask=torch.zeros(pts1.shape[0], dtype=torch.bool),
        feature_keys=fks
    )

    pc2 = FeaturePointCloudTorch.from_torch(
        pts=pts2,
        features=torch.zeros(0, 1),
        mask=torch.zeros(pts2.shape[0], dtype=torch.bool),
        feature_keys=fks
    )

    voxel_localmapper = VoxelLocalMapper(
        metadata = metadata,
        feature_keys = fks,
        ema = 1.0
    )

    pos = torch.zeros(7)
    pos[-1] = 1.
    voxel_localmapper.update_pose(pos)
    voxel_localmapper.add_feature_pc(pos, pc1)
    voxel_localmapper.add_feature_pc(pos, pc2)

    #get elevation maps
    elevation_stats_block = ElevationStats(
        voxel_localmapper.metadata,
        voxel_localmapper.feature_keys,
        use_voxel_centers=True
    )

    bev_map1 = BEVGrid(
        metadata=voxel_localmapper.metadata,
        feature_keys=elevation_stats_block.output_feature_keys
    )

    bev_map2 = BEVGrid(
        metadata=voxel_localmapper.metadata,
        feature_keys=elevation_stats_block.output_feature_keys
    )

    elevation_stats_block.run(voxel_localmapper.voxel_grid, bev_map1)
    elevation_stats_block.use_voxel_centers=False
    elevation_stats_block.run(voxel_localmapper.voxel_grid, bev_map2)

    return {
        'pos': pos,
        'metadata': metadata,
        'pc1': pc1,
        'pc2': pc2,
        'localmapper': voxel_localmapper,
        'bev_map_voxel': bev_map1,
        'bev_map_pts': bev_map2
    }

    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(2, 2)
    # axs = axs.flatten()
    # axs[0].imshow(z1s)
    # axs[1].imshow(z2s)
    # axs[2].imshow(torch.maximum(z1s, z2s))
    # axs[3].imshow(torch.minimum(z1s, z2s))

    # axs[0].set_title('z1')
    # axs[1].set_title('z2')
    # axs[2].set_title('max')
    # axs[3].set_title('min')

    # plt.show()

def test_subvoxel(insert_data):
    pcs = torch.stack([pc.pts for pc in insert_data["pcs"]], dim=0)
    voxel_grid = insert_data['localmapper'].voxel_grid

    assert pcs.shape[1] == len(voxel_grid.raster_indices)

    voxel_lower_corner = voxel_grid.grid_indices_to_pts(voxel_grid.raster_indices_to_grid_indices(voxel_grid.raster_indices), centers=False)
    voxel_upper_corner = voxel_lower_corner + voxel_grid.metadata.resolution

    #check coord values inside of voxel bounds
    assert (voxel_grid.min_coords >= voxel_lower_corner).all() and (voxel_grid.min_coords <= voxel_upper_corner).all()
    assert (voxel_grid.max_coords >= voxel_lower_corner).all() and (voxel_grid.max_coords <= voxel_upper_corner).all()
    assert (voxel_grid.midpoints >= voxel_lower_corner).all() and (voxel_grid.midpoints <= voxel_upper_corner).all()

    #check monotonicity of min->mid->max
    assert (voxel_grid.midpoints >= voxel_grid.min_coords).all()
    assert (voxel_grid.midpoints <= voxel_grid.max_coords).all()

    #check value correctness
    assert torch.allclose(pcs.min(dim=0)[0], voxel_grid.min_coords)
    assert torch.allclose(pcs.max(dim=0)[0], voxel_grid.max_coords)

def test_elevation(elevation_data):
    pc1 = elevation_data['pc1']
    pc2 = elevation_data['pc2']

    bev_map_voxel = elevation_data['bev_map_voxel']
    bev_map_pts = elevation_data['bev_map_pts']

    bev_metadata = bev_map_voxel.metadata
    voxel_metadata = elevation_data['localmapper'].metadata

    assert bev_map_voxel.feature_keys == bev_map_pts.feature_keys

    min_elev_idx = bev_map_voxel.feature_keys.index('min_elevation')
    max_elev_idx = bev_map_voxel.feature_keys.index('max_elevation')
    n_voxel_idx = bev_map_voxel.feature_keys.index('num_voxels')

    assert (bev_map_voxel.data[..., n_voxel_idx] == 2).all()
    assert (bev_map_pts.data[..., n_voxel_idx] == 2).all()

    pt_min_elev = torch.minimum(pc1.pts[:, 2], pc2.pts[:, 2]).view(*bev_map_voxel.metadata.N)
    pt_max_elev = torch.maximum(pc1.pts[:, 2], pc2.pts[:, 2]).view(*bev_map_voxel.metadata.N)

    bev_voxel_min_elev = bev_map_voxel.data[..., min_elev_idx]
    bev_voxel_max_elev = bev_map_voxel.data[..., max_elev_idx]

    bev_pts_min_elev = bev_map_pts.data[..., min_elev_idx]
    bev_pts_max_elev = bev_map_pts.data[..., max_elev_idx]

    bev_voxel_min_elev_err = (bev_voxel_min_elev - pt_min_elev).abs()
    bev_voxel_max_elev_err = (bev_voxel_max_elev - pt_max_elev).abs()

    bev_pts_min_elev_err = (bev_pts_min_elev - pt_min_elev).abs()
    bev_pts_max_elev_err = (bev_pts_max_elev - pt_max_elev).abs()

    #elev error for voxels should be < z resolution of a voxel
    assert bev_voxel_min_elev_err.max() < voxel_metadata.resolution[2]
    assert bev_voxel_max_elev_err.max() < voxel_metadata.resolution[2]

    #elev error for pts should be essentially zero
    assert bev_pts_min_elev_err.max() < 1e-8
    assert bev_pts_max_elev_err.max() < 1e-8

    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(2, 5)

    # axs[0, 0].imshow(pt_min_elev, cmap='jet')
    # axs[1, 0].imshow(pt_max_elev, cmap='jet')

    # axs[0, 1].imshow(bev_voxel_min_elev, cmap='jet')
    # axs[1, 1].imshow(bev_voxel_max_elev, cmap='jet')

    # axs[0, 2].imshow(bev_pts_min_elev, cmap='jet')
    # axs[1, 2].imshow(bev_pts_max_elev, cmap='jet')

    # axs[0, 3].imshow(bev_voxel_min_elev_err, cmap='jet')
    # axs[1, 3].imshow(bev_voxel_max_elev_err, cmap='jet')

    # axs[0, 4].imshow(bev_pts_min_elev_err, cmap='jet')
    # axs[1, 4].imshow(bev_pts_max_elev_err, cmap='jet')

    # axs[0, 0].set_ylabel('min elev')
    # axs[1, 0].set_ylabel('max elev')

    # axs[0, 0].set_title('GT')
    # axs[0, 1].set_title('Voxel Centers')
    # axs[0, 2].set_title('Pts')
    # axs[0, 3].set_title('Voxel Centers Err')
    # axs[0, 4].set_title('Pts err')

    # plt.show()

