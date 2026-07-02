import pytest

import copy
import torch
torch.manual_seed(37)
import numpy as np

from ros_torch_converter.datatypes.pointcloud import FeaturePointCloudTorch

from physics_atv_visual_mapping.feature_key_list import FeatureKeyList
from physics_atv_visual_mapping.localmapping.metadata import LocalMapperMetadata
from physics_atv_visual_mapping.localmapping.voxel.voxel_localmapper import VoxelGrid, VoxelLocalMapper

"""
Things to check for:
    1. Voxel iff. point
    2. Features in the right spots

Also make sure to check after add, shift merge
"""

def make_rand_pc(n_pts, n_feats, feat_dim, metadata):
    #avoid any float roundoff error
    rand = torch.rand(n_pts, 3) * 0.98 + 0.01
    pts = rand * metadata.length + metadata.origin

    #give every pt the same feature to make things easier
    feats = torch.rand(1, 4).tile(n_feats, 1)
    mask = torch.zeros(n_pts, dtype=torch.bool)
    idxs = torch.randperm(n_pts)[:n_feats]
    mask[idxs] = True

    fks = FeatureKeyList(
        label=[f"f_{i}" for i in range(feat_dim)],
        metainfo=["feat"] * feat_dim
    )

    return FeaturePointCloudTorch.from_torch(
        pts=pts,
        features=feats,
        mask=mask,
        feature_keys=fks
    )

@pytest.fixture(scope="module")
def voxel_insert_data():
    origin = torch.rand(3) * -10.
    res = torch.rand(3) * 0.05 + 0.05
    n = torch.randint(400, size=(3,))
    length = res * n
    #round origin to nearest multiple of resolution
    origin = (origin//res)*res

    metadata = LocalMapperMetadata(
        origin = origin,
        length = length,
        resolution = res
    )

    #random pts fine for this test
    n_pts = 10000
    n_feats = 5000
    feat_dim = 4

    feat_pc = make_rand_pc(n_pts, n_feats, feat_dim, metadata)

    voxel_localmapper = VoxelLocalMapper(
        metadata = metadata,
        feature_keys = feat_pc.feature_keys,
        ema = 1.0
    )

    pos = torch.zeros(7)
    pos[-1] = 1.
    voxel_localmapper.update_pose(pos)
    voxel_localmapper.add_feature_pc(pos, feat_pc)

    return {
        'pos': pos,
        'metadata': metadata,
        'feat_pc': feat_pc,
        'localmapper': voxel_localmapper
    }

@pytest.fixture(scope="module")
def voxel_merge_data():
    origin = torch.rand(3) * -10.
    res = torch.rand(3) * 0.05 + 0.05
    n = torch.randint(400, size=(3,))
    length = res * n
    #round origin to nearest multiple of resolution
    origin = (origin//res)*res

    metadata = LocalMapperMetadata(
        origin = origin,
        length = length,
        resolution = res
    )

    #random pts fine for this test
    n_pts = 10000
    n_feats = 5000
    feat_dim = 4

    feat_pc1 = make_rand_pc(n_pts, n_feats, feat_dim, metadata)
    feat_pc2 = make_rand_pc(n_pts, n_feats, feat_dim, metadata)

    voxel_localmapper = VoxelLocalMapper(
        metadata = metadata,
        feature_keys = feat_pc1.feature_keys,
        ema = 1.0
    )

    pos = torch.zeros(7)
    pos[-1] = 1.
    voxel_localmapper.update_pose(pos)
    voxel_localmapper.add_feature_pc(pos, feat_pc1)
    voxel_localmapper.add_feature_pc(pos, feat_pc2)

    return {
        'pos': pos,
        'metadata': metadata,
        'feat_pc1': feat_pc1,
        'feat_pc2': feat_pc2,
        'localmapper': voxel_localmapper
    }

@pytest.fixture(scope="module")
def voxel_shift_data():
    origin = torch.rand(3) * -10.
    res = torch.rand(3) * 0.05 + 0.05
    n = torch.randint(400, size=(3,))
    length = res * n
    #round origin to nearest multiple of resolution
    origin = (origin//res)*res
    shift = torch.randint(20, size=(3, )) - 10

    metadata = LocalMapperMetadata(
        origin = origin,
        length = length,
        resolution = res
    )

    #random pts fine for this test
    n_pts = 10000
    n_feats = 5000
    feat_dim = 4

    feat_pc = make_rand_pc(n_pts, n_feats, feat_dim, metadata)

    voxel_localmapper = VoxelLocalMapper(
        metadata = metadata,
        feature_keys = feat_pc.feature_keys,
        ema = 1.0
    )

    pos = torch.zeros(7)
    pos[-1] = 1.
    pos[:3] = shift * res
    voxel_localmapper.update_pose(pos)
    voxel_localmapper.add_feature_pc(pos, feat_pc)

    return {
        'pos': pos,
        'metadata': metadata,
        'feat_pc': feat_pc,
        'localmapper': voxel_localmapper
    }

@pytest.fixture(scope="module")
def voxel_timestamp_data():
    origin = torch.rand(3) * -10.
    res = torch.rand(3) * 0.05 + 0.05
    n = torch.randint(400, size=(3,))
    length = res * n
    #round origin to nearest multiple of resolution
    origin = (origin//res)*res
    shift = torch.randint(20, size=(3, )) - 10

    metadata = LocalMapperMetadata(
        origin = origin,
        length = length,
        resolution = res
    )

    feat_pcs = []
    poses = []

    #random pts fine for this test
    n_pts = 10000
    n_feats = 5000
    feat_dim = 4

    ## use a real unix timestamp to stress double presicion
    curr_time = 1775568034.9604232

    feat_pc = make_rand_pc(n_pts, n_feats, feat_dim, metadata)
    voxel_localmapper = VoxelLocalMapper(
        metadata = metadata,
        feature_keys = feat_pc.feature_keys,
        ema = 1.0
    )

    pos = torch.zeros(7)
    pos[-1] = 1.
    pos[:3] = shift * res
    voxel_localmapper.update_pose(pos)


    for i in range(10):
        feat_pc = make_rand_pc(n_pts, n_feats, feat_dim, metadata)
        feat_pc.stamp = curr_time + (i/10.)
        voxel_localmapper.add_feature_pc(pos, feat_pc)

        poses.append(pos)
        feat_pcs.append(feat_pc)

    return {
        'poses': poses,
        'metadata': metadata,
        'feat_pcs': feat_pcs,
        'localmapper': voxel_localmapper
    }

def test_metadata(voxel_insert_data):
    base_metadata = voxel_insert_data['metadata']
    voxel_metadata = voxel_insert_data['localmapper'].voxel_grid.metadata

    assert torch.allclose(base_metadata.origin, voxel_metadata.origin)
    assert torch.allclose(base_metadata.length, voxel_metadata.length)
    assert torch.allclose(base_metadata.resolution, voxel_metadata.resolution)

def test_point_insert(voxel_insert_data):
    """
    Test that point iff. voxel
    """
    pc = voxel_insert_data['feat_pc']
    voxel_grid = voxel_insert_data['localmapper'].voxel_grid

    N = voxel_grid.raster_indices.shape[0]
    assert voxel_grid.feature_mask.shape[0] == N
    assert voxel_grid.hits.shape[0] == N
    assert voxel_grid.misses.shape[0] == N
    assert voxel_grid.min_coords.shape[0] == N
    assert voxel_grid.max_coords.shape[0] == N

    ## theres a lot of ops that can only be efficient if raster idxs are sorted.
    sorted_idxs, _ = torch.sort(voxel_grid.raster_indices.clone())
    assert (sorted_idxs == voxel_grid.raster_indices).all()

    pc_grid_idxs, mask = voxel_grid.get_grid_idxs(pc.pts)

    assert mask.all(), "Expected all points to be in bounds of voxel grid"

    pc_raster_idxs = voxel_grid.grid_indices_to_raster_indices(pc_grid_idxs)
    pc_raster_idxs = pc_raster_idxs.unique(sorted=True)

    voxel_raster_idxs, _ = torch.sort(voxel_grid.raster_indices.clone())

    assert len(voxel_raster_idxs) == len(pc_raster_idxs), "incorrect number of voxels!"
    assert torch.all(pc_raster_idxs == voxel_raster_idxs), "voxel grid putting voxels in the wrong places!"

    ## also test features correct
    feat_pts = pc.pts[pc.feat_mask]
    pc_grid_idxs, mask = voxel_grid.get_grid_idxs(feat_pts)
    pc_raster_idxs = voxel_grid.grid_indices_to_raster_indices(pc_grid_idxs)
    pc_raster_idxs = pc_raster_idxs.unique(sorted=True)

    voxel_raster_idxs, _ = torch.sort(voxel_grid.feature_raster_indices.clone())
    assert len(voxel_raster_idxs) == len(pc_raster_idxs), "incorrect number of feature voxels!"
    assert torch.all(pc_raster_idxs == voxel_raster_idxs), "voxel grid putting feature voxels in the wrong places!"

    ## note that a nonfeature test doesnt work as the voxels dont match if a feat/nonfeat pt are in the same voxel

def test_coordinate_math(voxel_insert_data):
    """
    Test that raster->grid->pt->grid->raster is lossless
    """
    voxel_grid = voxel_insert_data['localmapper'].voxel_grid

    raster1 = voxel_grid.raster_indices.clone()
    grid1 = voxel_grid.raster_indices_to_grid_indices(raster1)
    pts1 = voxel_grid.grid_indices_to_pts(grid1, centers=True)
    grid2, _ = voxel_grid.get_grid_idxs(pts1)
    raster2 = voxel_grid.grid_indices_to_raster_indices(grid2)
    grid3 = voxel_grid.raster_indices_to_grid_indices(raster2)
    pts2 = voxel_grid.grid_indices_to_pts(grid3, centers=True)

    assert (raster1 == raster2).all()
    assert (grid1 == grid2).all()
    assert torch.allclose(pts1, pts2)

def test_voxel_merge(voxel_merge_data):
    all_pts = torch.cat([
        voxel_merge_data['feat_pc1'].pts,
        voxel_merge_data['feat_pc2'].pts
    ], dim=0)

    voxel_grid = voxel_merge_data['localmapper'].voxel_grid

    N = voxel_grid.raster_indices.shape[0]
    assert voxel_grid.feature_mask.shape[0] == N
    assert voxel_grid.hits.shape[0] == N
    assert voxel_grid.misses.shape[0] == N
    assert voxel_grid.min_coords.shape[0] == N
    assert voxel_grid.max_coords.shape[0] == N

    ## double check that stuff stays sorted
    sorted_idxs, _ = torch.sort(voxel_grid.raster_indices.clone())
    assert (sorted_idxs == voxel_grid.raster_indices).all()

    pc_grid_idxs, mask = voxel_grid.get_grid_idxs(all_pts)

    assert mask.all(), "Expected all points to be in bounds of voxel grid"

    pc_raster_idxs = voxel_grid.grid_indices_to_raster_indices(pc_grid_idxs)
    pc_raster_idxs = pc_raster_idxs.unique(sorted=True)

    voxel_raster_idxs, _ = torch.sort(voxel_grid.raster_indices.clone())

    assert len(voxel_raster_idxs) == len(pc_raster_idxs), "incorrect number of voxels!"
    assert torch.all(pc_raster_idxs == voxel_raster_idxs), "voxel grid putting voxels in the wrong places!"

    ## also test features correct
    all_feat_pts = torch.cat([
        voxel_merge_data['feat_pc1'].feature_pts,
        voxel_merge_data['feat_pc2'].feature_pts
    ], dim=0)
    pc_grid_idxs, mask = voxel_grid.get_grid_idxs(all_feat_pts)
    pc_raster_idxs = voxel_grid.grid_indices_to_raster_indices(pc_grid_idxs)
    pc_raster_idxs = pc_raster_idxs.unique(sorted=True)

    voxel_raster_idxs, _ = torch.sort(voxel_grid.feature_raster_indices.clone())
    assert len(voxel_raster_idxs) == len(pc_raster_idxs), "incorrect number of feature voxels!"
    assert torch.all(pc_raster_idxs == voxel_raster_idxs), "voxel grid putting feature voxels in the wrong places!"

def test_voxel_shift(voxel_shift_data):
    pc = voxel_shift_data['feat_pc']
    pos = voxel_shift_data['pos']
    voxel_grid = voxel_shift_data['localmapper'].voxel_grid
    base_metadata = voxel_shift_data['metadata']
    new_metadata = voxel_grid.metadata

    N = voxel_grid.raster_indices.shape[0]
    assert voxel_grid.feature_mask.shape[0] == N
    assert voxel_grid.hits.shape[0] == N
    assert voxel_grid.misses.shape[0] == N
    assert voxel_grid.min_coords.shape[0] == N
    assert voxel_grid.max_coords.shape[0] == N

    assert torch.allclose(pos[:3] + base_metadata.origin, new_metadata.origin)
    assert torch.allclose(base_metadata.resolution, new_metadata.resolution)
    assert torch.allclose(base_metadata.length, new_metadata.length)

    ## double check that stuff stays sorted
    sorted_idxs, _ = torch.sort(voxel_grid.raster_indices.clone())
    assert (sorted_idxs == voxel_grid.raster_indices).all()

    #check that the raster idxs of the post-filtered pc == the stored raster idxs
    all_pts = pc.pts
    mask = voxel_grid.pts_in_bounds(all_pts)
    in_bounds_pts = all_pts[mask]

    pc_grid_idxs, mask = voxel_grid.get_grid_idxs(in_bounds_pts)
    pc_raster_idxs = voxel_grid.grid_indices_to_raster_indices(pc_grid_idxs)
    pc_raster_idxs = pc_raster_idxs.unique(sorted=True)
    
    assert mask.all(), "Expected all points to be in bounds of voxel grid"

    pc_raster_idxs = voxel_grid.grid_indices_to_raster_indices(pc_grid_idxs)
    pc_raster_idxs = pc_raster_idxs.unique(sorted=True)

    voxel_raster_idxs, _ = torch.sort(voxel_grid.raster_indices.clone())

    assert len(voxel_raster_idxs) == len(pc_raster_idxs), "incorrect number of voxels!"
    assert torch.all(pc_raster_idxs == voxel_raster_idxs), "voxel grid putting voxels in the wrong places!"

def test_voxel_timestamp(voxel_timestamp_data):
    """
    Test timestamp math on a few pcs. Things that we can check:
        1. every voxel timestamp comes from a pc timestamp
        2. first pc timestamp is exactly first_update time
        3. last pc timestamp is exactly last update time
        4. all pts within the corresponding voxel bounds
    """
    pcs = voxel_timestamp_data['feat_pcs']
    voxel_grid = voxel_timestamp_data['localmapper'].voxel_grid

    pc_times = torch.tensor([x.stamp for x in pcs], dtype=torch.double)
    voxel_first_update_times = voxel_grid.first_update_time
    voxel_last_update_times = voxel_grid.last_update_time

    ## check that every voxel update time matches a pointcloud ##
    assert all([x in pc_times for x in voxel_first_update_times])
    assert all([x in pc_times for x in voxel_last_update_times])

    ## check that the first pc passes its stamp to first_update time
    first_pc = pcs[0]
    gidxs, valid_mask = voxel_grid.get_grid_idxs(first_pc.pts)
    ridxs = voxel_grid.grid_indices_to_raster_indices(gidxs)
    for ridx in ridxs[valid_mask]:
        vox_idx = torch.argwhere(voxel_grid.raster_indices == ridx)
        assert vox_idx.numel() == 1
        _ftime = voxel_grid.first_update_time[vox_idx.item()].item()
        assert abs(_ftime - first_pc.stamp) < 1e-10

    ## check that the last pc passes its stamp to last_update time
    last_pc = pcs[-1]
    gidxs, valid_mask = voxel_grid.get_grid_idxs(last_pc.pts)
    ridxs = voxel_grid.grid_indices_to_raster_indices(gidxs)
    for ridx in ridxs[valid_mask]:
        vox_idx = torch.argwhere(voxel_grid.raster_indices == ridx)
        assert vox_idx.numel() == 1
        _ltime = voxel_grid.last_update_time[vox_idx.item()].item()
        assert abs(_ltime - last_pc.stamp) < 1e-10

    ## for all pcs, check stamp within bounds
    for pc in pcs:
        gidxs, valid_mask = voxel_grid.get_grid_idxs(pc.pts)
        ridxs = voxel_grid.grid_indices_to_raster_indices(gidxs)
        for ridx in ridxs[valid_mask]:
            vox_idx = torch.argwhere(voxel_grid.raster_indices == ridx)
            assert vox_idx.numel() == 1
            _ftime = voxel_grid.first_update_time[vox_idx.item()].item()
            _ltime = voxel_grid.last_update_time[vox_idx.item()].item()

            assert pc.stamp >= (_ftime - 1e-10)
            assert pc.stamp <= (_ltime + 1e-10)