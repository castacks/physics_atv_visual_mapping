import pytest

import torch
torch.manual_seed(37)

from physics_atv_visual_mapping.feature_key_list import FeatureKeyList
from physics_atv_visual_mapping.localmapping.metadata import LocalMapperMetadata
from physics_atv_visual_mapping.localmapping.bev.bev_localmapper import BEVGrid

"""
check bilinear interpolation, etc for BEVGrids
"""
@pytest.fixture(scope="module")
def bev_data():
    metadata = LocalMapperMetadata(
        origin=[-5., -5.],
        length=[10., 10.],
        resolution=[0.05, 0.05]
    )
    
    fks = FeatureKeyList(
        label=['layer1', 'layer2', 'mask1', 'mask2'],
        metainfo = ['data']*2 + ['mask']*2
    )

    """
    fill with a bunch of random planes to debug stuff
    """
    A = torch.rand(2, 3) #[n_planes x {mx, my, b}]

    #lower-left
    coords = metadata.get_coords()
    data = data_fn(coords.view(-1, 2), A).reshape(metadata.N[0], metadata.N[1], 2)
    mask = (data > 0.).float()
    data = torch.cat([data, mask], dim=-1)

    bev_grid = BEVGrid(metadata=metadata, feature_keys=fks)
    bev_grid.data = data

    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(2, 2)
    # axs[0, 0].imshow(data[...,0])
    # axs[0, 1].imshow(data[...,1])
    # axs[1, 0].imshow(data[...,2])
    # axs[1, 1].imshow(data[...,3])
    # fig.suptitle(f'A = {A}')
    # plt.show()

    return {
        'bev_grid': bev_grid,
        'A': A
    }


def data_fn(X, A):
    """
    Args:
        X: [N x 2] coord tensor
        A: [B x 3] coeff tensor
    Returns:
        Y: [N x B] res tensor
    """
    _X = torch.cat([X, torch.ones_like(X[:, [0]])], dim=-1)

    return _X @ A.T

def test_bev_indexing(bev_data):
    bev_grid = bev_data['bev_grid']
    metadata = bev_grid.metadata
    A = bev_data['A']

    coords = bev_grid.metadata.get_coords(centers=True)

    ## first check that all indexing functions return the same thing
    sample_data1, valid_mask1 = bev_grid.grid_sample_feature_keys(coords, feature_keys=['layer1', 'layer2'])
    sample_data2, valid_mask2 = bev_grid.grid_sample_metainfo(coords, metainfo='data')
    sample_data3, valid_mask3 = bev_grid.grid_sample_feature_idxs(coords, feature_idxs=[0, 1])

    assert torch.allclose(sample_data1, sample_data2) and torch.allclose(sample_data1, sample_data3)
    assert torch.allclose(valid_mask1, valid_mask2) and torch.allclose(valid_mask1, valid_mask3)

    ## check that sampling on bev cell coords yelds the BEV map
    all_sample_data, valid_mask = bev_grid.grid_sample_feature_idxs(coords, feature_idxs=[0,1,2,3], bilinear=False)
    
    assert valid_mask.all()
    assert torch.allclose(bev_grid.data, all_sample_data)

    ## test subpixel indexing
    coords_flat = bev_grid.metadata.get_coords(centers=False).reshape(-1, 2)
    offsets = torch.rand_like(coords_flat) * metadata.resolution.view(1,2)
    coords_jitter = coords_flat + offsets

    gt_vals = data_fn(coords_jitter.view(-1, 2), A)
    sample_vals, valid_mask = bev_grid.grid_sample_feature_keys(coords_jitter, feature_keys=['layer1', 'layer2'], bilinear=True)

    assert valid_mask.all()
    ## note that bilinear interpolation is quadratic and will thus have nontrivial error
    assert torch.allclose(sample_vals, gt_vals, atol=0.1)

    ## test masking
    coords2 = coords_jitter + metadata.length.view(1,2)
    sample_vals, valid_mask = bev_grid.grid_sample_feature_keys(coords2, feature_keys=['layer1', 'layer2'], bilinear=True)
    assert not valid_mask.any()