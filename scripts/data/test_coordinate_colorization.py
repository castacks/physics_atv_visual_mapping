import os
import time
import copy
import yaml
import argparse

import torch
import torch_scatter
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# from ros_torch_converter.datatypes.voxel_grid import VoxelGridTorch
from ros_torch_converter.datatypes.image import FeatureImageTorch, ImageTorch
from ros_torch_converter.tf_manager import TfManager
from ros_torch_converter.converter import str_to_cvt_class

from torch_coordinator.setup_torch_coordinator import setup_torch_coordinator

from physics_atv_visual_mapping.pointcloud_colorization.torch_color_pcl_utils import bilinear_interpolate_batch
from physics_atv_visual_mapping.utils import normalize_dino

def get_coord_data_from_features(feats):
    return feats.reshape(feats.shape[0], 5, -1).permute(0,2,1)

def get_unique_seq_cam(coord_data):
    """
    Find all the images to load from coord_data
    """
    valid_mask = coord_data[:, :, -1] > 1e-8
    seq_cam_ids = coord_data[valid_mask][:, :2].round().long()
    ws = coord_data[valid_mask][:, -1]
    seq_cam_ids, inv_idxs = seq_cam_ids.unique(dim=0, return_inverse=True, sorted=True)

    weight_contribution = torch_scatter.scatter(
        src=ws,
        index=inv_idxs,
        reduce='sum'
    )
    weight_contribution /= weight_contribution.sum()

    return seq_cam_ids, inv_idxs, weight_contribution

def get_recolor_feats(cam_idxs, uvw, feat_images, do_bilinear_interp=False):
    """
    Args:
        cam_idxs: [PxN] Tensor telling which image to index into (inv_idxs output of get_unique_seq_cam in 99% of cases)
        uvw: [PxNx3] Tensor of image coord data (last 3 channels of coord_data in 99% of cases)
        feat_images: [KxWxHxC] Tensor of images to recolor with
    Returns:
        [PxC] Tensor of recolorized features
    """
    _uv = uvw[:, :, :2]
    _w = uvw[:, :, -1]

    #filter out empty voxel feats
    valid_mask = _w > 1e-8
    _uv = _uv[valid_mask]
    _w = _w[valid_mask]

    scatter_idxs = torch.argwhere(valid_mask)[:, 0]

    if do_bilinear_interp:
        feats = bilinear_interpolate_batch(_uv, feat_images, cam_idxs)
    else:
        ui = _uv[:, 0].long()
        vi = _uv[:, 1].long()

        feats = feat_images[cam_idxs, vi, ui]

    weighted_feats = feats * _w.unsqueeze(-1)
    agg_feats = torch_scatter.scatter(
        src=weighted_feats,
        index=scatter_idxs,
        reduce='sum',
        dim=0
    )

    return agg_feats

def recolor_voxel_grid(coord_vg, img_dirs, do_bilinear_interp=True):
    """
    Re-colorize a voxel grid
    Args:
        coord_vg: Coordinate VoxelGrid, with feature keys of the form:
            [seq_idxN, cam_idxN, uxN, vxN, wxN]
        img_dirs: list of dirs to find kitti images from (ordering should match the cam_ids in the vg)
    """
    nvox = coord_vg.voxel_grid.features.shape[0]
    coord_data = coord_vg.voxel_grid.features.view(nvox, 5, -1).permute(0,2,1)
    ncams = coord_data[:, :, 1].max().round().long().item() + 1
    assert ncams == len(img_dirs), "number of image dirs != max cam id in vg!"

    ## step 1: get all the images that the voxel grid attends to
    torch.cuda.synchronize()
    t1 = time.time()
    uniq_seq_cam, inv_idxs, weight_contrib = get_unique_seq_cam(coord_data)
    uvw = coord_data[:,:, 2:]

    ## step 2: load all images
    torch.cuda.synchronize()
    t2 = time.time()
    res_imgs = [None] * uniq_seq_cam.shape[0]
    fks = None

    for i, (sid, cid) in enumerate(uniq_seq_cam):
        fp = img_dirs[cid.item()]
        img = FeatureImageTorch.from_kitti(fp, sid, device=coord_vg.device)
        res_imgs[i] = img.image
        fks = img.feature_keys

    feat_imgs = torch.stack(res_imgs, dim=0)

    ## step 3: recolor
    torch.cuda.synchronize()
    t3 = time.time()
    recolor_feats = get_recolor_feats(inv_idxs, uvw, feat_imgs, do_bilinear_interp=do_bilinear_interp)

    vg_out = copy.deepcopy(coord_vg)
    vg_out.voxel_grid.feature_keys = copy.deepcopy(fks)
    vg_out.voxel_grid.features = recolor_feats
    vg_out.voxel_grid.n_features = recolor_feats.shape[-1]

    torch.cuda.synchronize()
    t4 = time.time()

    timing_stats = {
        'img_check': t2-t1,
        'img_load': t3-t2,
        'recolor': t4-t3
    }

    return vg_out, weight_contrib, timing_stats

def recolor_bev_grid(coord_bev, img_dirs, do_bilinear_interp=True):
    """
    Re-colorize a voxel grid
    Args:
        coord_vg: Coordinate BEVGrid, with feature keys of the form:
            [seq_idxN, cam_idxN, uxN, vxN, wxN]
        img_dirs: list of dirs to find kitti images from (ordering should match the cam_ids in the vg)
    """
    nx, ny = coord_bev.bev_grid.metadata.N
    coord_fidxs = [i for i,k in enumerate(coord_bev.bev_grid.feature_keys.metainfo) if k=='img_coords']
    noncoord_fidxs = [i for i,k in enumerate(coord_bev.bev_grid.feature_keys.metainfo) if k!='img_coords']
    coord_data_2d = coord_bev.bev_grid.data[:, :, coord_fidxs].reshape(nx, ny, 5, -1).permute(0,1,3,2)

    has_feats_mask = coord_data_2d[:, :, :, -1].max(dim=-1)[0] > 1e-4
    ncells = has_feats_mask.sum()

    coord_data = coord_data_2d[has_feats_mask]
    ncams = coord_data[:, :, 1].max().round().long().item() + 1

    assert ncams == len(img_dirs), "number of image dirs != max cam id in vg!"

    ## step 1: get all the images that the voxel grid attends to
    torch.cuda.synchronize()
    t1 = time.time()
    uniq_seq_cam, inv_idxs, weight_contrib = get_unique_seq_cam(coord_data)
    uvw = coord_data[:,:, 2:]

    ## step 2: load all images
    torch.cuda.synchronize()
    t2 = time.time()
    res_imgs = [None] * uniq_seq_cam.shape[0]
    fks = None

    for i, (sid, cid) in enumerate(uniq_seq_cam):
        fp = img_dirs[cid.item()]
        img = FeatureImageTorch.from_kitti(fp, sid, device=coord_vg.device)
        res_imgs[i] = img.image
        fks = img.feature_keys

    feat_imgs = torch.stack(res_imgs, dim=0)

    ## step 3: recolor
    torch.cuda.synchronize()
    t3 = time.time()
    recolor_feats = get_recolor_feats(inv_idxs, uvw, feat_imgs, do_bilinear_interp=do_bilinear_interp)

    recolor_feats_2d = torch.zeros(nx, ny, len(fks), device=coord_bev.device)
    recolor_feats_2d[has_feats_mask] = recolor_feats

    out_keys = coord_bev.bev_grid.feature_keys[noncoord_fidxs] + fks

    bev_out = copy.deepcopy(coord_bev)
    bev_out.bev_grid.feature_keys = out_keys
    bev_out.bev_grid.n_features = len(out_keys)
    bev_out.bev_grid.data = torch.cat([bev_out.bev_grid.data[:, :, noncoord_fidxs], recolor_feats_2d], dim=-1)

    torch.cuda.synchronize()
    t4 = time.time()

    timing_stats = {
        'img_check': t2-t1,
        'img_load': t3-t2,
        'recolor': t4-t3
    }

    return bev_out, weight_contrib, timing_stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coordinator_config', type=str, required=True, help='path to coordinator config')
    parser.add_argument('--run_dir', type=str, required=True, help='kitti dir with coordinate voxels')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.coordinator_config, 'r'))

    N = len(np.loadtxt(os.path.join(args.run_dir, 'target_timestamps.txt')))
    torch_coordinator = setup_torch_coordinator(config)
    tf_manager = TfManager.from_kitti(args.run_dir, device=torch_coordinator.device)
    converters = {topic_conf['name']:str_to_cvt_class[topic_conf['type']] for topic_conf in config['ros_converter']['topics']}

    ## do some extra checks
    pc_colorizer_node = torch_coordinator.nodes['pointcloud_colorizer']
    coord_colorizer_node = torch_coordinator.nodes['pointcloud_coordinate_colorizer']

    assert pc_colorizer_node.image_keys == coord_colorizer_node.image_keys
    do_bilinear_interp = pc_colorizer_node.bilinear_interpolation
    
    img_keys = pc_colorizer_node.image_keys

    for ii in range(N):
        # load data
        torch_coordinator.data = {}

        for topic_conf in config['ros_converter']['topics']:
            label = topic_conf['name']
            base_dir = os.path.join(args.run_dir, label)
            cvt = converters[label]

            data = cvt.from_kitti(base_dir, ii, device=torch_coordinator.device)
            torch_coordinator.data[label] = data

        # load tfs
        for tf_name, tf_conf in config['tfs'].items():
            src_frame = tf_conf['src_frame']
            dst_frame = tf_conf['dst_frame']
            timestamp = torch_coordinator.data[tf_conf['src_data']].stamp

            if tf_manager.can_transform(src_frame, dst_frame, timestamp):
                torch_coordinator.data[tf_name] = tf_manager.get_transform(src_frame, dst_frame, timestamp)
            else:
                print('uh oh tf manager bug')

        # run torch coordinator
        t1 = time.time()
        timing_stats = torch_coordinator.run()
        t2 = time.time()

        for data_key in config['offline_proc']:
            data_dir = os.path.join(args.run_dir, data_key)
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            
            torch_coordinator.data[data_key].to_kitti(data_dir, ii)

        for k,v in torch_coordinator.data.items():
            print('{}:\n\t{}'.format(k,v), end='\n\n')

        print('proc dpt {}/{}'.format(ii+1, N))

        if ii % 200 != 0:
            continue

        #####################
        ##  Sanity checks  ##
        #####################
        feat_vg = torch_coordinator.data['voxel_map']
        feat_pc = torch_coordinator.data['feature_pointcloud_in_odom']
        feat_bev = torch_coordinator.data['bev_map']

        coord_vg = torch_coordinator.data['coord_voxel_map']
        coord_pc = torch_coordinator.data['coord_pointcloud_in_odom']
        coord_bev = torch_coordinator.data['coord_bev_map']

        curr_feat_imgs = torch.stack([torch_coordinator.data[k].image for k in img_keys], dim=0)

        ## check that points/voxels are spatially identical
        assert torch.allclose(feat_pc.pts, coord_pc.pts), "featpc.pts != coordpc.pts"
        assert torch.all(feat_pc.feat_mask == coord_pc.feat_mask), "featpc.mask != coordpc.mask"
        assert torch.all(feat_vg.voxel_grid.raster_indices == coord_vg.voxel_grid.raster_indices), "feat_vg.idxs != coord_vg.idxs"
        assert torch.all(feat_vg.voxel_grid.feature_mask == coord_vg.voxel_grid.feature_mask), "feat_vg.mask != coord_vg.mask"
        assert torch.all(feat_bev.bev_grid.hits == coord_bev.bev_grid.hits), "feat_bev.hits != coord_bev.hits"

        ## check that indexing into the instantaneous feat images with the coord pc matches the featpc
        coord_pc_coords = get_coord_data_from_features(coord_pc.features)
        _uv = coord_pc_coords[:, :, 2:4].long()
        _w = coord_pc_coords[:, :, -1]
        assert torch.allclose(_w.sum(-1), torch.ones_like(_w.sum(-1)))

        res_feats = []
        for cam_i in range(_uv.shape[1]):
            img = curr_feat_imgs[cam_i]
            _u = _uv[:, cam_i, 0]
            _v = _uv[:, cam_i, 1]
            _w = _w[:, cam_i]
            feats = img[_v, _u] * _w.unsqueeze(-1)
            res_feats.append(feats)

        res_feats = torch.stack(res_feats, dim=0).sum(dim=0)

        error = torch.linalg.norm(res_feats - feat_pc.features, dim=-1)

        img_dirs = [os.path.join(args.run_dir, k) for k in img_keys]

        ###################
        ## Voxel Recolor ##
        ###################

        vg_recolor, weight_contrib, vg_recolor_timing = recolor_voxel_grid(coord_vg, img_dirs, do_bilinear_interp=do_bilinear_interp)
        voxel_timing_str = " ".join([f"{k}:{v:.4f}s" for k,v in vg_recolor_timing.items()])

        ## compare
        feat_vg_o3d = feat_vg.voxel_grid.visualize()
        vg_recolor_o3d = vg_recolor.voxel_grid.visualize()

        assert (feat_vg.voxel_grid.feature_raster_indices == vg_recolor.voxel_grid.feature_raster_indices).all()

        orig_feats = feat_vg.voxel_grid.features
        recolor_feats = vg_recolor.voxel_grid.features

        ## metrics ##
        err = torch.linalg.norm(orig_feats - recolor_feats, dim=-1).cpu().numpy()
        cdf = np.cumsum(weight_contrib.cpu().numpy()[::-1]) 

        fig, axs = plt.subplots(1, 2)
        fig.suptitle(f'Voxel Recolor metrics')

        axs[0].plot(cdf)
        axs[0].set_title('CDF of image contribution')
        axs[0].set_xlabel('Num images')
        axs[0].set_ylabel('Frac of features')

        axs[1].hist(err, bins=100)
        axs[1].set_title('Voxel error dist  (if itr==0, there should be no error)')
        axs[1].set_xlabel('L2 error')
        axs[1].set_ylabel('density')

        plt.show()

        o3d.visualization.draw_geometries([feat_vg_o3d], window_name="Original Voxel Grid")
        o3d.visualization.draw_geometries([vg_recolor_o3d], window_name=f"Recolor Voxel Grid {voxel_timing_str}")

        #################
        ## BEV Recolor ##
        #################

        bev_recolor, weight_contrib, bev_recolor_timing = recolor_bev_grid(coord_bev, img_dirs, do_bilinear_interp=do_bilinear_interp)

        assert torch.all(bev_recolor.bev_grid.hits == feat_bev.bev_grid.hits)
        recolor_fidxs = [i for i,k in enumerate(bev_recolor.bev_grid.feature_keys.metainfo) if k == 'vfm']
        feat_fidxs = [i for i,k in enumerate(feat_bev.bev_grid.feature_keys.metainfo) if k == 'vfm']
        valid_mask = feat_bev.bev_grid.hits > 0.5

        orig_feats = feat_bev.bev_grid.data[:, :, feat_fidxs]
        recolor_feats = bev_recolor.bev_grid.data[:, :, recolor_fidxs]

        orig_viz = normalize_dino(orig_feats[:, :, :3])
        recolor_viz = normalize_dino(recolor_feats[:, :, :3])

        ## metrics ##
        err = torch.linalg.norm(orig_feats - recolor_feats, dim=-1)
        cdf = np.cumsum(weight_contrib.cpu().numpy()[::-1]) 

        fig, axs = plt.subplots(2, 3)
        fig.suptitle(f'BEV Recolor metrics')

        axs[0, 0].plot(cdf)
        axs[0, 0].set_title('CDF of image contribution')
        axs[0, 0].set_xlabel('Num images')
        axs[0, 0].set_ylabel('Frac of features')

        axs[0, 1].hist(err[valid_mask].cpu().numpy(), bins=100)
        axs[0, 1].set_title('Voxel error dist  (if itr==0, there should be no error)')
        axs[0, 1].set_xlabel('L2 error')
        axs[0, 1].set_ylabel('density')

        axs[1, 0].imshow(orig_viz.cpu())
        axs[1, 0].set_title('orig feats')

        axs[1, 1].imshow(recolor_viz.cpu())
        axs[1, 1].set_title('recolor feats')

        axs[1, 2].imshow(err.cpu(), cmap='jet')
        axs[1, 2].set_title('error')
        plt.show()