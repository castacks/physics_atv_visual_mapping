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

def get_coord_data_from_features(feats):
    return feats.reshape(feats.shape[0], 5, -1).permute(0,2,1)

def get_images(img_keys, coord_vg, max_images=-1):
    """
    Find all the images to load from the coord vg
    """
    ncams = len(img_keys)
    coord_data = get_coord_data_from_features(coord_vg.voxel_grid.features)
    valid_mask = coord_data[:, :, -1] > 1e-8
    seq_cam_ids = coord_data[valid_mask][:, :2].long()
    ws = coord_data[valid_mask][:, -1]
    seq_cam_ids, inv_idxs = seq_cam_ids.unique(dim=0, return_inverse=True, sorted=True)

    weight_contribution = torch_scatter.scatter(
        src=ws,
        index=inv_idxs,
        reduce='sum'
    )
    weight_contribution /= weight_contribution.sum()

    if max_images > 0:
        import pdb;pdb.set_trace()
        mask = torch.zeros_like(weight_contribution).bool()
        sort_idxs = weight_contribution.argsort(descending=True)
        mask[sort_idxs[:max_images]] = True

        seq_cam_ids = seq_cam_ids[mask]
        weight_contribution = weight_contribution[mask]

    return seq_cam_ids, weight_contribution

def colorize_voxel_grid(voxel_grid, feat_images, seq_img_ids, bilinear_interpolation=False):
    """
    Args:
        voxel_grid: VoxelGridTorch containing coordinate info
        feat_images: The list of K feature images to colorize with
        seq_img_idxs: sorted Kx2 tensor containing the [seq_id, cam_id] of each image
    """
    coord_data = get_coord_data_from_features(voxel_grid.voxel_grid.features)
    nvox, ncams, _ = coord_data.shape
    raster_idxs = voxel_grid.voxel_grid.feature_raster_indices

    assert (raster_idxs == raster_idxs.sort()[0]).all(), "need voxel raster idxs to be sorted"

    _vox_seq_cam = torch.cat([
        raster_idxs.view(nvox,1,1).tile(1,ncams,1),
        coord_data[:, :, :2],
    ], dim=-1).long()

    _uv = coord_data[:, :, 2:4]
    _w = coord_data[:, :, 4]

    #filter out empty voxel feats
    valid_mask = coord_data[:, :, -1] > 1e-8
    _vox_seq_cam = _vox_seq_cam[valid_mask]
    _uv = _uv[valid_mask]
    _w = _w[valid_mask]

    uniq_seq_cam, seq_cam_inv_idxs = torch.unique(_vox_seq_cam[:, 1:], dim=0, sorted=True, return_inverse=True)
    assert (uniq_seq_cam == seq_img_ids).all()

    ii = seq_cam_inv_idxs
    if bilinear_interpolation:
        import pdb;pdb.set_trace()
    else:
        ui = _uv[:, 0].long()
        vi = _uv[:, 1].long()

        feat_imgs = torch.stack([x.image for x in feat_images], dim=0)

        vox_feats = feat_imgs[ii, vi, ui]
        weighted_vox_feats = vox_feats * _w.unsqueeze(-1)

    vox_idxs = _vox_seq_cam[:, 0]

    uniq_vox_idxs, vox_inv_idxs = torch.unique(vox_idxs, sorted=True, return_inverse=True)
    assert (uniq_vox_idxs == raster_idxs).all()

    new_feats = torch_scatter.scatter(
        src=weighted_vox_feats,
        index=vox_inv_idxs,
        reduce='sum',
        dim=0
    )

    vg_out = copy.deepcopy(voxel_grid)
    vg_out.voxel_grid.feature_keys = copy.deepcopy(feat_images[0].feature_keys)
    vg_out.voxel_grid.features = new_feats
    vg_out.voxel_grid.n_features = new_feats.shape[-1]

    return vg_out

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
        curr_feat_imgs = torch.stack([torch_coordinator.data[k].image for k in img_keys], dim=0)

        coord_vg = torch_coordinator.data['coord_voxel_map']
        coord_pc = torch_coordinator.data['coord_pointcloud_in_odom']

        ## check that points/voxels are spatially identical
        assert torch.allclose(feat_pc.pts, coord_pc.pts), "featpc.pts != coordpc.pts"
        assert torch.all(feat_pc.feat_mask == coord_pc.feat_mask), "featpc.mask != coordpc.mask"
        assert torch.all(feat_vg.voxel_grid.raster_indices == coord_vg.voxel_grid.raster_indices), "feat_vg.idxs != coord_vg.idxs"
        assert torch.all(feat_vg.voxel_grid.feature_mask == coord_vg.voxel_grid.feature_mask), "feat_vg.mask != coord_vg.mask"

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

        # plt.imshow(curr_feat_imgs[0].cpu().numpy())
        # plt.scatter(_uv[:, 0, 0].cpu().numpy(), _uv[:, 0, 1].cpu().numpy())
        # plt.show()

        # pc1 = o3d.geometry.PointCloud()
        # pc1.points = o3d.utility.Vector3dVector(feat_pc.feature_pts.cpu().numpy())
        # pc1.colors = o3d.utility.Vector3dVector(feat_pc.features.cpu().numpy())
        # o3d.visualization.draw_geometries([pc1])

        # pc2 = o3d.geometry.PointCloud()
        # pc2.points = o3d.utility.Vector3dVector(coord_pc.feature_pts.cpu().numpy())
        # pc2.colors = o3d.utility.Vector3dVector(res_feats.cpu().numpy())
        # o3d.visualization.draw_geometries([pc2])

        #hmm the indexing seems right but the featpc features dont match

        assert torch.allclose(res_feats, feat_pc.features), "colorized feats dont match!"

        t1 = time.time()
        ## queue up all the images to load
        seq_img_keys, weight_contrib = get_images(img_keys, coord_vg)
       
        ## get features
        feat_images = []
        for ik in seq_img_keys:
            img = FeatureImageTorch.from_kitti(os.path.join(args.run_dir, img_keys[ik[1]]), ik[0], device='cuda')
            # img = ImageTorch.from_kitti(os.path.join(args.run_dir, img_keys[ik[1]]), ik[0], device='cuda')
            feat_images.append(img)

        ## re-colorize voxel grid
        torch.cuda.synchronize()
        t2 = time.time()
        vg_recolor = colorize_voxel_grid(coord_vg, feat_images, seq_img_keys, bilinear_interpolation=do_bilinear_interp)
        torch.cuda.synchronize()
        t3 = time.time()

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
        fig.suptitle(f'Recolor metrics ({len(feat_images)} images)')

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
        o3d.visualization.draw_geometries([vg_recolor_o3d], window_name=f"Recolor Voxel Grid (img load: {t2-t1:.4f}s, recolor {t3-t2:.4f}s)")