import os
import tqdm
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt

from tartandriver_utils.os_utils import kitti_n_frames

from ros_torch_converter.datatypes.rb_state import OdomRBStateTorch
from ros_torch_converter.datatypes.bev_grid import BEVGridTorch

from physics_atv_visual_mapping.feature_key_list import FeatureKeyList

"""
Script to postprocess in a "distance_to_traj" layer into BEV maps

This will append an extra channel called "dist_to_traj" into a bev map,
where each cell will contain the mindist from that cell to the trajectory
(note that this includes both past/future trajs)

This will be useful for a number of perception-related filtering things
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True, help='dir to run on')
    parser.add_argument('--odom_dir', type=str, required=False, default='super_odometry/odometry', help='dir ot get odom data from')
    parser.add_argument('--bev_dir', type=str, required=False, default='mapping/bev_map_reduce')
    parser.add_argument('--device', type=str, required=False, default='cuda')
    args = parser.parse_args()

    odom_dir = os.path.join(args.run_dir, args.odom_dir)
    bev_dir = os.path.join(args.run_dir, args.bev_dir)

    #check frames
    sample_odom = OdomRBStateTorch.from_kitti(odom_dir, 0)
    sample_bev = BEVGridTorch.from_kitti(bev_dir, 0)

    # if sample_odom.frame_id != sample_bev.frame_id:
    #     x = input(f"Got odom frame id = {sample_odom.frame_id} and bev frame id = {sample_bev.frame_id}. This is likely bad. Continue? [Y/n]")
    #     if x == 'n':
    #         exit(0)

    #load trajdata
    N = kitti_n_frames(args.run_dir)
    trajdata = OdomRBStateTorch.from_kitti_multi(odom_dir, torch.arange(N), device=args.device)
    poses = torch.stack([x.state[:2] for x in trajdata])

    #compute sdf for all bev maps
    for i in tqdm.tqdm(range(N)):
        bev_grid = BEVGridTorch.from_kitti(bev_dir, i, device=args.device)
        metadata = bev_grid.bev_grid.metadata
        coords = metadata.get_coords().flatten(end_dim=-2)

        mindists = torch.linalg.norm(coords.unsqueeze(1) - poses.unsqueeze(0), dim=-1).min(dim=1)[0]
        mindists = mindists.reshape(*metadata.N)

        ## remove spatial if there
        idxs = [i for i,k in enumerate(bev_grid.bev_grid.feature_keys.label) if k != 'dist_to_traj']
        _fks = bev_grid.bev_grid.feature_keys[idxs]
        _data = bev_grid.bev_grid.data[..., idxs]

        bev_grid.bev_grid.feature_keys = _fks + FeatureKeyList(label=['dist_to_traj'], metainfo=['spatial'])
        bev_grid.bev_grid.data = torch.cat([_data, mindists.unsqueeze(-1)], dim=-1)

        bev_grid.to_kitti(bev_dir, i)

        # ##debug viz
        # import matplotlib.pyplot as plt
        # plt.plot(poses[:, 0].cpu().numpy(), poses[:, 1].cpu().numpy())
        # plt.imshow(mindists.T.cpu().numpy(), cmap='jet', origin='lower', extent=metadata.extent())
        # plt.show()