import torch
import copy
from torch_scatter import scatter
import numpy as np
"""
Collection of functions for localmapping
"""

def localmap_from_pointcloud(pcl_pos, pcl_data, metadata, reduction='mean'):
    """
    Build a localmap from a pointcloud and corresponding data
    """
    # Stack pcl_data then height, so we can do height thresholding
    # Unsqueeze the 1D tensor to match dimensions
    pcl_pos_z = pcl_pos[:,2].unsqueeze(1)
    # Concatenate along the last dimension
    pcl_data = torch.cat((pcl_data, pcl_pos_z), dim=1)
    
    # original code
    ox = metadata['origin'][0].item()
    oy = metadata['origin'][1].item()
    lx = metadata['length_x'].item()
    ly = metadata['length_y'].item()
    res = metadata['resolution'].item()
    nx = round(lx/res)
    ny = round(ly/res)

    raster_idxs, valid_mask = get_raster_idxs(pcl_pos, metadata)
    raster_idxs = raster_idxs[valid_mask]

    res_map = torch.zeros(nx, ny, pcl_data.shape[-1], device=pcl_data.device)
    raster_map = res_map.view(-1, pcl_data.shape[-1])

    scatter(pcl_data[valid_mask], raster_idxs, dim=0, out=raster_map, reduce=reduction)
    res_unk_map = torch.linalg.norm(res_map, dim=-1) > 1e-6

    return res_map, res_unk_map, metadata

def aggregate_localmaps(agg_map, update_map, ema=0.5):
    """
    Add update_map into agg_map
    """
    ## first shift update map to agg_map ##
    update_map_shift = shift_localmap(copy.deepcopy(update_map), agg_map['metadata'])
#    update_map_shift = shift_localmap(update_map, agg_map['metadata'])

    ## map update
    res_known = update_map_shift['known'] | agg_map['known']

    agg_data_weighted = ema * agg_map['data']
    update_data_weighted = (1.-ema) * update_map_shift['data']
    denom = update_map_shift['known']*(1.-ema) + agg_map['known']*(ema)
    res_data = (agg_data_weighted + update_data_weighted) / denom.unsqueeze(-1)
    res_data[~res_known] = 0.

    res = {
        'data': res_data,
        'known': res_known,
        'metadata': update_map_shift['metadata']
    }

#    ## debug viz
#    if debug:
#        fig, axs = plt.subplots(2, 4, figsize=(48, 24))
#        for i,k in enumerate(['agg', 'update', 'shift_update', 'result']):
#            axs[0, i].set_title(k)
#
#        for i, _map in enumerate([agg_map, update_map, update_map_shift, res]):
#            axs[0, i].imshow(_map['data'].permute(1,0,2).cpu(), origin='lower', extent=get_viz_extent(_map['metadata']))
#            axs[1, i].imshow(_map['known'].T.cpu(), origin='lower', extent=get_viz_extent(_map['metadata']))
#            axs[1, i].set_title('origin: {:.2f}, {:.2f}'.format(_map['metadata']['origin'][0].item(), _map['metadata']['origin'][1].item()))
#        plt.show()

    return res

def shift_localmap(src_map, metadata):
    """
    Translate src_map to the position in metadata
    """
    dx = (metadata['origin'][0] - src_map['metadata']['origin'][0]).item()
    dy = (metadata['origin'][1] - src_map['metadata']['origin'][1]).item()

    dgx = round(dx/metadata['resolution'].item())
    dgy = round(dy/metadata['resolution'].item())

    #update map data
    src_map['data'] = torch.roll(src_map['data'], shifts=[-dgx, -dgy], dims=[0, 1])
    src_map['known'] = torch.roll(src_map['known'], shifts=[-dgx, -dgy], dims=[0, 1])

    if dgx > 0:
        src_map['data'][-dgx:] = 0.
        src_map['known'][-dgx:] = False
    elif dgx < 0:
        src_map['data'][:-dgx] = 0.
        src_map['known'][:-dgx] = False
    if dgy > 0:
        src_map['data'][:, -dgy:] = 0.
        src_map['known'][:, -dgy:] = False
    elif dgy < 0:
        src_map['data'][:, :-dgy] = 0.
        src_map['known'][:, :-dgy] = False

    #update metadata
    src_map['metadata']['origin'][0] += dgx*metadata['resolution']
    src_map['metadata']['origin'][1] += dgy*metadata['resolution']

    return src_map

def get_viz_extent(metadata):
    return (
        metadata['origin'][0].item(),
        metadata['origin'][0].item() + metadata['length_x'].item(),
        metadata['origin'][1].item(),
        metadata['origin'][1].item() + metadata['length_y'].item(),
    )

def get_raster_idxs(pos, metadata):
    """
    Get indexes for positions given map metadata
    """
    ox = metadata['origin'][0].item()
    oy = metadata['origin'][1].item()
    lx = metadata['length_x'].item()
    ly = metadata['length_y'].item()
    res = metadata['resolution'].item()
    nx = round(lx/res)
    ny = round(ly/res)

    ix = ((pos[..., 0]-ox)/res).long()
    iy = ((pos[..., 1]-oy)/res).long()

    mask = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    raster_idxs = ix * ny + iy

    return raster_idxs, mask

if __name__ == '__main__':
    import os
    import copy
    import time
    import open3d as o3d
    import matplotlib.pyplot as plt

    device = 'cuda'
    data_fp = '/home/physics_atv/workspace/datasets/train_turnpike_flat_2023-09-12-14-21-03'
    dpt_fps = sorted(os.listdir(data_fp))

    metadata = {
        'origin': torch.tensor([-100., -100.], device=device),
        'length_x': torch.tensor(200., device=device),
        'length_y': torch.tensor(200., device=device),
        'resolution': torch.tensor(0.5, device=device)
    }

    map_agg = None

#    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
#    plt.show(block=False)

    for i, dfp in enumerate(dpt_fps):
        dpt = torch.load(os.path.join(data_fp, dfp), map_location=device)

        pcl = dpt['pointcloud']
        pcl_pos = pcl[:, :3]
        pcl_data = pcl[:, 3:]

        mask = torch.linalg.norm(pcl_data, dim=-1) > 0.01
        pcl_pos = pcl_pos[mask]
        pcl_data = pcl_data[mask]

        pose = dpt['traj'][0]

        _metadata = copy.deepcopy(metadata)
        _metadata['origin'] += pose[:2]

        ## debug viz ##
#        pcl_viz = o3d.geometry.PointCloud()
#        pcl_viz.points = o3d.utility.Vector3dVector(pcl_pos.cpu().numpy())
#        pcl_viz.colors = o3d.utility.Vector3dVector(pcl_data.cpu().numpy())
#        o3d.visualization.draw_geometries([pcl_viz])

        print(i, pose[:3])

        t1 = time.time()
        localmap, known_mask, localmap_metadata = localmap_from_pointcloud(pcl_pos, pcl_data, _metadata)
        t2 = time.time()

        localmap = {
            'data': localmap,
            'known': known_mask,
            'metadata': localmap_metadata
        }

        t3 = time.time()
        if map_agg is None:
            map_agg = localmap
        else:
            map_agg = aggregate_localmaps(localmap, map_agg, ema=0.5)
        t4 = time.time()

        print('device = {}, BEV projection: {:.4f}s, accumulate: {:.4f}s'.format(device, t2-t1, t4-t3))

        """
        for ax in axs.flatten():
            ax.cla()

        for i,k in enumerate(['update', 'acc']):
            axs[0, i].set_title(k)

        for i, _map in enumerate([localmap, map_agg]):
            axs[0, i].imshow(_map['data'].permute(1,0,2).cpu(), origin='lower', extent=get_viz_extent(_map['metadata']))
            axs[1, i].imshow(_map['known'].T.cpu(), origin='lower', extent=get_viz_extent(_map['metadata']))
            axs[1, i].set_title('origin: {:.2f}, {:.2f}'.format(_map['metadata']['origin'][0].item(), _map['metadata']['origin'][1].item()))

        plt.pause(0.1)
        """
