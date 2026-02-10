import torch
import torch_scatter

from physics_atv_visual_mapping.terrain_estimation.processing_blocks.base import TerrainEstimationBlock
from physics_atv_visual_mapping.terrain_estimation.processing_blocks.utils import setup_kernel, apply_kernel
from physics_atv_visual_mapping.feature_key_list import FeatureKeyList

class MRFTerrainEstimation(TerrainEstimationBlock):
    """
    Compute a per-cell min and max height
    """
    def __init__(self, voxel_metadata, voxel_feature_keys, input_layer, itrs, alpha, beta, lr, kernel_params, mask_layer=None, output_layer='terrain', device='cpu'):
        super().__init__(voxel_metadata, voxel_feature_keys, device)
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.mask_layer = mask_layer
        self.itrs = itrs
        self.alpha = alpha
        self.beta = beta
        self.lr = lr

        #since we want to count neighbors with this kernel, set the middle element to zero.
        self.kernel = setup_kernel(**kernel_params, metadata=voxel_metadata).to(self.device)
        kmid = self.kernel.shape[0]//2, self.kernel.shape[1]//2
        self.kernel[kmid] = 0.

    def to(self, device):
        self.kernel = self.kernel.to(device)
        self.device = device
        return self

    @property
    def output_feature_keys(self):
        return FeatureKeyList(
            label = [self.output_layer],
            metainfo = ["terrain_estimation"]
        )

    def run(self, voxel_grid, bev_grid):
        input_idx = bev_grid.feature_keys.index(self.input_layer)
        input_data = bev_grid.data[..., input_idx].clone()

        if self.mask_layer is None:
            mask = torch.ones(*bev_grid.metadata.N, dtype=torch.bool, device=self.device)
        else:
            mask_idx = bev_grid.feature_keys.index(self.mask_layer)
            mask = bev_grid.data[..., mask_idx] > 1e-4

        terrain_estimate = input_data.clone()

        for i in range(self.itrs):
            dz = torch.zeros_like(terrain_estimate)

            measurement_update = input_data - terrain_estimate
            measurement_update[~mask] = 0.

            neighbor_vals = apply_kernel(kernel=self.kernel, data=terrain_estimate)
            neighbor_weights = apply_kernel(kernel=self.kernel, data=mask.float())
            neighbor_update = (neighbor_vals - neighbor_weights*terrain_estimate) / (neighbor_weights + 1e-8)
            neighbor_update[~mask] = 0.

            dz = self.lr * (self.alpha*measurement_update + self.beta*neighbor_update)

            terrain_estimate += dz
        
        output_data_idx = bev_grid.feature_keys.index(self.output_feature_keys.label[0])
        bev_grid.data[..., output_data_idx] = terrain_estimate

        return bev_grid

#test kernel-based mrf update
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from physics_atv_visual_mapping.terrain_estimation.processing_blocks.utils import apply_kernel

    xs = torch.linspace(-10, 10, 100)
    ys = torch.linspace(-20, 20, 100)
    xs, ys = torch.meshgrid(xs, ys, indexing='ij')
    zs = xs.cos() + ys.sin() * xs/20. + (xs + ys)/5.

    mask = zs > 0
    zs[~mask] = 0.

    """
    Some reasoning on why this kernel trick works:
        MRF computes an average of diffs to adjacent cells, i.e. sum[x-xn]/sum[I(xn=/=0)], xn = neighbors
        We are basically just distributing the sum to allow us to use convolution ops.
        i.e. update = sum[x]/sum[I(xn=/=0)] - sum[xn]/sum[I(xn=/=0)]
        and all the above sums are calculable with convolution
    """

    ## orig mrf update
    zs_adj = get_adjacencies(data=zs)
    valid_adj = get_adjacencies(data=mask)
    neighbor_update = zs_adj - zs.unsqueeze(0)
    neighbor_update[~valid_adj] = 0.
    neighbor_update = neighbor_update.sum(dim=0) / (valid_adj.sum(dim=0)+1e-8)

    ## kernel mrf update
    kernel = torch.tensor([
        [0., 1., 0.],
        [1., 0., 1.],
        [0., 1., 0.]
    ])

    neighbor_vals = apply_kernel(kernel=kernel, data=zs)
    neighbor_weights = apply_kernel(kernel=kernel, data=mask.float())
    neighbor_update2 = (neighbor_vals - neighbor_weights*zs) / (neighbor_weights + 1e-8)

    #im not checking the edges because the adjacency method same-pads and the kernel (correctly) zero-pads
    print(torch.allclose(neighbor_update[1:-1, 1:-1], neighbor_update2[1:-1, 1:-1], atol=1e-6))
    err = (neighbor_update[1:-1, 1:-1] - neighbor_update2[1:-1, 1:-1]).abs()
    print('max error = {}'.format(err.max()))

    fig, axs = plt.subplots(2, 2)
    axs = axs.flatten()
    axs[0].imshow(zs)
    axs[1].imshow(neighbor_update)
    axs[2].imshow(neighbor_update2)
    axs[3].imshow(err)
    
    axs[0].set_title('zs')
    axs[1].set_title('orig')
    axs[2].set_title('kernel')
    axs[3].set_title('diff')
    plt.show()