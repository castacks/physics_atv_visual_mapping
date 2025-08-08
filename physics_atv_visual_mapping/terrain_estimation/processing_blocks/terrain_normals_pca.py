import torch

from physics_atv_visual_mapping.terrain_estimation.processing_blocks.base import TerrainEstimationBlock

class TerrainNormalsPCA(TerrainEstimationBlock):

    def __init__(self, voxel_metadata, voxel_n_features, terrain_layer, mask_layer, device='cpu'):
        super().__init__(voxel_metadata, voxel_n_features, device)
        self.terrain_layer = terrain_layer
        self.mask_layer = mask_layer

    def run(self, voxel_grid, bev_grid):
        """
        Args:
            voxel_grid: the input voxel grid
            bev_grid: the input BEV (Bird's Eye View) grid

        Returns:
            bev_grid: the output BEV grid
        """

        # Compute the BEV grid spatial extent (min and max in x and y dimensions)
        bev_extent = (
            bev_grid.metadata.origin[0].item(),
            bev_grid.metadata.origin[0].item() + bev_grid.metadata.length[0].item(),
            bev_grid.metadata.origin[1].item(),
            bev_grid.metadata.origin[1].item() + bev_grid.metadata.length[1].item(),
        )

        # Generate a meshgrid of x and y coordinates over the BEV extent
        bev_x = torch.linspace(bev_extent[0], bev_extent[1], bev_grid.metadata.N[0], device='cuda')
        bev_y = torch.linspace(bev_extent[2], bev_extent[3], bev_grid.metadata.N[1], device='cuda')
        bev_x, bev_y = torch.meshgrid(bev_x, bev_y, indexing='ij')

        # Get index of the terrain feature layer from the BEV grid
        terrain_idx = bev_grid.feature_keys.index(self.terrain_layer)
        # Extract the terrain height data
        terrain_data = bev_grid.data[..., terrain_idx].clone()

        # Define the size of the neighborhood patch used for local surface analysis
        patch_size = torch.tensor([3, 3])  # 3x3 patch

        # Create an unfolding operator to extract patches from the terrain
        get_patches = torch.nn.Unfold(patch_size, padding=1)

        # Stack x, y, and terrain height into a 3-channel 3D tensor
        terrain_3D = torch.stack((bev_x, bev_y, terrain_data), dim=0)

        # Extract local 3x3 patches for PCA
        patches = get_patches(terrain_3D.unsqueeze(0)).squeeze()  # shape: (3 * 9, H*W)
        patches = patches.reshape(3, torch.prod(patch_size), -1).permute(2, 1, 0)  # shape: (H*W, 9, 3)

        # Perform PCA to find principal directions in each patch
        u, s, v = torch.pca_lowrank(patches)

        # The third principal component (v[..., 2]) is the estimated surface normal
        normals = v[..., 2]
        normals = normals.reshape(bev_grid.metadata.N[0], bev_grid.metadata.N[1], 3)

        # Ensure normals point upwards (positive Z direction)
        posotive_mask = normals[..., 2] < 0
        normals[posotive_mask] = -normals[posotive_mask]

        # Get mask index and mask out invalid regions
        mask_idx = bev_grid.feature_keys.index(self.mask_layer)
        mask = bev_grid.data[..., mask_idx] > 1e-4

        # Separate and apply mask to normal components
        normals_x = normals[..., 0]
        normals_x[~mask] = 0
        normals_y = normals[..., 1]
        normals_y[~mask] = 0
        normals_z = normals[..., 2]
        normals_z[~mask] = 0

        # Store computed normals into BEV grid output

        normals_x_idx = bev_grid.feature_keys.index(self.output_keys[0])
        bev_grid.data[..., normals_x_idx] = normals_x

        normals_y_idx = bev_grid.feature_keys.index(self.output_keys[1])
        bev_grid.data[..., normals_y_idx] = normals_y

        normals_z_idx = bev_grid.feature_keys.index(self.output_keys[2])
        bev_grid.data[..., normals_z_idx] = normals_z

        return bev_grid


    @property
    def output_keys(self):
        """
        define the layer keys to output to 
        Note that for some layers such as BEVSplat, the output keys depend on the input
        """
        return ['normals_x','normals_y','normals_z']

    def to(self, device):
        self.device = device
        return self