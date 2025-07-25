"""
Base class for terrain estimation blocks

Terrain estimation blocks will take in a voxel grid and a partially computed BEVGrid and return a modified BEVGrid
"""

import abc

class TerrainEstimationBlock(abc.ABC):
    def __init__(self, voxel_metadata, voxel_feature_keys, device='cpu'):
        self.voxel_metadata = voxel_metadata
        self.voxel_feature_keys = voxel_feature_keys
        self.device=device

    @abc.abstractmethod
    def run(self, voxel_grid, bev_grid):
        """
        Args:
            voxel_grid: the input voxel grid
            bev_grid: the input BEV grid
        Returns:
            bev_grid: the output BEV grid (modifies the input in-place)
        """
        pass

    @property
    @abc.abstractmethod
    def output_feature_keys(self):
        """
        define the layer keys to output to 
        Note that for some layers such as BEVSplat, the output keys depend on the input
        """
        pass

    @abc.abstractmethod
    def to(self, device):
        pass