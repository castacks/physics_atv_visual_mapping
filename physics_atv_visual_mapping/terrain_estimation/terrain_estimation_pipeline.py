import torch

from physics_atv_visual_mapping.localmapping.voxel.voxel_localmapper import VoxelGrid
from physics_atv_visual_mapping.localmapping.bev.bev_localmapper import BEVGrid
from physics_atv_visual_mapping.localmapping.metadata import LocalMapperMetadata

from physics_atv_visual_mapping.terrain_estimation.processing_blocks.elevation_stats import ElevationStats
from physics_atv_visual_mapping.terrain_estimation.processing_blocks.elevation_filter import ElevationFilter
from physics_atv_visual_mapping.terrain_estimation.processing_blocks.terrain_inflation import TerrainInflation
from physics_atv_visual_mapping.terrain_estimation.processing_blocks.mrf_terrain_estimation import MRFTerrainEstimation
from physics_atv_visual_mapping.terrain_estimation.processing_blocks.porosity import Porosity
from physics_atv_visual_mapping.terrain_estimation.processing_blocks.slope import Slope
from physics_atv_visual_mapping.terrain_estimation.processing_blocks.terrain_diff import TerrainDiff
from physics_atv_visual_mapping.terrain_estimation.processing_blocks.bev_feature_splat import BEVFeatureSplat
from physics_atv_visual_mapping.terrain_estimation.processing_blocks.terrain_aware_bev_feature_splat import TerrainAwareBEVFeatureSplat

def setup_terrain_estimation_pipeline(config):
    blocks = []
    feature_keys = [] #keep track of feature keys to check validity of pipeline

    voxel_metadata = LocalMapperMetadata(**config["localmapping"]["metadata"])
    voxel_n_features = config["localmapping"]["n_features"]

    for block_config in config["terrain_estimation"]:
        btype = block_config["type"]
        block_config["args"]["device"] = config["device"]
        block_config["args"]["voxel_metadata"] = voxel_metadata
        block_config["args"]["voxel_n_features"] = voxel_n_features
        
        if btype == "elevation_stats":
            block = ElevationStats(**block_config["args"])
        elif btype == "elevation_filter":
            block = ElevationFilter(**block_config["args"])
        elif btype == "sdf":
            block = SDF(**block_config["args"])
        elif btype == "terrain_inflation":
            block = TerrainInflation(**block_config["args"])
        elif btype == "mrf_terrain_estimation":
            block = MRFTerrainEstimation(**block_config["args"])
        elif btype == "porosity":
            block = Porosity(**block_config["args"])
        elif btype == "slope":
            block = Slope(**block_config["args"])
        elif btype == "terrain_diff":
            block = TerrainDiff(**block_config["args"])
        elif btype == "bev_feature_splat":
            block = BEVFeatureSplat(**block_config["args"])
        elif btype == "terrain_aware_bev_feature_splat":
            block = TerrainAwareBEVFeatureSplat(**block_config["args"])
        else:
            print('unknown terrain estimation block type {}'.format(btype))
            exit(1)

        blocks.append(block)
    
    pipeline = TerrainEstimationPipeline(blocks=blocks, voxel_metadata=voxel_metadata, voxel_n_features=voxel_n_features, device=config["device"])
    return pipeline

class TerrainEstimationPipeline:
    """
    Main driver class for performing terrain estimation (in this scope, terrain estimation
    means extracting BEV features from a VoxelGrid)
    """
    def __init__(self, blocks, voxel_metadata, voxel_n_features, device):
        self.blocks = blocks
        self.voxel_metadata = voxel_metadata
        self.voxel_n_features = voxel_n_features

        self.bev_metadata = LocalMapperMetadata(
            origin=voxel_metadata.origin[:2],
            length=voxel_metadata.length[:2],
            resolution=voxel_metadata.resolution[:2],
        )
        self.device = device

    def compute_feature_keys(self, voxel_grid):
        """
        Precompute the set of output feature keys
        """
        fks = []
        for block in self.blocks:
            for fk in block.output_keys:
                if fk not in fks:
                    fks.append(fk)
        return  fks

    def run(self, voxel_grid):
        bev_metadata = LocalMapperMetadata(
            origin=voxel_grid.metadata.origin[:2],
            length=voxel_grid.metadata.length[:2],
            resolution=voxel_grid.metadata.resolution[:2],
        )

        bev_feature_keys = self.compute_feature_keys(voxel_grid)

        bev_grid = BEVGrid(
            metadata = bev_metadata,
            n_features = len(bev_feature_keys),
            feature_keys = bev_feature_keys,
            device = self.device
        )

        for block in self.blocks:
            block.run(voxel_grid, bev_grid)

        return bev_grid

    def to(self, device):
        self.device = device
        self.blocks = [block.to(device) for block in self.blocks]
        return self