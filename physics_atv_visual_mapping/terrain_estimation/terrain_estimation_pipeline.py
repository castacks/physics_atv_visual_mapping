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
from physics_atv_visual_mapping.terrain_estimation.processing_blocks.traversability_prototype_scores import TraversabilityPrototypeScore
from physics_atv_visual_mapping.feature_key_list import FeatureKeyList


class TerrainEstimationPipeline:
    """
    Main driver class for performing terrain estimation (in this scope, terrain estimation
    means extracting BEV features from a VoxelGrid)
    """
    def __init__(self, config, blocks=None, voxel_metadata=None, voxel_n_features=None, device='cpu'):
        self.blocks = blocks
        self.voxel_metadata = voxel_metadata
        self.voxel_n_features = voxel_n_features
        self.bev_metadata = None
        self.config = config
        self.device = device
        self.cnt = 0

    def init_terrain_estimation_pipeline(self, voxel_grid):
        config = self.config

        voxel_metadata = LocalMapperMetadata(**config["localmapping"]["metadata"])
        voxel_n_features = len(voxel_grid.feature_key_list)

        self.bev_metadata = LocalMapperMetadata(
            origin=voxel_metadata.origin[:2],
            length=voxel_metadata.length[:2],
            resolution=voxel_metadata.resolution[:2],
        )

        blocks = []

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
            elif btype == "traversability_prototype_scores":
                block = TraversabilityPrototypeScore(**block_config["args"])
            else:
                print('unknown terrain estimation block type {}'.format(btype))
                exit(1)

            blocks.append(block)

        self.blocks = blocks

    def compute_feature_keys(self):
        """
        Precompute the set of output feature keys
        """
        fk_list = FeatureKeyList(label=[], metadata=[])
        for block in self.blocks:
            for fk in block.output_keys:
                if fk not in fk_list.label:
                    fk_list.label.append(fk)
                    fk_list.metadata.append("terrain_estimation")
        return fk_list

    def run(self, voxel_grid):
        if self.cnt == 0:
            self.init_terrain_estimation_pipeline(voxel_grid)
            self = self.to(self.device)

        self.cnt += 1
        bev_metadata = LocalMapperMetadata(
            origin=voxel_grid.metadata.origin[:2],
            length=voxel_grid.metadata.length[:2],
            resolution=voxel_grid.metadata.resolution[:2],
        )

        bev_feature_key_list = self.compute_feature_keys()
        feature_key_list = voxel_grid.feature_key_list + bev_feature_key_list
        bev_grid = BEVGrid(
            metadata = bev_metadata,
            n_features = len(feature_key_list),
            feature_key_list = feature_key_list,
            device = self.device
        )

        for block in self.blocks:
            block.run(voxel_grid, bev_grid)

        return bev_grid

    def to(self, device):
        self.device = device
        if self.cnt != 0:
            self.blocks = [block.to(device) for block in self.blocks]
        return self