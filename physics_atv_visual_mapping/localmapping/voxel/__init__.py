from physics_atv_visual_mapping.localmapping.voxel.voxel_localmapper import VoxelLocalMapper
from physics_atv_visual_mapping.localmapping.voxel.voxel_cov_localmapper import VoxelCovarianceLocalMapper

MAPPER_CLASSES = {
    "voxel": VoxelLocalMapper,
    "voxel_covariance": VoxelCovarianceLocalMapper,
}