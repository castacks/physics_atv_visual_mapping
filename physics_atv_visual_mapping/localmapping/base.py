import abc
import copy
import torch

from physics_atv_visual_mapping.localmapping.metadata import LocalMapperMetadata


class LocalMapper(abc.ABC):
    """
    Base mapper class (interface for BEV, voxel mapping)
    """

    def __init__(self, metadata: LocalMapperMetadata, device="cpu"):
        """
        Args:
            metadata: Metadata to define the map coordinates relative to poses
                e.g. if my pose is [3, 4] and origin is [-2, -1], the lower-left of my map is [1, 3]
                Note that base_metadata will stay constant, and metadata will update with pose
        """
        self.base_metadata = metadata.to(device)
        self.metadata = copy.deepcopy(metadata).to(device)
        self.device = device

    @abc.abstractmethod
    def update_pose(self, pose: torch.Tensor) -> None:
        """Update the pose of the localmapper

        Args:
            pose: the pose to update (assumed to be the center of the localmap)
        """
        pass

    @abc.abstractmethod
    def add_feature_pc(self, pts: torch.Tensor, features: torch.Tensor) -> None:
        """Add new feature points to the local mapper

        Args:
            pts: [Nx3] Tensor of points to add
            features: [NxK] Tensor of features corresponding to pts
        """
        pass
