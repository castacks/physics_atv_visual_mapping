import torch


class LocalMapperMetadata:
    """n-dimensional definition of local mapper metadata"""

    def __init__(self, origin, length, resolution, device="cpu"):
        self.origin = (
            origin if isinstance(origin, torch.Tensor) else torch.tensor(origin, device=device)
        )
        self.length = (
            length if isinstance(length, torch.Tensor) else torch.tensor(length, device=device)
        )
        self.resolution = (
            resolution
            if isinstance(resolution, torch.Tensor)
            else torch.tensor(resolution, device=device)
        )
        self.N = torch.round(self.length / self.resolution).long()
        self.ndims = self.origin.shape[0]
        self.device = device

    def __repr__(self):
        return "LocalMapperMetadata with \n\tOrigin: {}\n\tLength: {}\n\tResolution: {}\n\t(N: {})".format(
            self.origin, self.length, self.resolution, self.N
        )

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return LocalMapperMetadata(
                origin=self.origin[[idx]],
                length=self.length[[idx]],
                resolution=self.resolution[[idx]],
                device=self.device
            )
        else:
            return LocalMapperMetadata(
                origin=self.origin[idx],
                length=self.length[idx],
                resolution=self.resolution[idx],
                device=self.device
            )

    def to(self, device):
        self.device = device
        self.origin = self.origin.to(device)
        self.length = self.length.to(device)
        self.resolution = self.resolution.to(device)
        self.N = self.N.to(device)

        return self
