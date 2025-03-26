import torch

class FeaturePointCloud:
    """
    Class for pointclouds with an arbitrary set of "features"
    """
    def __init__(self, pts, feats, pt_to_feat_idxs, device='cpu'):
        """
        Args:
            pts: A [N1x3] FloatTensor of spatial positions
            feats: A [N2xF] Float tensor of features, N2 <= N1
            pt_to_feat_idxs: A [N1] LongTensor of indices, suich that
                the k-th point in points should have features corresponding to the idxs[k]-th row in feats
        """
        self.pts = pts
        self.feats = feats
        self.pt_to_feat_idxs = pt_to_feat_idxs
        self.device = device

    def to(self, device):
        self.device = device
        self.pts = self.pts.to(device)
        self.feats = self.feats.to(device)
        self.pt_to_feat_idxs = self.pt_to_feat_idxs.to(device)
        return self