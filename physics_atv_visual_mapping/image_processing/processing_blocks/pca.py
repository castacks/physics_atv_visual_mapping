import torch

from physics_atv_visual_mapping.image_processing.processing_blocks.base import ImageProcessingBlock

class PCABlock(ImageProcessingBlock):
    """
    Block that applies PCA to the image
    """
    def __init__(self, fp, device):
        self.compute_pca = fp == "on_the_fly"
        self.pca = None
        if not self.compute_pca:
            self.pca = {k:v.to(device) for k,v in torch.load(fp).items()}

    def run(self, image, intrinsics, image_orig):
        if image.dim() == 3:
            image = image.unsqueeze(1)
        image_feats = image.flatten(start_dim=-2).permute(0, 2, 1)
        
        if self.compute_pca:
            # Compute PCA on the fly
            _pmean = image_feats.mean(dim=1, keepdim=True)
            img_norm = image_feats - _pmean
            print("img_norm shape", img_norm.shape)
            img_cov = img_norm.transpose(1, 2) @ img_norm / img_norm.size(0)
            print("img_cov shape", img_cov.shape)
            _, _, _pv = torch.svd(img_cov.float())
            _pv = _pv[:, :16].half().transpose(1, 2)
        else:
            _pmean = self.pca['mean'].view(1, 1, -1)
            _pv = self.pca['V'].unsqueeze(0)

        print("pmean shape", _pmean.shape)
        print("pv shape", _pv.shape)

        # Move to channels-last
        img_norm = image_feats - _pmean
        img_pca = img_norm @ _pv
        img_out = img_pca.permute(0, 2, 1).view(image.shape[0], _pv.shape[-1], image.shape[-2], image.shape[-1])

        return img_out, intrinsics

    # def run(self, image, intrinsics, image_orig):
        
    #     # add an option for computing the PCA on the fly
        
        
    #     _pmean = self.pca['mean'].view(1, 1, -1)
    #     _pv = self.pca['V'].unsqueeze(0)
        
    #     print("pmean shape", _pmean.shape)
    #     print("pv shape", _pv.shape)

    #     #move to channels-last
    #     if image.dim() == 3:
    #         image = image.unsqueeze(1)
    #     image_feats = image.flatten(start_dim=-2).permute(0, 2, 1)
    #     img_norm = image_feats - _pmean
    #     img_pca = img_norm @ _pv
    #     img_out = img_pca.permute(0,2,1).view(image.shape[0], _pv.shape[-1], image.shape[-2], image.shape[-1])

    #     return img_out, intrinsics
