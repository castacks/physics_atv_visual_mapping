import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class GradientLoss(nn.Module):
    """GradientLoss.

    Adapted from https://www.cs.cornell.edu/projects/megadepth/

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
    """

    def __init__(self, valid_mask=True, loss_weight=1.0, max_depth=None, loss_name="loss_grad"):
        super(GradientLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth
        self.loss_name = loss_name

        self.eps = 0.001  # avoid grad explode

    def gradientloss(self, input, target):
        input_downscaled = [input] + [input[:, :, :: 2 * i, :: 2 * i] for i in range(1, 4)]
        target_downscaled = [target] + [target[:, :, :: 2 * i, :: 2 * i] for i in range(1, 4)]

        gradient_loss = 0
        for input, target in zip(input_downscaled, target_downscaled):
            if self.valid_mask:
                mask = target > 0
                if self.max_depth is not None:
                    mask = torch.logical_and(target > 0, target <= self.max_depth)
                N = torch.sum(mask)
            else:
                mask = torch.ones_like(target)
                N = input.numel()
            input_log = torch.log(input + self.eps)
            target_log = torch.log(target + self.eps)
            log_d_diff = input_log - target_log

            log_d_diff = torch.mul(log_d_diff, mask)

            v_gradient = torch.abs(log_d_diff[0:-2, :] - log_d_diff[2:, :])
            v_mask = torch.mul(mask[0:-2, :], mask[2:, :])
            v_gradient = torch.mul(v_gradient, v_mask)

            h_gradient = torch.abs(log_d_diff[:, 0:-2] - log_d_diff[:, 2:])
            h_mask = torch.mul(mask[:, 0:-2], mask[:, 2:])
            h_gradient = torch.mul(h_gradient, h_mask)

            gradient_loss += (torch.sum(h_gradient) + torch.sum(v_gradient)) / N

        return gradient_loss

    def forward(self, depth_pred, depth_gt):
        """Forward function."""

        gradient_loss = self.loss_weight * self.gradientloss(depth_pred, depth_gt)
        return gradient_loss


class SigLoss(nn.Module):
    """SigLoss.

        This follows `AdaBins <https://arxiv.org/abs/2011.14141>`_.

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
        warm_up (bool): A simple warm up stage to help convergence. Default: False.
        warm_iter (int): The number of warm up stage. Default: 100.
    """

    def __init__(
        self,
        valid_mask=True,
        loss_weight=1.0,
        max_depth=None,
        warm_up=False,
        warm_iter=100,
        loss_name="sigloss",
    ):
        super(SigLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth
        self.loss_name = loss_name

        self.eps = 0.001  # avoid grad explode

        # HACK: a hack implementation for warmup sigloss
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

    def sigloss(self, input, target):
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
            input = input[valid_mask]
            target = target[valid_mask]

        if self.warm_up:
            if self.warm_up_counter < self.warm_iter:
                g = torch.log(input + self.eps) - torch.log(target + self.eps)
                g = 0.15 * torch.pow(torch.mean(g), 2)
                self.warm_up_counter += 1
                return torch.sqrt(g)

        g = torch.log(input + self.eps) - torch.log(target + self.eps)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)

    def forward(self, depth_pred, depth_gt):
        """Forward function."""

        loss_depth = self.loss_weight * self.sigloss(depth_pred, depth_gt)
        return loss_depth


class Cosine_MSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.cosine_loss = torch.nn.CosineEmbeddingLoss()

    def forward(self, pred, target):
        pred = rearrange(pred, "b c h w -> (b h w) c")
        target = rearrange(target, "b c h w -> (b h w) c")

        gt = torch.ones_like(target[:, 0])

        # If you must normalize (example: min-max scaling)
        min_val = torch.min(target, dim=1, keepdim=True).values
        max_val = torch.max(target, dim=1, keepdim=True).values
        pred_normalized = (pred - min_val) / (max_val - min_val + 1e-6)
        target_normalized = (target - min_val) / (max_val - min_val + 1e-6)

        return self.cosine_loss(pred, target, gt) + self.mse_loss(pred_normalized, target_normalized)


class Loss(nn.Module):

    def __init__(
        self,
        loss_type,
        dim=384,
    ):
        super().__init__()
        self.dim = dim

        if loss_type == "cosine_mse":
            loss = Cosine_MSE()
        else:
            raise NotImplementedError(f"Loss type {loss_type} not implemented")

        self.loss_func = loss

    def __call__(self, pred, target):
        loss = self.loss_func(pred, target)
        return {"total": loss}
