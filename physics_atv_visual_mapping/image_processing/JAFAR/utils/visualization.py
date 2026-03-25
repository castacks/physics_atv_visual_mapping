from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.decomposition import PCA

@torch.no_grad()
def viz_pca(
    image,
    lr,
    hr_or_seg,
    legend=["Image", "HR Features", "Pred Features"],
    save_path=None,
    is_segmentation=False,
    num_classes=None,
):
    """
    Plots features or segmentation results in a grid format.

    Args:
        image (torch.Tensor): The input image tensor (C, H, W).
        lr (torch.Tensor): The low-resolution features or HR features (C, H, W).
        hr_or_seg (list[torch.Tensor]): List of HR features or segmentation masks.
        legend (list[str]): Titles for each subplot.
        save_path (str): Path to save the plot.
        is_segmentation (bool): Whether to plot segmentation masks instead of features.
        num_classes (int): Number of classes for segmentation (used for color mapping).
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    # Ensure hr_or_seg is a list
    if not isinstance(hr_or_seg, list):
        hr_or_seg = [hr_or_seg]

    # Check input dimensions
    assert len(image.shape) == 3

    # Prepare inputs for PCA
    if not is_segmentation:
        feats_for_pca = [lr.unsqueeze(0)] + [h.unsqueeze(0) for h in hr_or_seg]
        reduced_feats, _ = pca(feats_for_pca)  # pca outputs a list of reduced tensors

        lr_img = reduced_feats[0]
        hr_imgs = reduced_feats[1:]
    else:
        # Use segmentation masks directly
        lr_img = lr
        hr_imgs = hr_or_seg


    return lr_img[0], hr_imgs[0]

@torch.no_grad()
def plot_feats(
    image,
    lr,
    hr_or_seg,
    legend=["Image", "HR Features", "Pred Features"],
    save_path=None,
    is_segmentation=False,
    num_classes=None,
):
    """
    Plots features or segmentation results in a grid format.

    Args:
        image (torch.Tensor): The input image tensor (C, H, W).
        lr (torch.Tensor): The low-resolution features or HR features (C, H, W).
        hr_or_seg (list[torch.Tensor]): List of HR features or segmentation masks.
        legend (list[str]): Titles for each subplot.
        save_path (str): Path to save the plot.
        is_segmentation (bool): Whether to plot segmentation masks instead of features.
        num_classes (int): Number of classes for segmentation (used for color mapping).
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    # Ensure hr_or_seg is a list
    if not isinstance(hr_or_seg, list):
        hr_or_seg = [hr_or_seg]

    # Check input dimensions
    assert len(image.shape) == 3

    # Prepare inputs for PCA
    if not is_segmentation:
        feats_for_pca = [lr.unsqueeze(0)] + [h.unsqueeze(0) for h in hr_or_seg]
        reduced_feats, _ = pca(feats_for_pca)  # pca outputs a list of reduced tensors

        lr_img = reduced_feats[0]
        hr_imgs = reduced_feats[1:]
    else:
        # Use segmentation masks directly
        lr_img = lr
        hr_imgs = hr_or_seg

    # --- Plot ---
    n_cols = 2 + len(hr_imgs)  # image + lr + multiple hr
    fig, ax = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    ax[0].imshow(image.permute(1, 2, 0).detach().cpu())
    ax[0].set_title(legend[0])

    # Plot the low-resolution features or segmentation mask
    if not is_segmentation:
        ax[1].imshow(lr_img[0].permute(1, 2, 0).detach().cpu())
    else:
        cmap = plt.get_cmap("Paired", num_classes)
        norm = mcolors.Normalize(vmin=0, vmax=num_classes - 1) if num_classes else None
        ax[1].imshow(lr_img.detach().cpu(), cmap=cmap, norm=norm)
    ax[1].set_title(legend[1])

    # Plot HR features or segmentation masks
    for idx, hr_item in enumerate(hr_imgs):
        if not is_segmentation:
            ax[idx + 2].imshow(hr_item[0].permute(1, 2, 0).detach().cpu())
        else:
            ax[idx + 2].imshow(hr_item.detach().cpu(), cmap=cmap, norm=norm)
        if len(legend) > idx + 2:
            ax[idx + 2].set_title(legend[idx + 2])
        else:
            ax[idx + 2].set_title(f"HR Features {idx}" if not is_segmentation else f"Segmentation {idx}")

    remove_axes(ax)

    # Save each axis independently
    if save_path:
        # Save the combined plot
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)  # Close the figure to free memory

        save_dir = Path(save_path).with_suffix("") / "individual_axes"
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, axis in enumerate(ax):
            axis_legend = legend[i] if i < len(legend) else f"Axis_{i}"
            axis_save_path = save_dir / f"{axis_legend.replace(' ', '_').lower()}.png"
            fig_single, ax_single = plt.subplots(figsize=(5, 5))
            ax_single.imshow(axis.images[0].get_array(), cmap=axis.images[0].get_cmap(), norm=axis.images[0].norm)
            ax_single.set_title(axis_legend)
            plt.savefig(axis_save_path, bbox_inches="tight", dpi=300)
            plt.close(fig_single)

    else:
        plt.show()


def _remove_axes(ax):
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xticks([])
    ax.set_yticks([])


def remove_axes(axes):
    if len(axes.shape) == 2:
        for ax1 in axes:
            for ax in ax1:
                _remove_axes(ax)
    else:
        for ax in axes:
            _remove_axes(ax)


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        if len(image2.shape) == 4:
            # batched
            image2 = image2.permute(1, 0, 2, 3)
        for t, m, s in zip(image2, self.mean, self.std):
            t = t * s + m  # .mul_(s).add_(m)
        return image2.permute(1, 0, 2, 3)


norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class TorchPCA(object):

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)
        U, S, V = torch.pca_lowrank(unbiased, q=self.n_components, center=False, niter=4)
        self.components_ = V.T
        self.singular_values_ = S
        return self

    def transform(self, X):
        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_.T
        return projected


def pca(image_feats_list, dim=3, fit_pca=None, use_torch_pca=True, max_samples=None):
    device = image_feats_list[0].device

    def flatten(tensor, target_size=None):
        if target_size is not None and fit_pca is None:
            tensor = F.interpolate(tensor, (target_size, target_size), mode="area")
        B, C, H, W = tensor.shape
        return tensor.permute(1, 0, 2, 3).reshape(C, B * H * W).permute(1, 0).detach().cpu()

    if len(image_feats_list) > 1 and fit_pca is None:
        target_size = image_feats_list[0].shape[2]
    else:
        target_size = None

    flattened_feats = []
    for feats in image_feats_list:
        flattened_feats.append(flatten(feats, target_size))
    x = torch.cat(flattened_feats, dim=0)

    # Subsample the data if max_samples is set and the number of samples exceeds max_samples
    if max_samples is not None and x.shape[0] > max_samples:
        indices = torch.randperm(x.shape[0])[:max_samples]
        x = x[indices]

    if fit_pca is None:
        if use_torch_pca:
            fit_pca = TorchPCA(n_components=dim).fit(x)
        else:
            fit_pca = PCA(n_components=dim).fit(x)

    reduced_feats = []
    for feats in image_feats_list:
        x_red = fit_pca.transform(flatten(feats))
        if isinstance(x_red, np.ndarray):
            x_red = torch.from_numpy(x_red)
        x_red -= x_red.min(dim=0, keepdim=True).values
        x_red /= x_red.max(dim=0, keepdim=True).values
        B, C, H, W = feats.shape
        reduced_feats.append(x_red.reshape(B, H, W, dim).permute(0, 3, 1, 2).to(device))

    return reduced_feats, fit_pca


def plot_image_label_prediction(image, bilinear_feats, pred_feats, label, pred_prob, backbone, save_dir=None):
    """
    Plots the original image, PCA of bilinear features, PCA of predictions,
    label, prediction, and an overlay of the original image with predicted labels.
    Optionally saves all images in a common folder.

    Args:
        image (torch.Tensor): The input image tensor (C, H, W).
        bilinear_feats (torch.Tensor): Bilinear features tensor.
        pred_feats (torch.Tensor): Predicted features tensor.
        label (torch.Tensor): Ground truth label tensor.
        pred_prob (torch.Tensor): Predicted probabilities tensor.
        backbone: Backbone model for unnormalizing the image.
        save_dir (str or Path, optional): Directory to save the images. If None, images are not saved.
    """

    # Unnormalize the image using the backbone's mean and std
    img = image.cpu().clone()
    for t, m, s in zip(img, backbone.config["mean"], backbone.config["std"]):
        t.mul_(s).add_(m)

    # Perform PCA on both feature types in the same space
    combined_feats = [bilinear_feats, pred_feats.cpu().detach()]
    reduced_feats, _ = pca(combined_feats, dim=3)

    # Process PCA results
    bilinear_pca = reduced_feats[0].squeeze(0).permute(1, 2, 0)
    ms_pred_pca = reduced_feats[1].squeeze(0).permute(1, 2, 0)

    # Create the save directory if it doesn't exist
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Plot and optionally save each image
    def save_and_show(image, title):
        if save_dir:
            save_path = save_dir / f"{title.replace(' ', '_').lower()}.png"
            plt.imsave(save_path, image)
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")

    plt.figure(figsize=(25, 5))

    # Original image
    plt.subplot(1, 6, 1)
    save_and_show(img.permute(1, 2, 0).clip(0, 1).numpy(), "Original Image")

    # Bilinear features PCA
    plt.subplot(1, 6, 2)
    save_and_show(bilinear_pca.numpy(), "LR Features PCA")

    # MS Predictions PCA
    plt.subplot(1, 6, 3)
    save_and_show(ms_pred_pca.numpy(), "JAFAR PCA")

    # Label
    plt.subplot(1, 6, 4)
    cmap = plt.get_cmap("Paired")
    norm = mcolors.Normalize(vmin=0, vmax=label.max().item())
    save_and_show(cmap(norm(label.cpu().squeeze().numpy())), "Label")

    # Prediction
    plt.subplot(1, 6, 5)
    save_and_show(cmap(norm(pred_prob.cpu().squeeze().numpy())), "Prediction")

    # Overlay of original image with predicted labels
    plt.subplot(1, 6, 6)
    overlay = img.permute(1, 2, 0).clip(0, 1).numpy()
    pred_overlay = cmap(norm(pred_prob.cpu().squeeze().numpy()))
    overlay_with_pred = 0.3 * overlay + 0.7 * pred_overlay[..., :3]  # Combine with opacity
    save_and_show(overlay_with_pred, "Overlay with Prediction")

    if not save_dir:
        plt.tight_layout()
        plt.show()
