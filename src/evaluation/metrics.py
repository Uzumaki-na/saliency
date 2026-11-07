"""Multi-dataset perceptual evaluation pipeline and saliency representation benchmarks.

Implements ROI and global perceptual distortion metrics:
- ROI-PSNR, BG-PSNR, Global-PSNR (with Jensen-consistent batch averaging)
- ROI-MS-SSIM, BG-MS-SSIM, Global-MS-SSIM (Wang et al. 2003, 5-scale formulation)
- ROI-LPIPS, BG-LPIPS, Global-LPIPS (Zhang et al. 2018 deep feature distance)

Implements visual saliency representation benchmark metrics:
- AUC-Judd (Judd et al. 2012 / MIT Benchmark)
- NSS (Normalized Scanpath Saliency; Peters et al. 2005)
- CC (Pearson Correlation Coefficient)
- KL Divergence (Kullback-Leibler)
- SIM (Histogram Similarity)

Supports evaluation on Kodak (with independent DeepGaze IIE masks)
and MIT1003 (with human eye-tracking fixation test splits).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

# NumPy trapezoid compatibility helper across 1.x and 2.x
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


@dataclass
class PerceptualDistortion:
    """Perceptual distortion metrics partitioned across ROI, Background, and Global image.

    Attributes:
        roi: Metric value computed over the Region of Interest (top fixation mask).
        bg: Metric value computed over the background region (complement of ROI mask).
        global_val: Metric value computed over the entire valid image area.
    """

    roi: float
    bg: float
    global_val: float

    def __repr__(self) -> str:
        return f"PerceptualDistortion(roi={self.roi:.4f}, bg={self.bg:.4f}, global={self.global_val:.4f})"


@dataclass
class SaliencyRepresentationMetrics:
    """Benchmark evaluation metrics for continuous and discrete visual saliency fidelity.

    Attributes:
        auc_judd: Area Under ROC Curve against discrete eye fixations (Judd et al. 2012).
        nss: Normalized Scanpath Saliency against discrete eye fixations (Peters et al. 2005).
        cc: Pearson Linear Correlation Coefficient against continuous teacher saliency.
        kl: Kullback-Leibler divergence D_KL(S_target || S_pred) as spatial probability.
        sim: Spatial histogram similarity intersection between normalized saliency maps.
    """

    auc_judd: float
    nss: float
    cc: float
    kl: float
    sim: float

    def __repr__(self) -> str:
        return (
            f"SaliencyRepresentationMetrics(auc_judd={self.auc_judd:.4f}, nss={self.nss:.4f}, "
            f"cc={self.cc:.4f}, kl={self.kl:.4f}, sim={self.sim:.4f})"
        )


@dataclass
class ImageEvaluationResult:
    """Complete evaluation report for a single image reconstruction.

    Attributes:
        psnr: ROI, background, and global PSNR in decibels (dB).
        msssim: ROI, background, and global Multi-Scale SSIM.
        lpips: ROI, background, and global LPIPS perceptual distance.
        saliency: Optional saliency representation benchmark metrics.
        bpp: Optional rate in bits per pixel.
    """

    psnr: PerceptualDistortion
    msssim: PerceptualDistortion
    lpips: PerceptualDistortion
    saliency: Optional[SaliencyRepresentationMetrics] = None
    bpp: Optional[float] = None


@dataclass
class DatasetEvaluationResult:
    """Aggregated dataset-level evaluation summary across test images.

    Attributes:
        mean_psnr: Dataset mean ROI, background, and global PSNR.
        mean_msssim: Dataset mean ROI, background, and global MS-SSIM.
        mean_lpips: Dataset mean ROI, background, and global LPIPS.
        mean_saliency: Dataset mean saliency representation metrics.
        mean_bpp: Dataset mean bitrate in bits per pixel.
        per_image_results: Individual image evaluation records.
    """

    mean_psnr: PerceptualDistortion
    mean_msssim: PerceptualDistortion
    mean_lpips: PerceptualDistortion
    mean_saliency: Optional[SaliencyRepresentationMetrics] = None
    mean_bpp: Optional[float] = None
    per_image_results: List[ImageEvaluationResult] = None  # type: ignore[assignment]


def _prepare_image_inputs(
    x: Union[torch.Tensor, np.ndarray],
    x_hat: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Standardize image and mask inputs into 4D FP32 PyTorch tensors (B, C, H, W)."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(x_hat, np.ndarray):
        x_hat = torch.from_numpy(x_hat)

    target_device = device if device is not None else x.device

    # Ensure float32 detached tensors
    x_t = x.detach().float().to(target_device)
    x_hat_t = x_hat.detach().float().to(target_device)

    # Standardize dimensions to 4D (B, C, H, W)
    if x_t.ndim == 2:
        x_t = x_t.unsqueeze(0).unsqueeze(0)
    elif x_t.ndim == 3:
        if x_t.shape[0] in (1, 3) and x_t.shape[1] > 3 and x_t.shape[2] > 3:
            # (C, H, W) -> unsqueeze batch: (1, C, H, W)
            x_t = x_t.unsqueeze(0)
        elif x_t.shape[-1] in (1, 3):
            # (H, W, C) -> permute to (C, H, W) and unsqueeze batch: (1, C, H, W)
            x_t = x_t.permute(2, 0, 1).unsqueeze(0)
        else:
            # (B, H, W) batched 2D grayscale -> unsqueeze channel: (B, 1, H, W)
            x_t = x_t.unsqueeze(1)

    if x_hat_t.ndim == 2:
        x_hat_t = x_hat_t.unsqueeze(0).unsqueeze(0)
    elif x_hat_t.ndim == 3:
        if x_hat_t.shape[0] in (1, 3) and x_hat_t.shape[1] > 3 and x_hat_t.shape[2] > 3:
            x_hat_t = x_hat_t.unsqueeze(0)
        elif x_hat_t.shape[-1] in (1, 3):
            x_hat_t = x_hat_t.permute(2, 0, 1).unsqueeze(0)
        else:
            x_hat_t = x_hat_t.unsqueeze(1)

    if x_t.shape != x_hat_t.shape:
        raise ValueError(f"Shape mismatch: x has shape {tuple(x_t.shape)}, x_hat has shape {tuple(x_hat_t.shape)}.")

    mask_t: Optional[torch.Tensor] = None
    if mask is not None:
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        m = mask.detach().float().to(target_device)
        if m.ndim == 2:
            m = m.unsqueeze(0).unsqueeze(0)
        elif m.ndim == 3:
            if m.shape[0] in (1, 3) and m.shape[1] > 3 and m.shape[2] > 3:
                m = m.unsqueeze(0)
            elif m.shape[-1] == 1:
                m = m.permute(2, 0, 1).unsqueeze(0)
            else:
                m = m.unsqueeze(1)

        if m.shape[1] != 1:
            m = m[:, :1, :, :]

        if m.shape[0] == 1 and x_t.shape[0] > 1:
            m = m.expand(x_t.shape[0], -1, -1, -1)
        elif m.shape[0] != x_t.shape[0]:
            raise ValueError(f"Batch mismatch: mask has {m.shape[0]}, x has {x_t.shape[0]}.")

        # Resize mask if spatial dimensions differ
        if m.shape[-2:] != x_t.shape[-2:]:
            m = F.interpolate(m, size=x_t.shape[-2:], mode="bilinear", align_corners=False)

        mask_t = torch.clamp(m, 0.0, 1.0)

    return x_t, x_hat_t, mask_t


# ---------------------------------------------------------------------------
# 1. ROI and Global PSNR
# ---------------------------------------------------------------------------


def compute_roi_psnr(
    x: Union[torch.Tensor, np.ndarray],
    x_hat: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    data_range: float = 1.0,
    max_psnr: float = 100.0,
    eps: float = 1e-15,
) -> PerceptualDistortion:
    """Compute ROI-PSNR, Background-PSNR, and Global-PSNR in decibels (dB).

    For batched inputs (B > 1), computes per-sample PSNR before averaging across
    the batch, strictly adhering to Jensen's inequality and MPEG/CompressAI conventions.

    Args:
        x: Ground truth target image in [0, 1].
        x_hat: Reconstructed image matching x in dimensions.
        mask: Optional binary or continuous ROI mask in [0, 1]. If None, all regions equal global.
        data_range: Dynamic peak signal range (1.0 for normalized images).
        max_psnr: Maximum upper bound clamp returned when MSE is near zero.
        eps: Small epsilon to prevent division by zero.

    Returns:
        PerceptualDistortion containing roi, bg, and global_val PSNR in dB.
    """
    x_t, x_hat_t, m_t = _prepare_image_inputs(x, x_hat, mask)

    # Respect Jensen's inequality for batched inputs: average PSNR across samples
    if x_t.shape[0] > 1:
        b_size = x_t.shape[0]
        roi_vals, bg_vals, glob_vals = [], [], []
        for b in range(b_size):
            m_b = m_t[b : b + 1] if m_t is not None else None
            res_b = compute_roi_psnr(
                x_t[b : b + 1],
                x_hat_t[b : b + 1],
                mask=m_b,
                data_range=data_range,
                max_psnr=max_psnr,
                eps=eps,
            )
            roi_vals.append(res_b.roi)
            bg_vals.append(res_b.bg)
            glob_vals.append(res_b.global_val)
        return PerceptualDistortion(
            roi=float(np.mean(roi_vals)),
            bg=float(np.mean(bg_vals)),
            global_val=float(np.mean(glob_vals)),
        )

    diff_sq = (x_t - x_hat_t) ** 2
    global_mse = float(diff_sq.mean().item())
    global_psnr = max_psnr if global_mse <= eps else min(10.0 * math.log10((data_range**2) / global_mse), max_psnr)

    if m_t is None:
        return PerceptualDistortion(roi=global_psnr, bg=global_psnr, global_val=global_psnr)

    # Expand mask across color channels (B, C, H, W)
    c = x_t.shape[1]
    m_expanded = m_t.expand(-1, c, -1, -1)
    bg_mask = 1.0 - m_expanded

    sum_m = float(m_expanded.sum().item())
    sum_bg = float(bg_mask.sum().item())

    if sum_m > eps:
        roi_mse = float((diff_sq * m_expanded).sum().item()) / sum_m
        roi_psnr = max_psnr if roi_mse <= eps else min(10.0 * math.log10((data_range**2) / roi_mse), max_psnr)
    else:
        roi_psnr = global_psnr

    if sum_bg > eps:
        bg_mse = float((diff_sq * bg_mask).sum().item()) / sum_bg
        bg_psnr = max_psnr if bg_mse <= eps else min(10.0 * math.log10((data_range**2) / bg_mse), max_psnr)
    else:
        bg_psnr = global_psnr

    return PerceptualDistortion(roi=roi_psnr, bg=bg_psnr, global_val=global_psnr)


# ---------------------------------------------------------------------------
# 2. ROI and Global MS-SSIM
# ---------------------------------------------------------------------------


def _fspecial_gauss_1d(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    """Create normalized 1D Gaussian filter."""
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
    gauss = torch.exp(-(coords**2) / (2.0 * sigma**2))
    return gauss / gauss.sum()


def compute_roi_msssim(
    x: Union[torch.Tensor, np.ndarray],
    x_hat: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    levels: int = 5,
    scale_weights: Optional[Sequence[float]] = None,
    eps: float = 1e-12,
) -> PerceptualDistortion:
    """Compute Multi-Scale Structural Similarity (MS-SSIM) partitioned over ROI, BG, and Global.

    Evaluates contrast and structure measures across dyadic downsampling scales,
    weighting them by spatial masks according to the canonical 5-scale formulation
    (Wang et al. 2003 Eq. 7), incorporating both luminance and contrast-structure at the coarsest scale.

    Args:
        x: Target image in [0, 1].
        x_hat: Reconstructed image matching x.
        mask: Optional spatial ROI mask in [0, 1].
        data_range: Dynamic range (1.0).
        window_size: Gaussian kernel window size (default 11).
        sigma: Standard deviation of Gaussian kernel (default 1.5).
        levels: Number of downsampling scales (default 5).
        scale_weights: Exponents for each scale (default standard Wang et al. 5-scale weights).
        eps: Epsilon for numerical stability.

    Returns:
        PerceptualDistortion with roi, bg, and global_val MS-SSIM scores in [0.0, 1.0].
    """
    x_t, x_hat_t, m_t = _prepare_image_inputs(x, x_hat, mask)
    device = x_t.device
    c = x_t.shape[1]

    # Validate minimum spatial dimensions for dyadic downsampling
    min_dim = 2 ** (levels - 1)
    if x_t.shape[-2] < min_dim or x_t.shape[-1] < min_dim:
        raise ValueError(
            f"Image spatial dimensions {x_t.shape[-2]}x{x_t.shape[-1]} are too small for "
            f"{levels}-level MS-SSIM; minimum is {min_dim}x{min_dim}."
        )

    if scale_weights is None:
        weights_tuple = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    else:
        weights_tuple = tuple(scale_weights)

    if len(weights_tuple) != levels:
        raise ValueError(f"scale_weights length {len(weights_tuple)} does not match levels {levels}.")

    # Gaussian 2D separable filter
    g1d = _fspecial_gauss_1d(window_size, sigma, device=device)
    g2d = torch.outer(g1d, g1d)
    kernel = g2d.expand(c, 1, window_size, window_size).contiguous()
    padding = window_size // 2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mcs_roi_list: List[torch.Tensor] = []
    mcs_bg_list: List[torch.Tensor] = []
    mcs_glob_list: List[torch.Tensor] = []

    l_roi_final: Optional[torch.Tensor] = None
    l_bg_final: Optional[torch.Tensor] = None
    l_glob_final: Optional[torch.Tensor] = None

    x_curr = x_t
    x_hat_curr = x_hat_t
    m_curr = m_t

    for scale_idx in range(levels):
        # Reflection padding prevents boundary darkening artifacts when dimensions permit;
        # replicate padding fallback ensures valid kernel overlap at coarse downsampled scales
        if x_curr.shape[-2] > padding and x_curr.shape[-1] > padding:
            x_pad = F.pad(x_curr, (padding, padding, padding, padding), mode="reflect")
            x_hat_pad = F.pad(x_hat_curr, (padding, padding, padding, padding), mode="reflect")
        else:
            x_pad = F.pad(x_curr, (padding, padding, padding, padding), mode="replicate")
            x_hat_pad = F.pad(x_hat_curr, (padding, padding, padding, padding), mode="replicate")

        mu_x = F.conv2d(x_pad, kernel, padding=0, groups=c)
        mu_y = F.conv2d(x_hat_pad, kernel, padding=0, groups=c)
        mu_x_sq = mu_x**2
        mu_y_sq = mu_y**2
        mu_xy = mu_x * mu_y

        sig_x_sq = torch.clamp(F.conv2d(x_pad * x_pad, kernel, padding=0, groups=c) - mu_x_sq, min=0.0)
        sig_y_sq = torch.clamp(F.conv2d(x_hat_pad * x_hat_pad, kernel, padding=0, groups=c) - mu_y_sq, min=0.0)
        sig_xy = F.conv2d(x_pad * x_hat_pad, kernel, padding=0, groups=c) - mu_xy

        cs_map = (2.0 * sig_xy + c2) / (sig_x_sq + sig_y_sq + c2)
        l_map = (2.0 * mu_xy + c1) / (mu_x_sq + mu_y_sq + c1)

        # Global average
        cs_glob = torch.clamp(cs_map.mean(dim=(1, 2, 3)), min=0.0)
        mcs_glob_list.append(cs_glob)

        if m_curr is not None:
            m_scaled = F.interpolate(m_curr, size=cs_map.shape[-2:], mode="bilinear", align_corners=False)
            m_exp = m_scaled.expand(-1, c, -1, -1)
            bg_exp = 1.0 - m_exp

            denom_roi = m_exp.sum(dim=(1, 2, 3))
            denom_bg = bg_exp.sum(dim=(1, 2, 3))

            # Guard against zero-sum ROI or BG mask: fallback to global
            cs_roi = torch.where(
                denom_roi > eps,
                torch.clamp((cs_map * m_exp).sum(dim=(1, 2, 3)) / (denom_roi + eps), min=0.0),
                cs_glob,
            )
            cs_bg = torch.where(
                denom_bg > eps,
                torch.clamp((cs_map * bg_exp).sum(dim=(1, 2, 3)) / (denom_bg + eps), min=0.0),
                cs_glob,
            )

            mcs_roi_list.append(cs_roi)
            mcs_bg_list.append(cs_bg)
        else:
            mcs_roi_list.append(cs_glob)
            mcs_bg_list.append(cs_glob)

        if scale_idx == levels - 1:
            l_glob_final = torch.clamp(l_map.mean(dim=(1, 2, 3)), min=0.0)
            if m_curr is not None:
                l_roi_final = torch.where(
                    denom_roi > eps,
                    torch.clamp((l_map * m_exp).sum(dim=(1, 2, 3)) / (denom_roi + eps), min=0.0),
                    l_glob_final,
                )
                l_bg_final = torch.where(
                    denom_bg > eps,
                    torch.clamp((l_map * bg_exp).sum(dim=(1, 2, 3)) / (denom_bg + eps), min=0.0),
                    l_glob_final,
                )
            else:
                l_roi_final = l_glob_final
                l_bg_final = l_glob_final

        if scale_idx < levels - 1:
            x_curr = F.avg_pool2d(x_curr, kernel_size=2, stride=2)
            x_hat_curr = F.avg_pool2d(x_hat_curr, kernel_size=2, stride=2)
            if m_curr is not None:
                m_curr = F.avg_pool2d(m_curr, kernel_size=2, stride=2)

    assert l_glob_final is not None and l_roi_final is not None and l_bg_final is not None

    # Wang et al. 2003 Eq. 7: MS-SSIM = (l_M ** beta_M) * prod_{j=1}^M (cs_j ** beta_j)
    overall_glob = l_glob_final ** weights_tuple[-1]
    overall_roi = l_roi_final ** weights_tuple[-1]
    overall_bg = l_bg_final ** weights_tuple[-1]

    for j in range(levels):
        overall_glob = overall_glob * (mcs_glob_list[j] ** weights_tuple[j])
        overall_roi = overall_roi * (mcs_roi_list[j] ** weights_tuple[j])
        overall_bg = overall_bg * (mcs_bg_list[j] ** weights_tuple[j])

    return PerceptualDistortion(
        roi=float(overall_roi.mean().item()),
        bg=float(overall_bg.mean().item()),
        global_val=float(overall_glob.mean().item()),
    )


# ---------------------------------------------------------------------------
# 3. ROI and Global LPIPS (Learned Perceptual Image Patch Similarity)
# ---------------------------------------------------------------------------


class LPIPSMetric(nn.Module):
    """Deep feature perceptual distance extractor with spatial mask pooling.

    Implements deep feature perceptual distance (Zhang et al. 2018) using a VGG16
    backbone with unit channel normalization and spatial pooling.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        try:
            from torchvision.models import VGG16_Weights, vgg16

            vgg = vgg16(weights=VGG16_Weights.DEFAULT if pretrained else None)
        except Exception as e:
            logger.info("Initializing VGG16 backbone without network download: %s", e)
            from torchvision.models import vgg16

            vgg = vgg16(weights=None)

        vgg_features = vgg.features
        # Slice into 5 perceptual stages
        self.slice1 = nn.Sequential(*[vgg_features[x] for x in range(4)])  # relu1_2
        self.slice2 = nn.Sequential(*[vgg_features[x] for x in range(4, 9)])  # relu2_2
        self.slice3 = nn.Sequential(*[vgg_features[x] for x in range(9, 16)])  # relu3_3
        self.slice4 = nn.Sequential(*[vgg_features[x] for x in range(16, 23)])  # relu4_3
        self.slice5 = nn.Sequential(*[vgg_features[x] for x in range(23, 30)])  # relu5_3

        # Freeze all backbone parameters
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

        # ImageNet normalization constants
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize [0, 1] RGB image to ImageNet standardization."""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return (x - self.mean) / self.std

    @staticmethod
    def _unit_normalize_channels(feat: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """Normalize feature activations to unit L2 norm along the channel axis."""
        norm = torch.sqrt(torch.sum(feat**2, dim=1, keepdim=True) + eps)
        return feat / norm

    def forward(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        eps: float = 1e-10,
    ) -> PerceptualDistortion:
        """Compute LPIPS perceptual distance partitioned over ROI, BG, and Global."""
        x_norm = self._normalize(x)
        x_hat_norm = self._normalize(x_hat)

        # Forward pass through VGG slices
        slices = [self.slice1, self.slice2, self.slice3, self.slice4, self.slice5]
        h_x = x_norm
        h_hat = x_hat_norm

        total_glob: float = 0.0
        total_roi: float = 0.0
        total_bg: float = 0.0

        for slice_layer in slices:
            h_x = slice_layer(h_x)
            h_hat = slice_layer(h_hat)

            norm_x = self._unit_normalize_channels(h_x, eps=eps)
            norm_hat = self._unit_normalize_channels(h_hat, eps=eps)

            # Spatial squared difference map (B, 1, H_l, W_l)
            diff_sq = torch.sum((norm_x - norm_hat) ** 2, dim=1, keepdim=True)

            glob_layer = float(diff_sq.mean().item())
            total_glob += glob_layer

            if mask is not None:
                m_layer = F.interpolate(mask, size=diff_sq.shape[-2:], mode="bilinear", align_corners=False)
                bg_layer = 1.0 - m_layer

                denom_roi = float(m_layer.sum().item())
                denom_bg = float(bg_layer.sum().item())

                # Safe fallback to global if mask is zero-sum
                roi_layer = (
                    float((diff_sq * m_layer).sum().item()) / max(denom_roi, eps) if denom_roi > eps else glob_layer
                )
                bg_layer_val = (
                    float((diff_sq * bg_layer).sum().item()) / max(denom_bg, eps) if denom_bg > eps else glob_layer
                )

                total_roi += roi_layer
                total_bg += bg_layer_val
            else:
                total_roi += glob_layer
                total_bg += glob_layer

        return PerceptualDistortion(roi=total_roi, bg=total_bg, global_val=total_glob)


_GLOBAL_LPIPS_METRIC: Optional[LPIPSMetric] = None


def get_lpips_metric(device: Optional[torch.device] = None) -> LPIPSMetric:
    """Retrieve or initialize singleton LPIPS metric instance."""
    global _GLOBAL_LPIPS_METRIC
    if _GLOBAL_LPIPS_METRIC is None:
        _GLOBAL_LPIPS_METRIC = LPIPSMetric(pretrained=True)
    if device is not None:
        _GLOBAL_LPIPS_METRIC = _GLOBAL_LPIPS_METRIC.to(device)
    return _GLOBAL_LPIPS_METRIC


def compute_roi_lpips(
    x: Union[torch.Tensor, np.ndarray],
    x_hat: Union[torch.Tensor, np.ndarray],
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    lpips_model: Optional[LPIPSMetric] = None,
) -> PerceptualDistortion:
    """Compute ROI, Background, and Global LPIPS perceptual distance.

    Args:
        x: Target image in [0, 1].
        x_hat: Reconstructed image in [0, 1].
        mask: Optional spatial ROI mask in [0, 1].
        lpips_model: Optional pre-instantiated LPIPSMetric model.

    Returns:
        PerceptualDistortion with roi, bg, and global_val LPIPS distances (lower is better).
    """
    x_t, x_hat_t, m_t = _prepare_image_inputs(x, x_hat, mask)
    model = lpips_model if lpips_model is not None else get_lpips_metric(device=x_t.device)
    model = model.to(x_t.device)

    with torch.no_grad():
        return model(x_t, x_hat_t, mask=m_t)


# ---------------------------------------------------------------------------
# 4. Saliency Representation Metrics
# ---------------------------------------------------------------------------


def compute_auc_judd(
    s_pred: Union[torch.Tensor, np.ndarray],
    fixations: Union[torch.Tensor, np.ndarray],
    valid_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    jitter: bool = True,
    seed: int = 42,
) -> float:
    """Compute Area Under ROC Curve (AUC-Judd) against discrete eye fixations.

    Follows the standard MIT Saliency Benchmark definition (Judd et al. 2012).
    Evaluates True Positive Rate (TPR) and False Positive Rate (FPR) across thresholds
    sampled at fixated pixel values. Correctly handles flat maps and duplicate thresholds.

    Args:
        s_pred: Predicted continuous visual saliency map in [0, 1].
        fixations: Binary ground-truth eye-tracking fixation map (1 for fixation, 0 otherwise).
        valid_mask: Optional mask indicating unpadded valid image coordinates.
        jitter: If True, adds minimal Gaussian jitter (1e-7) to break ties in flat regions.
        seed: Random seed for deterministic jitter.

    Returns:
        Scalar AUC-Judd score in [0.0, 1.0]. Perfect prediction = 1.0; random = 0.50.
    """
    if isinstance(s_pred, torch.Tensor):
        s_arr = s_pred.detach().cpu().numpy().astype(np.float64)
    else:
        s_arr = np.asarray(s_pred, dtype=np.float64)

    if isinstance(fixations, torch.Tensor):
        f_arr = fixations.detach().cpu().numpy().astype(bool)
    else:
        f_arr = np.asarray(fixations, dtype=bool)

    if valid_mask is not None:
        if isinstance(valid_mask, torch.Tensor):
            v_arr = valid_mask.detach().cpu().numpy().astype(bool)
        else:
            v_arr = np.asarray(valid_mask, dtype=bool)
        if len(v_arr) == 0 or not np.any(v_arr):
            return 0.50
        s_arr = s_arr[v_arr]
        f_arr = f_arr[v_arr]
    else:
        s_arr = s_arr.ravel()
        f_arr = f_arr.ravel()

    n_fix = int(np.sum(f_arr))
    if n_fix == 0:
        return 0.50

    s_min, s_max = float(s_arr.min()), float(s_arr.max())
    # Flat saliency maps represent pure chance performance (0.50)
    if np.isclose(s_max, s_min, atol=1e-12):
        return 0.50

    if jitter:
        rng = np.random.default_rng(seed)
        s_arr = s_arr + rng.normal(0, 1e-7, size=s_arr.shape)
        s_min, s_max = float(s_arr.min()), float(s_arr.max())

    if s_max > s_min:
        s_arr = (s_arr - s_min) / (s_max - s_min)

    s_fix = s_arr[f_arr]
    s_non_fix = s_arr[~f_arr]

    n_non_fix = len(s_non_fix)
    if n_non_fix == 0:
        return 1.0

    s_fix_sorted = np.sort(s_fix)
    s_non_fix_sorted = np.sort(s_non_fix)

    # Unique thresholds in descending order
    unique_thresh = np.unique(s_fix)[::-1]

    # Evaluate true positives and false positives for each distinct threshold
    count_fix_ge = len(s_fix) - np.searchsorted(s_fix_sorted, unique_thresh, side="left")
    tpr = count_fix_ge / n_fix

    count_non_fix_ge = len(s_non_fix) - np.searchsorted(s_non_fix_sorted, unique_thresh, side="left")
    fpr = count_non_fix_ge / n_non_fix

    # Prepend (0, 0) and append (1, 1)
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])

    # Trapezoidal integration along ascending FPR
    order = np.argsort(fpr)
    fpr_sorted = fpr[order]
    tpr_sorted = tpr[order]

    auc = float(_trapezoid(tpr_sorted, fpr_sorted))
    return float(np.clip(auc, 0.0, 1.0))


def compute_nss(
    s_pred: Union[torch.Tensor, np.ndarray],
    fixations: Union[torch.Tensor, np.ndarray],
    valid_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    eps: float = 1e-7,
) -> float:
    """Compute Normalized Scanpath Saliency (NSS) against discrete eye fixations.

    Standardizes the predicted saliency map to zero mean and unit variance,
    then evaluates the average value at fixated locations (Peters et al. 2005).

    Args:
        s_pred: Predicted continuous visual saliency map.
        fixations: Binary ground-truth eye-tracking fixation map.
        valid_mask: Optional mask indicating unpadded valid image coordinates.
        eps: Epsilon to prevent division by zero in zero-variance maps.

    Returns:
        Scalar NSS value. Higher is better; 0.0 indicates chance performance.
    """
    if isinstance(s_pred, torch.Tensor):
        s_arr = s_pred.detach().cpu().numpy().astype(np.float64)
    else:
        s_arr = np.asarray(s_pred, dtype=np.float64)

    if isinstance(fixations, torch.Tensor):
        f_arr = fixations.detach().cpu().numpy().astype(bool)
    else:
        f_arr = np.asarray(fixations, dtype=bool)

    if valid_mask is not None:
        if isinstance(valid_mask, torch.Tensor):
            v_arr = valid_mask.detach().cpu().numpy().astype(bool)
        else:
            v_arr = np.asarray(valid_mask, dtype=bool)
        if len(v_arr) == 0 or not np.any(v_arr):
            return 0.0
        s_arr = s_arr[v_arr]
        f_arr = f_arr[v_arr]
    else:
        s_arr = s_arr.ravel()
        f_arr = f_arr.ravel()

    n_fix = int(np.sum(f_arr))
    if n_fix == 0:
        return 0.0

    mu = float(np.mean(s_arr))
    sigma = float(np.std(s_arr))
    if sigma < eps:
        return 0.0

    s_norm = (s_arr - mu) / (sigma + eps)
    nss_val = float(np.mean(s_norm[f_arr]))
    return nss_val


def compute_cc(
    s_pred: Union[torch.Tensor, np.ndarray],
    s_target: Union[torch.Tensor, np.ndarray],
    valid_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    eps: float = 1e-7,
) -> float:
    """Compute Pearson Linear Correlation Coefficient (CC) between continuous saliency maps.

    Args:
        s_pred: Predicted visual saliency map.
        s_target: Ground truth continuous visual saliency map.
        valid_mask: Optional valid image region mask.
        eps: Small constant for denominator stability.

    Returns:
        Scalar CC value in [-1.0, 1.0].
    """
    if isinstance(s_pred, torch.Tensor):
        p = s_pred.detach().cpu().numpy().astype(np.float64)
    else:
        p = np.asarray(s_pred, dtype=np.float64)

    if isinstance(s_target, torch.Tensor):
        t = s_target.detach().cpu().numpy().astype(np.float64)
    else:
        t = np.asarray(s_target, dtype=np.float64)

    if valid_mask is not None:
        if isinstance(valid_mask, torch.Tensor):
            m = valid_mask.detach().cpu().numpy().astype(bool)
        else:
            m = np.asarray(valid_mask, dtype=bool)
        if len(m) == 0 or not np.any(m):
            return 0.0
        p = p[m]
        t = t[m]
    else:
        p = p.ravel()
        t = t.ravel()

    if len(p) == 0:
        return 0.0

    p_dev = p - np.mean(p)
    t_dev = t - np.mean(t)
    denom = float(np.sqrt(np.sum(p_dev**2) * np.sum(t_dev**2)) + eps)
    cc_val = float(np.sum(p_dev * t_dev) / denom)
    return float(np.clip(cc_val, -1.0, 1.0))


def compute_kl_divergence(
    s_pred: Union[torch.Tensor, np.ndarray],
    s_target: Union[torch.Tensor, np.ndarray],
    valid_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    eps: float = 1e-12,
) -> float:
    """Compute Kullback-Leibler Divergence D_KL(S_target || S_pred) as spatial distributions.

    Uses epsilon eps = 1e-12 to prevent distortion on high-resolution image maps.

    Args:
        s_pred: Predicted visual saliency map.
        s_target: Ground truth visual saliency map.
        valid_mask: Optional valid image region mask.
        eps: Small constant for numerical stability (default 1e-12).

    Returns:
        Non-negative scalar KL divergence. Lower is better; 0.0 indicates identity.
    """
    if isinstance(s_pred, torch.Tensor):
        p = s_pred.detach().cpu().numpy().astype(np.float64)
    else:
        p = np.asarray(s_pred, dtype=np.float64)

    if isinstance(s_target, torch.Tensor):
        t = s_target.detach().cpu().numpy().astype(np.float64)
    else:
        t = np.asarray(s_target, dtype=np.float64)

    if valid_mask is not None:
        if isinstance(valid_mask, torch.Tensor):
            m = valid_mask.detach().cpu().numpy().astype(bool)
        else:
            m = np.asarray(valid_mask, dtype=bool)
        if len(m) == 0 or not np.any(m):
            return 0.0
        p = p[m]
        t = t[m]
    else:
        p = p.ravel()
        t = t.ravel()

    if len(p) == 0:
        return 0.0

    p = np.clip(p, a_min=0.0, a_max=None)
    t = np.clip(t, a_min=0.0, a_max=None)

    sum_p = np.sum(p)
    sum_t = np.sum(t)
    if sum_p <= eps or sum_t <= eps:
        return 0.0

    p_prob = p / sum_p
    t_prob = t / sum_t

    kl_val = float(np.sum(t_prob * np.log((t_prob + eps) / (p_prob + eps))))
    return max(0.0, kl_val)


def compute_similarity(
    s_pred: Union[torch.Tensor, np.ndarray],
    s_target: Union[torch.Tensor, np.ndarray],
    valid_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    eps: float = 1e-7,
) -> float:
    """Compute Spatial Histogram Similarity (SIM) intersection between normalized saliency maps.

    Args:
        s_pred: Predicted visual saliency map.
        s_target: Ground truth visual saliency map.
        valid_mask: Optional valid image region mask.
        eps: Small constant for numerical stability.

    Returns:
        Scalar similarity in [0.0, 1.0]. Higher is better; 1.0 indicates identity.
    """
    if isinstance(s_pred, torch.Tensor):
        p = s_pred.detach().cpu().numpy().astype(np.float64)
    else:
        p = np.asarray(s_pred, dtype=np.float64)

    if isinstance(s_target, torch.Tensor):
        t = s_target.detach().cpu().numpy().astype(np.float64)
    else:
        t = np.asarray(s_target, dtype=np.float64)

    if valid_mask is not None:
        if isinstance(valid_mask, torch.Tensor):
            m = valid_mask.detach().cpu().numpy().astype(bool)
        else:
            m = np.asarray(valid_mask, dtype=bool)
        if len(m) == 0 or not np.any(m):
            return 0.0
        p = p[m]
        t = t[m]
    else:
        p = p.ravel()
        t = t.ravel()

    if len(p) == 0:
        return 0.0

    p = np.clip(p, a_min=0.0, a_max=None)
    t = np.clip(t, a_min=0.0, a_max=None)

    sum_p = np.sum(p)
    sum_t = np.sum(t)
    if sum_p <= eps or sum_t <= eps:
        return 0.0

    p_prob = p / sum_p
    t_prob = t / sum_t

    sim_val = float(np.sum(np.minimum(p_prob, t_prob)))
    return float(np.clip(sim_val, 0.0, 1.0))


def compute_saliency_metrics(
    s_pred: Union[torch.Tensor, np.ndarray],
    s_target: Optional[Union[torch.Tensor, np.ndarray]] = None,
    fixations: Optional[Union[torch.Tensor, np.ndarray]] = None,
    valid_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    gaussian_sigma: float = 3.0,
) -> SaliencyRepresentationMetrics:
    """Compute complete suite of visual saliency fidelity metrics.

    If only discrete fixations are provided, applies a spatial Gaussian filter
    to generate the continuous comparison target for CC, KL, and SIM.
    """
    if fixations is None and s_target is not None:
        t_np = s_target.detach().cpu().numpy() if isinstance(s_target, torch.Tensor) else np.asarray(s_target)
        cutoff = np.percentile(t_np, 80)
        fixations = t_np >= cutoff

    if s_target is None and fixations is not None:
        fix_np = fixations.detach().cpu().numpy() if isinstance(fixations, torch.Tensor) else np.asarray(fixations)
        # Smooth discrete fixations with Gaussian filter for continuous metric evaluation
        s_target = gaussian_filter(fix_np.astype(np.float64), sigma=gaussian_sigma)

    assert s_target is not None and fixations is not None

    auc = compute_auc_judd(s_pred, fixations, valid_mask=valid_mask)
    nss = compute_nss(s_pred, fixations, valid_mask=valid_mask)
    cc = compute_cc(s_pred, s_target, valid_mask=valid_mask)
    kl = compute_kl_divergence(s_pred, s_target, valid_mask=valid_mask)
    sim = compute_similarity(s_pred, s_target, valid_mask=valid_mask)

    return SaliencyRepresentationMetrics(
        auc_judd=auc,
        nss=nss,
        cc=cc,
        kl=kl,
        sim=sim,
    )


# ---------------------------------------------------------------------------
# 5. High-Level Perceptual Evaluation Pipeline
# ---------------------------------------------------------------------------


class PerceptualEvaluator:
    """Unified pipeline orchestrating ROI, background, and global perceptual and saliency evaluation."""

    def __init__(
        self,
        device: Optional[torch.device] = None,
        lpips_pretrained: bool = True,
    ) -> None:
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lpips_metric = LPIPSMetric(pretrained=lpips_pretrained).to(self.device)

    def evaluate_image(
        self,
        x: Union[torch.Tensor, np.ndarray],
        x_hat: Union[torch.Tensor, np.ndarray],
        mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        s_pred: Optional[Union[torch.Tensor, np.ndarray]] = None,
        s_target: Optional[Union[torch.Tensor, np.ndarray]] = None,
        fixations: Optional[Union[torch.Tensor, np.ndarray]] = None,
        bpp: Optional[float] = None,
    ) -> ImageEvaluationResult:
        """Evaluate perceptual distortion and saliency metrics for a single image reconstruction."""
        psnr = compute_roi_psnr(x, x_hat, mask=mask)
        msssim = compute_roi_msssim(x, x_hat, mask=mask)
        lpips = compute_roi_lpips(x, x_hat, mask=mask, lpips_model=self.lpips_metric)

        saliency_res: Optional[SaliencyRepresentationMetrics] = None
        if s_pred is not None and (s_target is not None or fixations is not None):
            saliency_res = compute_saliency_metrics(s_pred, s_target=s_target, fixations=fixations)

        return ImageEvaluationResult(
            psnr=psnr,
            msssim=msssim,
            lpips=lpips,
            saliency=saliency_res,
            bpp=bpp,
        )

    def evaluate_dataset(
        self,
        images: Sequence[Union[torch.Tensor, np.ndarray]],
        reconstructions: Sequence[Union[torch.Tensor, np.ndarray]],
        masks: Optional[Sequence[Optional[Union[torch.Tensor, np.ndarray]]]] = None,
        s_preds: Optional[Sequence[Optional[Union[torch.Tensor, np.ndarray]]]] = None,
        s_targets: Optional[Sequence[Optional[Union[torch.Tensor, np.ndarray]]]] = None,
        fixations: Optional[Sequence[Optional[Union[torch.Tensor, np.ndarray]]]] = None,
        bpps: Optional[Sequence[Optional[float]]] = None,
    ) -> DatasetEvaluationResult:
        """Evaluate perceptual distortion and saliency across a dataset of test images."""
        n = len(images)
        if len(reconstructions) != n:
            raise ValueError(f"Images count {n} does not match reconstructions count {len(reconstructions)}.")

        per_image_results: List[ImageEvaluationResult] = []

        for i in range(n):
            m = masks[i] if masks is not None else None
            sp = s_preds[i] if s_preds is not None else None
            st = s_targets[i] if s_targets is not None else None
            fix = fixations[i] if fixations is not None else None
            rate = bpps[i] if bpps is not None else None

            res = self.evaluate_image(
                x=images[i],
                x_hat=reconstructions[i],
                mask=m,
                s_pred=sp,
                s_target=st,
                fixations=fix,
                bpp=rate,
            )
            per_image_results.append(res)

        # Aggregate means
        mean_roi_psnr = float(np.mean([r.psnr.roi for r in per_image_results]))
        mean_bg_psnr = float(np.mean([r.psnr.bg for r in per_image_results]))
        mean_glob_psnr = float(np.mean([r.psnr.global_val for r in per_image_results]))

        mean_roi_msssim = float(np.mean([r.msssim.roi for r in per_image_results]))
        mean_bg_msssim = float(np.mean([r.msssim.bg for r in per_image_results]))
        mean_glob_msssim = float(np.mean([r.msssim.global_val for r in per_image_results]))

        mean_roi_lpips = float(np.mean([r.lpips.roi for r in per_image_results]))
        mean_bg_lpips = float(np.mean([r.lpips.bg for r in per_image_results]))
        mean_glob_lpips = float(np.mean([r.lpips.global_val for r in per_image_results]))

        mean_sal: Optional[SaliencyRepresentationMetrics] = None
        sal_records = [r.saliency for r in per_image_results if r.saliency is not None]
        if sal_records:
            mean_sal = SaliencyRepresentationMetrics(
                auc_judd=float(np.mean([s.auc_judd for s in sal_records])),
                nss=float(np.mean([s.nss for s in sal_records])),
                cc=float(np.mean([s.cc for s in sal_records])),
                kl=float(np.mean([s.kl for s in sal_records])),
                sim=float(np.mean([s.sim for s in sal_records])),
            )

        mean_bpp = (
            float(np.mean([r.bpp for r in per_image_results if r.bpp is not None]))
            if any(r.bpp is not None for r in per_image_results)
            else None
        )

        return DatasetEvaluationResult(
            mean_psnr=PerceptualDistortion(roi=mean_roi_psnr, bg=mean_bg_psnr, global_val=mean_glob_psnr),
            mean_msssim=PerceptualDistortion(roi=mean_roi_msssim, bg=mean_bg_msssim, global_val=mean_glob_msssim),
            mean_lpips=PerceptualDistortion(roi=mean_roi_lpips, bg=mean_bg_lpips, global_val=mean_glob_lpips),
            mean_saliency=mean_sal,
            mean_bpp=mean_bpp,
            per_image_results=per_image_results,
        )
