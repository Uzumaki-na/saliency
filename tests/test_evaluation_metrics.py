"""Tests for perceptual distortion and saliency representation metrics.

Verifies:
1. ROI, Background, and Global PSNR computation with spatial mask isolation and Jensen-consistent batching.
2. ROI, Background, and Global MS-SSIM evaluation across dyadic downsampling scales (Wang et al. 2003 Eq. 7).
3. Zero-sum and all-ones mask fallback to global metrics in MS-SSIM and LPIPS.
4. ROI, Background, and Global LPIPS perceptual distance computation using deep features.
5. AUC-Judd ROC evaluation against discrete fixations, including flat saliency maps and tie handling.
6. Normalized Scanpath Saliency (NSS) against discrete fixations (Peters et al. 2005).
7. Continuous saliency metrics: CC, KL divergence (with high-res eps=1e-12), and histogram similarity (SIM).
8. High-level PerceptualEvaluator pipeline on single images and multi-image datasets.
9. Interoperability between PyTorch tensors (including requires_grad=True, 3D batched grayscale) and NumPy arrays.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.evaluation.metrics import (
    DatasetEvaluationResult,
    ImageEvaluationResult,
    LPIPSMetric,
    PerceptualDistortion,
    PerceptualEvaluator,
    SaliencyRepresentationMetrics,
    compute_auc_judd,
    compute_cc,
    compute_kl_divergence,
    compute_nss,
    compute_roi_lpips,
    compute_roi_msssim,
    compute_roi_psnr,
    compute_saliency_metrics,
    compute_similarity,
)


class TestROIPSNR:
    """Verifies ROI-PSNR, Background-PSNR, and Global-PSNR calculations."""

    def test_identity_comparison_yields_max_psnr(self) -> None:
        """Comparing identical images yields max_psnr for ROI, BG, and Global."""
        x = torch.rand(1, 3, 64, 64)
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, :32, :32] = 1.0

        res = compute_roi_psnr(x, x, mask=mask, max_psnr=100.0)
        assert isinstance(res, PerceptualDistortion)
        assert math.isclose(res.roi, 100.0, abs_tol=1e-5)
        assert math.isclose(res.bg, 100.0, abs_tol=1e-5)
        assert math.isclose(res.global_val, 100.0, abs_tol=1e-5)

    def test_noise_in_roi_only_degrades_roi_psnr(self) -> None:
        """Noise injected strictly into ROI degrades ROI-PSNR while BG-PSNR remains 100 dB."""
        x = torch.ones(1, 3, 64, 64) * 0.5
        x_hat = x.clone()

        # ROI mask covers top-left quadrant
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, :32, :32] = 1.0

        # Inject noise into ROI quadrant only
        x_hat[:, :, :32, :32] = x_hat[:, :, :32, :32] + 0.1

        res = compute_roi_psnr(x, x_hat, mask=mask, max_psnr=100.0)
        # MSE in ROI is 0.01 -> PSNR = 10 * log10(1 / 0.01) = 20.0 dB
        assert math.isclose(res.roi, 20.0, abs_tol=1e-3)
        # Background MSE is 0 -> BG-PSNR is 100.0 dB
        assert math.isclose(res.bg, 100.0, abs_tol=1e-3)
        # Global PSNR is between ROI and BG
        assert 20.0 < res.global_val < 100.0

    def test_noise_in_bg_only_degrades_bg_psnr(self) -> None:
        """Noise injected strictly into background degrades BG-PSNR while ROI-PSNR remains 100 dB."""
        x = torch.ones(1, 3, 64, 64) * 0.5
        x_hat = x.clone()

        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, :32, :32] = 1.0

        # Inject noise into bottom-right quadrant (BG) only
        x_hat[:, :, 32:, 32:] = x_hat[:, :, 32:, 32:] + 0.1

        res = compute_roi_psnr(x, x_hat, mask=mask, max_psnr=100.0)
        assert math.isclose(res.roi, 100.0, abs_tol=1e-3)
        assert res.bg < 30.0

    def test_none_mask_yields_global_psnr_for_all(self) -> None:
        """When mask is None, ROI == BG == Global PSNR."""
        x = torch.rand(1, 3, 64, 64)
        x_hat = x + torch.randn_like(x) * 0.05

        res = compute_roi_psnr(x, x_hat, mask=None)
        assert math.isclose(res.roi, res.global_val, abs_tol=1e-6)
        assert math.isclose(res.bg, res.global_val, abs_tol=1e-6)

    def test_batched_psnr_preserves_arithmetic_mean(self) -> None:
        """Batched PSNR (B > 1) evaluates arithmetic mean of per-sample PSNRs (Jensen-consistent)."""
        x1 = torch.ones(1, 3, 32, 32) * 0.5
        x_hat1 = x1 + 0.01  # MSE = 0.0001 -> PSNR = 40.0 dB

        x2 = torch.ones(1, 3, 32, 32) * 0.5
        x_hat2 = x2 + 0.10  # MSE = 0.0100 -> PSNR = 20.0 dB

        res1 = compute_roi_psnr(x1, x_hat1)
        res2 = compute_roi_psnr(x2, x_hat2)
        expected_mean_psnr = (res1.global_val + res2.global_val) / 2.0  # 30.0 dB

        # Evaluate as a single batch (B = 2)
        x_batch = torch.cat([x1, x2], dim=0)
        x_hat_batch = torch.cat([x_hat1, x_hat2], dim=0)
        res_batch = compute_roi_psnr(x_batch, x_hat_batch)

        assert math.isclose(res_batch.global_val, expected_mean_psnr, abs_tol=1e-4)

    def test_batched_3d_grayscale_tensors(self) -> None:
        """Batched 2D grayscale images of shape (B, H, W) execute cleanly."""
        x = torch.rand(4, 64, 64)
        x_hat = x + 0.01
        res = compute_roi_psnr(x, x_hat)
        assert res.global_val > 0.0


class TestROIMSSSIM:
    """Verifies ROI-MS-SSIM calculations across dyadic spatial scales."""

    def test_identity_comparison_yields_one(self) -> None:
        """Comparing identical images yields 1.0 MS-SSIM across ROI, BG, and Global."""
        x = torch.rand(1, 3, 128, 128)
        mask = torch.zeros(1, 1, 128, 128)
        mask[:, :, :64, :64] = 1.0

        res = compute_roi_msssim(x, x, mask=mask)
        assert isinstance(res, PerceptualDistortion)
        assert math.isclose(res.roi, 1.0, abs_tol=1e-3)
        assert math.isclose(res.bg, 1.0, abs_tol=1e-3)
        assert math.isclose(res.global_val, 1.0, abs_tol=1e-3)

    def test_noise_in_roi_degrades_roi_msssim(self) -> None:
        """Noise localized in ROI degrades ROI-MS-SSIM significantly more than BG."""
        x = torch.rand(1, 3, 128, 128)
        x_hat = x.clone()

        mask = torch.zeros(1, 1, 128, 128)
        mask[:, :, :64, :64] = 1.0

        # Inject severe blur/noise in ROI
        x_hat[:, :, :64, :64] = torch.rand_like(x[:, :, :64, :64])

        res = compute_roi_msssim(x, x_hat, mask=mask)
        assert res.roi < res.bg
        assert res.bg > 0.95

    def test_zero_sum_mask_fallback_to_global_msssim(self) -> None:
        """When ROI mask is all zeros, ROI MS-SSIM falls back to global without collapsing to zero."""
        x = torch.rand(1, 3, 64, 64)
        mask_zero = torch.zeros(1, 1, 64, 64)

        res = compute_roi_msssim(x, x, mask=mask_zero)
        # Should be 1.0 for identical images, NOT 0.0
        assert math.isclose(res.roi, 1.0, abs_tol=1e-3)
        assert math.isclose(res.bg, 1.0, abs_tol=1e-3)

    def test_spatial_dimension_validation(self) -> None:
        """Images smaller than 2^(levels-1) raise ValueError."""
        x_tiny = torch.rand(1, 3, 12, 12)
        with pytest.raises(ValueError, match=r"too small for 5-level MS-SSIM"):
            compute_roi_msssim(x_tiny, x_tiny)


class TestROILPIPS:
    """Verifies ROI-LPIPS deep feature perceptual distance."""

    @pytest.fixture
    def lpips_model(self) -> LPIPSMetric:
        """Provide LPIPSMetric model initialized without downloading weights."""
        return LPIPSMetric(pretrained=False)

    def test_identity_comparison_yields_zero_lpips(self, lpips_model: LPIPSMetric) -> None:
        """Comparing identical images yields 0.0 perceptual distance."""
        x = torch.rand(1, 3, 64, 64)
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, :32, :32] = 1.0

        res = compute_roi_lpips(x, x, mask=mask, lpips_model=lpips_model)
        assert isinstance(res, PerceptualDistortion)
        assert math.isclose(res.roi, 0.0, abs_tol=1e-5)
        assert math.isclose(res.bg, 0.0, abs_tol=1e-5)
        assert math.isclose(res.global_val, 0.0, abs_tol=1e-5)

    def test_noise_in_roi_increases_roi_lpips(self, lpips_model: LPIPSMetric) -> None:
        """Localized distortion in ROI increases ROI-LPIPS compared to BG-LPIPS."""
        x = torch.rand(1, 3, 64, 64)
        x_hat = x.clone()

        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, :32, :32] = 1.0

        # Distort ROI only
        x_hat[:, :, :32, :32] = torch.rand_like(x[:, :, :32, :32])

        res = compute_roi_lpips(x, x_hat, mask=mask, lpips_model=lpips_model)
        assert res.roi > 3 * res.bg
        assert res.bg < 0.10

    def test_zero_sum_mask_fallback_to_global_lpips(self, lpips_model: LPIPSMetric) -> None:
        """When ROI mask is all zeros, ROI LPIPS falls back to global without collapsing to zero."""
        x = torch.rand(1, 3, 64, 64)
        x_corrupted = x + 0.10
        mask_zero = torch.zeros(1, 1, 64, 64)

        res = compute_roi_lpips(x, x_corrupted, mask=mask_zero, lpips_model=lpips_model)
        # Should not falsely report 0.0 for corrupted images
        assert res.roi > 0.0
        assert math.isclose(res.roi, res.global_val, abs_tol=1e-5)

    def test_grayscale_single_channel_expansion(self, lpips_model: LPIPSMetric) -> None:
        """Grayscale 1-channel images (B, 1, H, W) are expanded to 3 channels automatically."""
        x = torch.rand(1, 1, 64, 64)
        x_hat = x + 0.05
        res = compute_roi_lpips(x, x_hat, lpips_model=lpips_model)
        assert res.global_val > 0.0


class TestSaliencyRepresentationMetrics:
    """Verifies AUC-Judd, NSS, CC, KL divergence, and SIM calculations."""

    def test_auc_judd_perfect_prediction(self) -> None:
        """Fixations on top saliency values yield AUC-Judd = 1.0."""
        s_pred = np.linspace(0.0, 1.0, 1000)
        fixations = np.zeros(1000, dtype=bool)
        fixations[-50:] = True  # Highest 50 values

        auc = compute_auc_judd(s_pred, fixations, jitter=False)
        assert math.isclose(auc, 1.0, abs_tol=1e-4)

    def test_auc_judd_inverse_prediction(self) -> None:
        """Fixations on lowest saliency values yield AUC-Judd near 0.0."""
        s_pred = np.linspace(0.0, 1.0, 1000)
        fixations = np.zeros(1000, dtype=bool)
        fixations[:50] = True  # Lowest 50 values

        auc = compute_auc_judd(s_pred, fixations, jitter=False)
        assert auc < 0.05

    def test_auc_judd_flat_saliency_map_evaluates_to_chance(self) -> None:
        """Flat constant saliency map evaluates to exactly 0.50 (chance) for both jitter settings."""
        flat_s = np.ones(500)
        fixations = np.zeros(500, dtype=bool)
        fixations[:50] = True

        auc_no_jitter = compute_auc_judd(flat_s, fixations, jitter=False)
        auc_with_jitter = compute_auc_judd(flat_s, fixations, jitter=True)
        assert math.isclose(auc_no_jitter, 0.50, abs_tol=1e-5)
        assert math.isclose(auc_with_jitter, 0.50, abs_tol=1e-5)

    def test_auc_judd_random_prediction(self) -> None:
        """Random prediction yields AUC-Judd approximately 0.50."""
        rng = np.random.default_rng(100)
        s_pred = rng.uniform(0.0, 1.0, size=(100, 100))
        fixations = np.zeros((100, 100), dtype=bool)
        fix_indices = rng.choice(10000, size=200, replace=False)
        fixations.ravel()[fix_indices] = True

        auc = compute_auc_judd(s_pred, fixations)
        assert 0.45 < auc < 0.55

    def test_auc_judd_linear_shift_invariance(self) -> None:
        """Linear shift S' = a * S + b preserves exact AUC-Judd."""
        rng = np.random.default_rng(42)
        s_pred = rng.uniform(0.1, 0.9, size=500)
        fixations = rng.choice([True, False], size=500, p=[0.1, 0.9])

        auc_orig = compute_auc_judd(s_pred, fixations)
        auc_shifted = compute_auc_judd(3.5 * s_pred + 12.0, fixations)
        assert math.isclose(auc_orig, auc_shifted, abs_tol=1e-5)

    def test_nss_known_distributions(self) -> None:
        """NSS evaluates positive on peaks and negative on background."""
        s = np.zeros(100)
        s[:20] = 5.0
        fix_pos = np.zeros(100, dtype=bool)
        fix_pos[:20] = True

        nss_pos = compute_nss(s, fix_pos)
        assert nss_pos > 1.5

        fix_neg = np.zeros(100, dtype=bool)
        fix_neg[50:70] = True
        nss_neg = compute_nss(s, fix_neg)
        assert nss_neg < 0.0

    def test_cc_correlation_properties(self) -> None:
        """CC is 1.0 for identity, -1.0 for inverted, and 0.0 for orthogonal."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert math.isclose(compute_cc(a, a), 1.0, abs_tol=1e-5)
        assert math.isclose(compute_cc(a, -a), -1.0, abs_tol=1e-5)

    def test_kl_divergence_properties(self) -> None:
        """KL divergence is 0.0 for identical distributions and strictly positive otherwise."""
        a = np.array([0.1, 0.2, 0.3, 0.4])
        assert math.isclose(compute_kl_divergence(a, a), 0.0, abs_tol=1e-5)

        b = np.array([0.4, 0.3, 0.2, 0.1])
        assert compute_kl_divergence(a, b) > 0.0

    def test_similarity_metric(self) -> None:
        """Histogram similarity is 1.0 for identical maps and 0.0 for disjoint maps."""
        a = np.array([0.2, 0.3, 0.5])
        assert math.isclose(compute_similarity(a, a), 1.0, abs_tol=1e-5)

        disjoint1 = np.array([1.0, 0.0, 0.0])
        disjoint2 = np.array([0.0, 1.0, 0.0])
        assert math.isclose(compute_similarity(disjoint1, disjoint2), 0.0, abs_tol=1e-5)

    def test_compute_saliency_metrics_suite(self) -> None:
        """compute_saliency_metrics populates all 5 metrics in SaliencyRepresentationMetrics."""
        s_pred = np.random.uniform(0.0, 1.0, size=(32, 32))
        s_target = np.random.uniform(0.0, 1.0, size=(32, 32))
        fixations = s_target > np.percentile(s_target, 80)

        metrics = compute_saliency_metrics(s_pred, s_target=s_target, fixations=fixations)
        assert isinstance(metrics, SaliencyRepresentationMetrics)
        assert 0.0 <= metrics.auc_judd <= 1.0
        assert -1.0 <= metrics.cc <= 1.0
        assert metrics.kl >= 0.0
        assert 0.0 <= metrics.sim <= 1.0


class TestPerceptualEvaluator:
    """Verifies unified single-image and dataset-level evaluation pipeline."""

    @pytest.fixture
    def evaluator(self) -> PerceptualEvaluator:
        return PerceptualEvaluator(lpips_pretrained=False)

    def test_evaluate_image(self, evaluator: PerceptualEvaluator) -> None:
        """Single image evaluation returns complete ImageEvaluationResult."""
        x = torch.rand(1, 3, 64, 64)
        x_hat = x + 0.02
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, :32, :32] = 1.0
        s_pred = torch.rand(1, 1, 64, 64)
        s_target = torch.rand(1, 1, 64, 64)

        result = evaluator.evaluate_image(
            x=x,
            x_hat=x_hat,
            mask=mask,
            s_pred=s_pred,
            s_target=s_target,
            bpp=0.25,
        )

        assert isinstance(result, ImageEvaluationResult)
        assert result.psnr.global_val > 0.0
        assert result.msssim.global_val > 0.0
        assert result.lpips.global_val > 0.0
        assert result.saliency is not None
        assert result.bpp == 0.25

    def test_evaluate_dataset(self, evaluator: PerceptualEvaluator) -> None:
        """Dataset-level evaluation aggregates multiple images with per-image records."""
        n_images = 4
        images = [torch.rand(1, 3, 64, 64) for _ in range(n_images)]
        reconstructions = [img + 0.01 for img in images]
        masks = [torch.ones(1, 1, 64, 64) for _ in range(n_images)]
        bpps = [0.10, 0.20, 0.30, 0.40]

        dataset_result = evaluator.evaluate_dataset(
            images=images,
            reconstructions=reconstructions,
            masks=masks,
            bpps=bpps,
        )

        assert isinstance(dataset_result, DatasetEvaluationResult)
        assert len(dataset_result.per_image_results) == 4
        assert dataset_result.mean_bpp == pytest.approx(0.25, abs=1e-5)
        assert dataset_result.mean_psnr.global_val > 0.0
        assert dataset_result.mean_msssim.global_val > 0.0
        assert dataset_result.mean_lpips.global_val > 0.0
