"""Tests for spatial bit density mapping and Relative Allocation Ratio (RAR) computation."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from src.bitstream.container import create_model
from src.evaluation.spatial_allocation import (
    SpatialBitMap,
    compute_allocation_ratio,
    compute_relative_allocation_ratio,
    compute_spatial_bit_map,
    evaluate_rar_across_models,
    extract_top_percentile_mask,
)


class TestSpatialBitMapExtraction:
    """Verifies per-cell Shannon bit derivation and conservation properties."""

    def test_bit_map_conservation(self) -> None:
        """Asserts sum of spatial bit map cells exactly conserves total likelihood bits."""
        b, m, h_lat, w_lat = 2, 16, 8, 8
        n, h_hyp, w_hyp = 8, 2, 2

        # Create synthetic valid likelihoods in (0, 1]
        torch.manual_seed(42)
        y_lik = torch.rand((b, m, h_lat, w_lat)) * 0.9 + 0.05
        z_lik = torch.rand((b, n, h_hyp, w_hyp)) * 0.9 + 0.05

        out_dict = {"y_likelihoods": y_lik, "z_likelihoods": z_lik}
        bm = compute_spatial_bit_map(out_dict)

        assert isinstance(bm, SpatialBitMap)
        assert bm.y_bits.shape == (b, 1, h_lat, w_lat)
        assert bm.z_bits.shape == (b, 1, h_lat, w_lat)
        assert bm.total_bits.shape == (b, 1, h_lat, w_lat)

        # Expected total Shannon bits
        expected_y_total = float((-torch.sum(torch.log2(y_lik.float()))).item())
        expected_z_total = float((-torch.sum(torch.log2(z_lik.float()))).item())

        assert pytest.approx(bm.y_bits.sum().item(), rel=1e-5) == expected_y_total
        assert pytest.approx(bm.z_bits.sum().item(), rel=1e-5) == expected_z_total
        assert pytest.approx(bm.total_bits.sum().item(), rel=1e-5) == expected_y_total + expected_z_total

    def test_bit_map_with_valid_mask(self) -> None:
        """Asserts valid mask correctly isolates unpadded cells."""
        b, m, h_lat, w_lat = 1, 8, 4, 4
        n, h_hyp, w_hyp = 4, 1, 1

        y_lik = torch.full((b, m, h_lat, w_lat), 0.5)  # -log2(0.5) = 1 bit per channel -> 8 bits/cell
        z_lik = torch.full((b, n, h_hyp, w_hyp), 0.5)  # 4 bits per hyper-cell -> 4 / 16 = 0.25 bits/cell

        # Mask: only top-left 2x2 cells are valid
        valid_mask = torch.zeros((h_lat, w_lat), dtype=torch.bool)
        valid_mask[:2, :2] = True

        bm = compute_spatial_bit_map(
            {"y_likelihoods": y_lik, "z_likelihoods": z_lik},
            valid_mask=valid_mask,
        )

        assert bm.valid_mask is not None
        assert bm.valid_mask.shape == (1, 1, h_lat, w_lat)
        assert bm.valid_mask.sum().item() == 4

        # 4 valid cells * (8 y_bits + 0.25 z_bits) = 33.0 total valid bits
        assert pytest.approx(bm.total_valid_bits(), abs=1e-4) == 33.0
        assert pytest.approx(bm.total_y_bits(), abs=1e-4) == 32.0
        assert pytest.approx(bm.total_z_bits(), abs=1e-4) == 1.0

    def test_bit_map_sidecar_overhead(self) -> None:
        """Asserts sidecar bit overhead is conserved and partitioned across valid cells."""
        b, m, h_lat, w_lat = 1, 4, 4, 4
        y_lik = torch.full((b, m, h_lat, w_lat), 0.5)
        z_lik = torch.full((b, 2, 1, 1), 0.5)

        # 16-byte compressed sidecar -> 128 bits
        sidecar_bytes = b"0" * 16

        bm = compute_spatial_bit_map({"y_likelihoods": y_lik, "z_likelihoods": z_lik, "sidecar_bytes": sidecar_bytes})

        assert bm.sidecar_bits is not None
        # 128 bits across 16 cells = 8 bits per cell
        assert pytest.approx(bm.sidecar_bits.mean().item(), abs=1e-4) == 8.0
        assert pytest.approx(bm.sidecar_bits.sum().item(), abs=1e-4) == 128.0


class TestTopPercentileMaskExtraction:
    """Verifies top-percentile saliency segmentation."""

    def test_top_20_percent_area_ratio(self) -> None:
        """Asserts 80th percentile threshold extracts approximately top 20% salient cells."""
        h, w = 32, 32
        y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
        # Gaussian peak at center
        saliency = torch.exp(-(x**2 + y**2) / 0.2)

        roi_mask, bg_mask = extract_top_percentile_mask(
            saliency,
            percentile=80.0,
            target_shape=(h, w),
        )

        assert roi_mask.shape == (1, 1, h, w)
        assert bg_mask.shape == (1, 1, h, w)

        # Disjoint partitions
        assert (roi_mask & bg_mask).sum().item() == 0
        assert (roi_mask | bg_mask).sum().item() == h * w

        # Area ratio ~ 0.20
        n_roi = roi_mask.sum().item()
        n_tot = h * w
        ratio = n_roi / n_tot
        assert 0.18 <= ratio <= 0.22

    def test_uniform_saliency_tie_breaking(self) -> None:
        """Asserts uniform saliency map produces non-empty partitions without dividing by zero."""
        h, w = 10, 10
        uniform_saliency = torch.full((1, 1, h, w), 0.75)

        roi_mask, bg_mask = extract_top_percentile_mask(
            uniform_saliency,
            percentile=80.0,
            target_shape=(h, w),
        )

        n_roi = roi_mask.sum().item()
        n_bg = bg_mask.sum().item()

        assert n_roi > 0
        assert n_bg > 0
        assert n_roi + n_bg == h * w
        assert pytest.approx(n_roi / (h * w), abs=0.05) == 0.20


class TestAllocationRatioAndRAR:
    """Verifies mathematical properties of AllocationRatio and RelativeAllocationRatio."""

    def test_uniform_bit_density_yields_unit_ratio(self) -> None:
        """Asserts spatially uniform bit distribution yields AllocationRatio == 1.0."""
        bits = torch.full((1, 1, 8, 8), 4.0)
        roi_mask = torch.zeros((1, 1, 8, 8), dtype=torch.bool)
        roi_mask[0, 0, :4, :4] = True  # 16 cells ROI, 48 cells BG

        ar = compute_allocation_ratio(bits, roi_mask)
        assert pytest.approx(ar, abs=1e-6) == 1.0

    def test_concentrated_bit_density(self) -> None:
        """Asserts 3x bit concentration in ROI yields AllocationRatio == 3.0."""
        bits = torch.zeros((1, 1, 8, 8))
        roi_mask = torch.zeros((1, 1, 8, 8), dtype=torch.bool)
        roi_mask[0, 0, :4, :4] = True
        bg_mask = ~roi_mask

        # 6.0 bits/cell in ROI, 2.0 bits/cell in BG -> density ratio = 3.0
        bits[roi_mask] = 6.0
        bits[bg_mask] = 2.0

        ar = compute_allocation_ratio(bits, roi_mask, bg_mask)
        assert pytest.approx(ar, abs=1e-6) == 3.0

    def test_relative_allocation_ratio_identity_and_scaling(self) -> None:
        """Asserts RelativeAllocationRatio against baseline behaves consistently."""
        base_bits = torch.full((1, 1, 4, 4), 2.0)
        model_bits = torch.zeros((1, 1, 4, 4))
        roi_mask = torch.zeros((1, 1, 4, 4), dtype=torch.bool)
        roi_mask[0, 0, :2, :2] = True
        bg_mask = ~roi_mask

        # Baseline: AR = 1.0
        # Model: ROI = 4.0 bits, BG = 2.0 bits -> AR = 2.0
        model_bits[roi_mask] = 4.0
        model_bits[bg_mask] = 2.0

        rar = compute_relative_allocation_ratio(model_bits, base_bits, roi_mask, bg_mask)
        assert pytest.approx(rar, abs=1e-6) == 2.0

    def test_multi_model_rar_evaluation_end_to_end(self) -> None:
        """Verifies evaluate_rar_across_models executes across Models A, B, C, D, C-Transmitted."""
        torch.manual_seed(42)
        h, w = 64, 64
        x = torch.rand((1, 3, h, w))
        s = torch.rand((1, 1, h, w))

        models = {
            "A": create_model("A", n_channels=64, m_channels=96),
            "B": create_model("B", n_channels=64, m_channels=96),
            "C": create_model("C", n_channels=64, m_channels=96),
            "D": create_model("D", n_channels=64, m_channels=96),
            "C-Transmitted": create_model("C-Transmitted", n_channels=64, m_channels=96),
        }

        outputs: dict[str, Any] = {}
        with torch.no_grad():
            outputs["A"] = models["A"](x, training=False)
            outputs["B"] = models["B"](x, training=False)
            outputs["C"] = models["C"](x, training=False)
            outputs["D"] = models["D"](x, training=False)
            outputs["C-Transmitted"] = models["C-Transmitted"](x, s=s, training=False)

        results = evaluate_rar_across_models(outputs, s=s, percentile=80.0, baseline_key="A")

        for m_name in ("A", "B", "C", "D", "C-Transmitted"):
            assert m_name in results
            m_res = results[m_name]
            assert "allocation_ratio_y" in m_res
            assert "allocation_ratio_total" in m_res
            assert "rar_y" in m_res
            assert "rar_total" in m_res
            assert m_res["allocation_ratio_y"] > 0.0
            assert m_res["allocation_ratio_total"] > 0.0

        # Model A baseline RAR against itself must equal 1.0
        assert pytest.approx(results["A"]["rar_y"], abs=1e-6) == 1.0
        assert pytest.approx(results["A"]["rar_total"], abs=1e-6) == 1.0

    def test_batched_sidecar_partitioning(self) -> None:
        """Asserts batch items receive independent sidecar bits without cross-sample leakage."""
        b, m, h_lat, w_lat = 2, 4, 4, 4
        y_lik = torch.full((b, m, h_lat, w_lat), 0.5)
        z_lik = torch.full((b, 2, 1, 1), 0.5)

        # Sample 0 has 10 bytes (80 bits), sample 1 has 100 bytes (800 bits)
        sidecar_bytes = [b"a" * 10, b"b" * 100]

        bm = compute_spatial_bit_map({"y_likelihoods": y_lik, "z_likelihoods": z_lik, "sidecar_bytes": sidecar_bytes})

        assert bm.sidecar_bits is not None
        # Sample 0: 80 bits / 16 cells = 5.0 bits/cell
        assert pytest.approx(bm.sidecar_bits[0].mean().item(), abs=1e-4) == 5.0
        assert pytest.approx(bm.sidecar_bits[0].sum().item(), abs=1e-4) == 80.0

        # Sample 1: 800 bits / 16 cells = 50.0 bits/cell
        assert pytest.approx(bm.sidecar_bits[1].mean().item(), abs=1e-4) == 50.0
        assert pytest.approx(bm.sidecar_bits[1].sum().item(), abs=1e-4) == 800.0

    def test_3d_mask_resolution_b1(self) -> None:
        """Asserts 3D mask with shape (3, H, W) is handled cleanly without indexing error."""
        y_lik = torch.full((1, 4, 4, 4), 0.5)
        z_lik = torch.full((1, 2, 1, 1), 0.5)
        mask_3d = torch.ones((3, 4, 4), dtype=torch.bool)

        bm = compute_spatial_bit_map(
            {"y_likelihoods": y_lik, "z_likelihoods": z_lik},
            valid_mask=mask_3d,
        )
        assert bm.valid_mask is not None
        assert bm.valid_mask.shape == (1, 1, 4, 4)
        assert bm.total_valid_bits() > 0.0

    def test_inf_over_inf_rar_guard(self) -> None:
        """Asserts inf / inf returns 1.0 rather than NaN when background bit density is zero."""
        roi_mask = torch.zeros((1, 1, 4, 4), dtype=torch.bool)
        roi_mask[0, 0, :2, :2] = True
        bg_mask = ~roi_mask

        # Both model and baseline have zero bits in background
        model_bits = torch.zeros((1, 1, 4, 4))
        model_bits[roi_mask] = 10.0
        base_bits = torch.zeros((1, 1, 4, 4))
        base_bits[roi_mask] = 10.0

        rar = compute_relative_allocation_ratio(model_bits, base_bits, roi_mask, bg_mask)
        assert not math.isnan(rar)
        assert rar == 1.0

    def test_missing_baseline_key_raises_key_error(self) -> None:
        """Asserts evaluate_rar_across_models raises KeyError if baseline_key is not found."""
        out = {"y_likelihoods": torch.full((1, 4, 4, 4), 0.5), "z_likelihoods": torch.full((1, 2, 1, 1), 0.5)}
        with pytest.raises(KeyError, match="baseline_key 'NONEXISTENT' not found"):
            evaluate_rar_across_models({"ModelC": out}, s=torch.rand((1, 1, 64, 64)), baseline_key="NONEXISTENT")
