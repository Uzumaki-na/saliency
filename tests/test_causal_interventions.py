"""Tests for causal intervention protocol via latent block-shuffling and transplants.

Verifies:
1. Spatial block-shuffling and exact un-shuffling invertibility (4x4 and 8x8 tiles) across 2D/3D/4D tensors.
2. 2D permutation tensors, detailed BlockShuffleResult metadata, and non-square grids (16x32).
3. Targeted block swapping, self-swap error checking, and round-trip identity.
4. Numerical stability, empty valid mask handling, and batch correlation averaging for spatial Pearson correlation.
5. Saliency map transplanting, multi-channel inputs (C > 1), and spatial alignment.
6. End-to-end causal block-shuffling intervention on Model C and Model D, proving bit reallocation
   follows the permuted saliency map.
7. Auto-extraction of S_pred when saliency_map is None.
8. Control invariance on Model A across block shuffling and tile swap.
9. Targeted tile-swap micro-intervention proving bit density flips between salient and background tiles.
10. Cross-image transplant intervention on batches (B > 1) proving bit reallocation follows donor ROI masks.
"""

from __future__ import annotations

import pytest
import torch

from src.codec.model_a import ModelA
from src.codec.model_c import ModelC
from src.codec.model_d import ModelD
from src.evaluation.causal import (
    compute_spatial_correlation,
    run_causal_block_intervention,
    run_causal_tile_swap_intervention,
    run_causal_transplant_intervention,
    shuffle_spatial_blocks,
    shuffle_spatial_blocks_detailed,
    swap_spatial_blocks,
    transplant_saliency_map,
    unshuffle_spatial_blocks,
)


class TestSpatialBlockShuffling:
    """Tests spatial block partitioning, shuffling, and exact invertibility."""

    @pytest.mark.parametrize("block_size", [4, 8])
    @pytest.mark.parametrize("grid_size", [16, 32])
    def test_round_trip_invertibility(self, block_size: int, grid_size: int) -> None:
        """Verifies unshuffle(shuffle(T)) == T identically."""
        tensor = torch.randn(2, 3, grid_size, grid_size)
        shuffled, perm = shuffle_spatial_blocks(tensor, block_size=block_size, seed=123)
        restored = unshuffle_spatial_blocks(shuffled, perm, block_size=block_size)

        assert torch.equal(restored, tensor), "Un-shuffled tensor must identically match original tensor."
        assert not torch.equal(shuffled, tensor), "Shuffled tensor must differ from original tensor."

    def test_non_square_grid_shuffling(self) -> None:
        """Verifies block shuffling works on non-square latent grids (e.g. 16x32)."""
        tensor = torch.randn(2, 1, 16, 32)
        shuffled, perm = shuffle_spatial_blocks(tensor, block_size=4, seed=77)
        restored = unshuffle_spatial_blocks(shuffled, perm, block_size=4)

        assert torch.equal(restored, tensor)
        assert perm.shape[0] == (16 // 4) * (32 // 4) == 32

    def test_detailed_block_shuffle_result(self) -> None:
        """Verifies shuffle_spatial_blocks_detailed returns complete BlockShuffleResult container."""
        tensor = torch.randn(1, 1, 16, 16)
        res = shuffle_spatial_blocks_detailed(tensor, block_size=4, seed=42)

        assert res.block_size == 4
        assert res.num_blocks_h == 4
        assert res.num_blocks_w == 4
        assert res.num_blocks_total == 16
        assert res.shuffled_tensor.shape == (1, 1, 16, 16)
        assert res.permutation.shape == (16,)
        assert res.inverse_permutation.shape == (16,)

        # Inverting using inverse_permutation recovers original
        restored, _ = shuffle_spatial_blocks(res.shuffled_tensor, block_size=4, permutation=res.inverse_permutation)
        assert torch.equal(restored, tensor)

    def test_2d_permutation_tensor_per_batch_element(self) -> None:
        """Verifies distinct 2D permutations per batch item."""
        b = 2
        tensor = torch.randn(b, 1, 16, 16)
        p1 = torch.randperm(16)
        p2 = torch.randperm(16)
        perm2d = torch.stack([p1, p2], dim=0)

        shuffled, perm = shuffle_spatial_blocks(tensor, block_size=4, permutation=perm2d)
        assert torch.equal(perm, perm2d)

        restored = unshuffle_spatial_blocks(shuffled, perm, block_size=4)
        assert torch.equal(restored, tensor)

    def test_2d_3d_4d_and_boolean_mask_support(self) -> None:
        """Verifies block shuffling works across 2D, 3D, and 4D tensors, as well as boolean masks."""
        # 2D tensor
        t2d = torch.arange(16 * 16, dtype=torch.float32).view(16, 16)
        shuff_2d, perm = shuffle_spatial_blocks(t2d, block_size=4, seed=42)
        assert shuff_2d.shape == (16, 16)
        assert torch.equal(unshuffle_spatial_blocks(shuff_2d, perm, block_size=4), t2d)

        # 3D tensor
        t3d = torch.randn(5, 16, 16)
        shuff_3d, perm = shuffle_spatial_blocks(t3d, block_size=4, seed=42)
        assert shuff_3d.shape == (5, 16, 16)
        assert torch.equal(unshuffle_spatial_blocks(shuff_3d, perm, block_size=4), t3d)

        # Boolean mask
        mask = torch.rand(1, 1, 16, 16) > 0.5
        shuff_mask, perm = shuffle_spatial_blocks(mask, block_size=4, seed=42)
        assert shuff_mask.dtype == torch.bool
        assert torch.equal(unshuffle_spatial_blocks(shuff_mask, perm, block_size=4), mask)

    def test_deterministic_reproducibility_with_seed(self) -> None:
        """Verifies identical seed produces identical permutation and shuffled tensor."""
        t = torch.randn(1, 1, 16, 16)
        shuff1, perm1 = shuffle_spatial_blocks(t, block_size=4, seed=999)
        shuff2, perm2 = shuffle_spatial_blocks(t, block_size=4, seed=999)

        assert torch.equal(perm1, perm2)
        assert torch.equal(shuff1, shuff2)

    def test_single_block_tensor(self) -> None:
        """Verifies that when grid size equals block size, identity permutation is returned."""
        t = torch.randn(1, 1, 8, 8)
        shuff, perm = shuffle_spatial_blocks(t, block_size=8)
        assert torch.equal(shuff, t)
        assert torch.equal(perm, torch.tensor([0]))

    def test_invalid_dimensions_and_block_size_raise_error(self) -> None:
        """Verifies error handling on non-divisible spatial dimensions or invalid block size."""
        t_bad = torch.randn(1, 1, 15, 16)
        with pytest.raises(ValueError, match="divisible by block_size"):
            shuffle_spatial_blocks(t_bad, block_size=4)

        t_ok = torch.randn(1, 1, 16, 16)
        with pytest.raises(ValueError, match="block_size must be positive"):
            shuffle_spatial_blocks(t_ok, block_size=0)

        with pytest.raises(ValueError, match="Permutation length"):
            shuffle_spatial_blocks(t_ok, block_size=4, permutation=torch.arange(10))


class TestTargetedTileSwap:
    """Tests targeted block swapping."""

    def test_swap_spatial_blocks_inversion(self) -> None:
        """Verifies swapping block i and block j twice restores original tensor."""
        t = torch.arange(16 * 16, dtype=torch.float32).view(1, 1, 16, 16)
        swapped, _ = swap_spatial_blocks(t, block_idx_1=0, block_idx_2=15, block_size=4)

        # Block 0 and block 15 should be swapped
        assert not torch.equal(swapped, t)
        assert torch.equal(swapped[..., :4, :4], t[..., 12:16, 12:16])
        assert torch.equal(swapped[..., 12:16, 12:16], t[..., :4, :4])

        # Swap again
        restored, _ = swap_spatial_blocks(swapped, block_idx_1=0, block_idx_2=15, block_size=4)
        assert torch.equal(restored, t)

    def test_swap_out_of_bounds_raises_error(self) -> None:
        """Verifies out-of-bounds block index raises ValueError."""
        t = torch.randn(1, 1, 16, 16)
        with pytest.raises(ValueError, match=r"Block indices .* must be in range"):
            swap_spatial_blocks(t, block_idx_1=0, block_idx_2=20, block_size=4)

    def test_swap_identical_indices_raises_error(self) -> None:
        """Verifies that self-swapping (block_idx_1 == block_idx_2) raises ValueError."""
        t = torch.randn(1, 1, 16, 16)
        with pytest.raises(ValueError, match="must be distinct"):
            swap_spatial_blocks(t, block_idx_1=2, block_idx_2=2, block_size=4)


class TestSpatialCorrelation:
    """Tests spatial Pearson correlation computation."""

    def test_correlation_properties(self) -> None:
        """Verifies standard mathematical correlation properties."""
        m1 = torch.randn(16, 16)
        # Identical map -> 1.0
        assert compute_spatial_correlation(m1, m1) == pytest.approx(1.0, abs=1e-5)

        # Perfectly anti-correlated map -> -1.0
        assert compute_spatial_correlation(m1, -m1) == pytest.approx(-1.0, abs=1e-5)

        # Shift and positive scaling invariance
        assert compute_spatial_correlation(m1, 3.0 * m1 + 7.0) == pytest.approx(1.0, abs=1e-5)

    def test_constant_maps_avoid_nan(self) -> None:
        """Verifies zero variance / flat maps return 0.0 or 1.0 without NaN."""
        m_flat = torch.zeros(16, 16)
        m_rand = torch.randn(16, 16)

        # One flat, one random -> 0.0
        assert compute_spatial_correlation(m_flat, m_rand) == 0.0
        assert compute_spatial_correlation(m_rand, m_flat) == 0.0

        # Both flat and identical -> 1.0
        assert compute_spatial_correlation(m_flat, m_flat) == 1.0

        # Both flat but different constants -> 0.0
        assert compute_spatial_correlation(m_flat, m_flat + 5.0) == 0.0

    def test_valid_mask_filtering(self) -> None:
        """Verifies correlation is computed strictly over valid unpadded cells."""
        m1 = torch.randn(16, 16)
        m2 = m1.clone()

        # Add heavy noise in outer border
        m2[0, :] = 1000.0
        m2[-1, :] = -1000.0

        mask = torch.zeros(16, 16, dtype=torch.bool)
        mask[1:-1, :] = True  # Exclude outer top and bottom rows

        # With mask, correlation should be 1.0
        corr_masked = compute_spatial_correlation(m1, m2, valid_mask=mask)
        assert corr_masked == pytest.approx(1.0, abs=1e-5)

        # Without mask, correlation is corrupted by border
        corr_unmasked = compute_spatial_correlation(m1, m2)
        assert corr_unmasked < 0.9

    def test_empty_valid_mask_returns_zero(self) -> None:
        """Verifies that an empty valid mask (N=0) returns 0.0 without crash or NaN."""
        m1 = torch.randn(16, 16)
        m2 = torch.randn(16, 16)
        empty_mask = torch.zeros(16, 16, dtype=torch.bool)

        corr = compute_spatial_correlation(m1, m2, valid_mask=empty_mask)
        assert corr == 0.0

    def test_batched_spatial_correlation_averages_per_sample(self) -> None:
        """Verifies batch B > 1 computes per-sample correlation rather than conflating mean offsets."""
        m1 = torch.randn(2, 1, 16, 16)
        m2 = m1.clone()
        # Add massive offset to sample 0 only: should not impact sample correlation
        m2[0] += 500.0

        corr = compute_spatial_correlation(m1, m2)
        assert corr == pytest.approx(1.0, abs=1e-5)


class TestSaliencyTransplant:
    """Tests cross-image saliency transplanting and alignment."""

    def test_transplant_saliency_map_resampling_and_clamping(self) -> None:
        """Verifies spatial resampling and range clamping."""
        donor = torch.tensor([[-0.5, 0.2], [0.8, 1.5]])  # Contains out-of-range values
        transplanted = transplant_saliency_map(donor, (4, 4))

        assert transplanted.shape == (1, 1, 4, 4)
        assert float(transplanted.min().item()) >= 0.0
        assert float(transplanted.max().item()) <= 1.0

    def test_multi_channel_saliency_inputs(self) -> None:
        """Verifies 3-channel saliency maps (e.g. RGB heatmap) are averaged to 1 channel."""
        s_3ch = torch.rand(3, 16, 16)
        aligned = transplant_saliency_map(s_3ch, (8, 8))
        assert aligned.shape == (1, 1, 8, 8)

        s_4d_3ch = torch.rand(2, 3, 16, 16)
        aligned_4d = transplant_saliency_map(s_4d_3ch, (8, 8))
        assert aligned_4d.shape == (2, 1, 8, 8)


class TestCausalBlockIntervention:
    """Tests causal intervention protocols on neural compression architectures."""

    @pytest.fixture
    def small_model_c(self) -> ModelC:
        """Creates a lightweight Model C instance for fast CPU execution."""
        torch.manual_seed(42)
        model = ModelC(n_channels=32, m_channels=48)
        model.eval()
        return model

    @pytest.fixture
    def small_model_d(self) -> ModelD:
        """Creates a lightweight Model D instance for mechanism-only validation."""
        torch.manual_seed(42)
        model = ModelD(n_channels=32, m_channels=48)
        model.eval()
        return model

    @pytest.fixture
    def small_model_a(self) -> ModelA:
        """Creates a lightweight Model A instance as an unconditioned control baseline."""
        torch.manual_seed(42)
        model = ModelA(n_channels=32, m_channels=48)
        model.eval()
        return model

    def test_causal_block_intervention_model_c(self, small_model_c: ModelC) -> None:
        """Verifies that block-shuffling saliency causally reallocates bits in Model C."""
        # 128x128 image -> 8x8 latent grid
        x = torch.randn(1, 3, 128, 128)

        # Construct a synthetic saliency map with a high-saliency patch in top-left
        s = torch.zeros(1, 1, 128, 128)
        s[:, :, :64, :64] = 1.0  # Top-left quadrant has saliency 1.0, rest 0.0

        # Run block intervention with block_size=4 (on 8x8 latent grid, this is 2x2 blocks)
        result = run_causal_block_intervention(
            small_model_c,
            x,
            block_size=4,
            saliency_map=s,
            seed=42,
        )

        assert result.model_name == "C"
        # Intervened bit map should correlate with the shuffled saliency much more than with original
        assert result.corr_interv_vs_shuffled > result.corr_interv_vs_original
        assert result.causal_shift > 0.0, f"Expected positive causal shift, got {result.causal_shift}"

        # Inverting the permutation on intervened bits should closely recover baseline bit map
        assert result.corr_unshuffled_vs_baseline > 0.7, (
            f"Expected high correlation after un-shuffling, got {result.corr_unshuffled_vs_baseline}"
        )

    def test_causal_block_intervention_auto_s_pred_extraction(self, small_model_c: ModelC) -> None:
        """Verifies run_causal_block_intervention extracts out_base.s_pred when saliency_map is None."""
        x = torch.randn(1, 3, 128, 128)
        result = run_causal_block_intervention(
            small_model_c,
            x,
            block_size=4,
            saliency_map=None,  # Rely on predicted saliency
            seed=42,
        )

        assert result.model_name == "C"
        assert result.original_saliency.shape == (1, 1, 8, 8)
        assert result.shuffled_saliency.shape == (1, 1, 8, 8)
        assert result.corr_unshuffled_vs_baseline > 0.7

    def test_causal_block_intervention_batch_gt_one(self, small_model_c: ModelC) -> None:
        """Verifies causal block intervention executes cleanly on batch size B = 2."""
        x = torch.randn(2, 3, 128, 128)
        s = torch.zeros(2, 1, 128, 128)
        s[:, :, :64, :64] = 1.0

        result = run_causal_block_intervention(
            small_model_c,
            x,
            block_size=4,
            saliency_map=s,
            seed=42,
        )

        assert result.baseline_bit_map.y_bits.shape[0] == 2
        assert result.intervened_bit_map.y_bits.shape[0] == 2
        assert result.causal_shift > 0.0

    def test_causal_block_intervention_model_d(self, small_model_d: ModelD) -> None:
        """Verifies that Model D (mechanism-only, trained with uniform MSE) exhibits causal bit shift."""
        x = torch.randn(1, 3, 128, 128)
        s = torch.zeros(1, 1, 128, 128)
        s[:, :, :64, :64] = 1.0

        result = run_causal_block_intervention(
            small_model_d,
            x,
            block_size=4,
            saliency_map=s,
            seed=42,
        )

        assert result.model_name == "D"
        assert result.corr_interv_vs_shuffled > result.corr_interv_vs_original
        assert result.causal_shift > 0.0
        assert result.corr_unshuffled_vs_baseline > 0.7

    def test_causal_block_intervention_model_a_control_invariance(self, small_model_a: ModelA) -> None:
        """Verifies that Model A ignores saliency interventions, showing zero causal shift."""
        x = torch.randn(1, 3, 128, 128)
        s = torch.zeros(1, 1, 128, 128)
        s[:, :, :64, :64] = 1.0

        result = run_causal_block_intervention(
            small_model_a,
            x,
            block_size=4,
            saliency_map=s,
            seed=42,
        )

        # Baseline and intervened bit maps must be identically equal for Model A,
        # confirming zero causal responsiveness to the intervened saliency map.
        assert torch.equal(result.baseline_bit_map.y_bits, result.intervened_bit_map.y_bits)
        assert torch.equal(result.baseline_bit_map.total_bits, result.intervened_bit_map.total_bits)
        corr_interv_vs_base = compute_spatial_correlation(
            result.intervened_bit_map.y_bits, result.baseline_bit_map.y_bits
        )
        assert corr_interv_vs_base == pytest.approx(1.0, abs=1e-5)


class TestCausalTileSwapIntervention:
    """Tests targeted tile swap micro-interventions."""

    def test_tile_swap_reversal_model_c(self) -> None:
        """Verifies that swapping a salient and background tile flips local bit consumption."""
        model = ModelC(n_channels=32, m_channels=48)
        model.eval()

        # 128x128 image -> 8x8 latent grid (4 blocks of size 4x4: indices 0, 1, 2, 3)
        x = torch.randn(1, 3, 128, 128)

        # Saliency map where tile 0 (top-left) has high saliency and tile 3 (bottom-right) has low saliency
        s = torch.zeros(1, 1, 128, 128)
        s[:, :, :64, :64] = 0.95  # Tile 0
        s[:, :, 64:, 64:] = 0.05  # Tile 3

        result = run_causal_tile_swap_intervention(
            model,
            x,
            block_idx_1=0,
            block_idx_2=3,
            block_size=4,
            saliency_map=s,
        )

        # In baseline: tile 0 has more bits than tile 3
        assert result.baseline_bit_diff > 0.0, (
            f"Expected tile 0 to have more bits than tile 3 in baseline, got diff={result.baseline_bit_diff}"
        )

        # In swapped: tile 0 now gets low saliency and tile 3 gets high saliency, so diff flips sign!
        assert result.swapped_bit_diff < 0.0, (
            f"Expected swapped diff to be negative (tile 3 having more bits), got {result.swapped_bit_diff}"
        )

        # Redistribution ratio should be substantial
        assert result.bit_redistribution_ratio > 1.0

    def test_tile_swap_model_a_control_invariance(self) -> None:
        """Verifies that for Model A, tile swap has zero effect (ratio = 1.0, diffs identical)."""
        model = ModelA(n_channels=32, m_channels=48)
        model.eval()

        x = torch.randn(1, 3, 128, 128)
        s = torch.zeros(1, 1, 128, 128)
        s[:, :, :64, :64] = 0.95
        s[:, :, 64:, 64:] = 0.05

        result = run_causal_tile_swap_intervention(
            model,
            x,
            block_idx_1=0,
            block_idx_2=3,
            block_size=4,
            saliency_map=s,
        )

        assert result.bit_redistribution_ratio == pytest.approx(1.0, abs=1e-5)
        assert result.baseline_bit_diff == pytest.approx(result.swapped_bit_diff, abs=1e-5)


class TestCausalTransplantIntervention:
    """Tests cross-image saliency transplanting."""

    def test_transplant_intervention_model_c_and_model_a(self) -> None:
        """Verifies donor saliency causes Model C to reallocate bits to donor ROI while Model A is unchanged."""
        model_c = ModelC(n_channels=32, m_channels=48)
        model_a = ModelA(n_channels=32, m_channels=48)
        model_c.eval()
        model_a.eval()

        # Target image with flat/uniform texture
        target_image = torch.randn(1, 3, 128, 128)

        # Target image has natural uniform saliency
        target_saliency = torch.full((1, 1, 128, 128), 0.5)

        # Donor image saliency has a concentrated hot spot in bottom-right quadrant
        donor_saliency = torch.zeros(1, 1, 128, 128)
        donor_saliency[:, :, 64:, 64:] = 1.0

        # Run transplant on Model C
        res_c = run_causal_transplant_intervention(
            model_c,
            target_image,
            donor_saliency,
            target_saliency=target_saliency,
        )

        # Model C: Allocation Ratio on donor ROI must increase following transplant
        assert res_c.transplanted_ar_on_donor_roi > res_c.baseline_ar_on_donor_roi
        assert res_c.ar_gain_on_donor_roi > 0.0
        assert res_c.corr_transplanted_vs_donor_saliency > res_c.corr_baseline_vs_donor_saliency

        # Run transplant on Model A (control)
        res_a = run_causal_transplant_intervention(
            model_a,
            target_image,
            donor_saliency,
            target_saliency=target_saliency,
        )

        # Model A: AR gain must be 0.0 because it does not respond to saliency
        assert res_a.ar_gain_on_donor_roi == pytest.approx(0.0, abs=1e-5)
        assert res_a.corr_transplanted_vs_donor_saliency == pytest.approx(
            res_a.corr_baseline_vs_donor_saliency, abs=1e-5
        )

    def test_transplant_intervention_batch_broadcast(self) -> None:
        """Verifies transplant intervention works when target image has batch size B = 2 and donor has B = 1."""
        model_c = ModelC(n_channels=32, m_channels=48)
        model_c.eval()

        target_batch = torch.randn(2, 3, 128, 128)
        donor_single = torch.zeros(1, 1, 128, 128)
        donor_single[:, :, 64:, 64:] = 1.0

        res = run_causal_transplant_intervention(
            model_c,
            target_batch,
            donor_single,
        )

        assert res.transplanted_bit_map.y_bits.shape[0] == 2
        assert res.ar_gain_on_donor_roi > 0.0
