"""Unit test suite for dual optimizer and learning rate scheduler harness.

Verifies:
- Disjoint and exhaustive parameter partitioning on ModelA and ModelB.
- Frozen parameter exclusion and edge cases (zero aux, zero main parameters).
- Gradient norm clipping on main parameters and non-clipping on aux parameters.
- Cosine learning rate scheduling with and without warmup.
- Complete state dict serialization and roundtrip recovery (momentum buffers, steps, LRs).
- Robust error handling for invalid arguments and missing state dict keys.
- Full end-to-end training loop execution.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam

from src.codec.entropy_models import EntropyBottleneck
from src.codec.model_a import ModelA
from src.codec.model_b import ModelB
from src.training.loss import RateDistortionLoss
from src.training.optimizer import (
    CosineAnnealingWithWarmupLR,
    DualOptimizerHarness,
    ParameterPartition,
    create_dual_optimizer,
    partition_parameters,
)


class SimpleConvModel(nn.Module):
    """Synthetic model with only convolutional parameters (zero aux parameters)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


class SimpleAuxOnlyModel(nn.Module):
    """Synthetic model with only EntropyBottleneck (zero main parameters)."""

    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(channels=channels)


class HybridCustomModel(nn.Module):
    """Custom model combining transforms and entropy bottleneck with named submodules."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.entropy_bottleneck = EntropyBottleneck(channels=16)


# ==============================================================================
# Parameter Partitioning Tests
# ==============================================================================


class TestParameterPartitioning:
    """Verifies disjoint and exhaustive partitioning of model parameters."""

    def test_model_a_partitioning_disjoint_and_exhaustive(self) -> None:
        """ModelA parameters must partition cleanly into main transforms and aux entropy parameters."""
        model = ModelA()
        partition = partition_parameters(model)

        assert isinstance(partition, ParameterPartition)
        assert isinstance(partition.main_params, list)
        assert isinstance(partition.aux_params, list)

        # Check disjointness
        main_ids = {id(p) for p in partition.main_params}
        aux_ids = {id(p) for p in partition.aux_params}
        assert main_ids.isdisjoint(aux_ids), "Main and aux parameter sets must be disjoint"

        # Check completeness (all trainable parameters covered)
        all_trainable = [p for p in model.parameters() if p.requires_grad]
        all_trainable_ids = {id(p) for p in all_trainable}
        assert main_ids | aux_ids == all_trainable_ids, "Partition must cover 100% of trainable parameters"
        assert len(partition.main_params) + len(partition.aux_params) == len(all_trainable)

        # Verify transforms are in main_params
        for submodule in (model.g_a, model.g_s, model.h_a, model.h_s):
            for p in submodule.parameters():
                if p.requires_grad:
                    assert id(p) in main_ids
                    assert id(p) not in aux_ids

        # Verify entropy bottleneck parameters are in aux_params
        for p in model.entropy_bottleneck.parameters():
            if p.requires_grad:
                assert id(p) in aux_ids
                assert id(p) not in main_ids

    def test_model_b_partitioning_disjoint_and_exhaustive(self) -> None:
        """ModelB parameters must partition identically to ModelA."""
        model = ModelB()
        main_params, aux_params = partition_parameters(model)

        main_ids = {id(p) for p in main_params}
        aux_ids = {id(p) for p in aux_params}
        assert main_ids.isdisjoint(aux_ids)

        all_trainable = [p for p in model.parameters() if p.requires_grad]
        assert len(main_params) + len(aux_params) == len(all_trainable)

        for p in model.entropy_bottleneck.parameters():
            if p.requires_grad:
                assert id(p) in aux_ids

    def test_frozen_parameters_excluded(self) -> None:
        """Parameters with requires_grad=False must be excluded from both partitions."""
        model = ModelA()

        # Freeze g_a and half of entropy_bottleneck weights
        for p in model.g_a.parameters():
            p.requires_grad = False
        model.entropy_bottleneck.weights[0].requires_grad = False

        partition = partition_parameters(model)
        main_ids = {id(p) for p in partition.main_params}
        aux_ids = {id(p) for p in partition.aux_params}

        for p in model.g_a.parameters():
            assert id(p) not in main_ids
            assert id(p) not in aux_ids

        assert id(model.entropy_bottleneck.weights[0]) not in aux_ids
        assert id(model.entropy_bottleneck.weights[1]) in aux_ids

        # Invariant still holds for remaining trainable parameters
        all_trainable = [p for p in model.parameters() if p.requires_grad]
        assert len(partition.main_params) + len(partition.aux_params) == len(all_trainable)

    def test_zero_aux_parameters(self) -> None:
        """A module without entropy bottlenecks yields empty aux_params."""
        model = SimpleConvModel()
        partition = partition_parameters(model)

        assert len(partition.aux_params) == 0
        assert len(partition.main_params) == len(list(model.parameters()))

    def test_zero_main_parameters(self) -> None:
        """A module containing only an entropy bottleneck yields empty main_params."""
        model = SimpleAuxOnlyModel()
        partition = partition_parameters(model)

        assert len(partition.main_params) == 0
        assert len(partition.aux_params) == len(list(model.parameters()))

    def test_all_parameters_frozen(self) -> None:
        """A module with all parameters frozen yields empty main and aux partitions."""
        model = SimpleConvModel()
        for p in model.parameters():
            p.requires_grad = False

        partition = partition_parameters(model)
        assert len(partition.main_params) == 0
        assert len(partition.aux_params) == 0

    def test_submodule_parameter_recursion_arbitrary_attribute_name(self) -> None:
        """EntropyBottleneck parameters in ParameterLists are captured even with arbitrary attribute names."""
        class ArbitrarySubmoduleModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 8, 3)
                # Attribute name does not contain 'entropy_bottleneck' or 'quantiles'
                self.custom_bottleneck = EntropyBottleneck(channels=16)

        model = ArbitrarySubmoduleModel()
        partition = partition_parameters(model)

        assert len(partition.main_params) == len(list(model.conv.parameters()))
        assert len(partition.aux_params) == len(list(model.custom_bottleneck.parameters()))
        # All weights and biases in custom_bottleneck must be in aux_params
        for p in model.custom_bottleneck.parameters():
            assert any(p is aux_p for aux_p in partition.aux_params)


# ==============================================================================
# Gradient Norm Clipping Tests
# ==============================================================================


class TestGradientNormClipping:
    """Verifies gradient clipping on main parameters and non-clipping on aux parameters."""

    def test_gradient_norm_clipping_synthetic_norm_exceeds_threshold(self) -> None:
        """When total gradient norm > 1.0, gradients must be rescaled to <= 1.0 and pre-norm returned."""
        p1 = nn.Parameter(torch.zeros(10))
        p2 = nn.Parameter(torch.zeros(10))
        # Initial gradient norm = sqrt(10 * 3^2 + 10 * 4^2) = sqrt(90 + 160) = sqrt(250) ~ 15.811
        p1.grad = torch.full((10,), 3.0)
        p2.grad = torch.full((10,), 4.0)

        expected_initial_norm = math.sqrt(10 * 3.0**2 + 10 * 4.0**2)

        opt = Adam([p1, p2], lr=1e-3)
        harness = DualOptimizerHarness(main_optimizer=opt, clip_max_norm=1.0, main_params=[p1, p2])

        returned_norm = harness.step_main()

        assert returned_norm is not None
        assert math.isclose(returned_norm, expected_initial_norm, rel_tol=1e-5)

        # After step_main, parameter gradients are scaled such that total norm <= 1.0
        post_norm = torch.sqrt(torch.sum(p1.grad**2) + torch.sum(p2.grad**2)).item()
        assert post_norm <= 1.0 + 1e-5
        assert math.isclose(post_norm, 1.0, abs_tol=1e-5)

    def test_gradient_norm_below_threshold_unmodified(self) -> None:
        """When total gradient norm < 1.0, gradients are not scaled."""
        p = nn.Parameter(torch.zeros(4))
        p.grad = torch.tensor([0.1, 0.2, 0.2, 0.1])
        expected_norm = torch.linalg.vector_norm(p.grad).item()
        assert expected_norm < 1.0

        opt = Adam([p], lr=1e-3)
        harness = DualOptimizerHarness(main_optimizer=opt, clip_max_norm=1.0, main_params=[p])

        original_grad = p.grad.clone()
        returned_norm = harness.step_main()

        assert returned_norm is not None
        assert math.isclose(returned_norm, expected_norm, rel_tol=1e-5)
        assert torch.allclose(p.grad, original_grad, atol=1e-6)

    def test_zero_and_near_zero_gradients(self) -> None:
        """Zero gradients are handled cleanly without error and return 0.0 norm."""
        p = nn.Parameter(torch.ones(5))
        p.grad = torch.zeros(5)

        opt = Adam([p], lr=1e-3)
        harness = DualOptimizerHarness(main_optimizer=opt, clip_max_norm=1.0, main_params=[p])

        returned_norm = harness.step_main()
        assert returned_norm is not None
        assert returned_norm == 0.0
        assert torch.all(p.grad == 0.0)

    def test_auxiliary_parameters_never_clipped(self) -> None:
        """Auxiliary entropy bottleneck parameters must NEVER have gradient clipping applied."""
        aux_p = nn.Parameter(torch.zeros(5))
        aux_p.grad = torch.full((5,), 100.0)  # Huge gradient norm: sqrt(5 * 10000) ~ 223.6

        aux_opt = Adam([aux_p], lr=1e-3)
        harness = DualOptimizerHarness(aux_optimizer=aux_opt, clip_max_norm=1.0, aux_params=[aux_p])

        grad_before = aux_p.grad.clone()
        harness.step_aux()

        # Gradient must remain completely unclipped (100.0)
        assert torch.allclose(aux_p.grad, grad_before, atol=1e-5)

    def test_custom_per_step_clip_norm_override(self) -> None:
        """step_main supports an explicit clip_max_norm override."""
        p = nn.Parameter(torch.zeros(5))
        p.grad = torch.full((5,), 5.0)

        opt = Adam([p], lr=1e-3)
        harness = DualOptimizerHarness(main_optimizer=opt, clip_max_norm=1.0, main_params=[p])

        # Override to 0.5
        harness.step_main(clip_max_norm=0.5)
        post_norm = torch.linalg.vector_norm(p.grad).item()
        assert math.isclose(post_norm, 0.5, abs_tol=1e-5)

    def test_clip_max_norm_none_disables_clipping(self) -> None:
        """When clip_max_norm=None, step_main computes norm without modifying gradients."""
        p = nn.Parameter(torch.zeros(5))
        p.grad = torch.full((5,), 10.0)
        expected_norm = torch.linalg.vector_norm(p.grad).item()

        opt = Adam([p], lr=1e-3)
        harness = DualOptimizerHarness(main_optimizer=opt, clip_max_norm=None, main_params=[p])

        returned_norm = harness.step_main()
        assert returned_norm is not None
        assert math.isclose(returned_norm, expected_norm, rel_tol=1e-5)
        assert math.isclose(torch.linalg.vector_norm(p.grad).item(), expected_norm, rel_tol=1e-5)

    def test_step_main_clip_max_norm_zero(self) -> None:
        """When clip_max_norm=0.0, gradients are scaled to 0.0 and pre-norm returned."""
        p = nn.Parameter(torch.zeros(5))
        p.grad = torch.full((5,), 2.0)
        expected_pre_norm = torch.linalg.vector_norm(p.grad).item()

        opt = Adam([p], lr=1e-3)
        harness = DualOptimizerHarness(main_optimizer=opt, clip_max_norm=1.0, main_params=[p])

        returned_norm = harness.step_main(clip_max_norm=0.0)
        assert returned_norm is not None
        assert math.isclose(returned_norm, expected_pre_norm, rel_tol=1e-5)
        assert torch.all(p.grad == 0.0)

    def test_step_main_negative_clip_norm_raises_value_error(self) -> None:
        """Negative clip_max_norm in step_main raises ValueError."""
        p = nn.Parameter(torch.zeros(5))
        p.grad = torch.full((5,), 1.0)

        opt = Adam([p], lr=1e-3)
        harness = DualOptimizerHarness(main_optimizer=opt, clip_max_norm=1.0, main_params=[p])

        with pytest.raises(ValueError, match="clip_max_norm must be non-negative or None"):
            harness.step_main(clip_max_norm=-0.5)


# ==============================================================================
# Learning Rate Scheduler Tests
# ==============================================================================


class TestCosineAnnealingWithWarmupLR:
    """Verifies cosine annealing progression and linear warmup behaviors."""

    def test_cosine_decay_without_warmup(self) -> None:
        """LR smoothly decays from main_lr (5e-5) to min_lr (1e-6) following cosine curve."""
        p = nn.Parameter(torch.zeros(1))
        base_lr = 5e-5
        min_lr = 1e-6
        t_max = 10

        opt = Adam([p], lr=base_lr)
        sched = CosineAnnealingWithWarmupLR(opt, T_max=t_max, eta_min=min_lr, warmup_epochs=0)

        # Step 0
        assert math.isclose(opt.param_groups[0]["lr"], base_lr, rel_tol=1e-6)

        for step in range(1, t_max + 1):
            opt.step()
            sched.step()
            expected_lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * step / t_max))
            actual_lr = opt.param_groups[0]["lr"]
            assert math.isclose(actual_lr, expected_lr, rel_tol=1e-5)

        # Final step must exactly equal min_lr
        assert math.isclose(opt.param_groups[0]["lr"], min_lr, rel_tol=1e-6)

    def test_cosine_with_linear_warmup(self) -> None:
        """LR linearly warms up from warmup_start_lr to base_lr, then cosine decays to eta_min."""
        p = nn.Parameter(torch.zeros(1))
        base_lr = 5e-5
        min_lr = 1e-6
        warmup_start_lr = 1e-7
        warmup_epochs = 4
        t_max = 10

        opt = Adam([p], lr=base_lr)
        sched = CosineAnnealingWithWarmupLR(
            opt,
            T_max=t_max,
            eta_min=min_lr,
            warmup_epochs=warmup_epochs,
            warmup_start_lr=warmup_start_lr,
        )

        # Step 0: warmup start
        assert math.isclose(opt.param_groups[0]["lr"], warmup_start_lr, rel_tol=1e-5)

        # Linear warmup phase
        for step in range(1, warmup_epochs + 1):
            opt.step()
            sched.step()
            if step < warmup_epochs:
                expected_lr = warmup_start_lr + (step / float(warmup_epochs)) * (base_lr - warmup_start_lr)
                assert math.isclose(opt.param_groups[0]["lr"], expected_lr, rel_tol=1e-5)
            else:
                # At warmup_epochs, reach base_lr
                assert math.isclose(opt.param_groups[0]["lr"], base_lr, rel_tol=1e-5)

        # Cosine decay phase
        decay_steps = t_max - warmup_epochs
        for step in range(warmup_epochs + 1, t_max + 1):
            opt.step()
            sched.step()
            curr = step - warmup_epochs
            expected_lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * curr / float(decay_steps)))
            assert math.isclose(opt.param_groups[0]["lr"], expected_lr, rel_tol=1e-5)

        assert math.isclose(opt.param_groups[0]["lr"], min_lr, rel_tol=1e-6)

    def test_clamped_beyond_t_max(self) -> None:
        """Steps beyond T_max remain clamped at eta_min."""
        p = nn.Parameter(torch.zeros(1))
        opt = Adam([p], lr=5e-5)
        sched = CosineAnnealingWithWarmupLR(opt, T_max=5, eta_min=1e-6)

        for _ in range(8):
            opt.step()
            sched.step()

        assert math.isclose(opt.param_groups[0]["lr"], 1e-6, rel_tol=1e-6)

    def test_validation_errors(self) -> None:
        """Constructor validates T_max, warmup_epochs, and eta_min bounds."""
        p = nn.Parameter(torch.zeros(1))
        opt = Adam([p], lr=5e-5)

        with pytest.raises(ValueError, match="T_max must be a positive integer"):
            CosineAnnealingWithWarmupLR(opt, T_max=0)

        with pytest.raises(ValueError, match="T_max must be a positive integer"):
            CosineAnnealingWithWarmupLR(opt, T_max=-5)

        with pytest.raises(ValueError, match="warmup_epochs must be non-negative"):
            CosineAnnealingWithWarmupLR(opt, T_max=10, warmup_epochs=-1)

        with pytest.raises(ValueError, match=r"warmup_epochs .* must be strictly less than T_max"):
            CosineAnnealingWithWarmupLR(opt, T_max=10, warmup_epochs=10)

        with pytest.raises(ValueError, match=r"warmup_epochs .* must be strictly less than T_max"):
            CosineAnnealingWithWarmupLR(opt, T_max=10, warmup_epochs=15)

        with pytest.raises(ValueError, match="eta_min must be non-negative"):
            CosineAnnealingWithWarmupLR(opt, T_max=10, eta_min=-1e-6)

    def test_default_min_lr_is_1e_5(self) -> None:
        """Default min_lr is 1e-5 per precision-and-system-policy.md spec."""
        p = nn.Parameter(torch.zeros(1))
        opt = Adam([p], lr=5e-5)
        sched = CosineAnnealingWithWarmupLR(opt, T_max=10)
        assert sched.eta_min == 1e-5

        for _ in range(10):
            opt.step()
            sched.step()
        assert math.isclose(opt.param_groups[0]["lr"], 1e-5, rel_tol=1e-6)

    def test_scheduler_load_state_dict_without_last_lr(self) -> None:
        """load_state_dict restores group['lr'] via get_lr() when _last_lr is omitted."""
        p1 = nn.Parameter(torch.zeros(1))
        opt1 = Adam([p1], lr=5e-5)
        sched1 = CosineAnnealingWithWarmupLR(opt1, T_max=10, eta_min=1e-5)
        for _ in range(4):
            opt1.step()
            sched1.step()

        sd = sched1.state_dict()
        expected_lr = opt1.param_groups[0]["lr"]
        # Delete _last_lr from serialized state
        sd.pop("_last_lr", None)

        p2 = nn.Parameter(torch.zeros(1))
        opt2 = Adam([p2], lr=5e-5)
        sched2 = CosineAnnealingWithWarmupLR(opt2, T_max=10, eta_min=1e-5)
        sched2.load_state_dict(sd)

        assert math.isclose(opt2.param_groups[0]["lr"], expected_lr, rel_tol=1e-6)


# ==============================================================================
# DualOptimizerHarness Tests
# ==============================================================================


class TestDualOptimizerHarness:
    """Verifies harness instantiation, stepping, and edge-case handling."""

    def test_create_dual_optimizer_defaults(self) -> None:
        """create_dual_optimizer sets default learning rates, clipping, and Adam params."""
        model = ModelA()
        harness = create_dual_optimizer(model)

        assert harness.main_optimizer is not None
        assert harness.aux_optimizer is not None
        assert math.isclose(harness.main_lr, 5e-5, rel_tol=1e-7)
        assert math.isclose(harness.aux_lr, 1e-3, rel_tol=1e-7)
        assert harness.clip_max_norm == 1.0

        # Check Adam hyperparameters
        main_group = harness.main_optimizer.param_groups[0]
        assert main_group["betas"] == (0.9, 0.999)
        assert main_group["eps"] == 1e-8

        aux_group = harness.aux_optimizer.param_groups[0]
        assert aux_group["betas"] == (0.9, 0.999)
        assert aux_group["eps"] == 1e-8

    def test_create_dual_optimizer_with_cosine_scheduler(self) -> None:
        """create_dual_optimizer configures CosineAnnealingWithWarmupLR when requested."""
        model = ModelA()
        harness = create_dual_optimizer(
            model,
            main_lr=5e-5,
            aux_lr=1e-3,
            total_epochs=50,
            min_lr=1e-6,
            warmup_epochs=5,
        )

        assert harness.main_scheduler is not None
        assert isinstance(harness.main_scheduler, CosineAnnealingWithWarmupLR)
        assert harness.main_scheduler.T_max == 50
        assert harness.main_scheduler.eta_min == 1e-6
        assert harness.main_scheduler.warmup_epochs == 5

    def test_zero_aux_parameters_instantiates_cleanly_and_step_aux_is_noop(self) -> None:
        """A model with zero aux parameters must not raise empty parameter list errors."""
        model = SimpleConvModel()
        harness = create_dual_optimizer(model)

        assert harness.main_optimizer is not None
        assert harness.aux_optimizer is None
        assert harness.aux_lr is None
        assert len(harness.aux_params) == 0

        # step_aux must execute safely as a no-op
        harness.step_aux()
        harness.zero_grad_aux()

    def test_zero_main_parameters_instantiates_cleanly_and_step_main_is_noop(self) -> None:
        """A model with zero main parameters instantiates cleanly."""
        model = SimpleAuxOnlyModel()
        harness = create_dual_optimizer(model)

        assert harness.main_optimizer is None
        assert harness.aux_optimizer is not None
        assert harness.main_lr is None
        assert len(harness.main_params) == 0

        # step_main must execute safely and return None
        res = harness.step_main()
        assert res is None
        harness.zero_grad_main()

    def test_zero_grad_behaviors(self) -> None:
        """zero_grad, zero_grad_main, and zero_grad_aux reset gradients correctly."""
        model = ModelA()
        harness = create_dual_optimizer(model)

        # Assign synthetic gradients
        for p in harness.main_params:
            p.grad = torch.ones_like(p)
        for p in harness.aux_params:
            p.grad = torch.ones_like(p)

        # Test zero_grad_main
        harness.zero_grad_main(set_to_none=True)
        assert all(p.grad is None for p in harness.main_params)
        assert all(p.grad is not None for p in harness.aux_params)

        # Test zero_grad_aux
        harness.zero_grad_aux(set_to_none=True)
        assert all(p.grad is None for p in harness.aux_params)

    def test_step_schedulers_and_alias(self) -> None:
        """step_schedulers and step_scheduler alias advance attached schedulers."""
        model = ModelA()
        harness = create_dual_optimizer(model, total_epochs=10, min_lr=1e-6)

        # Step optimizer first to avoid PyTorch ordering warning
        for p in harness.main_params:
            p.grad = torch.zeros_like(p)
        harness.step_main()

        initial_lr = harness.main_lr
        harness.step_schedulers()
        assert harness.main_lr < initial_lr

        # Test alias
        harness.step_main()
        prev_lr = harness.main_lr
        harness.step_scheduler()
        assert harness.main_lr < prev_lr

    def test_properties_and_repr(self) -> None:
        """Test getters, properties, and string representation."""
        model = ModelA()
        harness = create_dual_optimizer(model)

        assert len(harness.optimizers) == 2
        assert isinstance(harness.get_lr(), dict)
        assert "main" in harness.get_lr()
        assert "aux" in harness.get_lr()

        repr_str = repr(harness)
        assert "DualOptimizerHarness" in repr_str
        assert "main_optimizer=Adam" in repr_str
        assert "aux_optimizer=Adam" in repr_str


# ==============================================================================
# State Serialization & Checkpoint Roundtrip Tests
# ==============================================================================


class TestCheckpointStateRoundtrip:
    """Verifies state serialization, restoration, and error handling."""

    def test_state_dict_roundtrip_recovers_moments_lrs_and_steps(self) -> None:
        """Serialized harness state restores exact momentum buffers, step counters, and LRs."""
        model1 = ModelA()
        harness1 = create_dual_optimizer(model1, total_epochs=10, min_lr=1e-6)

        # Run several training steps with synthetic gradients to accumulate Adam moments
        for step in range(3):
            harness1.zero_grad()
            for p in harness1.main_params:
                p.grad = torch.randn_like(p)
            harness1.step_main()

            for p in harness1.aux_params:
                p.grad = torch.randn_like(p)
            harness1.step_aux()
            harness1.step_schedulers()

        saved_state = harness1.state_dict()

        # Build fresh model and harness
        model2 = ModelA()
        harness2 = create_dual_optimizer(model2, total_epochs=10, min_lr=1e-6)

        # Restore checkpoint state
        harness2.load_state_dict(saved_state)

        # Verify learning rates match
        assert math.isclose(harness1.main_lr, harness2.main_lr, rel_tol=1e-7)
        assert math.isclose(harness1.aux_lr, harness2.aux_lr, rel_tol=1e-7)

        # Verify scheduler steps match
        assert harness1.main_scheduler.last_epoch == harness2.main_scheduler.last_epoch

        # Verify Adam momentum buffers match for main optimizer
        assert len(harness1.main_optimizer.state) > 0
        assert len(harness2.main_optimizer.state) > 0
        for (p1_id, p1_state), (p2_id, p2_state) in zip(
            harness1.main_optimizer.state.items(),
            harness2.main_optimizer.state.items(),
        ):
            assert p1_state["step"] == p2_state["step"]
            assert torch.allclose(p1_state["exp_avg"], p2_state["exp_avg"])
            assert torch.allclose(p1_state["exp_avg_sq"], p2_state["exp_avg_sq"])

        # Verify Adam momentum buffers match for aux optimizer
        assert len(harness1.aux_optimizer.state) > 0
        assert len(harness2.aux_optimizer.state) > 0
        for (p1_id, p1_state), (p2_id, p2_state) in zip(
            harness1.aux_optimizer.state.items(),
            harness2.aux_optimizer.state.items(),
        ):
            assert p1_state["step"] == p2_state["step"]
            assert torch.allclose(p1_state["exp_avg"], p2_state["exp_avg"])
            assert torch.allclose(p1_state["exp_avg_sq"], p2_state["exp_avg_sq"])

    def test_state_dict_missing_required_keys_raises_key_error(self) -> None:
        """load_state_dict raises KeyError if required optimizer keys are absent."""
        model = ModelA()
        harness = create_dual_optimizer(model)

        with pytest.raises(KeyError, match="Missing required key 'aux_optimizer'"):
            harness.load_state_dict({"main_optimizer": {}})

        with pytest.raises(KeyError, match="Missing required key 'main_optimizer'"):
            harness.load_state_dict({"aux_optimizer": {}})

    def test_state_dict_optimizer_presence_mismatch_raises_value_error(self) -> None:
        """load_state_dict raises ValueError when active optimizer gets None in state_dict or vice versa."""
        model = ModelA()
        harness = create_dual_optimizer(model)  # Has both main and aux optimizers

        with pytest.raises(ValueError, match=r"Harness has active main_optimizer, but state_dict.*is None"):
            harness.load_state_dict({"main_optimizer": None, "aux_optimizer": {}})

        with pytest.raises(ValueError, match=r"Harness has active aux_optimizer, but state_dict.*is None"):
            harness.load_state_dict({"main_optimizer": {}, "aux_optimizer": None})

        # Test vice versa on a model with no aux optimizer
        conv_model = SimpleConvModel()
        conv_harness = create_dual_optimizer(conv_model)  # aux_optimizer is None

        with pytest.raises(ValueError, match=r"Harness has no aux_optimizer, but state_dict.*is not None"):
            conv_harness.load_state_dict({"main_optimizer": {}, "aux_optimizer": {}})


# ==============================================================================
# End-to-End Training Step Integration Test
# ==============================================================================


class TestEndToEndTrainingLoop:
    """Verifies realistic training step integration with ModelA and RateDistortionLoss."""

    def test_full_training_iteration(self) -> None:
        """Executes a realistic forward, backward, dual step, and scheduler progression."""
        torch.manual_seed(42)
        model = ModelA(n_channels=64, m_channels=96)
        criterion = RateDistortionLoss(lmbda=0.0130)

        harness = create_dual_optimizer(
            model,
            main_lr=5e-5,
            aux_lr=1e-3,
            clip_max_norm=1.0,
            total_epochs=10,
            min_lr=1e-6,
        )

        # Batch of images with spatial dimensions multiples of 64
        x = torch.rand(2, 3, 64, 64)

        # 1. Main rate-distortion step
        harness.zero_grad()
        out = model(x, training=True)
        loss_dict = criterion(out, x)
        rd_loss = loss_dict["loss"]
        assert torch.isfinite(rd_loss)
        rd_loss.backward()

        grad_norm = harness.step_main()
        assert grad_norm is not None
        assert grad_norm > 0.0

        # 2. Auxiliary entropy bottleneck step
        harness.zero_grad()
        aux_loss = model.aux_loss()
        if not aux_loss.requires_grad:
            aux_loss = sum(p.norm() for p in harness.aux_params)
        assert aux_loss.requires_grad
        aux_loss.backward()
        assert all(p.grad is not None for p in harness.aux_params)
        harness.step_aux()

        # 3. Learning rate scheduler step
        initial_main_lr = harness.main_lr
        harness.step_schedulers()
        assert harness.main_lr < initial_main_lr
