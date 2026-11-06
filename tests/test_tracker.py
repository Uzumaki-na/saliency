"""Unit and integration test suite for MLflow tracking, metrics, and checkpoint persistence engine.

Verifies:
- PSNR evaluation without division by zero on identical tensors, and exact mathematical scaling.
- 5-scale MS-SSIM evaluation according to Wang et al. (2003) with 4D and spatial dimension validation.
- BPP evaluation from continuous likelihoods, dictionaries, and raw byte counts.
- Post-training CDF table serialization gate (compilation of _quantized_cdf before saving).
- Checkpoint persistence and roundtrip restoration for models and DualOptimizerHarness.
- MLflowTracker lifecycle (experiment creation, run management, context managers, tags, params, metrics, artifacts).
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest
import torch
import torch.nn as nn

from src.codec.entropy_models import EntropyBottleneck
from src.codec.model_a import ModelA
from src.training import (
    MLflowTracker,
    compute_bpp,
    compute_mse,
    compute_msssim,
    compute_psnr,
    create_dual_optimizer,
    load_checkpoint,
    save_checkpoint,
)

# ---------------------------------------------------------------------------
# Test Metrics (PSNR, MSE, MS-SSIM, BPP)
# ---------------------------------------------------------------------------


class TestMetrics:
    """Tests covering scalar metric computation functions."""

    def test_compute_mse_identical(self) -> None:
        """MSE of identical tensors must be exactly 0.0."""
        x = torch.rand(2, 3, 64, 64)
        mse = compute_mse(x, x)
        assert mse == 0.0

    def test_compute_mse_known_diff(self) -> None:
        """MSE of tensors differing by constant offset delta must be delta^2."""
        x = torch.zeros(1, 3, 32, 32)
        x_hat = torch.full((1, 3, 32, 32), 0.5)
        mse = compute_mse(x, x_hat)
        assert pytest.approx(mse, abs=1e-6) == 0.25

    def test_compute_mse_as_tensor(self) -> None:
        """as_tensor=True returns scalar torch.Tensor."""
        x = torch.rand(1, 3, 32, 32)
        mse_t = compute_mse(x, x, as_tensor=True)
        assert isinstance(mse_t, torch.Tensor)
        assert mse_t.item() == 0.0

    def test_compute_mse_shape_mismatch(self) -> None:
        """MSE raises ValueError on shape mismatch."""
        x = torch.rand(1, 3, 32, 32)
        y = torch.rand(1, 3, 64, 64)
        with pytest.raises(ValueError, match="shape mismatch"):
            compute_mse(x, y)

    def test_compute_psnr_identical_returns_max_clamp(self) -> None:
        """PSNR of identical tensors must return 100.0 dB without division by zero."""
        x = torch.rand(2, 3, 64, 64)
        psnr = compute_psnr(x, x)
        assert psnr == 100.0

    def test_compute_psnr_known_values(self) -> None:
        """PSNR with known MSE must match 10 * log10(1 / MSE)."""
        # When MSE = 0.01 and data_range = 1.0, PSNR = 10 * log10(100) = 20.0 dB
        x = torch.zeros(1, 1, 32, 32)
        x_hat = torch.full((1, 1, 32, 32), 0.1)
        psnr = compute_psnr(x, x_hat, data_range=1.0)
        assert pytest.approx(psnr, abs=1e-4) == 20.0

        # When MSE = 0.001 and data_range = 1.0, PSNR = 10 * log10(1000) = 30.0 dB
        diff = float(10.0**-1.5)
        x_hat_30 = torch.full((1, 1, 32, 32), diff)
        psnr_30 = compute_psnr(x, x_hat_30, data_range=1.0)
        assert pytest.approx(psnr_30, abs=1e-4) == 30.0

    def test_compute_psnr_no_premature_clamp(self) -> None:
        """PSNR does not prematurely clamp for small non-zero MSE (e.g. 1e-12)."""
        x = torch.zeros(1, 1, 32, 32)
        x_hat = torch.full((1, 1, 32, 32), 1e-6)  # MSE = 1e-12 -> PSNR = 120 dB
        psnr = compute_psnr(x, x_hat, data_range=1.0, max_psnr=150.0)
        assert pytest.approx(psnr, abs=1e-2) == 120.0

    def test_compute_psnr_custom_max_clamp(self) -> None:
        """PSNR respects custom max_psnr clamp."""
        x = torch.rand(1, 3, 32, 32)
        psnr = compute_psnr(x, x, max_psnr=80.0)
        assert psnr == 80.0

    def test_compute_psnr_as_tensor(self) -> None:
        """PSNR returns torch.Tensor when as_tensor=True."""
        x = torch.rand(1, 3, 32, 32)
        psnr_t = compute_psnr(x, x, as_tensor=True)
        assert isinstance(psnr_t, torch.Tensor)
        assert psnr_t.item() == 100.0

    def test_compute_psnr_validation(self) -> None:
        """PSNR raises appropriate errors for invalid inputs."""
        x = torch.rand(1, 3, 32, 32)
        y = torch.rand(1, 3, 64, 64)
        with pytest.raises(ValueError, match="shape mismatch"):
            compute_psnr(x, y)

        with pytest.raises(TypeError, match=r"torch\.Tensor"):
            compute_psnr([1, 2, 3], x)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="positive"):
            compute_psnr(x, x, data_range=0.0)

    def test_compute_msssim_identical_returns_one(self) -> None:
        """MS-SSIM of identical 4D tensors must be exactly 1.0."""
        x = torch.rand(2, 3, 64, 64)
        msssim = compute_msssim(x, x)
        assert pytest.approx(msssim, abs=1e-6) == 1.0

    def test_compute_msssim_noisy_high_fidelity(self) -> None:
        """MS-SSIM of slightly perturbed images should be close to 1.0 (> 0.95)."""
        torch.manual_seed(42)
        x = torch.rand(1, 3, 64, 64)
        noise = torch.randn_like(x) * 0.01
        x_noisy = torch.clamp(x + noise, 0.0, 1.0)
        msssim = compute_msssim(x, x_noisy)
        assert 0.95 < msssim <= 1.0

    def test_compute_msssim_shape_validation(self) -> None:
        """MS-SSIM requires 4D tensors and spatial dimensions >= 32."""
        # Non-4D tensor
        x_3d = torch.rand(3, 64, 64)
        with pytest.raises(ValueError, match="4D input tensors"):
            compute_msssim(x_3d, x_3d)

        # Spatial dimension < 32
        x_small = torch.rand(1, 3, 16, 16)
        with pytest.raises(ValueError, match=r"spatial dimensions .* must be >= 32"):
            compute_msssim(x_small, x_small)

        # Mismatched shapes
        x_a = torch.rand(1, 3, 64, 64)
        x_b = torch.rand(1, 3, 64, 128)
        with pytest.raises(ValueError, match="shape mismatch"):
            compute_msssim(x_a, x_b)

        # Non-tensor input
        with pytest.raises(TypeError, match=r"torch\.Tensor"):
            compute_msssim("invalid", x_a)  # type: ignore[arg-type]

    def test_compute_msssim_as_tensor(self) -> None:
        """MS-SSIM returns torch.Tensor when as_tensor=True."""
        x = torch.rand(1, 3, 64, 64)
        res_t = compute_msssim(x, x, as_tensor=True)
        assert isinstance(res_t, torch.Tensor)
        assert pytest.approx(res_t.item(), abs=1e-6) == 1.0

    def test_compute_bpp_from_likelihood_tensor(self) -> None:
        """BPP calculated from likelihoods tensor matches -sum(log2(p)) / num_pixels."""
        # 1 item, 1 channel, 4x4 image
        likelihoods = torch.full((1, 1, 2, 2), 0.5)  # log2(0.5) = -1.0 -> 4 bits total
        bpp = compute_bpp(likelihoods, num_pixels=16)  # 4 bits / 16 pixels = 0.25 bpp
        assert pytest.approx(bpp, abs=1e-6) == 0.25

    def test_compute_bpp_from_dict(self) -> None:
        """BPP calculated from dict of likelihoods sums bit contributions."""
        y_lik = torch.full((1, 1, 4, 4), 0.5)  # 16 bits
        z_lik = torch.full((1, 1, 2, 2), 0.25)  # 4 * 2 = 8 bits
        # Total bits = 24. Image size = 64 pixels -> 24 / 64 = 0.375 bpp
        bpp = compute_bpp({"y": y_lik, "z": z_lik}, num_pixels=64)
        assert pytest.approx(bpp, abs=1e-6) == 0.375

    def test_compute_bpp_from_bytes(self) -> None:
        """BPP calculated from byte count matches (bytes * 8) / num_pixels."""
        num_bytes = 1024  # 8192 bits
        num_pixels = 4096
        bpp = compute_bpp(num_bytes, num_pixels=num_pixels)
        assert pytest.approx(bpp, abs=1e-6) == 2.0

    def test_compute_bpp_from_image_shape(self) -> None:
        """BPP infers num_pixels from 4D image_shape."""
        likelihoods = torch.full((2, 1, 2, 2), 0.5)  # 8 bits total
        # Image shape: (2, 3, 4, 4) -> 2 * 4 * 4 = 32 pixels -> 8 / 32 = 0.25 bpp
        bpp = compute_bpp(likelihoods, image_shape=(2, 3, 4, 4))
        assert pytest.approx(bpp, abs=1e-6) == 0.25

    def test_compute_bpp_validation(self) -> None:
        """BPP raises errors for missing or invalid pixel counts."""
        with pytest.raises(ValueError, match="Either num_pixels or image_shape"):
            compute_bpp(100)

        # Likelihood tensor without num_pixels or image_shape must raise ValueError
        likelihoods = torch.full((1, 192, 4, 4), 0.5)
        with pytest.raises(ValueError, match="Either num_pixels or image_shape"):
            compute_bpp(likelihoods)

        with pytest.raises(ValueError, match="positive"):
            compute_bpp(100, num_pixels=0)

    def test_compute_bpp_autograd_graph_intact(self) -> None:
        """compute_bpp with as_tensor=True retains the autograd computation graph."""
        y_lik = torch.full((1, 1, 4, 4), 0.5, requires_grad=True)
        z_lik = torch.full((1, 1, 2, 2), 0.5, requires_grad=True)
        bpp_tensor = compute_bpp({"y": y_lik, "z": z_lik}, num_pixels=64, as_tensor=True)
        assert isinstance(bpp_tensor, torch.Tensor)
        assert bpp_tensor.requires_grad
        bpp_tensor.backward()
        assert y_lik.grad is not None
        assert z_lik.grad is not None


# ---------------------------------------------------------------------------
# Test Checkpoint Persistence & Post-Training CDF Gate
# ---------------------------------------------------------------------------


class MockUncompilableModel(nn.Module):
    """Toy model simulating failed CDF compilation."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1)
        self.register_buffer("_quantized_cdf", torch.empty((0, 0), dtype=torch.int32))

    def update(self, force: bool = False) -> None:
        # Intentionally leaves _quantized_cdf empty to trigger gate failure
        pass


class TestCheckpointPersistenceAndCDFGate:
    """Tests covering atomic checkpoint persistence, CDF gate enforcement, and state restoration."""

    def test_save_checkpoint_compiles_cdfs_model_a(self, tmp_path: Path) -> None:
        """save_checkpoint with compile_cdfs=True compiles empty CDF tables in ModelA."""
        model = ModelA()
        # Verify initial state: CDF buffers are empty before compilation
        assert model.entropy_bottleneck._quantized_cdf.numel() == 0
        assert model.gaussian_conditional._quantized_cdf.numel() == 0

        ckpt_file = tmp_path / "model_a_compiled.pt"
        saved_path = save_checkpoint(model, ckpt_file, compile_cdfs=True)
        assert saved_path.is_file()

        # Verify CDF buffers are populated in memory and in the saved checkpoint
        assert model.entropy_bottleneck._quantized_cdf.numel() > 0
        assert model.gaussian_conditional._quantized_cdf.numel() > 0

        data = torch.load(saved_path, map_location="cpu", weights_only=False)
        assert data["state_dict"]["entropy_bottleneck._quantized_cdf"].numel() > 0
        assert data["state_dict"]["gaussian_conditional._quantized_cdf"].numel() > 0

    def test_save_checkpoint_gate_raises_on_empty_cdf(self, tmp_path: Path) -> None:
        """save_checkpoint raises RuntimeError if empty _quantized_cdf remains with compile_cdfs=True."""
        model = MockUncompilableModel()
        ckpt_file = tmp_path / "failed_gate.pt"
        with pytest.raises(RuntimeError, match="Post-training CDF serialization gate failed"):
            save_checkpoint(model, ckpt_file, compile_cdfs=True)

        # File must not exist if gate failed
        assert not ckpt_file.exists()

    def test_save_checkpoint_uncompiled_when_flag_false(self, tmp_path: Path) -> None:
        """save_checkpoint with compile_cdfs=False permits saving without compiling CDF tables."""
        model = MockUncompilableModel()
        ckpt_file = tmp_path / "uncompiled.pt"
        saved_path = save_checkpoint(model, ckpt_file, compile_cdfs=False)
        assert saved_path.is_file()

    def test_save_checkpoint_strips_module_prefix_on_wrapped_model(self, tmp_path: Path) -> None:
        """save_checkpoint saves state_dict without 'module.' prefix when model is wrapped."""
        model = ModelA()
        wrapper = nn.Module()
        wrapper.module = model
        ckpt_file = tmp_path / "wrapped_model.pt"
        saved_path = save_checkpoint(wrapper, ckpt_file, compile_cdfs=True)
        data = torch.load(saved_path, map_location="cpu", weights_only=False)
        assert any(k.startswith("g_a.") for k in data["state_dict"])
        assert not any(k.startswith("module.") for k in data["state_dict"])

    def test_save_checkpoint_submodule_update_invoked(self, tmp_path: Path) -> None:
        """Submodule update(force=True) is invoked even if root module lacks update()."""

        class ModelWithoutRootUpdate(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.entropy_bottleneck = EntropyBottleneck(channels=16)

        model = ModelWithoutRootUpdate()
        assert model.entropy_bottleneck._quantized_cdf.numel() == 0
        ckpt_file = tmp_path / "submodule_update.pt"
        save_checkpoint(model, ckpt_file, compile_cdfs=True)
        assert model.entropy_bottleneck._quantized_cdf.numel() > 0

    def test_checkpoint_roundtrip_weights(self, tmp_path: Path) -> None:
        """Loaded model weights produce identical outputs on forward evaluation."""
        torch.manual_seed(42)
        model1 = ModelA()
        x = torch.rand(1, 3, 64, 64)

        # Evaluate model 1
        model1.eval()
        with torch.no_grad():
            out1 = model1(x)

        ckpt_file = tmp_path / "roundtrip_weights.pt"
        save_checkpoint(model1, ckpt_file, compile_cdfs=True)

        # Load into clean model 2
        model2 = ModelA()
        load_checkpoint(ckpt_file, model=model2)
        model2.eval()
        with torch.no_grad():
            out2 = model2(x)

        assert torch.allclose(out1.x_hat, out2.x_hat, atol=1e-6)
        assert torch.allclose(out1.y, out2.y, atol=1e-6)
        assert torch.allclose(out1.z, out2.z, atol=1e-6)
        assert torch.allclose(out1.log_sigma, out2.log_sigma, atol=1e-6)

    def test_checkpoint_roundtrip_dual_optimizer_harness(self, tmp_path: Path) -> None:
        """Checkpoint persists and restores DualOptimizerHarness optimizer and scheduler states."""
        torch.manual_seed(42)
        model = ModelA()
        harness = create_dual_optimizer(
            model,
            main_lr=1e-4,
            aux_lr=5e-4,
            lr_scheduler="cosine",
            total_epochs=50,
        )

        # Execute mock training step to populate optimizer state
        x = torch.rand(1, 3, 64, 64)
        out = model(x)
        loss = torch.mean(out.x_hat**2) + 0.1 * model.aux_loss()
        loss.backward()

        harness.step_main()
        harness.step_aux()
        harness.step_schedulers(epoch=5)

        saved_main_lr = harness.main_lr
        saved_aux_lr = harness.aux_lr
        assert saved_main_lr is not None

        ckpt_file = tmp_path / "roundtrip_optimizer.pt"
        save_checkpoint(
            model=model,
            path=ckpt_file,
            optimizer_harness=harness,
            compile_cdfs=True,
            epoch=5,
            step=150,
            metrics={"val/psnr": 31.5},
            experiment_tag="baseline",
        )

        # Create fresh model and harness
        fresh_model = ModelA()
        fresh_harness = create_dual_optimizer(
            fresh_model,
            main_lr=5e-5,  # initial different lr
            aux_lr=1e-3,
            lr_scheduler="cosine",
            total_epochs=50,
        )

        loaded_data = load_checkpoint(
            ckpt_file,
            model=fresh_model,
            optimizer_harness=fresh_harness,
        )

        assert loaded_data["epoch"] == 5
        assert loaded_data["step"] == 150
        assert loaded_data["metrics"]["val/psnr"] == 31.5
        assert loaded_data["metadata"]["experiment_tag"] == "baseline"

        # Verify learning rates and momentum restored
        assert pytest.approx(fresh_harness.main_lr, abs=1e-7) == saved_main_lr
        assert pytest.approx(fresh_harness.aux_lr, abs=1e-7) == saved_aux_lr

    def test_load_checkpoint_file_not_found(self, tmp_path: Path) -> None:
        """load_checkpoint raises FileNotFoundError if checkpoint file does not exist."""
        missing = tmp_path / "does_not_exist.pt"
        with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
            load_checkpoint(missing)


# ---------------------------------------------------------------------------
# Test MLflowTracker Run Lifecycle & Artifact Persistence
# ---------------------------------------------------------------------------


class TestMLflowTracker:
    """Tests covering MLflowTracker experiment tracking, lifecycle context manager, and artifact logging."""

    @pytest.fixture
    def tracking_dir(self, tmp_path: Path) -> Path:
        """Provides an isolated temporary tracking directory for MLflow."""
        td = tmp_path / "mlruns"
        return td

    def test_tracker_auto_creates_missing_directory(self, tracking_dir: Path) -> None:
        """MLflowTracker automatically creates missing directory paths."""
        assert not tracking_dir.exists()
        tracker = MLflowTracker(
            experiment_name="test_auto_create",
            tracking_uri=tracking_dir,
        )
        assert tracking_dir.exists()
        assert tracker.experiment_id is not None

    def test_tracker_run_context_manager_lifecycle(self, tracking_dir: Path) -> None:
        """tracker.run() manages clean run opening, parameter/metric logging, and termination."""
        tracker = MLflowTracker(
            experiment_name="test_lifecycle",
            tracking_uri=tracking_dir,
            tags={"framework": "pytorch"},
        )

        with tracker.run("run_1", tags={"dataset": "kodak"}) as run:
            assert tracker.active_run is not None
            assert tracker.run_id == run.info.run_id

            tracker.log_param("lmbda", 0.013)
            tracker.log_params({"batch_size": 4, "model": "ModelA"})

            tracker.log_metric("train/loss", 0.42, step=1)
            tracker.log_metrics(
                {"val/psnr": 32.1, "val/bpp": torch.tensor(0.245)},
                step=1,
            )

        # After exiting context, active run must be ended
        assert tracker.active_run is None

        # Verify run data recorded in MLflow client
        client = mlflow.tracking.MlflowClient(tracking_uri=str(tracking_dir))
        run_data = client.get_run(run.info.run_id)
        assert run_data.info.status == "FINISHED"
        assert run_data.data.params["lmbda"] == "0.013"
        assert run_data.data.params["model"] == "ModelA"
        assert pytest.approx(run_data.data.metrics["val/psnr"], abs=1e-4) == 32.1
        assert pytest.approx(run_data.data.metrics["val/bpp"], abs=1e-4) == 0.245
        assert run_data.data.tags["framework"] == "pytorch"
        assert run_data.data.tags["dataset"] == "kodak"

    def test_tracker_run_context_manager_exception_marks_failed(self, tracking_dir: Path) -> None:
        """tracker.run() marks run status as FAILED when an exception is raised inside the block."""
        tracker = MLflowTracker(
            experiment_name="test_failure",
            tracking_uri=tracking_dir,
        )

        run_id_recorded = None
        with pytest.raises(RuntimeError, match="Simulated training error"):
            with tracker.run("failing_run") as run:
                run_id_recorded = run.info.run_id
                tracker.log_metric("train/step", 1, step=1)
                raise RuntimeError("Simulated training error")

        assert run_id_recorded is not None
        assert tracker.active_run is None

        client = mlflow.tracking.MlflowClient(tracking_uri=str(tracking_dir))
        run_data = client.get_run(run_id_recorded)
        assert run_data.info.status == "FAILED"

    def test_tracker_run_catches_base_exception(self, tracking_dir: Path) -> None:
        """tracker.run() marks run status as FAILED on KeyboardInterrupt (BaseException)."""
        tracker = MLflowTracker(
            experiment_name="test_base_exception",
            tracking_uri=tracking_dir,
        )

        run_id_recorded = None
        with pytest.raises(KeyboardInterrupt):
            with tracker.run("keyboard_interrupt_run") as run:
                run_id_recorded = run.info.run_id
                raise KeyboardInterrupt("Simulated Ctrl+C")

        assert run_id_recorded is not None
        assert tracker.active_run is None

        client = mlflow.tracking.MlflowClient(tracking_uri=str(tracking_dir))
        run_data = client.get_run(run_id_recorded)
        assert run_data.info.status == "FAILED"

    def test_tracker_reapplies_tracking_uri(self, tracking_dir: Path, tmp_path: Path) -> None:
        """MLflowTracker re-applies its own tracking_uri in start_run and run."""
        tracker = MLflowTracker(
            experiment_name="test_reapply_uri",
            tracking_uri=tracking_dir,
        )
        other_dir = tmp_path / "other_mlruns"
        mlflow.set_tracking_uri(str(other_dir))

        with tracker.run("reapplied_uri_run") as run:
            assert mlflow.get_tracking_uri() == str(tracking_dir)
            tracker.log_param("test_key", "test_val")

        client = mlflow.tracking.MlflowClient(tracking_uri=str(tracking_dir))
        run_data = client.get_run(run.info.run_id)
        assert run_data.info.status == "FINISHED"

    def test_tracker_nested_runs(self, tracking_dir: Path) -> None:
        """MLflowTracker supports nested child runs within a parent run."""
        tracker = MLflowTracker(
            experiment_name="test_nested",
            tracking_uri=tracking_dir,
        )

        with tracker.run("parent_run") as parent:
            tracker.log_param("parent_param", "root")

            with tracker.run("child_run_1", nested=True) as child:
                tracker.log_param("child_param", "leaf")
                assert tracker.run_id == child.info.run_id

            # Active run should return to parent
            assert tracker.run_id == parent.info.run_id

        assert tracker.active_run is None

    def test_tracker_artifact_logging(self, tracking_dir: Path, tmp_path: Path) -> None:
        """tracker.log_artifact uploads local files to MLflow artifact store."""
        tracker = MLflowTracker(
            experiment_name="test_artifacts",
            tracking_uri=tracking_dir,
        )

        dummy_artifact = tmp_path / "config.json"
        dummy_artifact.write_text('{"learning_rate": 0.0001}')

        with tracker.run("artifact_run") as run:
            tracker.log_artifact(dummy_artifact, artifact_path="configs")

        client = mlflow.tracking.MlflowClient(tracking_uri=str(tracking_dir))
        artifacts = client.list_artifacts(run.info.run_id, path="configs")
        artifact_names = [a.path for a in artifacts]
        assert any("config.json" in name for name in artifact_names)

    def test_tracker_log_checkpoint(self, tracking_dir: Path, tmp_path: Path) -> None:
        """tracker.log_checkpoint saves compiled checkpoint and logs as MLflow artifact."""
        model = ModelA()
        tracker = MLflowTracker(
            experiment_name="test_chk_logging",
            tracking_uri=tracking_dir,
        )

        ckpt_target = tmp_path / "chk_test.pt"
        with tracker.run("checkpoint_run") as run:
            tracker.log_checkpoint(
                model=model,
                path=ckpt_target,
                compile_cdfs=True,
                step=500,
                epoch=2,
                metrics={"val/psnr": 30.5},
            )

        client = mlflow.tracking.MlflowClient(tracking_uri=str(tracking_dir))
        artifacts = client.list_artifacts(run.info.run_id, path="checkpoints")
        artifact_names = [a.path for a in artifacts]
        assert any("chk_test.pt" in name for name in artifact_names)

        # Step metric should be logged
        run_data = client.get_run(run.info.run_id)
        assert pytest.approx(run_data.data.metrics["val/psnr"], abs=1e-4) == 30.5

    def test_tracker_log_checkpoint_filters_non_numeric_metrics(self, tracking_dir: Path, tmp_path: Path) -> None:
        """log_checkpoint safely logs checkpoints with mixed non-numeric and numeric metadata."""
        model = ModelA()
        tracker = MLflowTracker(
            experiment_name="test_mixed_metrics",
            tracking_uri=tracking_dir,
        )
        ckpt_target = tmp_path / "chk_mixed.pt"
        with tracker.run("mixed_metrics_run") as run:
            tracker.log_checkpoint(
                model=model,
                path=ckpt_target,
                compile_cdfs=True,
                step=10,
                metrics={
                    "val/psnr": 33.2,
                    "model_family": "variational",
                    "nested_config": {"opt": "adam"},
                },
            )

        client = mlflow.tracking.MlflowClient(tracking_uri=str(tracking_dir))
        run_data = client.get_run(run.info.run_id)
        assert pytest.approx(run_data.data.metrics["val/psnr"], abs=1e-4) == 33.2
        assert "model_family" not in run_data.data.metrics
