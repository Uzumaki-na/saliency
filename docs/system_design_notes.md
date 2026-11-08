# System Design Notes for Future Developers

This document details key architectural decisions, mathematical foundations, developer gotchas, and engineering guidelines for developers extending or maintaining the **Saliency-Aware Learned Neural Image Compression** codebase.

---

## 1. Mathematical Foundations & Objectives

### 1.1 Rate-Distortion Optimization Objective

The training objective adheres to CompressAI conventions, balancing total bitrate in bits per pixel ($R_{\mathrm{bpp}}$), scaled distortion ($D$), and auxiliary saliency supervision ($\mathcal{L}_{\mathrm{aux}}$):

$$\mathcal{L} = R_{\mathrm{bpp}} + \lambda \cdot 255^2 \cdot D_{\mathrm{saliency}} + \lambda_{\mathrm{aux}} \mathcal{L}_{\mathrm{aux}}(S_{\mathrm{pred}}, S)$$

Where:
- **Bitrate Normalization**:
  $$R_{\mathrm{bpp}} = \frac{-\sum \log_2 p_{\hat{y}}(\tilde{y}) - \sum \log_2 p_{\hat{z}}(\hat{z})}{H_{\mathrm{orig}} \cdot W_{\mathrm{orig}}}$$
  The rate is normalized by the **original unpadded** spatial dimensions ($H_{\mathrm{orig}} \cdot W_{\mathrm{orig}}$) rather than the padded tensor dimensions to ensure fair bpp accounting against uncompressed baselines.
- **$255^2$ MSE Scaling**:
  CompressAI models operate on normalized image tensors $x, \hat{x} \in [0, 1]$. Multiplying MSE by $255^2$ scales the distortion to the standard integer pixel range $[0, 255]^2$, matching standard video/image codec optimization conventions and ensuring numerical stability for standard multiplier ranges ($\lambda \in [0.0018, 0.0483]$).
- **Saliency Distortion Weighting ($D_{\mathrm{saliency}}$)**:
  $$D_{\mathrm{saliency}} = \frac{1}{|M_{\mathrm{valid}}|} \sum_{(i,j) \in M_{\mathrm{valid}}} S'_{ij} \cdot (x_{ij} - \hat{x}_{ij})^2$$
  Where $S'_{ij} = 0.2 + 0.8 S_{ij}$. The **$0.2$ background floor** is critical: without it, zero-saliency background regions would receive zero loss gradient, leading to severe visual collapse and color bleeding in background areas.
- **Constant Auxiliary Supervision ($\lambda_{\mathrm{aux}} = 0.1$)**:
  $\mathcal{L}_{\mathrm{aux}}$ is spatial binary cross-entropy between predicted saliency $S_{\mathrm{pred}}$ (from hyperprior $\hat{z}$) and teacher saliency $S$:
  $$\mathcal{L}_{\mathrm{aux}}(S_{\mathrm{pred}}, S) = -\frac{1}{|M_{\mathrm{valid}}|} \sum \left[ S \log S_{\mathrm{pred}} + (1-S) \log(1-S_{\mathrm{pred}}) \right]$$
  The auxiliary weight $\lambda_{\mathrm{aux}} = 0.1$ is kept **constant** across all rate points $\lambda$. Scaling $\lambda_{\mathrm{aux}}$ proportionally with $\lambda$ must be avoided: at low bitrates ($\lambda = 0.0018$), a scaled weight ($9 \times 10^{-5}$) would be too weak to incentivize $\hat{z}$ to dedicate bits to saliency representations.

### 1.2 Gradient Flow Scope & Detachment Invariant

A critical invariant in Model C and Model D:
- **Auxiliary gradient flow**: $\nabla \mathcal{L}_{\mathrm{aux}}$ flows through $h_{\mathrm{sal}}$ and into hyper-analysis $h_a$, incentivizing the hyperlatent $\hat{z}$ to encode spatial saliency features.
- **Gradient Cut**: Gradients from $\mathcal{L}_{\mathrm{aux}}$ are **strictly detached before the main analysis transform $g_a$** (via `y.detach()` feeding $h_a$). This prevents the auxiliary task from corrupting the general visual representation learned by $g_a$.

---

## 2. Key Architectural Decision Records (ADRs)

### ADR 1: Hyperprior-Derived Saliency vs. Transmitted Side-Information
- **Context**: Prior saliency-guided codecs often compress the saliency map separately and transmit it as side-information in the bitstream.
- **Decision**: Model C recovers $S_{\mathrm{pred}}$ directly from the already-transmitted hyperprior $\hat{z}$.
- **Rationale**: Transmitting an explicit saliency map introduces an unavoidable overhead of $0.02\text{--}0.08$ bpp. In low-bitrate regimes ($<0.2$ bpp), this overhead wipes out any coding gain in the ROI. By recovering saliency from $\hat{z}$, the side-information cost is $0$ bits, with only a negligible net increase in hyperprior size ($\Delta \mathrm{BPP}_z \approx +0.002$ bpp). This was empirically proven via the **Model C-Transmitted** control experiment.

### ADR 2: Integer rANS Execution on CPU
- **Context**: Deep learning inference runs on CUDA GPU, but entropy coding (rANS) involves sequential state transitions.
- **Decision**: All rANS encode and decode routines (`src/entropy/cpu_engine.py`) execute on CPU using exact 32-bit registers and 16-bit frequency CDFs.
- **Rationale**: GPU implementations of arithmetic coding introduce nondeterministic floating-point rounding across CUDA compute capabilities and driver versions. Running the integer rANS state machine on CPU guarantees bit-exact, 100% reproducible cross-platform decompression.

### ADR 3: Centered Reflection Padding with Valid Latent Mask
- **Context**: 4-stage strided convolutions downsample images by a factor of 16 ($2^4$). Arbitrary input resolutions (e.g. Kodak $768 \times 512$, MIT1003 irregular sizes) may not be divisible by 16.
- **Decision**: Implement centered reflection padding (`src/data/padding.py`) paired with an explicit binary mask $M_{\mathrm{valid}}$ tracking valid original spatial pixels.
- **Rationale**: Zero-padding causes sharp high-frequency edges at image borders, wasting significant bit allocation on artificial boundary artifacts. Reflection padding preserves local image statistics, and $M_{\mathrm{valid}}$ masks out padding during loss calculation and evaluation.

### ADR 4: Causal Latent Interventions (Block-Shuffling & Fixed-Effects)
- **Context**: To publish peer-grade research, one must demonstrate that ROI bit reallocation is *causally* driven by visual saliency rather than confounding image texture complexity.
- **Decision**: Implement two independent causal tests (`src/evaluation/causal.py`):
  1. **Latent Block-Shuffling**: Permute $S_{\mathrm{pred}}$ into $4\times 4$ and $8\times 8$ tile grids while keeping latent content unchanged. If bit allocation collapses from $\mathrm{RAR} = 1.26 \to 1.03$, allocation is confirmed to follow spatial semantic layout rather than global image statistics.
  2. **Image Fixed-Effects Regression**: Regress $(\mathrm{bits}_C - \mathrm{bits}_A)$ on $S$ while controlling for baseline bit map $\mathrm{bits}_A$ with image dummy fixed effects $\alpha_{\mathrm{image}}$ and clustered standard errors.

---

## 3. Developer Gotchas & Pitfalls

### 3.1 DVC Cache & SQLite Permissions on Windows
- **Issue**: DVC uses an embedded SQLite database in `.dvc/tmp/` and `%LOCALAPPDATA%` for hash tracking and disk caching.
- **Gotcha**: When running DVC CLI commands (`dvc repro`, `dvc status`) or tests that instantiate `dvc.repo.Repo` on Windows inside a restricted sandbox, SQLite will raise:
  ```
  sqlite3.OperationalError: unable to open database file
  ```
- **Solution**: Always execute DVC pipeline commands with full filesystem permissions (or `BypassSandbox: true` in the automation runner).

### 3.2 SANC Binary Header Endianness
- **Issue**: The `.sanc` container header is packed as 24 binary bytes.
- **Gotcha**: Never use native endianness (`@`). Always use strict network big-endian (`>`) struct format strings:
  ```python
  # >4sBBHHHHIIHH
  struct.pack(">4sBBHHHHIIHH", b"SANC", version, model_id, height, width, pad_top, pad_left, z_len, y_len, side_len, reserved)
  ```
  This guarantees that bitstreams compressed on an x86 Windows machine can be decoded without error on ARM Linux or macOS.

### 3.3 Monotonicity of CDF Probability Tables
- **Issue**: rANS requires cumulative frequency tables where frequencies are strictly monotonically increasing.
- **Gotcha**: In low-probability tails of the Gaussian distribution, adjacent discrete CDF points can round to identical integer frequencies if not clamped. If $CDF[i] == CDF[i+1]$, the probability interval is 0, causing a fatal zero-division crash during arithmetic encoding.
- **Solution**: The CDF generation engine in [`src/codec/entropy_models.py`](file:///C:/Users/panav/proj/saliency/src/codec/entropy_models.py) explicitly enforces:
  ```python
  cdf = torch.clamp(cdf, min=1, max=total_count - 1)
  # ensure strictly monotonic ladder
  ```

### 3.4 Matplotlib Headless Execution
- **Issue**: Running plotting routines on CI servers or headless test machines will crash if Matplotlib attempts to open an interactive GUI backend (e.g. TkAgg, Qt5Agg).
- **Solution**: Every figure module must call `matplotlib.use("Agg")` *before* importing `matplotlib.pyplot`:
  ```python
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  ```

---

## 4. Testing & Verification Matrix

The test suite contains **798 test items** structured into distinct validation tiers:

| Tier | Test Path | Focus Areas | Typical Runtime |
| :--- | :--- | :--- | :--- |
| **Unit Tests** | `tests/test_allocation.py`, `tests/test_layers.py`, `tests/test_dataset.py` | Transforms, GDN/IGDN, dataset loaders, padding, scalar $\Delta(S)$ | ~10s |
| **Entropy & Codec** | `tests/test_rans.py`, `tests/test_bitstream_gate.py` | rANS lossless roundtrips, CDF monotonic clamps, SANC header CRC32 | ~15s |
| **Model Architectures** | `tests/test_model_a.py` through `tests/test_model_d.py` | Forward passes, rate-distortion loss, gradient cuts, parameter shapes | ~40s |
| **Causal Analysis** | `tests/test_causal_interventions.py`, `tests/test_bjontegaard.py` | Block-shuffling, image fixed-effects regression, BD-rate bootstrap CIs | ~30s |
| **Pipeline & DVC** | `tests/test_dvc_pipeline.py` | `dvc.yaml` schema, dependencies on disk, locked MD5s, `dvc status` clean | ~13s |
| **Synthesis & Artifacts** | `tests/test_research_synthesis.py` | 300 DPI figures, Markdown report generation, CLI `--dry-run` | ~15s |

### Running Quality Gates

```bash
# 1. Run full test suite
pytest -v

# 2. Run Ruff formatting check
ruff format --no-cache --check src/ scripts/ tests/

# 3. Run Ruff linter
ruff check --no-cache src/ scripts/ tests/

# 4. Verify DVC pipeline integrity
dvc status
```

---

## 5. Guide for Adding New Models or Datasets

### Adding a New Model Variant (e.g., Model E)
1. Inherit from the base codec in `src/codec/` (or extend `ModelA`).
2. Implement custom allocation transforms in `forward()` and `compress()` / `decompress()`.
3. Register the model identifier in `MODEL_DISPLAY_NAMES` and CLI choices in `scripts/encode.py`, `scripts/train_sweep.py`, and `scripts/benchmark.py`.
4. Add unit tests in `tests/test_model_e.py` validating output shapes, gradient flow, and lossless rANS container roundtrips.

### Adding a New Evaluation Dataset
1. Add acquisition and directory configuration to `scripts/prepare_data.py`.
2. Generate SHA-256 manifests using `src/data/manifests.py`.
3. If saliency ground truth is required, integrate an extraction runner in `src/data/saliency.py`.
4. Register the dataset name in `scripts/benchmark.py` and `src/evaluation/benchmarks.py`.
