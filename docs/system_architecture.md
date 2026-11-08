# System Architecture

This document describes the architectural design, component interactions, neural network topologies, and data representations of the **Saliency-Aware Learned Neural Image Compression** system.

---

## 1. High-Level Architectural Overview

The system extends variational autoencoder (VAE) image compression architectures (based on Ballé et al., 2018) with **adaptive spatial bit allocation** driven by visual saliency. The primary goal is to prioritize bitrate in high-salience Regions of Interest (ROI) while maintaining a matched or reduced total bitstream size without requiring auxiliary side-information transmission.

```mermaid
flowchart TD
    subgraph Input ["Image Input & Preprocessing"]
        X["Raw Input Image (x)"] --> PAD["Centered Reflection Padding<br/>(16-multiple boundary alignment)"]
        PAD --> XPAD["Padded Tensor (x_pad)"]
    end

    subgraph AnalysisTransforms ["Analysis & Encoding"]
        XPAD --> GA["Analysis Transform (g_a)<br/>Strided Conv + GDN"]
        GA --> Y["Latent Representation (y)"]
        Y --> HA["Hyper-Analysis Transform (h_a)<br/>Strided Conv + LeakyReLU"]
        HA --> Z["Hyperprior Latent (z)"]
        Z --> QZ["Hyperprior Quantization & rANS Encode"]
        QZ --> STREAM_Z["Hyperprior Bitstream (z.sanc)"]
    end

    subgraph HyperpriorSynthesis ["Hyperprior Decoding & Saliency Recovery"]
        STREAM_Z --> DQZ["Hyperprior rANS Decode & Dequant"]
        DQZ --> Z_HAT["Decoded Hyperlatent (z_hat)"]
        Z_HAT --> HS["Hyper-Synthesis (h_s)<br/>Deconv + LeakyReLU"]
        HS --> PARAMS["Entropy Parameters (mu, sigma)"]
        Z_HAT --> HSAL["Saliency Estimation Head (h_sal)<br/>Auxiliary Saliency Decoder"]
        HSAL --> S_PRED["Recovered Saliency Map (S_pred)"]
    end

    subgraph AdaptiveAllocationEngine ["Adaptive Allocation Engine"]
        S_PRED --> DELTA["Dynamic Quantization Step Delta(S)<br/>Delta = Delta_min + (1-Delta_min)(1-S_pred)"]
        Y --> MOD_Q["Quantization Scaling<br/>y_tilde = round(y / Delta) * Delta"]
        DELTA --> MOD_Q
        PARAMS --> MOD_PARAMS["Modulated Entropy Parameters<br/>sigma_eff = sigma * Delta"]
        DELTA --> MOD_PARAMS
        MOD_Q --> RANS_Y["Integer rANS Encoder"]
        MOD_PARAMS --> RANS_Y
        RANS_Y --> STREAM_Y["Main Latent Bitstream (y.sanc)"]
    end

    subgraph SynthesisTransform ["Decompression & Synthesis"]
        STREAM_Y --> DECODE_Y["Integer rANS Decoder"]
        MOD_PARAMS --> DECODE_Y
        DECODE_Y --> Y_HAT["Reconstructed Latent (y_hat)"]
        Y_HAT --> GS["Synthesis Transform (g_s)<br/>Transposed Conv + IGDN"]
        GS --> X_RECON["Reconstructed Image (x_hat)"]
        X_RECON --> UNPAD["Spatial Crop to Original Dimensions"]
        UNPAD --> OUT["Final Output Image (x_out)"]
    end
```

---

## 2. Model Comparative Matrix

The codebase defines five distinct model variants across the experimental matrix to rigorously isolate causal mechanisms and architectural components:

```mermaid
graph TD
    classDef baseline fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef implicit fill:#e8f8f5,stroke:#26a69a,stroke-width:2px;
    classDef control fill:#f3e5f5,stroke:#ab47bc,stroke-width:2px;

    A["Model A<br/>Standard Hyperprior<br/>(Uniform MSE, No Allocation)"]:::baseline
    B["Model B<br/>Saliency-Weighted Codec<br/>(Saliency-Weighted MSE, No Allocation)"]:::baseline
    C["Model C<br/>Implicit Hyperprior Allocation<br/>(Saliency MSE + Latent Delta(S), Zero Side Overhead)"]:::implicit
    D["Model D<br/>Learned Hyperprior Auto-Codec<br/>(Uniform MSE + Latent Delta(S_pred))"]:::implicit
    CT["Model C-Transmitted<br/>Explicit Side-Information Control<br/>(Transmits Saliency Map via PNG/rANS)"]:::control
```

| Model | Identifier | Distortion Loss | Allocation Mechanism | Saliency Transmission | Primary Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A** | `src/codec/model_a.py` | Uniform MSE | None ($\Delta = 1.0$) | None | Control baseline (Ballé 2018) |
| **Model B** | `src/codec/model_b.py` | Saliency MSE ($0.2$ floor) | None ($\Delta = 1.0$) | None | Tests loss gradient weighting alone |
| **Model C** | `src/codec/model_c.py` | Saliency MSE ($0.2$ floor) | Latent $\Delta(S_{\mathrm{pred}})$, $\tilde{\sigma}$ | Hyperprior-Derived ($0$ side-bits) | Primary innovation under study |
| **Model D** | `src/codec/model_d.py` | Uniform MSE | Latent $\Delta(S_{\mathrm{pred}})$, $\tilde{\sigma}$ | Hyperprior-Derived ($0$ side-bits) | Tests allocation without loss bias |
| **Model C-Trans** | `src/codec/model_c_transmitted.py` | Saliency MSE ($0.2$ floor) | Latent $\Delta(S)$, $\tilde{\sigma}$ | Explicit side bitstream | Measures side transmission penalty |

---

## 3. Subsystem Breakdown

### 3.1 Neural Transforms (`src/codec/layers.py`, `src/codec/model_a.py`)

- **Analysis Transform ($g_a$)**: Maps input image $x \in \mathbb{R}^{3 \times H \times W}$ to latent representation $y \in \mathbb{R}^{C \times H/16 \times W/16}$ ($C=128$).
  - Four convolution stages (stride 2) paired with Generalized Divisive Normalization (**GDN**).
  - GDN non-linearities decorrelate local activations and enforce Gaussian marginals:
    $$y_i = \frac{x_i}{\left(\beta_i + \sum_j \gamma_{ij} x_j^2\right)^{1/2}}$$
- **Synthesis Transform ($g_s$)**: Reconstructs image $\hat{x}$ from latent $\hat{y}$ using four transposed convolution stages paired with Inverse GDN (**IGDN**).
- **Hyper-Analysis ($h_a$)**: Maps $|y|$ to hyperprior latent $z \in \mathbb{R}^{N \times H/64 \times W/64}$ ($N=192$) using convolutions and LeakyReLU activations.
- **Hyper-Synthesis ($h_s$)**: Decodes $\hat{z}$ to conditional scale/mean parameters $(\sigma, \mu)$ for the main latent distribution $p(y \mid \hat{z}) \sim \mathcal{N}(\mu, \sigma^2)$.

### 3.2 Adaptive Allocation Engine (`src/codec/allocation.py`)

The adaptive allocation engine modulates latent quantization step size based on local visual saliency $S_{\mathrm{pred}} \in [0, 1]$:

1. **Quantization Step Modulation**:
   $$\Delta(S) = \Delta_{\min} + (1.0 - \Delta_{\min})(1.0 - S)$$
   Where $\Delta_{\min} = 0.5$. In high-salience regions ($S \approx 1.0$), $\Delta \to 0.5$ (finer step size $\implies$ higher bit density). In background regions ($S \approx 0.0$), $\Delta \to 1.0$ (standard step size $\implies$ coarse quantization).
2. **Effective Scale Adjustment**:
   $$\tilde{\sigma}_{ij} = \sigma_{ij} \cdot \Delta_{ij}$$
   Because quantization step size contracts, the effective scale parameter for the Gaussian conditional CDF tables is adjusted to maintain accurate probability density evaluation.

### 3.3 Integer rANS Entropy Coding Core (`src/entropy/`)

- **Streaming rANS Engine (`src/entropy/coder.py`)**: Asymmetric Numeral Systems (ANS) encoder/decoder operating with exact 32-bit state registers and 16-bit cumulative probability frequencies.
- **CDF Monotonicity & Clamping (`src/codec/entropy_models.py`)**: Probability tables are clamped to non-zero probabilities $[1, 2^{16} - 1]$ to prevent zero-frequency division exceptions during arithmetic coding.
- **Deterministic CPU Execution Engine (`src/entropy/cpu_engine.py`)**: Latents are transferred to CPU for arithmetic encoding and decoding to ensure 100% deterministic, bit-exact reproducibility across different hardware platforms.

### 3.4 Binary Bitstream Container (`src/bitstream/container.py`)

All compressed streams are packaged in the custom `.sanc` format:

```mermaid
classDiagram
    class SancHeader {
        +bytes magic "SANC (4B)"
        +uint8 version "0x01 (1B)"
        +uint8 model_id "Model ID (1B)"
        +uint16 height "Original Height (2B)"
        +uint16 width "Original Width (2B)"
        +uint16 pad_top "Top Reflection Padding (2B)"
        +uint16 pad_left "Left Reflection Padding (2B)"
        +uint32 z_length "Hyperprior Stream Length (4B)"
        +uint32 y_length "Main Latent Stream Length (4B)"
        +uint16 side_length "Optional Side Map Length (2B)"
        +uint16 reserved "Alignment Reserved (2B)"
    }

    class SancContainer {
        +SancHeader header "24-Byte Header"
        +bytes z_stream "rANS Encoded z Latents"
        +bytes y_stream "rANS Encoded y Latents"
        +bytes side_stream "Optional Saliency Payload"
        +uint32 crc32 "CRC32 Integrity Checksum"
        +save(path)
        +load(path)
    }

    SancContainer *-- SancHeader
```

---

## 4. Sequence Diagram: Encoding & Decoding Flow

The sequence diagram below details the operational interaction across components during an image compression and decompression cycle:

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant CLI as scripts/encode.py & decode.py
    participant Cont as SancContainer
    participant Model as Model C
    participant rANS as RansEncoder/Decoder
    participant Disk as Filesystem

    Note over User, Disk: Encoding Pipeline
    User->>CLI: python scripts/encode.py -i input.png -o output.sanc -m C
    CLI->>Model: Forward Analysis (x_pad)
    Model->>Model: y = g_a(x_pad), z = h_a(y)
    Model->>rANS: Compress z with EntropyBottleneck CDFs
    rANS-->>Model: z_bytes
    Model->>Model: z_hat = dequant(z), S_pred = h_sal(z_hat)
    Model->>Model: Delta = Delta(S_pred), y_tilde = quant(y, Delta)
    Model->>rANS: Compress y_tilde with GaussianConditional(mu, sigma*Delta)
    rANS-->>Model: y_bytes
    CLI->>Cont: Assemble(header, z_bytes, y_bytes, crc32)
    Cont->>Disk: Write output.sanc (binary)

    Note over User, Disk: Decoding Pipeline
    User->>CLI: python scripts/decode.py -i output.sanc -o recon.png
    CLI->>Cont: Load and verify CRC32
    Cont-->>CLI: SancContainer (header, z_bytes, y_bytes)
    CLI->>Model: Decompress(z_bytes, y_bytes, header)
    Model->>rANS: Decompress z_bytes
    rANS-->>Model: z_hat
    Model->>Model: sigma, mu = h_s(z_hat), S_pred = h_sal(z_hat)
    Model->>Model: Delta = Delta(S_pred)
    Model->>rANS: Decompress y_bytes using GaussianConditional(mu, sigma*Delta)
    rANS-->>Model: y_hat
    Model->>Model: x_recon = g_s(y_hat)
    Model->>CLI: Crop unpadded (H_orig, W_orig)
    CLI->>Disk: Save recon.png
    CLI-->>User: Compression & Decompression Verified
```
