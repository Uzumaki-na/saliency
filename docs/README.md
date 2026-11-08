# Documentation Hub

Welcome to the internal engineering and architectural documentation for the **Saliency-Aware Learned Neural Image Compression** system.

---

## Documentation Index

| Document | Primary Audience | Key Topics |
| :--- | :--- | :--- |
| **[System Architecture](system_architecture.md)** | Core Developers, ML Engineers | Top-level architecture, 5-model comparative matrix, neural transforms ($g_a, g_s, h_a, h_s, h_{\mathrm{sal}}$), adaptive allocation engine ($\Delta(S)$), integer rANS coding, and end-to-end sequence flow. |
| **[System Design Notes](system_design_notes.md)** | Maintainers, Researchers | Mathematical foundations, $255^2$ MSE scaling, auxiliary loss invariant, Architectural Decision Records (ADRs), developer gotchas, testing matrix, and guide for adding new models/datasets. |
| **[Visualizations & Figure Guide](visualizations.md)** | Researchers, Reviewers | Clean visual schematics, Mermaid flowcharts, `.sanc` binary container layout, and comprehensive reference guide for all 300 DPI publication plots in `experiments/research_artifacts/figures/`. |

---

## Quick Navigation

- Looking to understand how the **latent quantization contraction** works? See [System Architecture § 3.2](system_architecture.md#32-adaptive-allocation-engine-srccodecallocationpy).
- Looking for details on the **.sanc container binary layout**? See [Visualizations § 3](visualizations.md#3-sanc-binary-container-layout) and [System Architecture § 3.4](system_architecture.md#34-binary-bitstream-container-srcbitstreamcontainerpy).
- Wondering why we chose **integer rANS on CPU** instead of GPU? See [System Design Notes § 2 (ADR 2)](system_design_notes.md#adr-2-integer-rans-execution-on-cpu).
- Running into **DVC SQLite issues on Windows**? See [System Design Notes § 3.1](system_design_notes.md#31-dvc-cache--sqlite-permissions-on-windows).
- Analyzing the **causal block-shuffling and regression figures**? See [Visualizations § 4](visualizations.md#4-publication-grade-figures-reference).
