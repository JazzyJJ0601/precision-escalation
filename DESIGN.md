# Design: Uncertainty-Gated Precision Escalation

## Overview
This project proposes a hybrid inference architecture designed to maximize throughput on consumer-grade GPUs while maintaining high-quality generation. The core idea is to store a highly quantized base model (2-bit) directly in VRAM, while storing residual precision data (the difference between 2-bit and higher precision weights) on slower NVMe or system RAM. During inference, we run a cheap forward pass using only the 2-bit base. We then evaluate the uncertainty of the prediction using token-level entropy. If the entropy exceeds a configured threshold, we fetch the necessary residual bits for the active layers, add them to the weights, and recompute the token at higher precision.

## Architecture & Bit-Plane Weight Layout
Weights are decomposed into a base component and residual bit-planes.

- **Base Component:** Stored as 2-bit integers (0-3) in VRAM. This fits approximately 4x more parameters compared to standard 8-bit quantization, keeping the entire transformer (Qwen3-8B) resident on a 24GB GPU without OOM errors.
- **Residuals:** Stored as packed bit-planes on NVMe. For a target of 6-bit or 8-bit effective precision, we store the higher-order bits separately. Specifically, we subtract the reconstructed 2-bit value from the original 8-bit value and encode the delta as residual bits.
- **Access Pattern:** Residuals are indexed by layer ID and weight tensor offset. They are streamed into VRAM on demand, adding to the active weight buffer before computation.

## Entropy Gate Mechanism
The decision to escalate precision is driven by the Shannon entropy of the output logits from the 2-bit pass.

- **Calculation:** $H = -\sum_{i} p(x_i) \log p(x_i)$ where $p(x_i)$ is the normalized probability of the $i$-th token in the vocabulary.
- **Thresholding:** If $H > \tau$, the token is considered ambiguous or high-uncertainty. Tokens with low entropy (confident predictions) are accepted as-is from the 2-bit pass.
- **Action:** For ambiguous tokens, trigger the residual fetch for the current layer's active parameters. This ensures we only spend compute and IO resources on difficult tokens.

## Prefetch Strategy
To minimize latency from NVMe streaming, we employ speculative prefetching:

- **Layer Locality:** Predict which layers are likely to need escalation in the next few tokens based on attention sparsity patterns. If a token is ambiguous at layer L, layers L+1 and L+2 often exhibit similar uncertainty due to contextual dependencies.
- **Double Buffering:** While the current token computes with base weights, pre-load residual blocks for the subsequent layer into a staging buffer in VRAM. This overlaps IO with compute to hide NVMe latency.

## Metrics to Measure
We will evaluate the system against the following dimensions:

- **VRAM Footprint:** Track peak VRAM usage with and without residuals loaded. The target is to keep base weights + active residuals under 20GB on a 3090.
- **Throughput:** Tokens per second (tokens/s) compared to full 4-bit and 8-bit baselines.
- **Accuracy:** Perplexity on standard benchmarks (e.g., MMLU, GSM8K) to ensure the 2-bit base doesn't degrade quality too much.
- **IO Overhead:** Time spent fetching residuals vs. compute time to measure efficiency gains.

## Related Work and Novelty
Existing methods include static quantization (GGUF, AWQ) and Mixture of Experts (MoE).

- **Static Quantization:** Fixes precision for all tokens uniformly. Our approach is dynamic and context-dependent.
- **Adaptive Precision (BitNet b1.58):** Uses ternary weights but computes statically without residual reconstruction.
- **Sparsity/Pruning:** Removes weights permanently rather than storing them as retrievable residuals.
- **Novelty:** This approach uniquely combines per-token entropy gating with residual bit-plane retrieval from non-volatile memory. No prior work uses entropy to trigger high-precision retrieval from external storage on a per-token basis.

## Experiment Plan (Qwen3-8B)
Target Hardware: NVIDIA RTX 3090 (24GB VRAM, PCIe 4.0 NVMe).

1. **Base Setup:** Run Qwen3-8B with full 8-bit precision as a control. Record baseline perplexity and tokens/s.
2. **Quantized Base:** Run the 2-bit base-only version on a 10k token subset. Measure perplexity drop to determine baseline accuracy cost.
3. **Hybrid:** Implement the entropy gate (threshold $\tau = 2.0$ initially). Enable residual fetching.
4. **Benchmarks:** Run inference on the 10k token corpus. Measure:
   - Perplexity vs 8-bit and 4-bit baselines
   - Token generation rate (tokens/s)
   - Percentage of tokens triggering escalation (target <20%)
5. **Optimization:** Adjust $\tau$ and residual batch size to balance VRAM usage and perplexity.

## Risks & Mitigations
- **Latency Spikes:** NVMe fetches may cause inconsistent generation speed. Mitigation: Use double buffering and batch residual requests.
- **VRAM OOM:** Residuals may accumulate. Mitigation: Evict least recently used residual blocks when VRAM usage exceeds threshold.

This design aims to push inference beyond memory bandwidth limits by exploiting the sparsity of uncertainty in typical generation tasks.
