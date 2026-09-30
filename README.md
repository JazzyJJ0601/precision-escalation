# precision-escalation

Layer-wise precision escalation using 2-bit base + residual bit-planes,
targeting consumer GPUs with limited VRAM.

Keep a 2-bit base model resident in VRAM; store residual precision data
(the difference between 2-bit and higher-precision weights) on NVMe or
system RAM. During inference the entropy gate decides per token whether
to fetch residuals and recompute at higher precision — typical generation
only needs high precision for ~20% of tokens.

## Usage

```bash
# Synthetic test (2048×2048 random matrix, no model required)
python pe/eval_layer.py

# Real Qwen3-8B layer test (requires safetensors weights on disk)
python pe/eval_real.py
```

## Results (reconstruction error, Frobenius norm)

### Synthetic 2048×2048 matrix

| Bits | Relative Error |
|------|---------------|
| 2    | 0.682         |
| 4    | 0.657         |
| 6    | 0.557         |
| 8    | 0.005         |

### Real Qwen3-8B weights (model.layers.10)

| Bits | self_attn.q_proj | mlp.down_proj |
|------|------------------|---------------|
| 2    | 0.362            | 0.362         |
| 3    | 0.181            | 0.181         |
| 4    | 0.091            | 0.091         |
| 6    | 0.024            | 0.024         |
| 8    | 0.005            | 0.005         |

Real Qwen3-8B weights reconstruct much more accurately at low bit-widths
than the random synthetic matrix — 2-bit error is ~36 % vs ~68 %.  This
is because real weights have structure (clusters, low-rank subspaces) that
the group-wise quantiser exploits.

## Perplexity

Not yet measured on a real model run. The `pe/eval_ppl.py` script
loads the local Qwen3-8B checkpoint and evaluates a single quantised
layer. Run it when the model checkpoint is available:

```bash
cd repos/precision-escalation
python pe/eval_ppl.py
```

## Architecture

Weights are decomposed into a 2-bit base component (stays in VRAM) and
residual bit-planes (stored on NVMe, fetched on demand). The entropy gate
checks output uncertainty from the 2-bit pass and triggers a residual
fetch when entropy exceeds a threshold.  Double-buffering overlaps NVMe
IO with computation to hide latency.

## Design

See [DESIGN.md](DESIGN.md) for full architecture, entropy gate mechanism,
prefetch strategy, and experiment plan.
