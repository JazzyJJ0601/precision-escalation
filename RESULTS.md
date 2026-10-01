# Precision Escalation Results

**Status:** Only the FP16 Qwen3-8B baseline is measured (18.43 perplexity); the 2-bit and escalation runs ran out of memory. The method is not evaluated yet.

Run command: `python3 repos/precision-escalation/results/run_real.py`

| Method | Avg Perplexity |
|--------|---------------|
| Full precision (baseline) | 18.43 |
| 2-bit quantization | Not evaluated (OOM) |
| 2-bit + residual escalation | Not evaluated (OOM) |

The measured perplexity establishes a baseline for the Qwen3-8B model on short text prompts. The full precision model achieves 18.43 perplexity on average across the test prompts. This baseline enables comparison with quantized variants that will be evaluated separately with additional memory resources.
