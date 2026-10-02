# precision-escalation

Run a small low-bit copy of an LLM for every token, and only escalate to higher precision on
the tokens where the low-bit model is unsure. The gate is the low-bit model's own predictive
entropy. The idea for consumer GPUs: keep the low-bit base in VRAM and the residual bits
(base → 4-bit or → bf16) in system RAM or on NVMe, fetched only for escalated tokens.

Measured on **Qwen3-8B** with a 3-bit base: for the same share of escalated tokens, the
entropy gate recovers much more quality than escalating at random.

| Escalated tokens | 3-bit → bf16, entropy gate | 3-bit → bf16, random | 3-bit → 4-bit, entropy gate | 3-bit → 4-bit, random |
|---:|---:|---:|---:|---:|
| 0% | 16.44 | 16.44 | 16.44 | 16.44 |
| 6.3% | **15.59** | 16.06 | **15.80** | 16.15 |
| 13.8% | **14.86** | 15.58 | **15.22** | 15.72 |
| 22.5% | **14.17** | 15.01 | **14.65** | 15.31 |
| 42.7% | **12.90** | 13.94 | **13.61** | 14.41 |
| 100% | 11.20 | 11.20 | 12.08 | 12.08 |

Perplexity, WikiText-2 test windows 11–40 (15,330 predicted tokens). Escalating 22.5% of
tokens to bf16 recovers 43% of the 3-bit → bf16 gap with the gate, against 27% at random; at
42.7% it is 68% against 48%.

## What didn't work: a 2-bit base

The original plan was a 2-bit base. With group-128 round-to-nearest, 2-bit Qwen3-8B is broken
(perplexity over 1,000,000), and its entropy carries no useful signal: the gate did *worse* than
random at every share (e.g. 3,505 vs 2,282 at 53% escalated to bf16). A broken model is
confidently wrong exactly where it matters, so "unsure" stops meaning "wrong". The method needs a
base that is degraded but still sane; 3-bit is. A better 2-bit quantiser (e.g. GPTQ or
[activation-aware scaling](https://github.com/JazzyJJ0601/activation-aware-quant)) might rescue
a 2-bit base; not tested here.

## How it was measured

- Weights: all 252 decoder linears quantised with asymmetric RTN, one fp16 scale and zero-point
  per 128 weights (fake-quantised, evaluated in bf16). bf16, 4-bit, 3-bit and 2-bit are each run
  once over the same text, keeping every token's log-prob and the base's entropy, so only one
  copy of the weights is ever on the GPU.
- The entropy threshold for each target share is set on windows 1–10 (dev) and applied unchanged
  to windows 11–40 (test). Thresholds transfer imperfectly (a 10% target gives 6.3% on test), so
  the random baseline escalates exactly the share the gate actually used. Random rows are the
  mean of 5 draws (spread ≤ 0.6 perplexity; every gate result is outside it).
- Teacher-forced: an escalated token's prediction comes from the higher-precision model with its
  own context. In a real decoder that means a recompute at high precision.

## Honest limits

- No speed or memory measurements: the residual fetch/recompute pipeline is not built. Escalating
  43% of tokens is a lot of fetching, and at 42.7% the 3-bit base + gate (13.61) is still worse
  than simply running the 4-bit model (12.08), which is only 0.87 GB bigger packed. The result
  here is that entropy is a good escalation signal, not that the system is faster.
- One model, WikiText-2 only.
- Earlier versions of this repo reported an fp16 perplexity of 18.43 on short prompts, a
  "near-8-bit" 2-bit model in `blog-post.md`, and per-layer error tables. They came from untested
  code and are withdrawn.

## Reproduce

Needs a CUDA GPU with ~20 GB and a local Qwen3-8B checkpoint (path in `results/qcommon.py`).

```bash
python results/run_real.py      # ~5 min on an RTX 3090 Ti, writes results/real.json
pytest -q tests                 # residual bit-plane store and gate (NumPy prototype in pe/)
```

Details: [RESULTS.md](RESULTS.md). Original design: [DESIGN.md](DESIGN.md).

## License

MIT
