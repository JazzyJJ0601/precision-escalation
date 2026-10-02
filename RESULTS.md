# Precision Escalation: results

Model: Qwen3-8B. All 252 decoder linears quantised with asymmetric RTN, group 128.
Text: WikiText-2 test, 40 × 512-token windows; windows 1–10 set the entropy thresholds (dev),
windows 11–40 are scored (test, 15,330 predicted tokens).
Command: `python results/run_real.py` (raw output in `results/real.json`).

## Each precision on its own (test windows)

| Weights | Perplexity |
|---|---:|
| bf16 | 11.1976 |
| 4-bit | 12.0798 |
| 3-bit | 16.4365 |
| 2-bit | 1,059,789 |

## Escalation: entropy gate vs random (same share of tokens; random = mean of 5 draws)

| Base → high | Target share | Share on test | Entropy gate | Random |
|---|---:|---:|---:|---:|
| 3-bit → 4-bit | 10% | 6.3% | **15.7979** | 16.1488 |
| 3-bit → 4-bit | 20% | 13.8% | **15.2177** | 15.7186 |
| 3-bit → 4-bit | 30% | 22.5% | **14.6533** | 15.3130 |
| 3-bit → 4-bit | 50% | 42.7% | **13.6103** | 14.4086 |
| 3-bit → bf16 | 10% | 6.3% | **15.5896** | 16.0638 |
| 3-bit → bf16 | 20% | 13.8% | **14.8628** | 15.5756 |
| 3-bit → bf16 | 30% | 22.5% | **14.1662** | 15.0077 |
| 3-bit → bf16 | 50% | 42.7% | **12.9020** | 13.9427 |
| 2-bit → 4-bit | 10% | 12.0% | 394,400 | **271,679** |
| 2-bit → 4-bit | 20% | 23.5% | 115,449 | **73,832** |
| 2-bit → 4-bit | 30% | 33.6% | 37,474 | **23,251** |
| 2-bit → 4-bit | 50% | 53.3% | 3,668 | **2,372** |
| 2-bit → bf16 | 10% | 12.0% | 388,119 | **272,458** |
| 2-bit → bf16 | 20% | 23.5% | 113,117 | **71,497** |
| 2-bit → bf16 | 30% | 33.6% | 36,409 | **22,827** |
| 2-bit → bf16 | 50% | 53.3% | 3,505 | **2,282** |

**3-bit base:** the gate beats random at every share, for both targets, by more than the spread
of the 5 random draws.

**2-bit base:** the gate loses to random at every share. 2-bit RTN Qwen3-8B is broken
(perplexity ~10⁶), so its entropy no longer marks the tokens it gets wrong. The method needs a
base that still works; 3-bit does.

**Withdrawn.** Earlier versions reported fp16 18.43 on short prompts, 2-bit 13.27 / 4-bit 13.20 /
8-bit 13.05 (`results/ppl.json`, one quantised layer) and a "near-8-bit" 2-bit model in a blog post.
None of that held up; it is replaced by the tables above.
