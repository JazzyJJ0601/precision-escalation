#!/usr/bin/env python3
"""Real results: entropy-gated precision escalation on Qwen3-8B.

A low-bit base model (2 or 3 bits, group-wise RTN) answers every token. When its predictive
entropy is high, the token is escalated: its prediction is taken from a higher-precision
model (4-bit, i.e. base + residual bit-planes, or full bf16). The question: for a given
share of escalated tokens, does the entropy gate recover more quality than escalating the
same share of tokens at random?

Each precision is run once over the same text and per-token log-probs/entropies are kept,
so only one copy of the weights is ever on the GPU (the old version ran out of memory).
The entropy threshold for each escalation share is set on the first 10 windows (dev) and
applied unchanged to the other 30 (test). Teacher-forced evaluation: an escalated token
sees a context processed at high precision; in a real decoder that means recomputing it.
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from qcommon import decoder_linears, load_model, rtn, save, token_stats, windows  # noqa: E402

OUT = HERE / "real.json"
N_WIN, N_DEV = 40, 10
SHARES = [0.1, 0.2, 0.3, 0.5]


def ppl(lp):
    return round(float(torch.exp(-lp.mean())), 4)


def main():
    model, tok = load_model()
    wins = windows(tok, "test", N_WIN)
    linears = decoder_linears(model)
    originals = {n: m.weight.data.to("cpu", copy=True) for n, m in linears}

    stats = {}
    for name, bits in (("bf16", None), ("4bit", 4), ("3bit", 3), ("2bit", 2)):
        for n, m in linears:
            m.weight.data = originals[n].cuda()
            if bits:
                m.weight.data = rtn(m.weight.data, bits).to(torch.bfloat16)
        stats[name] = token_stats(model, wins)
        print(name, "done", flush=True)
        torch.cuda.empty_cache()

    per_win = 512 - 1  # predicted tokens per window
    dev = slice(0, N_DEV * per_win)
    test = slice(N_DEV * per_win, None)
    results = {"setup": {"model": "Qwen3-8B", "group": 128, "dev": f"WikiText-2 test windows 1-{N_DEV}",
                         "test": f"WikiText-2 test windows {N_DEV + 1}-{N_WIN} (x512 tokens)"},
               "test_ppl": {k: ppl(v[0][test]) for k, v in stats.items()}}
    gen = torch.Generator().manual_seed(0)
    for base in ("2bit", "3bit"):
        for hi in ("4bit", "bf16"):
            rows = []
            b_lp, b_ent = stats[base]
            h_lp = stats[hi][0]
            for share in SHARES:
                thr = torch.quantile(b_ent[dev], 1 - share)
                gate = b_ent[test] > thr
                mixed = torch.where(gate, h_lp[test], b_lp[test])
                rand_ppls = []
                for _ in range(5):
                    r = torch.rand(gate.numel(), generator=gen) < gate.float().mean()
                    rand_ppls.append(ppl(torch.where(r, h_lp[test], b_lp[test])))
                rows.append({"target_share": share, "test_share": round(float(gate.float().mean()), 3),
                             "entropy_gate_ppl": ppl(mixed),
                             "random_ppl_mean": round(sum(rand_ppls) / 5, 4),
                             "random_ppl_all": rand_ppls})
                print(base, hi, rows[-1], flush=True)
            results[f"{base}_to_{hi}"] = rows
    save(OUT, results)


if __name__ == "__main__":
    main()
