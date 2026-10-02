"""Shared helpers for the real Qwen3-8B runs: model, WikiText-2 windows, perplexity,
activation statistics and group-wise round-to-nearest (RTN) fake quantisation.

Weights are quantised and immediately dequantised back to bf16 ("fake quantisation"),
so perplexity is exact for the quantised weights; packed memory is computed, not measured.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn

MODEL_PATH = "/home/jasper/eirene-projects/03-inference-lab/ai-lab/models/Qwen--Qwen3-8B"
SEQ = 512
GROUP = 128


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, local_files_only=True, device_map="cuda")
    model.eval()
    return model, tok


def windows(tok, split, n):
    """n non-overlapping SEQ-token windows from WikiText-2 `split` (train = calibration, test = eval)."""
    from datasets import load_dataset
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    return [ids[i * SEQ:(i + 1) * SEQ] for i in range(n)]


@torch.no_grad()
def token_stats(model, wins):
    """Per-token log-prob of the true next token and predictive entropy (nats), on CPU."""
    lps, ents = [], []
    for w in wins:
        x = w.unsqueeze(0).cuda()
        logp = torch.log_softmax(model(x).logits[0, :-1].float(), -1)
        lps.append(logp.gather(1, x[0, 1:, None])[:, 0].cpu())
        ents.append(-(logp.exp() * logp).sum(-1).cpu())
    return torch.cat(lps), torch.cat(ents)


def perplexity(model, wins):
    lp, _ = token_stats(model, wins)
    return float(torch.exp(-lp.mean()))


def decoder_linears(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear) and ".layers." in n]


@torch.no_grad()
def act_stats(model, linears, calib):
    """Mean squared input activation per input channel, for each decoder linear."""
    sums, hooks = {}, []
    for name, mod in linears:
        def hook(m, inp, out, name=name):
            a = inp[0].float().pow(2).reshape(-1, inp[0].shape[-1]).mean(0)
            sums[name] = sums.get(name, 0) + a
        hooks.append(mod.register_forward_hook(hook))
    for w in calib:
        model(w.unsqueeze(0).cuda())
    for h in hooks:
        h.remove()
    return {k: v / len(calib) for k, v in sums.items()}


@torch.no_grad()
def rtn(w, bits, group=GROUP):
    """Asymmetric min-max RTN with one scale and zero-point per `group` input weights. Returns float32."""
    w32 = w.float()
    out, inp = w32.shape
    g = w32.reshape(out, inp // group, group)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    qmax = 2 ** bits - 1
    scale = ((hi - lo) / qmax).clamp_min(1e-8)
    zero = torch.round(-lo / scale)
    q = torch.clamp(torch.round(g / scale) + zero, 0, qmax)
    return ((q - zero) * scale).reshape(out, inp)


@torch.no_grad()
def weighted_err(w, wq, act):
    """Output-error proxy: sum over weights of (dW)^2 * E[x^2] of the input channel (diagonal Hessian)."""
    return float(((w.float() - wq) ** 2 * act.view(1, -1)).sum())


def packed_gb(model, bits_by_linear, group=GROUP):
    """Size of the weights if stored packed: `bits` per quantised weight plus an fp16 scale and
    zero-point per group; everything else (embeddings, lm_head, norms) stays bf16."""
    linear = dict(decoder_linears(model))
    total_bits = 0.0
    for n, p in model.named_parameters():
        mod = n.rsplit(".", 1)[0]
        if mod in linear and n.endswith("weight"):
            total_bits += p.numel() * (bits_by_linear[mod] + 32 / group)
        else:
            total_bits += p.numel() * 16
    return round(total_bits / 8 / 1e9, 2)


def save(path, results):
    Path(path).write_text(json.dumps(results, indent=2))
