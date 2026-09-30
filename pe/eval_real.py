"""Quantisation error on a real Qwen3-8B layer (safetensors on disk)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from quant import quantize_and_store, reconstruct_from_storage
from safetensors import safe_open

D = "/home/jasper/eirene-projects/03-inference-lab/ai-lab/models/Qwen--Qwen3-8B"
idx = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
names = ["model.layers.10.self_attn.q_proj.weight", "model.layers.10.mlp.down_proj.weight"]
out = {}
for n in names:
    with safe_open(f"{D}/{idx[n]}", "pt") as f:
        w = f.get_tensor(n).float().numpy()
    w = w[:2048, :2048]
    q = quantize_and_store(w, group_size=64, max_bits=8)
    out[n] = {str(b): float(np.linalg.norm(w - reconstruct_from_storage(q, b)) / np.linalg.norm(w)) for b in (2, 3, 4, 6, 8)}
    print(n, out[n])
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "real_layer_error.json"), "w"), indent=2)
