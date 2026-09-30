#!/usr/bin/env python3
"""Perplexity evaluation of Qwen3-8B at 2/4/8-bit quantization."""

import json
import math
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/jasper/eirene-projects/03-inference-lab/ai-lab/models/Qwen--Qwen3-8B"
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "ppl.json")
SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello world, this is a test sentence.",
    "Artificial intelligence is transforming the world.",
    "Machine learning models learn from data.",
    "Python is widely used for data science.",
    "Software engineering requires continuous learning.",
    "Natural language processing enables AI assistants.",
    "Optimization algorithms converge to solutions.",
    "Graph neural networks model relational data.",
    "The five boxing wizards jump quickly.",
    "Deep learning transforms modern computing systems.",
    "Quantum physics challenges classical intuition.",
    "Statistics provides tools for data analysis.",
    "Linear algebra underpins machine learning.",
    "Calculus enables gradient based optimization.",
    "Probability theory models uncertainty and risk.",
    "Computer vision interprets images and video.",
    "Reinforcement learning learns from interaction.",
    "Transformers revolutionize sequence modeling.",
    "Neural networks approximate complex functions.",
]


def quantize_linear_weights(weight, bits):
    """Quantize linear layer weights to specified bit width."""
    original_shape = weight.shape
    weight_flat = weight.flatten(1)
    
    if bits == 8:
        scale = weight_flat.abs().max(dim=1, keepdim=True).values / 127.0
        qweight = (weight_flat / scale).round().clamp(-128, 127).to(torch.int8)
        dequant = (qweight.float() * scale).view(original_shape)
    elif bits == 4:
        scale = weight_flat.abs().max(dim=1, keepdim=True).values / 7.0
        qweight = (weight_flat / scale).round().clamp(-8, 7).to(torch.int8)
        dequant = (qweight.float() * scale).view(original_shape)
    elif bits == 2:
        min_val, max_val = weight_flat.min(dim=1, keepdim=True).values, weight_flat.max(dim=1, keepdim=True).values
        scale = (max_val - min_val) / 3.0
        qweight = ((weight_flat - min_val) / scale).round().clamp(0, 3).to(torch.uint8)
        dequant = ((qweight.float() - 1.5) * scale + min_val).view(original_shape)
    else:
        raise ValueError(f"Unsupported bit width: {bits}")
    
    return dequant.to(weight.dtype)


def evaluate_perplexity(model, tokenizer, bit_width, device):
    """Evaluate perplexity on fixed sentences with quantized layer 10."""
    quantized_modules = []
    for name, module in model.named_modules():
        if "layers.10" in name and hasattr(module, 'weight'):
            if module.__class__.__name__ == "Linear":
                original_weight = module.weight.data.clone()
                quant_weight = quantize_linear_weights(original_weight, bit_width)
                module.weight.data = quant_weight
                quantized_modules.append((module, original_weight))
    
    total_loss = 0.0
    total_tokens = 0
    
    model.eval()
    with torch.no_grad():
        for sentence in SENTENCES:
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_ids = inputs["input_ids"]
            
            outputs = model(**inputs)
            logits = outputs.logits
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            non_pad = (shift_labels != tokenizer.pad_token_id).sum().item()
            total_loss += loss.item() * non_pad
            total_tokens += non_pad
    
    for module, orig_weight in quantized_modules:
        module.weight.data = orig_weight
    
    return math.exp(total_loss / total_tokens)


def main():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    results = {}
    
    for bits in [2, 4, 8]:
        print(f"Evaluating {bits}-bit quantization...")
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )
        
        try:
            ppl = evaluate_perplexity(model, tokenizer, bits, device)
            print(f"{bits}-bit Perplexity: {ppl:.2f}")
            results[str(bits)] = round(ppl, 2)
        except Exception as e:
            print(f"Error evaluating {bits}-bit: {e}")
            results[str(bits)] = None
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
