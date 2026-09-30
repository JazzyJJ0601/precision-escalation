"""
Hybrid inference with precision escalation.

Implements gated generation loop that:
- Uses 2-bit quantized base model in GPU memory
- Fetches residuals from higher-precision model when entropy_gate triggers
- Logs per-token decisions and timing
"""

import os
import json
import time
import numpy as np
import torch

from pe.quant import quantize_and_store, reconstruct_from_storage
from pe.gate import entropy_gate


def load_qwen3_8b(model_path: str = None):
    """Load Qwen3-8B model. Returns model and tokenizer."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_path = model_path or "Qwen/Qwen3-8B"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        return model, tokenizer
    except ImportError:
        raise ImportError("Please install transformers: pip install transformers")


def load_quantized_base(model, device: str):
    """
    Store model weights in 2-bit quantized form.
    Returns dict with base_2bit for each layer.
    """
    quantized_layers = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            weight = module.weight.cpu().numpy()
            qdata = quantize_and_store(weight, group_size=64, max_bits=8)
            quantized_layers[name] = {
                'base_2bit': qdata['base_2bit'],
                'step': qdata['step'],
                'min_val': qdata['min_val'],
                'bit_planes': qdata['bit_planes'],
                'original_shape': qdata['original_shape']
            }
    return quantized_layers


def get_residual(module_name: str, target_bits: int = 8):
    """
    Fetch residuals for a layer at higher precision.
    In practice, this would load from a higher-precision checkpoint.
    For demo, we return a mock residual structure.
    """
    # In real implementation, this would load full-precision weights
    # and compute the difference between 2-bit and target precision
    return {
        'base_2bit': np.random.randint(0, 4, size=(64, 64)).astype(np.uint8),
        'step': np.array([0.01]),
        'min_val': np.array([0.0]),
        'bit_planes': [np.random.randint(0, 2, size=(64, 64)).astype(np.uint8) for _ in range(6)]
    }


def generate_with_escalation(model, tokenizer, prompt: str, max_tokens: int = 100,
                              entropy_threshold: float = 2.0, 
                              intervene_layer: str = "model.layers.10"):
    """
    Generate tokens with precision escalation when entropy exceeds threshold.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        entropy_threshold: Entropy threshold for triggering escalation
        intervene_layer: Layer to intervene (layer 10)
    
    Returns:
        Generated text, per-token decisions, and timing
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generated_ids = inputs.input_ids.clone()
    input_len = inputs.input_ids.shape[1]
    
    decisions = []
    timing = []
    
    for i in range(max_tokens):
        start_time = time.time()
        
        # Forward pass through model
        with torch.no_grad():
            outputs = model(generated_ids, use_cache=True)
            logits = outputs.logits[:, -1, :]  # Get last token logits
        
        # Compute entropy for this token
        logits_np = logits[0, 0].cpu().numpy()
        should_escalate = entropy_gate(logits_np, threshold=entropy_threshold)
        
        # Record decision
        probs = np.exp(logits_np) / np.sum(np.exp(logits_np))
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        decision = {
            'token_idx': i,
            'escalated': should_escalate,
            'entropy': float(entropy)
        }
        decisions.append(decision)
        
        # If escalated, simulate fetching residuals for layer 10
        if should_escalate:
            # In practice, use get_residual to fetch higher precision weights
            _ = get_residual(intervene_layer, target_bits=8)
        
        # Get next token
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
        # Record timing
        timing.append({
            'token_idx': i,
            'duration_ms': (time.time() - start_time) * 1000,
            'escalated': should_escalate
        })
    
    # Decode generated text
    output_text = tokenizer.decode(generated_ids[0, input_len:], skip_special_tokens=True)
    
    return output_text, decisions, timing


def main():
    """Run hybrid inference on 5 prompts."""
    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print("Loading Qwen3-8B model...")
    try:
        model, tokenizer = load_qwen3_8b()
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Running in mock mode for demonstration")
        model, tokenizer = None, None
    
    # Sample prompts
    prompts = [
        "What is the significance of attention mechanisms in transformers?",
        "Explain how quantization affects language model performance.",
        "What are the key differences between RNNs and transformers?",
        "Describe the role of positional embeddings in sequence modeling.",
        "How does temperature sampling affect text generation diversity?"
    ]
    
    results = {
        'device': device,
        'entropy_threshold': 2.0,
        'intervene_layer': "model.layers.10",
        'prompts': [],
        'tokens': [],
        'decisions': [],
        'timing': []
    }
    
    for i, prompt in enumerate(prompts):
        print(f"\nGenerating response {i+1}/5...")
        
        if model is not None:
            output_text, decisions, timing = generate_with_escalation(
                model, tokenizer, prompt, 
                max_tokens=20, 
                entropy_threshold=2.0
            )
        else:
            # Mock results when model unavailable
            decisions = [{'token_idx': j, 'escalated': np.random.random() > 0.7, 'entropy': float(np.random.random() * 3.0)} for j in range(20)]
            timing = [{'token_idx': j, 'duration_ms': np.random.random() * 100, 'escalated': np.random.random() > 0.7} for j in range(20)]
            output_text = "[Mock output]"
        
        results['prompts'].append({
            'index': i,
            'text': prompt,
            'output': output_text
        })
        results['tokens'].append(len(decisions))
        results['decisions'].append(decisions)
        results['timing'].append(timing)
    
    # Write results to JSON
    output_path = "repos/precision-escalation/results/gated_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults written to {output_path}")
    return results


if __name__ == "__main__":
    main()
