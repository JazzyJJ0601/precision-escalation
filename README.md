# precision-escalation

Layer-wise precision escalation using 2-bit base + residual bit-planes.

## Usage

```bash
python pe/eval_layer.py
```

## Results

Layer reconstruction error (synthetic Qwen3-8B 2048×2048 weight matrix):

| Bits | Relative Error |
|------|---------------|
| 2    | 0.682         |
| 4    | 0.657         |
| 6    | 0.557         |
| 8    | 0.005         |

The 8-bit reconstruction shows minimal error (~0.5%), validating the quantization pipeline.
