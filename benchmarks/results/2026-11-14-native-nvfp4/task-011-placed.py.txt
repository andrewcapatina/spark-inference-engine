"""Streaming reader for Qwen compressed-tensors mixed-precision checkpoints.

Native cache remains BF16; this module only streams tensors from disk.
"""
import json
from pathlib import Path
from contextlib import contextmanager, ExitStack
from safetensors import safe_open

@contextmanager
def quantized_checkpoint(directory):
    root = Path(directory).resolve()
    cfg = json.loads((root / "config.json").read_text())
    qc = cfg.get("quantization_config", {})
    if qc.get("quant_method") != "compressed-tensors" or qc.get("format") != "mixed-precision":
        raise ValueError("not a compressed-tensors mixed-precision checkpoint")
    text_cfg = cfg.get("text_config", cfg)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    canonical = {}
    for name, shard in weight_map.items():
        if name.startswith("model.visual.") or name.startswith("mtp.") or name.startswith("model.mtp."):
            continue
        if name.startswith("model.language_model."):
            canon = name[len("model.language_model."):]
        elif name.startswith("lm_head."):
            canon = name
        else:
            raise ValueError(f"unexpected tensor namespace: {name}")
        if canon.endswith(".self_attn.k_scale") or canon.endswith(".self_attn.v_scale"):
            continue
        if canon in canonical:
            raise ValueError(f"duplicate canonical key: {canon}")
        canonical[canon] = (name, shard)
    shards = {}
    with ExitStack() as stack:
        for _, (_, shard) in canonical.items():
            if shard not in shards:
                path = root / shard
                if not path.resolve().is_relative_to(root):
                    raise ValueError(f"shard escapes directory: {shard}")
                shards[shard] = stack.enter_context(safe_open(str(path), framework="pt", device="cpu"))
        consumed = set()
        def get_tensor(canon):
            orig, shard = canonical[canon]
            t = shards[shard].get_tensor(orig)
            consumed.add(canon)
            return t
        yield text_cfg, get_tensor, set(canonical)
        missing = set(canonical) - consumed
        if missing:
            raise RuntimeError(f"unconsumed tensors: {sorted(missing)[:5]}")

import torch

def load_quantized_qwen(directory, dtype=torch.bfloat16, device='cuda'):
    if dtype != torch.bfloat16:
        raise ValueError("load_quantized_qwen requires dtype=torch.bfloat16")
    if torch.device(device).type != 'cuda':
        raise ValueError("load_quantized_qwen requires a CUDA device")

    from spark_engine.qwen_model import QwenCausalLM
    from spark_engine.qwen_nvfp4 import NVFP4Linear
    from spark_engine.qwen_fp8 import FP8Linear

    nvfp4_count = 0
    fp8_count = 0

    with quantized_checkpoint(directory) as (cfg, get, keys):
        with torch.device('meta'):
            model = QwenCausalLM(cfg)

        def _set_module(parent_path, name, module):
            if parent_path:
                parent = model.get_submodule(parent_path)
            else:
                parent = model
            setattr(parent, name, module)

        for name, module in list(model.named_modules()):
            if not isinstance(module, torch.nn.Linear):
                continue
            parent_path, _, mod_name = name.rpartition('.')
            if name + '.weight_packed' in keys:
                wp = get(name + '.weight_packed').to(device=device)
                ws = get(name + '.weight_scale').to(device=device)
                wg = get(name + '.weight_global_scale').to(device=device)
                ig = get(name + '.input_global_scale').to(device=device)
                if wp.shape[0] != module.out_features or wp.shape[1] != module.in_features // 2:
                    raise ValueError(f"NVFP4 packed shape mismatch for {name}")
                if ws.shape[0] != module.out_features or ws.shape[1] != module.in_features // 16:
                    raise ValueError(f"NVFP4 scale shape mismatch for {name}")
                _set_module(parent_path, mod_name, NVFP4Linear(wp, ws, wg, ig))
                nvfp4_count += 1
            elif name + '.weight' in keys:
                w = get(name + '.weight')
                if w.dtype == torch.float8_e4m3fn and w.shape == (module.out_features, module.in_features):
                    w = w.to(device=device)
                    scale = get(name + '.weight_scale').to(device=device)
                    _set_module(parent_path, mod_name, FP8Linear(w, scale))
                    fp8_count += 1

        for name, param in model.named_parameters():
            if param.device.type != 'meta':
                continue
            canonical = name
            if canonical not in keys:
                raise KeyError(f"Missing weight {canonical}")
            t = get(canonical)
            if t.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                raise ValueError(f"Unexpected dtype {t.dtype} for {canonical}")
            if t.shape != param.shape:
                raise ValueError(f"Shape mismatch for {canonical}: expected {param.shape}, got {t.shape}")
            parent_path, _, mod_name = name.rpartition('.')
            if mod_name == 'weight' or mod_name == 'bias':
                parent = model.get_submodule(parent_path) if parent_path else model
                setattr(parent, mod_name, torch.nn.Parameter(t.to(device=device, dtype=dtype), requires_grad=False))
            else:
                _set_module(parent_path, mod_name, torch.nn.Parameter(t.to(device=device, dtype=dtype), requires_grad=False))

    model.quantization_metadata = {
        'format': 'nvfp4_fp8',
        'kv_cache_dtype': 'bfloat16',
        'nvfp4_linears': nvfp4_count,
        'fp8_linears': fp8_count,
    }
    return model.eval()