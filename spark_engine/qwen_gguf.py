"""GGUF reader for Qwen3.5 models. Metadata and Q8_0 dequantization."""
import struct
import json
import numpy as np
import torch

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3

TYPE_U8 = 0
TYPE_I8 = 1
TYPE_U16 = 2
TYPE_I16 = 3
TYPE_U32 = 4
TYPE_I32 = 5
TYPE_F32 = 6
TYPE_BOOL = 7
TYPE_STRING = 8
TYPE_ARRAY = 9
TYPE_U64 = 10
TYPE_I64 = 11
TYPE_F64 = 12

def _read_u8(f):
    return struct.unpack('<B', f.read(1))[0]

def _read_i8(f):
    return struct.unpack('<b', f.read(1))[0]

def _read_u16(f):
    return struct.unpack('<H', f.read(2))[0]

def _read_i16(f):
    return struct.unpack('<h', f.read(2))[0]

def _read_u32(f):
    return struct.unpack('<I', f.read(4))[0]

def _read_i32(f):
    return struct.unpack('<i', f.read(4))[0]

def _read_u64(f):
    return struct.unpack('<Q', f.read(8))[0]

def _read_i64(f):
    return struct.unpack('<q', f.read(8))[0]

def _read_f32(f):
    return struct.unpack('<f', f.read(4))[0]

def _read_f64(f):
    return struct.unpack('<d', f.read(8))[0]

def _read_string(f):
    length = _read_u64(f)
    if length == 0:
        return ""
    data = f.read(length)
    return data.decode('utf-8')

def _read_value(f, type_code):
    if type_code == TYPE_U8:
        return _read_u8(f)
    elif type_code == TYPE_I8:
        return _read_i8(f)
    elif type_code == TYPE_U16:
        return _read_u16(f)
    elif type_code == TYPE_I16:
        return _read_i16(f)
    elif type_code == TYPE_U32:
        return _read_u32(f)
    elif type_code == TYPE_I32:
        return _read_i32(f)
    elif type_code == TYPE_F32:
        return _read_f32(f)
    elif type_code == TYPE_BOOL:
        return _read_u8(f) != 0
    elif type_code == TYPE_STRING:
        return _read_string(f)
    elif type_code == TYPE_ARRAY:
        array_type = _read_u32(f)
        length = _read_u64(f)
        items = []
        for _ in range(length):
            items.append(_read_value(f, array_type))
        return items
    elif type_code == TYPE_U64:
        return _read_u64(f)
    elif type_code == TYPE_I64:
        return _read_i64(f)
    elif type_code == TYPE_F64:
        return _read_f64(f)
    else:
        raise ValueError(f"Unknown GGUF type: {type_code}")

def dequantize_q8_0(raw_bytes):
    """Decode one or more Q8_0 GGML blocks: each has FP16 scale then 32 signed int8 values."""
    block_size = 34
    n = len(raw_bytes)
    if n % block_size != 0:
        raise ValueError(f"Q8_0 payload must be a multiple of {block_size} bytes, got {n}")
    num_blocks = n // block_size
    dtype = np.dtype([('scale', '<f2'), ('weights', 'i1', (32,))])
    blocks = np.frombuffer(raw_bytes, dtype=dtype)
    scales = blocks['scale'].astype(np.float32)
    ints = blocks['weights'].astype(np.float32)
    result = (ints * scales[:, np.newaxis]).ravel()
    return torch.tensor(result, dtype=torch.float32)

def read_metadata(path):
    """Read GGUF metadata, returning a dict of key -> value."""
    with open(path, 'rb') as f:
        magic = _read_u32(f)
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: {magic:#x}")
        version = _read_u32(f)
        if version != GGUF_VERSION:
            raise ValueError(f"Unsupported GGUF version: {version}")
        n_tensors = _read_u64(f)
        n_kv = _read_u64(f)

        metadata = {}
        for _ in range(n_kv):
            key = _read_string(f)
            type_code = _read_u32(f)
            value = _read_value(f, type_code)
            metadata[key] = value

        return metadata


def read_tensor(path, directory, name, device="cpu", dtype=torch.float32):
    """Read a single tensor payload from a GGUF file using a pre-parsed directory."""
    tensor_info = directory["tensors"].get(name)
    if tensor_info is None:
        raise KeyError(f"Tensor {name} not found in GGUF directory")

    dims = tensor_info["dimensions"]
    tensor_type = tensor_info["type"]
    offset = tensor_info["offset"]

    # Compute element count and payload size
    n_elements = 1
    for d in dims:
        n_elements *= d

    if tensor_type == 0:  # F32
        payload_size = n_elements * 4
    elif tensor_type == 30:  # BF16
        payload_size = n_elements * 2
    elif tensor_type == 8:  # Q8_0
        payload_size = (n_elements // 32) * 34
    else:
        raise ValueError(f"Unsupported tensor type: {tensor_type}")

    with open(path, "rb") as f:
        f.seek(directory["data_offset"] + offset)
        raw = f.read(payload_size)
        if len(raw) != payload_size:
            raise ValueError(f"Truncated tensor {name}: expected {payload_size} bytes, got {len(raw)}")

    # Decode based on type
    if tensor_type == 0:  # F32
        arr = np.frombuffer(raw, dtype=np.float32)
    elif tensor_type == 30:  # BF16 -> reinterpret as F32 upper bits
        # BF16: 16 bits are the upper 16 bits of an IEEE float32
        # Read as uint16, shift left 16, reinterpret as float32
        bf16_vals = np.frombuffer(raw, dtype=np.uint16)
        f32_bits = (bf16_vals.astype(np.uint32)) << 16
        arr = f32_bits.view(np.float32)
    elif tensor_type == 8:  # Q8_0
        arr = dequantize_q8_0(raw).numpy()
    else:
        raise ValueError(f"Unsupported tensor type: {tensor_type}")

    # Reverse GGML dimensions for PyTorch (GGML is column-major)
    shape = tuple(reversed(dims))
    arr = arr.reshape(shape)

    tensor = torch.tensor(arr, dtype=dtype, device=device)
    return tensor


def restore_tensor_value(gguf_name, tensor):
    """Invert Qwen3.5 GGUF export conventions back to native PyTorch values.

    Input tensor must already be decoded and have GGML dimensions reversed.
    Operates in FP32 internally; preserves device and final dtype.
    """
    if tensor.dtype != torch.float32:
        tensor = tensor.to(torch.float32)

    import re
    m = re.match(r"blk\.[0-9]+\.(.+)", gguf_name)
    suffix = m.group(1) if m else gguf_name
    if gguf_name == "output_norm.weight" or suffix in ("attn_norm.weight",
                     "post_attention_norm.weight", "attn_q_norm.weight",
                     "attn_k_norm.weight"):
        tensor = tensor - 1.0
    elif suffix == "ssm_a":
        tensor = torch.log(-tensor)
    elif suffix == "ssm_conv1d.weight":
        tensor = tensor.unsqueeze(1)

    return tensor


def map_tensor_name(gguf_name):
    """Map a GGUF tensor name to the engine's native state_dict name.

    Raises ValueError for unknown names instead of silently skipping.
    """
    if gguf_name == "token_embd.weight":
        return "embed_tokens.weight"
    if gguf_name == "output.weight":
        return "lm_head.weight"
    if gguf_name == "output_norm.weight":
        return "norm.weight"

    import re
    m = re.match(r"blk\.([0-9]+)\.(.+)", gguf_name)
    if not m:
        raise ValueError(f"Unknown GGUF tensor name: {gguf_name}")

    layer = int(m.group(1))
    rest = m.group(2)
    prefix = f"layers.{layer}."

    if rest == "attn_norm.weight":
        return prefix + "input_layernorm.weight"
    if rest == "post_attention_norm.weight":
        return prefix + "post_attention_layernorm.weight"
    if rest == "ffn_gate.weight":
        return prefix + "mlp.gate_proj.weight"
    if rest == "ffn_up.weight":
        return prefix + "mlp.up_proj.weight"
    if rest == "ffn_down.weight":
        return prefix + "mlp.down_proj.weight"

    if rest == "attn_q.weight":
        return prefix + "self_attn.q_proj.weight"
    if rest == "attn_k.weight":
        return prefix + "self_attn.k_proj.weight"
    if rest == "attn_v.weight":
        return prefix + "self_attn.v_proj.weight"
    if rest == "attn_output.weight":
        return prefix + "self_attn.o_proj.weight"
    if rest == "attn_q_norm.weight":
        return prefix + "self_attn.q_norm.weight"
    if rest == "attn_k_norm.weight":
        return prefix + "self_attn.k_norm.weight"

    if rest == "attn_qkv.weight":
        return prefix + "linear_attn.in_proj_qkv.weight"
    if rest == "attn_gate.weight":
        return prefix + "linear_attn.in_proj_z.weight"
    if rest == "ssm_alpha.weight":
        return prefix + "linear_attn.in_proj_a.weight"
    if rest == "ssm_beta.weight":
        return prefix + "linear_attn.in_proj_b.weight"
    if rest == "ssm_a":
        return prefix + "linear_attn.A_log"
    if rest == "ssm_dt.bias":
        return prefix + "linear_attn.dt_bias"
    if rest == "ssm_conv1d.weight":
        return prefix + "linear_attn.conv1d.weight"
    if rest == "ssm_norm.weight":
        return prefix + "linear_attn.norm.weight"
    if rest == "ssm_out.weight":
        return prefix + "linear_attn.out_proj.weight"

    raise ValueError(f"Unknown GGUF tensor name: {gguf_name}")


def config_from_gguf(metadata):
    """Build a flat QwenCausalLM config dict from GGUF metadata."""
    def get(key, default=None):
        return metadata.get(key, default)

    hidden_size = int(get("qwen35.embedding_length", 0))
    num_layers = int(get("qwen35.block_count", 0))
    num_heads = int(get("qwen35.attention.head_count", 0))
    num_kv_heads = int(get("qwen35.attention.head_count_kv", 0))
    head_dim = int(get("qwen35.attention.key_length", 0))
    intermediate_size = int(get("qwen35.feed_forward_length", 0))
    vocab_size = len(get("tokenizer.ggml.tokens", []))
    rope_theta = float(get("qwen35.rope.freq_base", 10000.0))
    rms_eps = float(get("qwen35.attention.layer_norm_rms_epsilon", 1e-6))

    ssm_group_count = int(get("qwen35.ssm.group_count", 0))
    ssm_state_size = int(get("qwen35.ssm.state_size", 0))
    ssm_time_step_rank = int(get("qwen35.ssm.time_step_rank", 0))
    ssm_inner_size = int(get("qwen35.ssm.inner_size", 0))
    ssm_conv_kernel = int(get("qwen35.ssm.conv_kernel", 4))

    full_attention_interval = int(get("qwen35.full_attention_interval", 0))

    rope_dim = int(get("qwen35.rope.dimension_count", head_dim))
    rotary_fraction = rope_dim / head_dim if head_dim > 0 else 1.0

    layer_types = []
    for i in range(num_layers):
        if full_attention_interval > 0 and (i + 1) % full_attention_interval == 0:
            layer_types.append("full_attention")
        else:
            layer_types.append("linear_attention")

    return {
        "hidden_size": hidden_size,
        "num_hidden_layers": num_layers,
        "intermediate_size": intermediate_size,
        "num_attention_heads": num_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "rms_norm_eps": rms_eps,
        "vocab_size": vocab_size,
        "linear_num_key_heads": ssm_group_count,
        "linear_key_head_dim": ssm_state_size,
        "linear_num_value_heads": ssm_time_step_rank,
        "linear_value_head_dim": ssm_inner_size // ssm_time_step_rank if ssm_time_step_rank > 0 else 0,
        "linear_conv_kernel_dim": ssm_conv_kernel,
        "layer_types": layer_types,
        "rope_parameters": {
            "rope_theta": rope_theta,
            "partial_rotary_factor": rotary_fraction,
        },
    }


def tokenizer_from_gguf(metadata):
    """Build a tokenizer from GGUF metadata using the installed tokenizers library.

    Constructs a BPE tokenizer from the GGUF vocabulary/merges, applies the
    model's chat template, and returns a PreTrainedTokenizerFast-compatible
    object with encode/decode and callable return_tensors support.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers, normalizers, decoders
    from transformers import PreTrainedTokenizerFast

    # Extract vocabulary and merges from GGUF metadata
    tokens = metadata.get("tokenizer.ggml.tokens", [])
    token_types = metadata.get("tokenizer.ggml.token_type", [])
    merges_raw = metadata.get("tokenizer.ggml.merges", [])

    # Build vocab dict from ALL metadata token strings at their original indices
    # (no filtering by type) so special/control/unused slots keep their correct IDs
    vocab = {}
    for i, token in enumerate(tokens):
        vocab[token] = i

    # Build merges list from provided ordered merges
    merges = []
    for m in merges_raw:
        parts = m.split(" ")
        if len(parts) == 2:
            merges.append((parts[0], parts[1]))

    # Build vocab for BPE model (token -> index)
    bpe_vocab = vocab

    # Add special tokens from GGUF metadata
    special_tokens = {}
    eos_id = metadata.get("tokenizer.ggml.eos_token_id")
    bos_id = metadata.get("tokenizer.ggml.bos_token_id")
    pad_id = metadata.get("tokenizer.ggml.padding_token_id")

    if eos_id is not None and eos_id < len(tokens):
        special_tokens["eos_token"] = tokens[eos_id]
    if bos_id is not None and bos_id < len(tokens):
        special_tokens["bos_token"] = tokens[bos_id]
    if pad_id is not None and pad_id < len(tokens):
        special_tokens["pad_token"] = tokens[pad_id]

    # Build BPE model
    bpe_model = models.BPE(
        vocab=bpe_vocab,
        merges=merges,
        unk_token=special_tokens.get("eos_token", "<|endoftext|>"),
        byte_fallback=True,
    )

    # Build tokenizer
    tokenizer = Tokenizer(bpe_model)

    # Qwen3.5 uses byte-level BPE with specific preprocessing
    # Check metadata for normalizer settings; default to no NFC (actual GGUF lacks it)
    normalizer_name = metadata.get("tokenizer.normalizer", "")
    if "nfc" in normalizer_name.lower():
        tokenizer.normalizer = normalizers.NFC()
    # No normalizer by default (matches actual GGUF metadata)

    # Byte-level pre-tokenizer for Qwen3.5
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Add special tokens
    pass

    # Wrap in PreTrainedTokenizerFast for return_tensors support
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token=special_tokens.get("eos_token"),
        bos_token=special_tokens.get("bos_token"),
        pad_token=special_tokens.get("pad_token"),
        unk_token=special_tokens.get("eos_token"),
    )

    # Apply chat template from GGUF metadata
    chat_template = metadata.get("tokenizer.chat_template", "")
    if chat_template:
        fast_tokenizer.chat_template = chat_template

    return fast_tokenizer


def read_tensor_directory(path):
    """Read GGUF metadata and tensor directory without loading tensor payloads."""
    with open(path, 'rb') as f:
        magic = _read_u32(f)
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: {magic:#x}")
        version = _read_u32(f)
        if version != GGUF_VERSION:
            raise ValueError(f"Unsupported GGUF version: {version}")
        n_tensors = _read_u64(f)
        n_kv = _read_u64(f)

        metadata = {}
        for _ in range(n_kv):
            key = _read_string(f)
            type_code = _read_u32(f)
            value = _read_value(f, type_code)
            metadata[key] = value

        alignment = 32
        if 'general.alignment' in metadata:
            alignment = int(metadata['general.alignment'])

        def align_up(offset):
            remainder = offset % alignment
            if remainder == 0:
                return offset
            return offset + (alignment - remainder)

        tensors = {}
        for _ in range(n_tensors):
            name = _read_string(f)
            n_dims = _read_u32(f)
            dimensions = []
            for _ in range(n_dims):
                dimensions.append(_read_u64(f))
            tensor_type = _read_u32(f)
            offset = _read_u64(f)
            tensors[name] = {
                'dimensions': dimensions,
                'type': tensor_type,
                'offset': offset,
            }

        data_offset = align_up(f.tell())

        return {
            'metadata': metadata,
            'data_offset': data_offset,
            'tensors': tensors,
        }
