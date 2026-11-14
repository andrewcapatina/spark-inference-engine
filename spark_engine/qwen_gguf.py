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
