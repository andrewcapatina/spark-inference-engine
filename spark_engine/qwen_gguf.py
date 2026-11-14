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
        data_offset = 0
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
            data_offset = offset

        data_offset = align_up(data_offset)

        return {
            'metadata': metadata,
            'data_offset': data_offset,
            'tensors': tensors,
        }
