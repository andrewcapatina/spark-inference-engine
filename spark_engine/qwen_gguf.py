"""GGUF reader for Qwen3.5 models. Metadata only; no model loading yet."""
import struct
import json
import numpy as np

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3

TYPE_I8 = 0
TYPE_I16 = 1
TYPE_I32 = 2
TYPE_I64 = 3
TYPE_F32 = 4
TYPE_F64 = 5
TYPE_BOOL = 6
TYPE_STRING = 7
TYPE_ARRAY = 8

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
    if type_code == TYPE_I8:
        return _read_i8(f)
    elif type_code == TYPE_I16:
        return _read_i16(f)
    elif type_code == TYPE_I32:
        return _read_i32(f)
    elif type_code == TYPE_I64:
        return _read_i64(f)
    elif type_code == TYPE_F32:
        return _read_f32(f)
    elif type_code == TYPE_F64:
        return _read_f64(f)
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
    else:
        raise ValueError(f"Unknown GGUF type: {type_code}")

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
