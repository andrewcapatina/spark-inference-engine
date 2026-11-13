"""Qwen GGUF metadata reader (metadata only, not model support)."""
import json
import struct
import torch

GGUF_MAGIC = 0x46554747

def read_metadata(path):
    """Read GGUF header and metadata tensors; return dict of metadata fields."""
    with open(path, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file (magic {magic:#x})")
        version = struct.unpack('<I', f.read(4))[0]
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        metadata_count = struct.unpack('<Q', f.read(8))[0]

        meta = {}
        for _ in range(metadata_count):
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8')
            val_type = struct.unpack('<I', f.read(4))[0]

            if val_type == 0:  # string
                slen = struct.unpack('<Q', f.read(8))[0]
                val = f.read(slen).decode('utf-8')
            elif val_type == 1:  # int8
                val = struct.unpack('<b', f.read(1))[0]
            elif val_type == 2:  # int16
                val = struct.unpack('<h', f.read(2))[0]
            elif val_type == 3:  # int32
                val = struct.unpack('<i', f.read(4))[0]
            elif val_type == 4:  # int64
                val = struct.unpack('<q', f.read(8))[0]
            elif val_type == 5:  # float32
                val = struct.unpack('<f', f.read(4))[0]
            elif val_type == 6:  # float64
                val = struct.unpack('<d', f.read(8))[0]
            elif val_type == 7:  # bool
                val = struct.unpack('<?', f.read(1))[0]
            elif val_type == 8:  # array
                arr_type = struct.unpack('<I', f.read(4))[0]
                arr_len = struct.unpack('<Q', f.read(8))[0]
                arr = []
                for _ in range(arr_len):
                    if arr_type == 0:  # string array
                        slen = struct.unpack('<Q', f.read(8))[0]
                        arr.append(f.read(slen).decode('utf-8'))
                    elif arr_type == 3:  # int32 array
                        arr.append(struct.unpack('<i', f.read(4))[0])
                    elif arr_type == 4:  # int64 array
                        arr.append(struct.unpack('<q', f.read(8))[0])
                    elif arr_type == 5:  # float32 array
                        arr.append(struct.unpack('<f', f.read(4))[0])
                        elif arr_type == 6:  # float64 array
                            arr.append(struct.unpack('<d', f.read(8))[0])
                    else:
                        raise ValueError(f"Unsupported array element type {arr_type}")
                val = arr
            else:
                raise ValueError(f"Unsupported GGUF value type {val_type}")

            meta[key] = val

    return meta
