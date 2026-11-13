"""Qwen3.5 text config parser.

Accepts either the nested real-model config (with text_config) or a flat
text-only config. Returns a dict with all text architecture fields preserved.
"""

def parse_config(raw):
    if "text_config" in raw:
        cfg = raw["text_config"]
    else:
        cfg = raw
    return dict(cfg)
