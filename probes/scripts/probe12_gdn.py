"""Probe 12b: how the GDN/KDA builder picks the spec state slots (align mode).

On steps with a prefill and a short (spec-verify) request, logs per request:
seq_len, query_len, start_idx = (seq_len-1)//mamba_block_size, the raw block
table row around start_idx, and the selected state row.
Usage: probe12_gdn.py <tree>/vllm/v1/attention/backends/gdn_attn.py
"""
import sys
from pathlib import Path

path = Path(sys.argv[1])
src = path.read_text()
old = '''            output=self.block_table_tensor,
        )
'''
assert src.count(old) == 1
new = old + '''        if _GLMX_P12B["on"]:
            _glmx_p12b_log(m, block_table_tensor, self.kv_cache_spec)
'''
src = src.replace(old, new)
helpers = '''

# --- glm-xid31 probe 12b (debug only; inert unless GLMX_KDA_PROBE=1) ---
import os as _glmx_os

_GLMX_P12B = {"on": _glmx_os.environ.get("GLMX_KDA_PROBE") == "1", "logs": 0, "meta": 0}


def _glmx_p12b_log(m, selected, spec):
    import logging

    import torch as _t

    if _t.cuda.is_current_stream_capturing() or _GLMX_P12B["logs"] >= 3000:
        return
    qs = m.query_start_loc_cpu
    qlens = (qs[1:] - qs[:-1]).tolist()
    if not (any(q > 4 for q in qlens) and any(2 <= q <= 4 for q in qlens)):
        return
    log = logging.getLogger("vllm")
    bs = spec.block_size
    if _GLMX_P12B["meta"] < 2:
        _GLMX_P12B["meta"] += 1
        log.warning("GLMX_PROBE p12b meta mamba_block_size=%d num_speculative_blocks=%s "
                    "block_table_shape=%s stride=%s storage_int32=%d spec=%r", bs,
                    getattr(spec, "num_speculative_blocks", None),
                    tuple(m.block_table_tensor.shape), tuple(m.block_table_tensor.stride()),
                    m.block_table_tensor.untyped_storage().nbytes() // 4, spec)
    _GLMX_P12B["logs"] += 1
    seq = m.seq_lens.tolist()
    rows = []
    for i, q in enumerate(qlens):
        s = int(seq[i])
        start = max((s - 1) // bs, 0)
        lo = max(start - 2, 0)
        raw = m.block_table_tensor[i, lo:start + 6].tolist()
        nz = int((m.block_table_tensor[i] != 0).sum())
        rows.append({"q": q, "seq": s, "start": start, "raw_from": lo, "raw": raw, "nonzero": nz,
                     "sel": selected[i].tolist()})
    log.warning("GLMX_PROBE p12b gdn #%d %s", _GLMX_P12B["logs"], rows)
'''
anchor = '\n\nclass GDNAttentionMetadataBuilder('
assert src.count(anchor) == 1
src = src.replace(anchor, helpers + anchor)
path.write_text(src)
print('probe 12b applied to', path)
