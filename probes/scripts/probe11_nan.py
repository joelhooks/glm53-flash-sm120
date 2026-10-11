from pathlib import Path

p = Path('vllm/model_executor/layers/sparse_attn_indexer_kpool.py')
s = p.read_text()


def once(old, new):
    global s
    assert s.count(old) == 1, old[:60]
    s = s.replace(old, new)


helpers = '''

# GLMX probe 11: on mixed steps, find where all-NaN decode logits come from.
# Logs per-row NaN in logits next to NaN/inf/zero stats of the kernel inputs
# (q fp8, weights, hidden_states, k, gate) and of the KPool pages and scales the
# rows read. Dumps the inputs on the first NaN events.
_glmx_p11_state = {"dumps": 0, "calls": 0, "clean_scans": 0}


def _glmx_rowstats(t):
    import torch as _t

    f = t.reshape(t.shape[0], -1).float()
    fin = _t.where(_t.isfinite(f), f.abs(), _t.zeros_like(f))
    return [(int(_t.isnan(f[r]).sum()), int(_t.isinf(f[r]).sum()), float(fin[r].max()),
             int((f[r] == 0).sum())) for r in range(f.shape[0])]


def _glmx_p11(prefix, score_rows, logits, seq_lens, q_quant, weights, hidden_states, k, gate_score,
              kv_cache_raw, block_table, num_decode_tokens, positions):
    import logging
    import os
    import torch as _t
    from vllm.models.glm5next.nvidia.kda import _glmx_step

    if not _glmx_step.get("mixed"):
        return
    log = logging.getLogger("vllm")
    _glmx_p11_state["calls"] += 1
    lens = [int(x) for x in seq_lens.tolist()]
    lnan = [int(_t.isnan(logits[r, :lens[r]]).sum()) for r in range(score_rows)]
    q8 = q_quant[:score_rows].reshape(score_rows, -1).view(_t.uint8)
    qnan = [int(((q8[r] & 0x7F) == 0x7F).sum()) for r in range(score_rows)]
    qzero = [int(((q8[r] & 0x7F) == 0).sum()) for r in range(score_rows)]
    w = _glmx_rowstats(weights[:score_rows])
    hs = _glmx_rowstats(hidden_states[:score_rows])
    kk = _glmx_rowstats(k[:score_rows]) if k is not None else None
    gs = _glmx_rowstats(gate_score[:score_rows]) if gate_score is not None else None
    pos = positions[:score_rows].tolist() if positions is not None else None
    any_nan = any(lnan)
    pages = None
    if any_nan or _glmx_p11_state["clean_scans"] < 24:
        if not any_nan:
            _glmx_p11_state["clean_scans"] += 1
        pages = []
        n_blocks = kv_cache_raw.shape[0]
        flat = kv_cache_raw.reshape(n_blocks, -1)
        for r in range(score_rows):
            npg = -(-lens[r] // 32)
            ids = block_table[r, :npg].long()
            bad_ids = int(((ids < 0) | (ids >= n_blocks)).sum())
            ids = ids.clamp(0, n_blocks - 1)
            pg = flat[ids]
            sc = pg[:, 4096:4096 + 128].contiguous().view(_t.float32).reshape(-1)[:lens[r]]
            kb = pg[:, :4096].reshape(-1)[:lens[r] * 128]
            pages.append({"npg": npg, "bad_ids": bad_ids, "sc_nan": int(_t.isnan(sc).sum()),
                          "sc_inf": int(_t.isinf(sc).sum()), "sc_zero": int((sc == 0).sum()),
                          "sc_max": float(sc[_t.isfinite(sc)].abs().max()) if _t.isfinite(sc).any() else None,
                          "k_nanbytes": int(((kb & 0x7F) == 0x7F).sum()),
                          "last_page": int(block_table[r, npg - 1]), "last_sc": [float(x) for x in sc[-4:].tolist()]})
    level = log.error if any_nan else log.warning
    level("GLMX_PROBE p11 nan check: call=%d prefix=%s ndec=%d lens=%s pos=%s logits_nan=%s q_nan=%s q_zero=%s "
          "w=%s hs=%s k=%s gate=%s pages=%s",
          _glmx_p11_state["calls"], prefix, num_decode_tokens, lens, pos, lnan, qnan, qzero, w, hs, kk, gs, pages)
    if any_nan and _glmx_p11_state["dumps"] < 2:
        _glmx_p11_state["dumps"] += 1
        path = "/cache/glmx-p11-%d-%d.pt" % (os.getpid(), _glmx_p11_state["dumps"])
        _t.save({"prefix": prefix, "lens": lens, "positions": pos, "q_quant": q_quant[:score_rows].view(_t.uint8).cpu(),
                 "weights": weights[:score_rows].cpu(), "hidden_states": hidden_states[:max(score_rows, num_decode_tokens) + 8].cpu(),
                 "k": k[:score_rows].cpu() if k is not None else None,
                 "gate_score": gate_score[:score_rows].cpu() if gate_score is not None else None,
                 "block_table": block_table[:score_rows].cpu()}, path)
        log.error("GLMX_PROBE p11 dumped %s", path)
'''

once('\n\ndef _glmx_p10_alloc(', helpers + '\n\ndef _glmx_p10_alloc(')
once('''            if _GLMX_PROBE:
                _glmx_p2_log(score_rows, work_k, max_pool_len, logits, pool_seq_lens, pool_block_table, positions, num_decode_tokens, topk_indices_buffer)
''', '''            if _GLMX_PROBE:
                _glmx_p2_log(score_rows, work_k, max_pool_len, logits, pool_seq_lens, pool_block_table, positions, num_decode_tokens, topk_indices_buffer)
            if _GLMX_PROBE:
                _glmx_p11(k_cache_prefix, score_rows, logits, pool_seq_lens, q_quant, weights, hidden_states, k, gate_score,
                          kv_cache_raw, pool_block_table, num_decode_tokens, positions)
''')
p.write_text(s)
print('ok')
