"""Probe 12: where the decode NaN starts in KDA layers 0-2.

On mixed steps (and on the first NaN outputs of other steps), logs per spec
request: num_accepted_tokens, the spec state slot ids, NaN counts in the
layer's spec input (qkv before conv), the conv state rows, the recurrent
state slots before the kernel, q/k/v after conv, the kernel output and the
recurrent state slots after. Also the prefill output NaN count.
Usage: probe12_kda.py <tree>/vllm/models/glm5next/nvidia/kda.py
"""
import sys
from pathlib import Path

path = Path(sys.argv[1])
src = path.read_text()


def once(old, new):
    global src
    assert src.count(old) == 1, old[:80]
    src = src.replace(old, new)


helpers = '''

# Probe 12: NaN origin in KDA layers 0-2 (spec-verify path).
_glmx_p12 = {"mixed_logs": 0, "other_logs": 0}
_GLMX_P12_LAYERS = (".layers.0.", ".layers.1.", ".layers.2.")


def _glmx_p12_rows(x, n):
    """NaN count per leading row of x (first n rows), as a list."""
    import torch as _t
    if x is None or n == 0:
        return None
    n = min(n, x.shape[0])
    return _t.isnan(x[:n].reshape(n, -1).float()).sum(-1).tolist()


def _glmx_p12_state(state, idx):
    import torch as _t
    if idx is None:
        return None
    flat = idx.reshape(-1).long()
    ok = (flat >= 0) & (flat < state.shape[0])
    out = _t.full((flat.numel(),), -1, dtype=_t.long, device=flat.device)
    if bool(ok.any()):
        sel = state[flat[ok]]
        out[ok] = _t.isnan(sel.reshape(sel.shape[0], -1).float()).sum(-1)
    return out.reshape(idx.shape).tolist()
'''
once('\n_glmx_step = {"mixed": False}\n', '\n_glmx_step = {"mixed": False}\n' + helpers)

# Pre-kernel snapshot, right before the spec conv update.
once('''        glmx_sync_point(f"{self.prefix} after token index_select")
''', '''        glmx_sync_point(f"{self.prefix} after token index_select")
        _p12 = None
        if (_GLMX_PROBE and use_spec and any(s in self.prefix for s in _GLMX_P12_LAYERS)
                and not __import__("torch").cuda.is_current_stream_capturing()):
            _mixed = attn_metadata_narrowed.num_prefills > 0
            _p12 = {"mixed": _mixed}
            _nsp = spec_state_indices_tensor.shape[-1]
            _ntok = num_spec_decodes * _nsp
            _p12["acc"] = num_accepted_tokens[:num_spec_decodes].tolist()
            _p12["slots"] = spec_state_indices_tensor[:num_spec_decodes].tolist()
            _p12["qkv_in"] = _glmx_p12_rows(qkv_spec, _ntok)
            # The slot the recurrent kernel reads its initial state from:
            # flat index i * stride + num_accepted_tokens[i] - 1, unchecked
            # in fused_recurrent.py:115-126 (out-of-row when acc > columns).
            _flat = spec_state_indices_tensor.reshape(-1)
            _st = spec_state_indices_tensor.stride(0)
            _read = []
            for _i, _a in enumerate(_p12["acc"]):
                _j = _i * _st + int(_a) - 1
                _read.append((_j, int(_flat[_j]) if 0 <= _j < _flat.numel() else "OOB"))
            _p12["read_slot"] = _read
            _p12["read_slot_nan"] = [
                _glmx_p12_state(recurrent_state, __import__("torch").tensor([s], device=recurrent_state.device))[0]
                if isinstance(s, int) else None for _, s in _read]
            if _mixed:
                # Prefill side of the same step: its state slots and overlap.
                _ns = non_spec_state_indices_tensor
                _p12["prefill_slots"] = _ns.tolist() if _ns is not None else None
                _p12["prefill_has_init"] = has_initial_state.tolist() if has_initial_state is not None else None
                _sp = set(int(x) for x in spec_state_indices_tensor[:num_spec_decodes].reshape(-1).tolist())
                _p12["overlap"] = sorted(_sp & set(_p12["prefill_slots"] or []))
                _p12["adjacent"] = sorted(s for s in (_p12["prefill_slots"] or [])
                                          if (s - 1) in _sp or (s + 1) in _sp)
                _p12["conv_pre"] = _glmx_p12_state(conv_state, spec_state_indices_tensor[:num_spec_decodes, 0])
                _p12["state_pre"] = _glmx_p12_state(recurrent_state, spec_state_indices_tensor[:num_spec_decodes])
''')
once('''            glmx_sync_point(f"{self.prefix} after spec conv1d_update")
''', '''            glmx_sync_point(f"{self.prefix} after spec conv1d_update")
            if _p12 is not None:
                _p12["qkv_conv"] = _glmx_p12_rows(qkv_spec, _ntok)
''')
once('''        glmx_sync_point(f"{self.prefix} after spec fused_recurrent_kda")
''', '''        glmx_sync_point(f"{self.prefix} after spec fused_recurrent_kda")
        if _p12 is not None:
            _out = core_attn_out_spec if core_attn_out_spec is not None else spec_out
            _p12["out"] = _glmx_p12_rows(_out[0], _ntok)
            _bad = any(_p12["out"] or []) or any(_p12["qkv_in"] or [])
            if _p12["mixed"] or _bad:
                _p12["state_post"] = _glmx_p12_state(recurrent_state, spec_state_indices_tensor[:num_spec_decodes])
''')
once('''            glmx_sync_point(f"{self.prefix} after chunk prefill + scatter_states")
''', '''            glmx_sync_point(f"{self.prefix} after chunk prefill + scatter_states")
            if _p12 is not None:
                _p12["prefill_out_nan"] = int(__import__("torch").isnan(core_attn_out_non_spec.float()).sum().item())
''')
once('''        # --- merge spec / non-spec outputs back into token order ---
''', '''        if _p12 is not None:
            _key = "mixed_logs" if _p12["mixed"] else "other_logs"
            _bad = any(_p12.get("out") or []) or any(_p12.get("qkv_in") or [])
            if (_p12["mixed"] and _glmx_p12[_key] < 3000) or (_bad and _glmx_p12[_key] < 60):
                _glmx_p12[_key] += 1
                _glmx_log.warning("GLMX_PROBE p12 kda %s #%d %s", self.prefix, _glmx_p12[_key], _p12)
        # --- merge spec / non-spec outputs back into token order ---
''')
path.write_text(src)
print('probe 12 applied to', path)
