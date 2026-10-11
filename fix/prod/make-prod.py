"""Turn the nankey image's guarded top-k hook into the production path.

The nankey image (guard/Dockerfile, GLMX_KEY_FIX=1) ships glmx_topk.so: the
persistent_topk kernel with a NaN-safe fp32 key and every output write bounded
to [0, k). There it is opt-in (GLMX_TOPK_GUARD=1) and polls a drop counter.
Here it is unconditional and silent: the library loads at import, every call
goes to torch.ops.glmx.persistent_topk, and nothing is polled or logged.
"""
import pathlib
import sys

path = pathlib.Path(sys.argv[1]) / 'vllm/model_executor/layers/sparse_attn_indexer_kpool.py'
text = path.read_text()
start = text.index('# GLMX guard: with GLMX_TOPK_GUARD=1')
end = text.index('def _can_use_b12x_kpool_prefill(')
prod = '''# glm-fixed: persistent_topk comes from /opt/glmx/guard/glmx_topk.so, built from
# this tree's kernel with a NaN-safe fp32 key (NVVM folds `bits | 0x80000000u`
# into abs/neg, which canonicalizes NaN and splits the histogram from the
# buffering) and every output write bounded to [0, k). Defence in depth: the
# origin fix keeps NaN logits from reaching the indexer.
torch.ops.load_library("/opt/glmx/guard/glmx_topk.so")


def _glmx_persistent_topk(logits, lengths, output, workspace, k, max_seq_len):
    torch.ops.glmx.persistent_topk(logits, lengths, output, workspace, k, max_seq_len)


'''
path.write_text(text[:start] + prod + text[end:])
assert text.count('_glmx_persistent_topk(') == path.read_text().count('_glmx_persistent_topk(')
print('make-prod: patched', path)
