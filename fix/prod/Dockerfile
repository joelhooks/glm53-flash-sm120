# Production image glm-fixed: our source build plus
#   1. origin-fix.patch: model runner v2 no longer DCP-scales the Mamba group's
#      block-table width (the root cause; upstream fixed it in vLLM PR #50493);
#   2. the nankey top-k build: NaN-safe fp32 key and bounded output writes in
#      persistent_topk (defence in depth), always on, no counter polling.
# No probes, no guard env, no debug env.
ARG BASE=glm53-flash-sm120:topk-nankey
FROM ${BASE}
COPY origin-fix.patch make-prod.py /opt/glmx/fix/
RUN cd /opt/infernal-invocation/vllm \
 && git apply --whitespace=nowarn /opt/glmx/fix/origin-fix.patch \
 && grep -q 'width_cp_size = 1 if isinstance(spec, MambaSpec)' vllm/v1/worker/gpu/model_runner.py \
 && /opt/venv/bin/python /opt/glmx/fix/make-prod.py /opt/infernal-invocation/vllm \
 && ! grep -q 'GLMX_TOPK_GUARD\|_glmx_guard_poll' vllm/model_executor/layers/sparse_attn_indexer_kpool.py \
 && /opt/venv/bin/python -m py_compile vllm/v1/worker/gpu/model_runner.py vllm/model_executor/layers/sparse_attn_indexer_kpool.py \
 && /opt/venv/bin/python -c "import torch; torch.ops.load_library('/opt/glmx/guard/glmx_topk.so'); print(torch.ops.glmx.persistent_topk)"
LABEL org.opencontainers.image.title="glm-fixed" \
      org.opencontainers.image.description="GLM-5.3-Flash EXL3 on sm120: Mamba block-table width fix plus NaN-safe bounded persistent_topk"
