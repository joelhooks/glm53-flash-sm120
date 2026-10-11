# Origin-fix image: our source build plus origin-fix.patch only (no probes, no
# top-k guard, no NaN-safe key). The patch stops model runner v2 from DCP-scaling
# the Mamba group's block-table width (vllm/v1/worker/gpu/model_runner.py).
ARG BASE=glm53-flash-sm120:source
FROM ${BASE}
COPY origin-fix.patch /opt/glmx/fix/origin-fix.patch
RUN cd /opt/infernal-invocation/vllm \
 && git apply --whitespace=nowarn /opt/glmx/fix/origin-fix.patch \
 && /opt/venv/bin/python -m py_compile vllm/v1/worker/gpu/model_runner.py \
 && grep -q 'width_cp_size = 1 if isinstance(spec, MambaSpec)' vllm/v1/worker/gpu/model_runner.py
