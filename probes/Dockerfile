# Debug probe image: our source build plus one inert-by-default probe patch.
# Enable at run time with GLMX_KDA_PROBE=1 (make-compose.py variant "probe").
#   kda-prefill-probe.patch  probe 1: KDA prefill metadata vs tensor sizes
#   kda-probe2.patch         probe 1 plus layer-entry sync and token-index checks
ARG BASE=glm53-flash-sm120:source
FROM ${BASE}
ARG PATCH=kda-prefill-probe.patch
COPY ${PATCH} /opt/glmx/probes/probe.patch
RUN cd /opt/infernal-invocation/vllm \
 && git apply --whitespace=nowarn /opt/glmx/probes/probe.patch \
 && /opt/venv/bin/python -m py_compile vllm/models/glm5next/nvidia/kda.py \
 && grep -q '_glmx_probe_prefill' vllm/models/glm5next/nvidia/kda.py
