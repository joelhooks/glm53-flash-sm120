# Guard image: our source build plus a guarded persistent_topk extension.
# Inert unless GLMX_TOPK_GUARD=1 (make-compose.py variant "guard").
# The guard bounds every top-k output write and counts dropped writes. It is a
# diagnostic and a stopgap, not the fix.
# Build with --build-arg GLMX_KEY_FIX=1 for the fix candidate (topk-keyfix):
# the same guard, plus a NaN-safe fp32 key, so the drop counter measures the fix.
ARG BASE=glm53-flash-sm120:source
FROM ${BASE}
ARG GLMX_KEY_FIX=0
COPY make-guard.py topk-guard.patch /opt/glmx/guard/
RUN cd /opt/glmx/guard \
 && /opt/venv/bin/python make-guard.py /opt/infernal-invocation/vllm/csrc/libtorch_stable gen \
 && TORCH=$(/opt/venv/bin/python -c 'import torch, os; print(os.path.dirname(torch.__file__))') \
 && nvcc -O3 -std=c++17 -shared -Xcompiler -fPIC -Xcompiler -fvisibility=hidden \
      -gencode arch=compute_120a,code=sm_120a \
      -D_GLIBCXX_USE_CXX11_ABI=1 -DTORCH_EXTENSION_NAME=glmx_topk \
      -I gen -I $TORCH/include -I $TORCH/include/torch/csrc/api/include \
      -L $TORCH/lib -lc10 -lc10_cuda -ltorch -ltorch_cpu -ltorch_cuda \
      -Xlinker -rpath -Xlinker $TORCH/lib \
      gen/glmx_topk.cu -o glmx_topk.so \
 && /opt/venv/bin/python -c "import torch; torch.ops.load_library('/opt/glmx/guard/glmx_topk.so'); print(torch.ops.glmx.persistent_topk, torch.ops.glmx.guard_stats)" \
 && cd /opt/infernal-invocation/vllm \
 && git apply --whitespace=nowarn /opt/glmx/guard/topk-guard.patch \
 && /opt/venv/bin/python -m py_compile vllm/model_executor/layers/sparse_attn_indexer_kpool.py \
 && grep -q '_glmx_persistent_topk' vllm/model_executor/layers/sparse_attn_indexer_kpool.py
