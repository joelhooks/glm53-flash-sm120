# CPU-only source overlay. Reuse the original native extensions and toolchain.
FROM verdictai/glm53-flash-exl3-k4@sha256:0f1cdcc8891f1cc3a444121eb61d366289a1cbba285f0892dcbb24bc94961692
USER root
COPY trees/vllm/ /opt/infernal-invocation/vllm/
COPY trees/b12x/ /opt/infernal-invocation/b12x/
COPY trees/lmcache/ /opt/infernal-invocation/lmcache/
COPY receipts/summary.json receipts/PROVENANCE.json /opt/glmx/
ENV PYTHONPATH=/opt/infernal-invocation/vllm:/opt/infernal-invocation/b12x:/opt/infernal-invocation/lmcache
# Build the editable Python package offline with the image's existing compiler
# dependencies. vLLM's copied native extensions are deliberately NOT rebuilt.
# uv is absent in the pinned image. Use its existing virtualenv's pip for
# B12X packaging only; do not install a new tool or rebuild vLLM through pip.
RUN /opt/venv/bin/python -m pip install \
      --no-index --no-deps --no-build-isolation --force-reinstall \
      -e /opt/infernal-invocation/b12x
RUN /opt/venv/bin/python -m compileall -q \
      /opt/infernal-invocation/vllm/vllm \
      /opt/infernal-invocation/b12x/b12x \
      /opt/infernal-invocation/lmcache/lmcache
RUN /opt/venv/bin/python - <<'PY'
import importlib.metadata as metadata
import importlib.util
import json
import shutil
from pathlib import Path
names = ['vllm', 'b12x', 'lmcache']
origins = {n: str(importlib.util.find_spec(n).origin) for n in names}
for n, origin in origins.items():
    assert origin.startswith(f'/opt/infernal-invocation/{n}/'), (n, origin)
assert not list(Path('/dev').glob('nvidia*')), 'Build must not receive GPU devices'
receipt = {
    'kind': 'CPU-built editable Python overlay; inherited native extensions',
    'import_origins': origins,
    'distributions': {n: metadata.version(n) for n in ['vllm', 'b12x', 'lmcache', 'torch', 'triton', 'nvidia-cutlass-dsl', 'apache-tvm-ffi']},
    'tools': {n: shutil.which(n) for n in ['nvcc', 'compute-sanitizer', 'nsys', 'ncu', 'cmake', 'ninja', 'uv']},
    'gpu_devices': [],
    'gpu_equivalence': 'not tested',
    'native_rebuild': False,
}
Path('/opt/glmx/BUILD.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2))
PY
RUN nvcc --version && compute-sanitizer --version
# Inherit the exact original vLLM entrypoint. No server is started during build.
