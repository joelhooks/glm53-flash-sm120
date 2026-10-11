"""Generate the guarded persistent_topk extension from the image's vLLM source.

Run inside the image at build time:
  make-guard.py <vllm csrc/libtorch_stable dir> <out dir>

Writes persistent_topk_guard.cuh (every top-k output write bounded to
[0, TopK); a dropped write bumps a device counter and records the first one)
and glmx_topk.cu (topk.cu ported to ATen, registered as torch.ops.glmx).
This is a diagnostic guard and stopgap, not the fix.
"""
import os
import re
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

# Output-write sites by line: (expected text, site id, medium-path forensics).
SITES = {
    281: ('output_indices[above_base + above_rank] = elem_idx;', 1, False),
    312: ('output_indices[base + i] = bufs[0][i];', 2, False),
    376: ('output_indices[pos] = idx;', 3, False),
    395: ('output_indices[pos] = idx;', 4, False),
    399: ('if (slot > 0) output_indices[TopK - slot] = idx;', 5, False),
    491: ('output_indices[output_pos] = idx;', 11, True),
    509: ('output_indices[output_pos] = idx;', 12, True),
    552: ('output_indices[output_pos] = idx;', 13, True),
    572: ('output_indices[output_pos] = idx;', 14, True),
    577: ('output_indices[TopK - slot] = idx;', 15, True),
    836: ('row_output[pos] = static_cast<int32_t>(my_chunk_start + i);', 21, False),
    852: ('row_output[pos] = static_cast<int32_t>(my_chunk_start + i);', 22, False),
}
WRITE = re.compile(r'(output_indices|row_output)\[([^\]]+)\] = ([^;]+);')

GUARD = r'''
// ---- GLMX guard: bounded top-k output writes (diagnostic stopgap) ----
// glmx_guard_first: site, pos, value, block, thread, then (medium path only)
// seq_len, threshold_bin, remaining_k, output_count, buffered[0], buffered[1],
// final_k.
__device__ long long glmx_guard_drops;
__device__ long long glmx_guard_first[15];
__device__ int glmx_guard_flag;

__device__ __noinline__ void glmx_guard_record(
    int site, long long pos, long long val, long long a, long long b,
    long long c, long long d, long long e, long long f, long long g) {
  atomicAdd(reinterpret_cast<unsigned long long*>(&glmx_guard_drops), 1ULL);
  if (atomicCAS(&glmx_guard_flag, 0, 1) == 0) {
    long long v[12] = {site, pos, val, (long long)blockIdx.x,
                       (long long)threadIdx.x, a, b, c, d, e, f, g};
    for (int i = 0; i < 12; ++i) glmx_guard_first[i] = v[i];
    __threadfence();
  }
}

#define GLMX_STORE(arr, pos, val, site)                                  \
  do {                                                                   \
    const long long glmx_p = (long long)(pos);                           \
    if (glmx_p >= 0 && glmx_p < (long long)TopK) {                       \
      arr[glmx_p] = (val);                                               \
    } else {                                                             \
      glmx_guard_record(site, glmx_p, (long long)(val), 0, 0, 0, 0, 0, 0, \
                        0);                                              \
    }                                                                    \
  } while (0)

#define GLMX_STORE_M(arr, pos, val, site)                                \
  do {                                                                   \
    const long long glmx_p = (long long)(pos);                           \
    if (glmx_p >= 0 && glmx_p < (long long)TopK) {                       \
      arr[glmx_p] = (val);                                               \
    } else {                                                             \
      glmx_guard_record(site, glmx_p, (long long)(val), seq_len,         \
                        shared_threshold_bin, remaining_k,               \
                        shared_output_count, shared_buffered_count[0],   \
                        shared_buffered_count[1], shared_final_k);       \
    }                                                                    \
  } while (0)
'''

lines = (src / 'persistent_topk.cuh').read_text().split('\n')
for no, (text, site, medium) in SITES.items():
    line = lines[no - 1]
    assert line.strip() == text, (no, line)
    macro = 'GLMX_STORE_M' if medium else 'GLMX_STORE'
    lines[no - 1], n = WRITE.subn(
        lambda m: '%s(%s, %s, %s, %d);' % (macro, m.group(1), m.group(2), m.group(3), site), line)
    assert n == 1, (no, line)
cuh = '\n'.join(lines)
# GLMX_KEY_FIX=1: the fix candidate. NVVM folds `bits | 0x80000000u` into
# abs.f32/neg.f32 at some call sites (compute_100a/120a), which turns NaN into
# a different key than the integer path gives. Inline PTX keeps the OR integer.
# The same key appears in convert_to_uint32_v2, ToOrdered and the 4096 header.
KEY_FIX = os.environ.get('GLMX_KEY_FIX') == '1'
KEY = re.compile(r'return \(bits & 0x80000000u\) \? ~bits : \(bits \| 0x80000000u\);')


def key_fix(text):
    return KEY.subn(
        'uint32_t glmx_or; asm("or.b32 %0, %1, 0x80000000;" : "=r"(glmx_or) : "r"(bits)); '
        'return (bits & 0x80000000u) ? ~bits : glmx_or;', text)


hist4096 = (src / 'topk_histogram_4096.cuh').read_text()
if KEY_FIX:
    cuh, n1 = key_fix(cuh)
    hist4096, n2 = key_fix(hist4096)
    assert (n1, n2) == (2, 1), (n1, n2)
    print('key fix applied: %d in persistent_topk.cuh, %d in topk_histogram_4096.cuh' % (n1, n2))
anchor = 'namespace vllm {\nnamespace persistent {\n'
assert cuh.count(anchor) == 1
cuh = cuh.replace(anchor, GUARD + '\n' + anchor)
cuh = cuh.replace('PERSISTENT_TOPK_CUH_', 'GLMX_PERSISTENT_TOPK_GUARD_CUH_')
(out / 'persistent_topk_guard.cuh').write_text(cuh)
(out / 'topk_histogram_4096.cuh').write_text(hist4096)

cu = (src / 'topk.cu').read_text()
subs = [
    ('#include "torch_utils.h"',
     '#include <ATen/ATen.h>\n#include <ATen/cuda/CUDAContext.h>\n'
     '#include <c10/cuda/CUDAGuard.h>\n#include <torch/library.h>'),
    ('#include "persistent_topk.cuh"', '#include "persistent_topk_guard.cuh"'),
    ('torch::stable::Tensor', 'at::Tensor'),
    ('get_current_cuda_stream()', 'at::cuda::getCurrentCUDAStream().stream()'),
    ('get_device_prop()', 'at::cuda::getCurrentDeviceProperties()'),
    ('STD_TORCH_CHECK', 'TORCH_CHECK'),
    ('torch::headeronly::ScalarType::', 'at::ScalarType::'),
    ('void persistent_topk(', 'void glmx_persistent_topk('),
]
for old, new in subs:
    assert old in cu, old
    cu = cu.replace(old, new)
cu, n = re.subn(r'const torch::stable::accelerator::DeviceGuard device_guard\(\s*logits\.get_device_index\(\)\);',
                'const c10::cuda::CUDAGuard device_guard(logits.device());', cu)
assert n == 2, n
assert 'torch::stable' not in cu and 'torch::headeronly' not in cu
cu += r'''

// Copies [drops, first[0..14]] into a CUDA int64 tensor of >= 16 elements.
void glmx_guard_stats(at::Tensor& out) {
  TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::ScalarType::Long &&
                  out.is_contiguous() && out.numel() >= 16,
              "guard_stats needs a contiguous CUDA int64 tensor of >= 16");
  const c10::cuda::CUDAGuard device_guard(out.device());
  cudaStream_t s = at::cuda::getCurrentCUDAStream().stream();
  char* dst = static_cast<char*>(out.data_ptr());
  TORCH_CHECK(cudaMemcpyFromSymbolAsync(dst, glmx_guard_drops, 8, 0,
                                        cudaMemcpyDeviceToDevice, s) == cudaSuccess);
  TORCH_CHECK(cudaMemcpyFromSymbolAsync(dst + 8, glmx_guard_first, 15 * 8, 0,
                                        cudaMemcpyDeviceToDevice, s) == cudaSuccess);
}

TORCH_LIBRARY(glmx, m) {
  m.def("persistent_topk(Tensor logits, Tensor lengths, Tensor! output, "
        "Tensor workspace, int k, int max_seq_len) -> ()");
  m.def("guard_stats(Tensor! out) -> ()");
}

TORCH_LIBRARY_IMPL(glmx, CUDA, m) {
  m.impl("persistent_topk", &glmx_persistent_topk);
  m.impl("guard_stats", &glmx_guard_stats);
}
'''
(out / 'glmx_topk.cu').write_text(cu)
print('guard sources written to', out)
