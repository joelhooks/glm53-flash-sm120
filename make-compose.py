"""Write one compose file: the base recipe plus one named variant.

CPU only. Every variant states the flags it changes, so a run's resolved compose
shows exactly one change (or a declared paired change) from baseline.

Host-specific values come from the environment:
  GLMX_ROOT         run and cache root (default /data/bench/glm-xid31)
  GLMX_HF_CACHE     host Hugging Face cache mounted at /models/hf (default /data/models/hf)
  GLMX_MODEL_PATH   model snapshot path inside the container (default: stock tr3 4bpw)
  GLMX_SERVED_NAME  --served-model-name (default glm-5.3-flash)
  GLMX_TAG          local image built by build-source.sh (default glm53-flash-sm120:source)
  GLMX_BENCH_LABEL  label key that marks bench containers (required)
  GLMX_VARIANT_LABEL label key for the variant name (default glmx.variant)
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT/'compose-base.json'
STOCK = '/models/hf/hub/models--brandonmusic--GLM-5.3-Flash-tr3-4bpw/snapshots/a5fee929cf4888b1824323e33e8a19b60129e025'
IMAGES = {
    'upstream': 'verdictai/glm53-flash-exl3-k4@sha256:0f1cdcc8891f1cc3a444121eb61d366289a1cbba285f0892dcbb24bc94961692',
    'source': os.environ.get('GLMX_TAG', 'glm53-flash-sm120:source'),
    # probes/Dockerfile: source plus inert-by-default debug probes.
    'probe': os.environ.get('GLMX_PROBE_TAG', 'glm53-flash-sm120:probe'),
    # guard/Dockerfile: source plus a guarded persistent_topk (inert unless GLMX_TOPK_GUARD=1).
    'guard': os.environ.get('GLMX_GUARD_TAG', 'glm53-flash-sm120:topk-guard'),
    # guard/Dockerfile with GLMX_KEY_FIX=1: the same guard plus a NaN-safe fp32
    # key (the fix candidate); with variant guard, drops should stay at 0.
    'nankey': os.environ.get('GLMX_NANKEY_TAG', 'glm53-flash-sm120:topk-nankey'),
    # fix/Dockerfile: source plus the origin fix only (Mamba block-table width
    # not DCP-scaled). No probes, no top-k guard, no NaN-safe key.
    'origin-fix': os.environ.get('GLMX_ORIGIN_FIX_TAG', 'glm53-flash-sm120:origin-fix'),
    # probes/Dockerfile with kda-probe12-fix.patch: probe 12 plus the origin fix.
    'probe-fix': os.environ.get('GLMX_PROBE_FIX_TAG', 'glm53-flash-sm120:probe-fix'),
    # fix/prod/Dockerfile: production image. Origin fix plus the nankey top-k build
    # (NaN-safe key, bounded writes) always on. No probes, no guard env, no debug env.
    'glm-fixed': os.environ.get('GLMX_GLM_FIXED_TAG', 'glm53-flash-sm120:glm-fixed'),
}
# Layer-3 suspects: DSA indexer K pool, sparse_utils, b12x sparse MLA, top-k ops.
MEMCHECK_REGEX = 'topk|kpool|expand_pools|gather_topk|global_index|ckv|page_table|prepare_uniform|mla|indexer'
SANITIZER = ['/usr/local/cuda/bin/compute-sanitizer', '--target-processes', 'all',
             '--print-limit', '200', '--log-file', '/cache/sanitizer/%p.log']


def set_arg(cmd, flag, value=None):
    if flag in cmd:
        i = cmd.index(flag)
        if value is None:
            return
        cmd[i+1] = value
    else:
        cmd.extend([flag] if value is None else [flag, value])


def drop_arg(cmd, flag, takes_value=True):
    if flag not in cmd:
        raise ValueError(f'{flag} not in baseline recipe')
    i = cmd.index(flag)
    del cmd[i:i+(2 if takes_value else 1)]


def spec_tokens(cmd, n):
    i = cmd.index('--speculative-config')
    value = json.loads(cmd[i+1])
    value['num_speculative_tokens'] = n
    cmd[i+1] = json.dumps(value, separators=(',', ':'))


# name: (description, function(cmd, env) -> None, paired change note)
VARIANTS = {
    'baseline': ('Unchanged base recipe', lambda c, e: None, None),
    'debug': ('CUDA_LAUNCH_BLOCKING=1 and vLLM DEBUG logging',
              lambda c, e: e.update(CUDA_LAUNCH_BLOCKING='1', VLLM_LOGGING_LEVEL='DEBUG'), None),
    'memcheck': ('compute-sanitizer memcheck around the engine, launch blocking, eager',
                 None, 'Sanitizer needs eager mode; CUDA graphs hide individual launches'),
    'dcp1': ('Decode context parallel 2 -> 1',
             lambda c, e: (drop_arg(c, '--decode-context-parallel-size'), drop_arg(c, '--dcp-comm-backend'), e.pop('VLLM_USE_B12X_DCP_A2A')),
             'Paired: the a2a backend flag and its env only exist with DCP>1'),
    'mbt1024': ('--max-num-batched-tokens 2048 -> 1024', lambda c, e: set_arg(c, '--max-num-batched-tokens', '1024'), None),
    'mbt8192': ('--max-num-batched-tokens 2048 -> 8192', lambda c, e: set_arg(c, '--max-num-batched-tokens', '8192'), None),
    'no-chunked': ('Chunked prefill off', lambda c, e: (drop_arg(c, '--enable-chunked-prefill', False), set_arg(c, '--no-enable-chunked-prefill')),
                   'Unchunked prefill needs a batch budget >= the prompt; engine may refuse at 2048'),
    'no-mtp': ('MTP speculative decoding off', lambda c, e: drop_arg(c, '--speculative-config'), None),
    'mtp1': ('MTP 3 -> 1 speculative token', lambda c, e: spec_tokens(c, 1), None),
    'no-ep': ('Expert parallel off (TP2 MoE)', lambda c, e: drop_arg(c, '--enable-expert-parallel', False), None),
    'eager': ('CUDA graphs off (--enforce-eager)', lambda c, e: set_arg(c, '--enforce-eager'), None),
    'no-prefix-cache': ('Prefix caching off', lambda c, e: (drop_arg(c, '--enable-prefix-caching', False), set_arg(c, '--no-enable-prefix-caching')), None),
    'seqs2': ('--max-num-seqs 4 -> 2', lambda c, e: set_arg(c, '--max-num-seqs', '2'), None),
    'block128': ('KV manager block 128 (kernel block stays 64)', lambda c, e: set_arg(c, '--block-size', '128'),
                 'Backend kernel block is fixed at 64; engine may refuse'),
    'probe': ('GLMX_KDA_PROBE=1: check KDA prefill metadata against tensor sizes (probe image only)',
              lambda c, e: e.update(GLMX_KDA_PROBE='1'), 'Diagnostic; syncs the device once per KDA prefill call'),
    # Async scheduling defaults on for MTP (vllm/config/vllm.py, EagleModelTypes).
    'no-async': ('Async scheduling off (--no-async-scheduling)',
                 lambda c, e: set_arg(c, '--no-async-scheduling'), None),
    'memcheck-k': ('memcheck filtered to kernels matching GLMX_MEMCHECK_REGEX; CUDA graphs kept unless GLMX_MEMCHECK_NOCACHE=1',
                   None, 'Sanitizer entrypoint; GLMX_MEMCHECK_NOCACHE=1 adds PYTORCH_NO_CUDA_MEMORY_CACHING=1 and --enforce-eager'),
    'guard': ('GLMX_TOPK_GUARD=1: bounded persistent_topk output writes, dropped-write counter (guard image only)',
              lambda c, e: e.update(GLMX_TOPK_GUARD='1'), 'Diagnostic stopgap, not the fix'),
    'probe-no-async': ('Probe plus async scheduling off',
                       lambda c, e: (e.update(GLMX_KDA_PROBE='1'), set_arg(c, '--no-async-scheduling')),
                       'Paired diagnostic: probe image and env, with --no-async-scheduling'),
    # Prefix caching forces mamba cache mode 'align' (models/config.py), where the
    # spec state slots are block-table columns; off, the mode is 'none'.
    'probe-no-prefix-cache': ('Probe plus prefix caching off (mamba cache mode none)',
                              lambda c, e: (e.update(GLMX_KDA_PROBE='1'),
                                            drop_arg(c, '--enable-prefix-caching', False),
                                            set_arg(c, '--no-enable-prefix-caching')),
                              'Paired diagnostic: causal test of the align-mode state slots'),
    'probe-dcp1': ('Probe plus decode context parallel 2 -> 1',
                   lambda c, e: (e.update(GLMX_KDA_PROBE='1'), drop_arg(c, '--decode-context-parallel-size'),
                                 drop_arg(c, '--dcp-comm-backend'), e.pop('VLLM_USE_B12X_DCP_A2A')),
                   'Paired diagnostic: tests rank-local seq_lens against the mamba block table (W)'),
}


def build(variant, image, port):
    label = os.environ.get('GLMX_BENCH_LABEL')
    if not label:
        raise SystemExit('GLMX_BENCH_LABEL is required: the label key that marks bench containers')
    root = os.environ.get('GLMX_ROOT', '/data/bench/glm-xid31')
    fill = {'${MODEL_PATH}': os.environ.get('GLMX_MODEL_PATH', STOCK),
            '${SERVED_NAME}': os.environ.get('GLMX_SERVED_NAME', 'glm-5.3-flash'),
            '${HF_CACHE}': os.environ.get('GLMX_HF_CACHE', '/data/models/hf'),
            '${RUN_CACHE}': f'{root}/cache/{variant}'}
    compose = json.loads(BASE.read_text())
    server = compose['services']['server']
    server['command'] = [fill.get(c, c) for c in server['command']]
    server['volumes'] = [v.replace('${HF_CACHE}', fill['${HF_CACHE}']).replace('${RUN_CACHE}', fill['${RUN_CACHE}'])
                         for v in server['volumes']]
    cmd, env = server['command'], server['environment']
    description, change, paired = VARIANTS[variant]
    if variant == 'memcheck':
        set_arg(cmd, '--enforce-eager')
        env.update(CUDA_LAUNCH_BLOCKING='1', VLLM_LOGGING_LEVEL='DEBUG')
        server['entrypoint'] = SANITIZER+['--tool', 'memcheck', '/opt/venv/bin/vllm']
    elif variant == 'memcheck-k':
        # memcheck bounds are per CUDA allocation, so with caching on it reports
        # writes outside the allocator's segments (the Xid 31 kind) but not into
        # a neighbouring tensor. GLMX_MEMCHECK_NOCACHE=1 gives every tensor its
        # own allocation; raw cudaMalloc can't run under graph capture, so eager.
        if os.environ.get('GLMX_MEMCHECK_NOCACHE') == '1':
            env.update(PYTORCH_NO_CUDA_MEMORY_CACHING='1')
            set_arg(cmd, '--enforce-eager')
        # The sanitizer takes ~2 GiB per GPU, so the default 0.986 fails startup.
        set_arg(cmd, '--gpu-memory-utilization', os.environ.get('GLMX_MEMCHECK_GPU_UTIL', '0.95'))
        server['entrypoint'] = SANITIZER+['--tool', 'memcheck', '--padding', '4096',
                                          '--kernel-name', 'regex='+os.environ.get('GLMX_MEMCHECK_REGEX', MEMCHECK_REGEX),
                                          '/opt/venv/bin/vllm']
    else:
        change(cmd, env)
    slug = 'glmx-'+variant
    compose['name'] = slug
    server['image'] = IMAGES[image]
    server['container_name'] = 'bench-'+slug
    server['labels'] = {label: '1', os.environ.get('GLMX_VARIANT_LABEL', 'glmx.variant'): variant}
    server['ports'] = [f'127.0.0.1:{port}:8000']
    return compose, {'variant': variant, 'image': IMAGES[image], 'description': description, 'paired_change': paired,
                     'base': BASE.name, 'model_path': fill['${MODEL_PATH}'], 'served_name': fill['${SERVED_NAME}']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', choices=sorted(VARIANTS))
    parser.add_argument('--image', choices=sorted(IMAGES), default='source')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    compose, meta = build(args.variant, args.image, args.port)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'compose.json').write_text(json.dumps(compose, indent=2)+'\n')
    (args.out/'variant.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(meta))
