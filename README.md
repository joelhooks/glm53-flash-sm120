# glm53-flash-sm120

Status: root cause found and fixed in this recipe.

A debugging kit and fix for one engine crash: GLM-5.3-Flash (EXL3) on two Blackwell SM120 GPUs (RTX PRO 6000), with tensor, expert and decode-context parallelism, in the `verdictai/glm53-flash-exl3-k4` image.

## Symptoms, cause, fix

### Symptoms

- The driver logs `NVRM: Xid 31` (GPU MMU fault, `FAULT_PDE`), sometimes `Xid 43` or `Xid 13`.
- The engine then dies with `CUDA error: an illegal memory access was encountered`, a cuBLAS internal error, or a device-side assert.
- It needs all of these together:
  - DCP=2;
  - MTP speculative decoding;
  - prefix caching on;
  - two or more concurrent requests;
  - prompts past about 109K tokens (`--max-model-len 204800`).

### Cause

- The image's vLLM fork is based before upstream vLLM PR #50493. So model runner v2 divides the block-table width of the Mamba-style linear-attention (KDA) state by the DCP size: 17 columns instead of the 30 needed. That state isn't sharded across DCP ranks, so it must not be divided.
- With prefix caching, the state slot is chosen by token position. Past about 109K tokens the selection reads past the end of the row.
- A verify step that accepts all of its draft tokens then reads slot 0, the null block.
- The KDA recurrent kernel returns early on slot 0 without writing its output, so the next layers read garbage, and it turns into NaN.
- NaN logits make the `persistent_topk` kernel write thousands of entries past its output (see the top-k section below). The corrupted memory ends in Xid 31.

### Fix

- Apply `fix/origin-fix.patch`, one line in `vllm/v1/worker/gpu/model_runner.py`: don't DCP-scale the Mamba group's block-table width.
- Or rebase the fork past vLLM PR #50493, which fixed the same line upstream (first released in v0.29.0).
- The recipe's serving image adds a bounded, NaN-safe `persistent_topk` as a backstop.

### Proof

- With the one-line fix alone, on the production config (DCP=2, prefix caching, MTP3, `--max-model-len 204800`): 16 of 16 waves clean, with two concurrent sessions at 122,880 prompt tokens each. That's 0 Xid, 0 failed requests, and no CUDA errors in the server logs.
- The fixed serving image (`fix/prod/`, origin fix plus the top-k backstop), on the same config and load: 16 of 16 waves clean, 0 Xid, 0 failed requests.
- Without the fix, every multi-wave engine we ran failed within 2–7 waves on the same load.

## The cause in detail

1. **The block table is too narrow.** The fork's model runner v2 (`vllm/v1/worker/gpu/model_runner.py`, vLLM base `6dc2f516`) divides every KV cache group's block-table width by the DCP size. Mamba-style state is replicated across DCP ranks, not sharded. With prefix caching, it runs in `align` mode, where its blocks are indexed by global token position. So with DCP=2, block size 7,808 and `--max-model-len 204800`, a row is `cdiv(204800, 2 × 7808) + 3 = 17` wide, but positions near 204,800 need 30. Upstream vLLM releases v0.24.0 through v0.28.0 ship the same scaled width; v0.29.0 is the first with PR #50493.
2. **The state selection reads past the row.** The align-mode selection kernel reads columns `start .. start + 3`, with `start = (seq_len - 1) // 7808`. It masks only by the number of state blocks, not by the row width. Past 109,312 tokens it reads the next row's entries.
3. **A verifying request gets slot 0**, the null block, or another request's state slot, with 4 tokens accepted.
4. **The recurrent kernel skips it.** `fused_recurrent` returns early when the state index is 0, without writing output or state. The request's output is whatever the buffer held.
5. **That garbage becomes NaN** in the hidden states by the next sparse-attention (DSA) layer, along with the indexer's inputs.
6. **Top-k writes out of bounds**, as described in the next section.
7. **Xid 31.** The overrun corrupts neighbouring allocations, and a later kernel faults.

The probes in `probes/` do nothing unless `GLMX_KDA_PROBE=1`. They are how we found links 1–5.

## A separate bug: persistent_topk overruns on NaN

This bug is in upstream vLLM, not the fork. NaN logits trigger it.

- `persistent_topk` turns each fp32 logit into a sortable key with `bits | 0x80000000u` and similar.
- On Blackwell targets (`compute_100a`, `compute_120a`, not 80 or 90), NVVM folds that integer operation into float `abs; neg` at one call site but not at another. `abs.f32` and `neg.f32` on NaN give an unspecified NaN, so the histogram pass and the buffering pass put the same NaN in different bins.
- A stale buffered count is never reset, and the write loop has no bound. One row of NaN logits writes up to 7,680 entries past `k`.
- The official vLLM 0.31.0 wheel (CUDA 13.0, sm_120) reproduces this on its own with a single-kernel test.
- We document it here; it is not filed upstream.
- The recipe's backstop (`guard/` with `GLMX_KEY_FIX=1`):
  - computes the key with inline PTX `or.b32`, so it can't be folded;
  - bounds every output write to `[0, k)`.
- With the origin fix, NaN no longer reaches this kernel. The backstop covers any other path that produces NaN logits.

## Tuning knobs and workarounds

For a build without the fix. Each change removes one trigger condition.

| Change | Effect | Cost | Tested |
| --- | --- | --- | --- |
| `--no-enable-prefix-caching` | Mamba cache mode becomes `none`, with a 4-wide table not chosen by position | Every request re-prefills | One run, 8 of 8 waves clean, no NaN and no slot-0 reads |
| `--decode-context-parallel-size 1` | The width is no longer divided | Less KV capacity per rank | Untested |
| `--max-model-len` at most twice the longest context you serve | Keeps `start + 3` inside the halved row | Shorter context | Untested; inferred from the formula |

These are not workarounds:

- Turning async scheduling off: the crash still happened, 3 of 3 engines within 3 waves.
- Changing `--kv-cache-dtype`: `nvfp4_ds_mla` is the only valid value for this model on this backend.

The other serving flags are unchanged by the fix.

## Build recipe

### 1. The source build

The verified path is an overlay on the published image. `Dockerfile` starts from the pinned image and copies the three Python trees over the originals. It installs b12x editable with the image's own pip, offline. The compiled vLLM extensions and the ExLlamaV3 extension come from the base image unchanged. The build runs with `--network none` and no GPU:

```sh
docker pull verdictai/glm53-flash-exl3-k4@sha256:0f1cdcc8891f1cc3a444121eb61d366289a1cbba285f0892dcbb24bc94961692
docker build --network none -f Dockerfile -t glm53-flash-sm120:local /path/to/new-dir
```

### Building from NVIDIA's container

Documented, not yet built by us. The image labels give this chain:

1. `nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c`
2. NCCL 2.31.2 from `local-inference-lab/nccl-canonical@fb6f4099`
3. PyTorch 2.13.0 built from `pytorch/pytorch@cf30153c` for `TORCH_CUDA_ARCH_LIST=12.0a`, then torchvision 0.28.0 and xgrammar 0.2.5
4. FlashInfer `1ac69427` wheels
5. vLLM, b12x and LMCache at the pins above, with the integration patches
6. The GLM-5.3 layer: ExLlamaV3 extension, calibration scales and the DFlash2 path

The original recipe is `local-inference-lab/blackwell-llm-docker@54a4e2da`. It has no licence file, so we link to it rather than copy it.

Using NVIDIA's container means accepting NVIDIA's licence terms for it. This repository publishes no image, for that reason.

### 2. The fixed serving image

Tag the source build as `glm53-flash-sm120:source`, then:

```sh
docker build --build-arg GLMX_KEY_FIX=1 -t glm53-flash-sm120:topk-nankey guard/
docker build -t glm53-flash-sm120:glm-fixed fix/prod/
```

- `guard/` rebuilds `persistent_topk` from the same source with every output write bounded to `[0, k)` and, with `GLMX_KEY_FIX=1`, a NaN-safe sort key (see the top-k section above).
- `fix/prod/` applies `fix/origin-fix.patch` and routes both top-k call sites to that build, always on. No probes and no debug env.
- The image keeps the base image's entrypoint (`vllm`) and needs no new flags or env. Serve it with the same command as the published image.

For the one-line fix alone, `fix/Dockerfile` applies only `fix/origin-fix.patch` on top of the source build.

## Affected build

- Image: `verdictai/glm53-flash-exl3-k4@sha256:0f1cdcc8891f1cc3a444121eb61d366289a1cbba285f0892dcbb24bc94961692`
- Release label: `r19-sm120-tp2-ep2-dcp2-v84-dflash2` (see `upstream/PROVENANCE.json`)
- Serving flags:
  - `--tensor-parallel-size 2 --enable-expert-parallel`
  - `--decode-context-parallel-size 2 --dcp-comm-backend a2a`
  - `--moe-backend b12x --attention-backend B12X_MLA_SPARSE --kv-cache-dtype nvfp4_ds_mla`
  - MTP with 3 speculative tokens
  - `--max-num-batched-tokens 2048 --max-num-seqs 4`, chunked prefill and prefix caching on

`make-compose.py` writes the full recipe and every one-factor variant we test.

## Source pins

The image ships its Python source as git checkouts with uncommitted changes on top. Each commit below can be fetched from its public repository.

| Tree | Repository | Commit | Changes in the image |
| --- | --- | --- | --- |
| vLLM | `local-inference-lab/vllm` | `6dc2f516688fe6f84c6994dcd20fddf296853a6c` | 157 tracked files plus 17 new files |
| b12x | `local-inference-lab/b12x` | `36bce2c1552ba2d47dc09f20a6f64fbfc8ec4ff8` | 29 tracked files plus 1 new file |
| LMCache | `local-inference-lab/LMCache` | `a128b2e286ebb3556cb43124149e600ff99fe481` | none |
| ExLlamaV3 extension | `brandonmmusic-max/exllamav3` | `704aefd743b390af4bd0fb429d1906f9b964c7d8` | compiled `.so` only; not part of this kit |

`upstream/upstream-image-labels.json` records the rest of the chain: the NVIDIA base, NCCL, PyTorch, FlashInfer and others.

## Contents

| Path | What it is |
| --- | --- |
| `image-patches/<tree>.head.patch` | The image's changes to tracked files (`git diff HEAD`) |
| `image-patches/<tree>.untracked.patch` | Files the image adds that git doesn't track |
| `image-patches/<tree>/`, `image-patches/<tree>-untracked/` | The same changes, one patch per file, with `INDEX.tsv` |
| `extract-source.sh` | Copies the trees out of a stopped container of the image |
| `fetch-source.sh` | Rebuilds the same trees from the public pins plus the patches |
| `split-patches.py` | Splits a combined patch into per-file patches |
| `Dockerfile`, `build-source.sh` | Builds a local image from the trees |
| `make-compose.py` | Writes compose files for the recipe and each variant |
| `run-repro.sh`, `run-block.sh`, `plans/`, `bench-lib.sh` | One attempt per fresh engine (one or more waves); serial blocks of attempts; telemetry and thermal stop guard |
| `repro-client.py` | Standalone client: N sessions at a token target, warmed one at a time, then one concurrent wave |
| `fix/origin-fix.patch`, `fix/Dockerfile`, `fix/prod/` | The one-line fix, and the fixed serving image |
| `guard/` | `persistent_topk` rebuilt with bounded writes and an optional NaN-safe key |
| `probes/` | Debug probes as patches over our source build, with `probes/Dockerfile`; inert unless `GLMX_KDA_PROBE=1` |
| `attribute-patches.py` | Adds the attribution header to every image patch (idempotent) |
| `licenses/` | Verbatim upstream licence texts: vLLM, b12x, LMCache (Apache-2.0), ExLlamaV3 (MIT, reference only) |
| `PUBLISH.txt`, `export-public.sh`, `redaction-scan.sh`, `redaction-allow.txt` | The public file list, the export that copies exactly that list, and the scan run on it before publishing |

`minimal-repro.py` is our lab client. It needs a private padding corpus and request template that aren't published here. `repro-client.py` is the public substitute: same target, same concurrency, different prompt text.

### Authorship

The patches are not our work. We extracted them from the image above. `upstream/PROVENANCE.json` names Brandon M. Music as the release author, and the trees are forks maintained by `local-inference-lab`. Every patch file starts with a comment header giving its source image, release author, upstream repository and commit, and licence. `git apply` ignores that header, and we checked that the attributed patches still reproduce the image's files byte for byte.

We wrote the scripts, the Dockerfile, the probes and this README.

## Rebuilding the source

```sh
./fetch-source.sh /path/to/new-dir   # public pins + patches
# or, with the image pulled locally:
./extract-source.sh /path/to/new-dir
```

We checked both paths against each other.

- `fetch-source.sh` gives 6,687 vLLM files, 686 b12x files and 2,145 LMCache files. Each one is byte-identical to the image's copy.
- The image also has files the fetch doesn't produce, and all of them are git-ignored build output: compiled extensions, generated kernels, vendored `vllm_flash_attn/cute` and `third_party/triton_kernels`, and `_version.py`.
- The `Dockerfile` copies the trees over the base image without deleting anything, so those build files come from the base image on either path.

## Running the repro

`run-repro.sh` starts one fresh engine bound to `127.0.0.1` and waits for it to report healthy. It then runs the client and stops the engine. Finally it compares the kernel's `NVRM: Xid` lines from before and after the attempt. The comparison ignores the timestamp, because `dmesg` can print the same event with a timestamp a microsecond different between reads.

The stop rule:

- Xid 31 is expected.
- Xid 43 is accepted only when the engine logged a device-side assert or CUDA error within 10 s of it.
- Anything else, or a missing GPU, writes `STOP_GPU_WORK`, and both runners refuse to continue until a `STOP_GPU_WORK.owner-cleared` record is written next to it.

Each attempt's outcome is one of: `xid31`, `xid43_assert`, `other_xid_STOP`, `clean`, `client_failure_no_xid`, `startup_failure`.

```sh
python3 repro-client.py --model glm-5.3-flash --padding any-large-text.txt \
  --output runs/first --target 122880 --sessions 2
```

The padding file needs about 4 characters per target token, so about 500 KB for 122,880 tokens.

The scripts assume a host with two GPUs, the model weights in a local Hugging Face cache, and Docker reached through `sudo -n`. Host settings are environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `GLMX_BENCH_LABEL` | none (required) | Label key on every engine container; the drain check and thermal stop guard use it |
| `GLMX_ROOT` | `/data/bench/glm-xid31` | Runs, caches and source trees |
| `GLMX_SCRATCH` | `~/.cache/glm53-flash-sm120` | `TMPDIR` for the scripts |
| `GLMX_HF_CACHE` | `/data/models/hf` | Host Hugging Face cache, mounted read-only |
| `GLMX_MODEL_PATH` | stock `brandonmusic/GLM-5.3-Flash-tr3-4bpw@a5fee929` | Snapshot path inside the container |
| `GLMX_SERVED_NAME` | `glm-5.3-flash` | Served model name |
| `GLMX_TAG` | `glm53-flash-sm120:source` | Local image from `build-source.sh` |
| `GLMX_PADDING` | none | Padding text for `repro-client.py` |

`run-repro.sh` sources `lab.env` from this directory if one exists. Ours is not published.

```sh
GLMX_BENCH_LABEL=glm53.bench GLMX_PADDING=padding.txt \
LEASE_START=2026-01-01T00:00:00Z LEASE_END=2026-01-01T06:00:00Z \
  ./run-block.sh plans/block1b.txt
```

A plan line is `IMAGE VARIANT SESSIONS ATTEMPT [WAVES]`. Every attempt starts a fresh engine. Each wave on that engine uses fresh sessions: it warms each one alone, then sends them all at once.

## Components and licences

| Component | What this repo contains | Licence |
| --- | --- | --- |
| This project: scripts, Dockerfiles, probes, README | Our own files | Apache-2.0 (`LICENSE`) |
| vLLM fork, `local-inference-lab/vllm@6dc2f516` | Patches only, each with an attribution header | Apache-2.0 (`licenses/vllm.LICENSE`, verbatim) |
| b12x, `local-inference-lab/b12x@36bce2c1` | Patches only, each with an attribution header | Apache-2.0 (`licenses/b12x.LICENSE`, verbatim) |
| LMCache, `local-inference-lab/LMCache@a128b2e2` | Pin only; the image doesn't change it | Apache-2.0 (`licenses/lmcache.LICENSE`, verbatim) |
| ExLlamaV3 fork, `brandonmmusic-max/exllamav3@704aefd7` | Pinned reference only, no code | MIT (`licenses/exllamav3.LICENSE`, verbatim from that commit) |
| Image build recipe, `local-inference-lab/blackwell-llm-docker@54a4e2da` | Linked, not copied | No licence file in the repository |
| NVIDIA PyTorch container and other NVIDIA components | Nothing; referenced by digest only | NVIDIA's terms |
| Model weights, DFlash2 draft weights | Nothing | Their own licences |

This repository publishes no container image yet. A pullable image is planned that doesn't use the NVIDIA PyTorch container. It will start from PyTorch's builder image, ship on NVIDIA's CUDA base image, and use the official PyTorch and CUDA pip wheels. Its README will list the redistribution terms for every NVIDIA component it contains.
