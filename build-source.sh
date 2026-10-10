#!/usr/bin/env bash
# Build only. No GPU devices, server startup, lease scripts or network fetches.
set -euo pipefail
export TMPDIR=${GLMX_SCRATCH:-${XDG_CACHE_HOME:-$HOME/.cache}/glm53-flash-sm120}
root_dir=${GLMX_ROOT:-/data/bench/glm-xid31}
source_dir=${1:-$root_dir/source}
[[ $source_dir == "$root_dir"/* ]] || { echo "Source must be under $root_dir/ (set GLMX_ROOT)"; exit 2; }
grep -qx SOURCE_EXTRACTED "$source_dir/receipts/COMPLETE"
root=$(cd "$(dirname "$0")" && pwd)
tag=${GLMX_TAG:-glm53-flash-sm120:source}
# Docker context ignores Git storage and old object/build intermediates, not
# source files or the packaged .so files in the vllm package itself.
printf '**/.git\n**/__pycache__\n**/build\n**/*.egg-info\nreceipts/*.bundle\n' >"$source_dir/.dockerignore"
sudo -n /usr/bin/docker build --network none --pull=false \
  -f "$root/Dockerfile" -t "$tag" "$source_dir"
sudo -n /usr/bin/docker image inspect "$tag" \
  --format '{{.Id}} {{.Size}} {{json .Config.Entrypoint}}' >"$source_dir/receipts/local-image.txt"
printf 'SOURCE_BUILT\n' >"$source_dir/receipts/BUILD_COMPLETE"
