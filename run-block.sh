#!/usr/bin/env bash
# Run a block of repro attempts serially inside the owner's GPU lease.
#   run-block.sh <plan-file>
# Each plan line: IMAGE VARIANT SESSIONS ATTEMPT [WAVES]   (# starts a comment)
# Stops the whole block on an uncleared STOP_GPU_WORK (any Xid outside the
# owner stop rule, or a missing GPU).
set -uo pipefail
: "${LEASE_START:?owner lease required}" "${LEASE_END:?owner lease required}"
root=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=/dev/null
[[ -f $root/lab.env ]] && source "$root/lab.env"
plan=${1:?plan file required}
runs=${GLMX_ROOT:-/data/bench/glm-xid31}/runs
export TMPDIR=${GLMX_SCRATCH:-${XDG_CACHE_HOME:-$HOME/.cache}/glm53-flash-sm120}
mkdir -p "$runs" "$TMPDIR"
while read -r image variant sessions attempt waves; do
  [[ -z ${image:-} || $image == \#* ]] && continue
  echo "== $(date -u +%FT%TZ) $image $variant c$sessions a$attempt w${waves:-1}"
  IMAGE=$image VARIANT=$variant SESSIONS=$sessions ATTEMPT=$attempt WAVES=${waves:-1} bash "$root/run-repro.sh"
  rc=$?
  echo "exit=$rc"
  # 2 is a guard refusal (lease, drain, STOP marker); 126/127 cannot run.
  if (( rc == 2 || rc >= 126 )); then echo 'Attempt refused or not runnable; block halted'; exit 3; fi
  if find "$runs" -maxdepth 2 -name STOP_GPU_WORK | while read -r f; do [[ -e $f.owner-cleared ]] || echo "$f"; done | grep -q .; then
    echo 'STOP_GPU_WORK recorded; block halted'; exit 3
  fi
  if [[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -ne 2 ]]; then
    echo 'GPU count is not 2; block halted'; exit 3
  fi
done <"$plan"
echo "BLOCK_DONE $(date -u +%FT%TZ)"
