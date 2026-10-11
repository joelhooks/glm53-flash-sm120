#!/usr/bin/env bash
# One fresh-engine repro attempt inside the owner's GPU lease. Never runs
# gpu-claim/gpu-release and never contacts llama-swap.
#   LEASE_START LEASE_END  owner lease (date strings)
#   VARIANT                make-compose.py variant (default baseline)
#   IMAGE                  source | upstream (default source)
#   SESSIONS               concurrent wave size (default 2; 1 = serial control)
#   ATTEMPT                attempt number for the output label (default 1)
#   WAVES                  fresh-salt warm and wave cycles on one engine (default 1)
# Host settings come from lab.env when present (see make-compose.py for the
# GLMX_* variables). GLMX_CLIENT=lab uses minimal-repro.py with our private
# corpus; the default public client is repro-client.py with GLMX_PADDING.
set -euo pipefail
: "${LEASE_START:?owner lease required}" "${LEASE_END:?owner lease required}"
variant=${VARIANT:-baseline} image=${IMAGE:-source} sessions=${SESSIONS:-2} attempt=${ATTEMPT:-1} waves=${WAVES:-1}
[[ $variant =~ ^[a-z0-9-]+$ && $attempt =~ ^[1-9][0-9]*$ && $sessions =~ ^[1-4]$ && $waves =~ ^[1-9][0-9]?$ ]] || exit 2
root=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=/dev/null
[[ -f $root/lab.env ]] && source "$root/lab.env"
: "${GLMX_BENCH_LABEL:?set GLMX_BENCH_LABEL, the label key the bench stop guard uses}"
glmx_root=${GLMX_ROOT:-/data/bench/glm-xid31}
client=${GLMX_CLIENT:-public}
served=${GLMX_SERVED_NAME:-glm-5.3-flash}
if [[ $client == public ]]; then : "${GLMX_PADDING:?public client needs GLMX_PADDING, a large text file}"; fi
start=$(date -d "$LEASE_START" +%s) end=$(date -d "$LEASE_END" +%s) now=$(date +%s)
if (( now < start || now >= end-900 || end <= start )); then
  echo 'Outside the owner lease, or under 15 minutes left'; exit 2
fi
export TMPDIR=${GLMX_SCRATCH:-${XDG_CACHE_HOME:-$HOME/.cache}/glm53-flash-sm120} PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR"
docker() { sudo -n /usr/bin/docker "$@"; }
# Prefer the host's own bench library (its stop guard and telemetry summary).
host_lib=$(git -C "$root" rev-parse --show-toplevel 2>/dev/null)/stack/bench/lib.sh
if [[ -f $host_lib ]]; then source "$host_lib"; else source "$root/bench-lib.sh"; fi
out=$glmx_root/runs/$(date -u +%Y%m%dT%H%M%SZ)-$variant-$image-c$sessions-a$attempt-w$waves
[[ ! -e $out ]] || { echo 'Refusing to overwrite a run'; exit 2; }
mkdir -p "$out"
RUN_LOG=$out/run.log
python3 "$root/make-compose.py" "$variant" --image "$image" --out "$out" >>"$RUN_LOG"
compose=(docker compose -f "$out/compose.json")
active=0 telemetry=0
xid_lines() { sudo -n dmesg --time-format iso 2>/dev/null | grep -E 'NVRM: Xid' || true; }
cleanup() {
  trap - EXIT TERM INT
  if (( active )); then
    "${compose[@]}" logs --no-color --timestamps >"$out/server.log" 2>&1 || true
    "${compose[@]}" down --timeout 10 || true
    mapfile -t left < <(docker ps -q --filter "label=$GLMX_BENCH_LABEL")
    if (( ${#left[@]} )); then docker stop --time 10 "${left[@]}" || true; fi
  fi
  (( telemetry )) && telemetry_stop "$out"
  xid_lines >"$out/xid-after.txt"
  # New kernel Xid lines since this attempt began, then the outcome class.
  # dmesg's ISO timestamps for one event can differ by a microsecond between
  # reads, so match on the text after the timestamp (it carries pid and fault).
  python3 - "$out/xid-before.txt" "$out/xid-after.txt" >"$out/xid-new.txt" <<'PY' || true
import sys
seen = {line.split(' ', 1)[-1] for line in open(sys.argv[1]).read().splitlines()}
for line in open(sys.argv[2]).read().splitlines():
    if line.split(' ', 1)[-1] not in seen:
        print(line)
PY
  nvidia-smi --query-gpu=index,pci.bus_id,memory.used --format=csv,noheader >"$out/final-gpus.txt" || true
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >"$out/final-compute.txt" || true
  python3 - "$out" <<'PY'
import datetime, json, re, sys
from pathlib import Path
out = Path(sys.argv[1])
xid_text = (out/'xid-new.txt').read_text()
codes = sorted({int(m) for m in re.findall(r'Xid \(PCI:[^)]*\): (\d+)', xid_text)})
status = json.loads((out/'client/status.json').read_text()) if (out/'client/status.json').exists() else {}
gpus = (out/'final-gpus.txt').read_text().strip().splitlines()

# Owner stop rule (2026-10-10): Xid 31 is allowed. Xid 43 is allowed only when
# the engine logged a device-side assert or CUDA error within 10 s of it. Xid 13
# (widened 19:57Z) is allowed only with an engine illegal-memory-access error
# within 10 s. Any other Xid, an unmatched 43 or 13, or a missing GPU stops all
# GPU work.
def kernel_time(line):
    m = re.match(r'(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)[,.](\d+)([+-]\d\d:\d\d)', line)
    if not m:
        return None
    return datetime.datetime.fromisoformat(f'{m[1]}.{m[2][:6]}{m[3]}').astimezone(datetime.timezone.utc)
engine_errors, illegal_access = [], []
log = out/'server.log'
if log.exists():
    pattern = re.compile(r'device-side assert|illegal memory access|Triton Error \[CUDA\]|CUDA error|cudaError|AcceleratorError')
    for line in log.read_text(errors='replace').splitlines():
        m = re.search(r'(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\.(\d+)Z', line)
        if m and pattern.search(line):
            t = datetime.datetime.fromisoformat(f'{m[1]}.{m[2][:6]}+00:00')
            engine_errors.append(t)
            if 'illegal memory access' in line:
                illegal_access.append(t)
def matched_xids(code, errors):
    found = []
    for line in xid_text.splitlines():
        if re.search(r'Xid \(PCI:[^)]*\): %d\b' % code, line):
            when = kernel_time(line)
            gap = min((abs((e-when).total_seconds()) for e in errors), default=None) if when else None
            found.append({'utc': when.isoformat() if when else None, 'nearest_engine_error_s': gap,
                          'matched': gap is not None and gap <= 10})
    return found
xid43 = matched_xids(43, engine_errors)
xid13 = matched_xids(13, illegal_access)
unmatched = any(not x['matched'] for x in xid43 + xid13)
other = [c for c in codes if c not in (13, 31, 43)]
if other or unmatched:
    outcome = 'other_xid_STOP'
elif 13 in codes:
    outcome = 'xid13_illegal_access'
elif 43 in codes:
    outcome = 'xid43_assert'
elif codes:
    outcome = 'xid31'
else:
    outcome = 'clean' if status.get('state') == 'wave_clean' else 'client_failure_no_xid' if status else 'startup_failure'
result = {'outcome': outcome, 'xid_codes': codes, 'xid43': xid43,
          'xid13_matched': sum(x['matched'] for x in xid13), 'xid13_unmatched': sum(not x['matched'] for x in xid13),
          'client_state': status.get('state'), 'gpus_after': gpus,
          'gpu_count_ok': len(gpus) == 2, 'compute_clear': not (out/'final-compute.txt').read_text().strip()}
(out/'outcome.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
if outcome == 'other_xid_STOP' or len(gpus) != 2:
    (out/'STOP_GPU_WORK').write_text('Xid outside the stop rule (not 31; 43 without an engine error, or 13 without an '
                                     'engine illegal-access error, within 10 s) '
                                     'or missing GPU: stop all GPU work and message the owner.\n')
PY
  echo "Run finished: $out"
}
trap cleanup EXIT
trap 'exit 124' TERM INT
xid_lines >"$out/xid-before.txt"
if [[ -n $(docker ps -q --filter "label=$GLMX_BENCH_LABEL") ]] || [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]; then
  echo 'Bench containers or GPU compute present; owner must drain first'; exit 2
fi
# A STOP file stays in place as evidence; the owner clears it by writing a
# STOP_GPU_WORK.owner-cleared record beside it.
if find "$glmx_root/runs" -maxdepth 2 -name STOP_GPU_WORK | while read -r f; do [[ -e $f.owner-cleared ]] || echo "$f"; done | grep -q .; then
  echo 'A previous run recorded STOP_GPU_WORK; owner must clear it'; exit 2
fi
sudo -n /usr/bin/mkdir -p "$glmx_root/cache/$variant/tmp" "$glmx_root/cache/$variant/sanitizer"
"${compose[@]}" config -q
img=$("${compose[@]}" config --images)
docker image inspect "$img" >/dev/null # never pull
record_env "$out" "$img"
telemetry=1; telemetry_start "$out"
active=1; "${compose[@]}" up -d --pull never
began=$(date +%s) healthy=0
limit=1200
# Under memcheck, model load and warm-up run many times slower.
if [[ $variant == memcheck* ]]; then limit=5400; fi
while (( $(date +%s)-began < limit && $(date +%s) < end-600 )); do
  if stopped "$out" || [[ -n $("${compose[@]}" ps -aq --status exited) ]]; then break; fi
  if curl -fsS -m 5 http://127.0.0.1:8000/health >/dev/null; then healthy=1; break; fi
  sleep 10
done
printf '{"healthy":%s,"startup_seconds":%s}\n' "$healthy" "$(( $(date +%s)-began ))" >"$out/load.json"
(( healthy )) || { echo 'Engine did not become healthy'; exit 1; }
if [[ $client == lab ]]; then
  python3 "$root/minimal-repro.py" prepare --scratch "$TMPDIR/corpus" >"$out/prepare.json"
  python3 "$root/minimal-repro.py" run --scratch "$TMPDIR/corpus" --output "$out/client" --model "$served" \
    --lease-end "$((end-300))" --sessions "$sessions" --waves "$waves" || true
else
  # The public client runs one wave per call; repeat it with fresh sessions.
  for (( w = 0; w < waves; w++ )); do
    python3 "$root/repro-client.py" --model "$served" --padding "$GLMX_PADDING" \
      --output "$out/client-w$w" --sessions "$sessions" || true
    mkdir -p "$out/client"
    cp "$out/client-w$w/status.json" "$out/client/status.json" 2>/dev/null || true
    grep -q '"wave_clean"' "$out/client-w$w/status.json" 2>/dev/null || break
  done
fi
# Let the engine log a late fault before collecting logs.
sleep 15
