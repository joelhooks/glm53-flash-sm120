#!/usr/bin/env bash
# Run helpers for run-repro.sh: environment receipt, telemetry and a thermal
# stop guard. Source this; do not run it. run-repro.sh prefers a host's own
# bench library when one exists and falls back to this file.

# Record image ids and digests, driver and kernel for a run.
record_env() {
  local out=$1
  shift
  {
    echo "date: $(date -Is)"
    echo "kernel: $(uname -r)"
    nvidia-smi --query-gpu=index,name,driver_version,power.limit,persistence_mode,pcie.link.gen.current,pcie.link.width.current --format=csv
    for img in "$@"; do
      echo "image: $img"
      docker image inspect "$img" --format '  id={{.Id}} digests={{json .RepoDigests}}'
    done
  } >"$out/env.txt" 2>&1
}

# GPU state at 1 Hz and dmon, plus a guard that stops labelled bench
# containers if hardware thermal or power-brake slowdown appears.
telemetry_start() {
  local out=$1
  mkdir -p "$out"
  TEL_PIDS=()
  nvidia-smi --query-gpu=timestamp,index,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used,pstate,clocks_event_reasons.hw_slowdown,clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.hw_power_brake_slowdown \
    --format=csv -l 1 >"$out/gpu.csv" 2>&1 &
  TEL_PIDS+=($!)
  nvidia-smi dmon -s pucvmet -d 1 -o DT >"$out/dmon.log" 2>&1 &
  TEL_PIDS+=($!)
  (while :; do
    if nvidia-smi --query-gpu=clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.hw_power_brake_slowdown --format=csv,noheader 2>/dev/null | grep -Eq '(^|, )Active'; then
      echo "STOP $(date -Is): hardware thermal slowdown or power-brake slowdown active" >"$out/STOPPED"
      mapfile -t ids < <(docker ps -q --filter "label=$GLMX_BENCH_LABEL")
      [ "${#ids[@]}" -gt 0 ] && docker stop "${ids[@]}" >/dev/null 2>&1
      break
    fi
    sleep 1
  done) &
  TEL_PIDS+=($!)
}

telemetry_stop() {
  for p in "${TEL_PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
}

stopped() { [ -e "$1/STOPPED" ]; }
