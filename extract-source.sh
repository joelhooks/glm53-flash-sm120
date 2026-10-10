#!/usr/bin/env bash
# CPU-only: copy from a stopped container. Never start it or request GPUs.
set -euo pipefail
export TMPDIR=${GLMX_SCRATCH:-${XDG_CACHE_HOME:-$HOME/.cache}/glm53-flash-sm120}
mkdir -p "$TMPDIR"
root_dir=${GLMX_ROOT:-/data/bench/glm-xid31}
image=verdictai/glm53-flash-exl3-k4@sha256:0f1cdcc8891f1cc3a444121eb61d366289a1cbba285f0892dcbb24bc94961692
out=${1:-$root_dir/source}
[[ $out == "$root_dir"/* && ! -e $out ]] || {
  echo "Use a new directory under $root_dir/ (set GLMX_ROOT to change it)"; exit 2;
}
mkdir -p "$out/trees" "$out/receipts"
docker() { sudo -n /usr/bin/docker "$@"; }
docker image inspect "$image" --format '{{.Id}} {{.Size}} {{json .Config.Entrypoint}}' >"$out/receipts/image.txt"
id=$(docker create --runtime=runc --network none --entrypoint /bin/true "$image")
printf '%s\n' "$id" >"$out/receipts/stopped-container-id.txt"
cleanup() { docker rm "$id" >/dev/null; }
trap cleanup EXIT
for tree in vllm b12x lmcache; do
  docker cp "$id:/opt/infernal-invocation/$tree" "$out/trees/$tree"
done
docker cp "$id:/opt/exllamav3" "$out/trees/exllamav3"
docker cp "$id:/opt/glm53/PROVENANCE.json" "$out/receipts/PROVENANCE.json"
docker cp "$id:/opt/glm53/calibration" "$out/receipts/calibration"
# Only this new extraction is changed; preserve original file bytes.
sudo -n chown -R "$(id -u):$(id -g)" "$out"
for tree in vllm b12x lmcache; do
  src="$out/trees/$tree"
  git -C "$src" rev-parse HEAD >"$out/receipts/$tree.head.txt"
  git -C "$src" status --porcelain=v1 >"$out/receipts/$tree.status.txt"
  git -C "$src" diff HEAD --binary >"$out/receipts/$tree.head.patch"
  git -C "$src" diff --binary >"$out/receipts/$tree.unstaged.patch"
  git -C "$src" diff --cached --binary >"$out/receipts/$tree.staged.patch"
  git -C "$src" ls-files --others --exclude-standard >"$out/receipts/$tree.untracked.txt"
  git -C "$src" bundle create "$out/receipts/$tree.bundle" --all
  git -C "$src" diff HEAD --numstat >"$out/receipts/$tree.numstat.txt"
done
python3 - "$out" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for tree in ('vllm', 'b12x', 'lmcache', 'exllamav3'):
    src = root/'trees'/tree
    files = []
    for p in sorted(src.rglob('*')):
        if '.git' in p.relative_to(src).parts or not p.is_file():
            continue
        h = hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(8*1024*1024), b''):
                h.update(chunk)
        files.append({'path': str(p.relative_to(src)), 'bytes': p.stat().st_size,
                      'sha256': h.hexdigest(), 'symlink': str(p.readlink()) if p.is_symlink() else None})
    (root/'receipts'/f'{tree}.files.json').write_text(json.dumps(files, indent=2)+'\n')
    rows.append({'tree': tree, 'files': len(files), 'bytes': sum(f['bytes'] for f in files),
                 'head': (root/'receipts'/f'{tree}.head.txt').read_text().strip() if tree != 'exllamav3' else None,
                 'status_lines': len((root/'receipts'/f'{tree}.status.txt').read_text().splitlines()) if tree != 'exllamav3' else None})
(root/'receipts'/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
PY
printf 'SOURCE_EXTRACTED\n' >"$out/receipts/COMPLETE"
echo "Source preserved at $out; no container was started."
