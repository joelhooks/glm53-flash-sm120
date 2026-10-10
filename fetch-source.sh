#!/usr/bin/env bash
# Rebuild the image's Python source trees from public git pins plus the
# patches in image-patches/, in the layout the Dockerfile expects.
#   fetch-source.sh <empty-output-dir>
# extract-source.sh gets the same trees out of the image itself; this script
# needs only network access to GitHub. The result must match file for file.
set -euo pipefail
out=${1:?output directory required}
root=$(cd "$(dirname "$0")" && pwd)
[[ ! -e $out ]] || { echo "Refusing to reuse $out"; exit 2; }
mkdir -p "$out/trees" "$out/receipts"
fetch() { # name repo commit
  local dir=$out/trees/$1
  git init -q "$dir"
  git -C "$dir" fetch -q --depth 1 "$2" "$3"
  git -C "$dir" checkout -q --detach FETCH_HEAD
  [[ $(git -C "$dir" rev-parse HEAD) == "$3" ]]
  printf '%s %s\n' "$3" "$2" >"$out/receipts/$1.head.txt"
}
fetch vllm https://github.com/local-inference-lab/vllm.git 6dc2f516688fe6f84c6994dcd20fddf296853a6c
fetch b12x https://github.com/local-inference-lab/b12x.git 36bce2c1552ba2d47dc09f20a6f64fbfc8ec4ff8
fetch lmcache https://github.com/local-inference-lab/LMCache.git a128b2e286ebb3556cb43124149e600ff99fe481
for tree in vllm b12x; do
  git -C "$out/trees/$tree" apply --whitespace=nowarn \
    "$root/image-patches/$tree.head.patch" "$root/image-patches/$tree.untracked.patch"
done
cp "$root/upstream/PROVENANCE.json" "$out/receipts/PROVENANCE.json"
printf '{"kind": "fetched from public pins plus image-patches", "trees": ["vllm", "b12x", "lmcache"]}\n' \
  >"$out/receipts/summary.json"
printf 'SOURCE_EXTRACTED\n' >"$out/receipts/COMPLETE"
echo "Source ready: $out"
