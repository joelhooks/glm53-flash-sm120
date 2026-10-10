#!/usr/bin/env bash
# Copy exactly the files listed in PUBLISH.txt into a new directory, write a
# sha256 manifest, and run the redaction scan inside the exported tree.
#   export-public.sh <new-output-dir>
set -euo pipefail
out=${1:?output directory required}
root=$(cd "$(dirname "$0")" && pwd)
[[ ! -e $out ]] || { echo "Refusing to reuse $out"; exit 2; }
mkdir -p "$out"
cd "$root"
! grep -qE '^\s*redaction-private\.txt' PUBLISH.txt || { echo "redaction-private.txt must not be published"; exit 2; }
grep -vE '^\s*(#|$)' PUBLISH.txt | while read -r path; do
  [[ -e $path ]] || { echo "Listed path missing: $path"; exit 2; }
  if [[ -d $path ]]; then
    find "$path" -type f ! -path '*/__pycache__/*' -print0 | xargs -0 -I{} cp --parents -p {} "$out/"
  else
    cp --parents -p "$path" "$out/"
  fi
done
(cd "$out" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum >MANIFEST.sha256)
echo "Exported $(wc -l <"$out/MANIFEST.sha256") files to $out"
[[ ! -e $out/redaction-private.txt ]] || { echo "private list leaked into export"; exit 2; }
GLMX_REDACTION_PRIVATE="$root/redaction-private.txt" bash "$out/redaction-scan.sh"
