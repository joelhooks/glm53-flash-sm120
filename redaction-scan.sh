#!/usr/bin/env bash
# Scan every file listed in PUBLISH.txt for credentials, home paths and
# private addresses, plus an optional lab-only name list. Exit 1 on any hit in our own files.
# Hits inside image-patches/ are upstream code; they are listed for review
# and fail the scan only for credential patterns.
#   redaction-scan.sh [extra-regex-file]
set -euo pipefail
cd "$(dirname "$0")"
mapfile -t files < <(grep -vE '^\s*(#|$)' PUBLISH.txt | while read -r p; do
  if [[ -d $p ]]; then find "$p" -type f; elif [[ -e $p ]]; then echo "$p"; else echo "MISSING:$p"; fi
done | sort)
missing=$(printf '%s\n' "${files[@]}" | grep '^MISSING:' || true)
[[ -z $missing ]] || { echo "$missing"; exit 1; }
credential='(hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|xox[abpr]-|BEGIN [A-Z ]*PRIVATE KEY|Bearer [A-Za-z0-9._-]{20,})'
generic='(/home/|/Users/|\.ts\.net|tailscale|tailnet|\b100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]+\.[0-9]+|\b10\.[0-9]+\.[0-9]+\.[0-9]+|\b192\.168\.[0-9]+\.[0-9]+|\b172\.(1[6-9]|2[0-9]|3[01])\.[0-9]+\.[0-9]+|api[_-]?key\s*[:=])'
# Machine, project and model names to keep out live in a lab-only list
# (one extended regex per line), never in this public script.
private_list=${GLMX_REDACTION_PRIVATE:-redaction-private.txt}
if [[ -f $private_list ]]; then echo "private list: $(grep -cvE '^\s*(#|$)' "$private_list") patterns"
else echo "private list not found, generic scan only"; private_list=; fi
extra=${1:-}
status=0
for f in "${files[@]}"; do
  case $f in image-patches/*) upstream=1 ;; *) upstream=0 ;; esac
  # The generic patterns would match themselves here; check this script
  # against the private list only.
  if [[ $f == redaction-scan.sh ]]; then
    if [[ -n $private_list ]] && grep -nIiE -f <(grep -vE '^\s*(#|$)' "$private_list") "$f" | sed "s|^|PRIVATE $f:|"; then status=1; fi
    continue
  fi
  if grep -nIE "$credential" "$f" | head -3 | sed "s|^|CREDENTIAL $f:|" | grep .; then status=1; fi
  hits=$(grep -nIiE "$generic" "$f" | head -5 || true)
  if [[ -n $private_list ]]; then hits+=$'\n'$(grep -nIiE -f <(grep -vE '^\s*(#|$)' "$private_list") "$f" | head -5 || true); fi
  if [[ -n $extra ]]; then hits+=$'\n'$(grep -nIiEf "$extra" "$f" | head -5 || true); fi
  hits=$(printf '%s\n' "$hits" | grep . || true)
  if [[ -n $hits ]]; then hits=$(printf '%s\n' "$hits" | grep -vF -f <(grep -v '^#' redaction-allow.txt) || true); fi
  if [[ -n $hits ]]; then
    if (( upstream )); then printf '%s\n' "$hits" | sed "s|^|REVIEW(upstream) $f:|" | cut -c1-220
    else printf '%s\n' "$hits" | sed "s|^|PRIVATE $f:|" | cut -c1-220; status=1; fi
  fi
done
echo "scanned ${#files[@]} files; status=$status"
exit "$status"
