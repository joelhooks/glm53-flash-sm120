#!/usr/bin/env python3
"""Split a tree's combined HEAD diff into one readable patch per file.

  split-patches.py image-patches/vllm.head.patch image-patches/vllm

Writes <out>/<path>.patch for every file in the diff and <out>/INDEX.tsv
(status, added lines, removed lines, path). The combined patch stays the
source of truth; applying every per-file patch reproduces it.
"""
import re
import sys
from pathlib import Path

source, out = Path(sys.argv[1]), Path(sys.argv[2])
chunks = re.split(r'(?m)^(?=diff --git )', source.read_text())
rows = []
for chunk in chunks:
    if not chunk.startswith('diff --git '):
        continue
    path = re.match(r'diff --git a/(\S+) b/(\S+)', chunk).group(2)
    status = ('added' if '\nnew file mode' in chunk else 'deleted' if '\ndeleted file mode' in chunk
              else 'binary' if '\nBinary files ' in chunk else 'modified')
    lines = chunk.splitlines()
    added = sum(1 for line in lines if line.startswith('+') and not line.startswith('+++'))
    removed = sum(1 for line in lines if line.startswith('-') and not line.startswith('---'))
    target = out/(path+'.patch')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(chunk)
    rows.append(f'{status}\t{added}\t{removed}\t{path}')
(out/'INDEX.tsv').write_text('status\tadded\tremoved\tpath\n'+'\n'.join(sorted(rows, key=lambda r: r.split('\t')[3]))+'\n')
rebuilt = ''.join(c for c in chunks if c.startswith('diff --git '))
assert rebuilt == source.read_text()[len(chunks[0]):], 'split does not reproduce the combined patch'
print(f'{len(rows)} files -> {out}')
