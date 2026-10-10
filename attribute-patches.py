#!/usr/bin/env python3
"""Prepend an attribution header to every image patch. Idempotent.

git apply skips text before the first "diff --git" line, so the header
does not change what a patch does. Run after split-patches.py.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MARK = '# glm53-flash-sm120 attribution'
IMAGE = 'verdictai/glm53-flash-exl3-k4@sha256:0f1cdcc8891f1cc3a444121eb61d366289a1cbba285f0892dcbb24bc94961692'
TREES = {
    'vllm': ('local-inference-lab/vllm', '6dc2f516688fe6f84c6994dcd20fddf296853a6c'),
    'b12x': ('local-inference-lab/b12x', '36bce2c1552ba2d47dc09f20a6f64fbfc8ec4ff8'),
}


def header(tree, kind):
    repo, commit = TREES[tree]
    return (f'{MARK}\n'
            f'# Source: changes found in the published image {IMAGE}\n'
            f'#   release r19-sm120-tp2-ep2-dcp2-v84-dflash2; release author per the image\'s\n'
            f'#   /opt/glm53/PROVENANCE.json: Brandon M. Music. Not written by this project.\n'
            f'# Applies to: https://github.com/{repo} at {commit} ({kind})\n'
            f'# Licence: Apache-2.0, as the upstream project (see licenses/{tree}.LICENSE).\n'
            f'# Apply with: git apply <this file>\n')


count = 0
for tree in TREES:
    for kind, paths in (('tracked files, git diff HEAD', [ROOT/f'image-patches/{tree}.head.patch',
                                                         *sorted((ROOT/f'image-patches/{tree}').rglob('*.patch'))]),
                        ('files untracked in the image', [ROOT/f'image-patches/{tree}.untracked.patch',
                                                          *sorted((ROOT/f'image-patches/{tree}-untracked').rglob('*.patch'))])):
        for path in paths:
            text = path.read_text()
            if text.startswith(MARK):
                continue
            path.write_text(header(tree, kind) + text)
            count += 1
print(f'attributed {count} patch files')
