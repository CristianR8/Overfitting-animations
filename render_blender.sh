#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if python -c 'import bpy, numpy' >/dev/null 2>&1; then
    exec python intro.py "$@"
elif command -v blender >/dev/null 2>&1; then
    exec blender --background --python intro.py -- "$@"
else
    echo 'Se necesita Blender o un Python con bpy y NumPy.' >&2
    exit 1
fi
