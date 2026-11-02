#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
task_resolution=1920x1080
task_fps=60
task_output=renders/manimgx/CVAIL-ManimGX.mp4
if [[ "${1:-}" == "--preview" ]]; then
    task_resolution=1280x720
    task_fps=30
    task_output=renders/manimgx/CVAIL-ManimGX-preview.mp4
    shift
fi
mkdir -p renders/manimgx
exec .venv-manimgx/bin/manimgx render manimgx/scene.py CVAILIntro \
    -r "$task_resolution" --fps "$task_fps" --crf 18 -o "$task_output" "$@"
