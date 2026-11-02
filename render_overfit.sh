#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
task_resolution=1920x1080
task_fps=60
task_output=renders/manimgx_overfit/CVAIL-Overfit-Regression.mp4
if [[ "${1:-}" == "--preview" ]]; then
    task_resolution=1280x720
    task_fps=30
    task_output=renders/manimgx_overfit/CVAIL-Overfit-Regression-preview.mp4
    shift
fi
mkdir -p renders/manimgx_overfit
task_native=renders/manimgx_overfit/CVAIL-Overfit-Regression-native.mp4
task_duration=$(.venv-manimgx/bin/python -c 'import json; print(json.load(open("manimgx_overfit/config.json"))["duration"])')
.venv-manimgx/bin/manimgx render manimgx_overfit/scene.py CVAILOverfitIntro \
    -r "$task_resolution" --fps "$task_fps" --crf 17 --preset fast -o "$task_native" "$@"
# Keep the static closing card for its full duration, at a constant frame rate.
task_encoder_python=.venv-manimgx/bin/python
if ! "$task_encoder_python" -c 'import av' >/dev/null 2>&1; then
    task_encoder_python=python3
fi
"$task_encoder_python" manimgx_overfit/finalize.py "$task_native" "$task_output" \
    --fps "$task_fps" --duration "$task_duration"
