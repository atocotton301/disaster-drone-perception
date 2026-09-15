#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if ! command -v ros2 >/dev/null 2>&1; then
  printf '%s\n' '먼저 설치된 ROS 2 환경의 setup.bash를 source 하세요. JetPack/CUDA를 자동 설치하지 않습니다.'
  exit 2
fi
exec "${DISASTER_PYTHON:-python3}" run.py "$@"
