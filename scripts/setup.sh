#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv sync --locked
uv run python -c \
  "from src.config import get_device; print('사용 장치:', get_device())"