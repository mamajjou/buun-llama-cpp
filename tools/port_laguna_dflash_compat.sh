#!/usr/bin/env bash
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"

POOL_REMOTE=${POOL_REMOTE:-poolside}
POOL_URL=${POOL_URL:-https://github.com/poolsideai/llama.cpp.git}
POOL_REF=${POOL_REF:-laguna}

if ! git remote get-url "$POOL_REMOTE" >/dev/null 2>&1; then
    git remote add "$POOL_REMOTE" "$POOL_URL"
fi

git fetch --depth=1 "$POOL_REMOTE" "$POOL_REF"
TMP=$(mktemp --suffix=-poolside-dflash.cpp)
trap 'rm -f "$TMP"' EXIT
git show FETCH_HEAD:src/models/dflash.cpp > "$TMP"

python3 tools/apply_laguna_dflash_compat.py --poolside-dflash "$TMP" --root "$ROOT"
git diff --check

cat <<'EOF'
Laguna DFlash source port applied.

Recommended validation:
  cmake -S . -B build-port-check \
    -DGGML_NATIVE=OFF -DGGML_CUDA=OFF -DGGML_HIP=OFF \
    -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build build-port-check --target llama-cli -j2

Then build your gfx1100 configuration and load the official 76-tensor Laguna drafter.
EOF
