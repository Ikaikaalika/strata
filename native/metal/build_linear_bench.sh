#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
BUILD_DIR=${STRATA_METAL_BUILD_DIR:-"$SCRIPT_DIR/build"}
SDK_PATH=$(xcrun --sdk macosx --show-sdk-path)
STRATA_BENCH_SDK=$(basename "$SDK_PATH")
STRATA_BENCH_REVISION=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)
if [ -z "$STRATA_BENCH_REVISION" ]; then
  STRATA_BENCH_REVISION=unknown
fi
if [ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=normal 2>/dev/null)" ]; then
  STRATA_BENCH_DIRTY=true
else
  STRATA_BENCH_DIRTY=false
fi

mkdir -p "$BUILD_DIR"

xcrun --sdk macosx metal \
  -std=metal3.1 \
  -c "$SCRIPT_DIR/linear_projection.metal" \
  -o "$BUILD_DIR/linear_projection.air"

xcrun --sdk macosx metallib \
  "$BUILD_DIR/linear_projection.air" \
  -o "$BUILD_DIR/strata-linear.metallib"

xcrun --sdk macosx clang++ \
  -std=c++20 \
  -O3 \
  -fobjc-arc \
  -isysroot "$SDK_PATH" \
  -framework Foundation \
  -framework Metal \
  "$SCRIPT_DIR/linear_projection_bench.mm" \
  -o "$BUILD_DIR/strata-linear-bench"

STRATA_BENCH_REVISION="$STRATA_BENCH_REVISION" \
STRATA_BENCH_DIRTY="$STRATA_BENCH_DIRTY" \
STRATA_BENCH_SDK="$STRATA_BENCH_SDK" \
  "$BUILD_DIR/strata-linear-bench"
