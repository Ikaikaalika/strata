#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
BUILD_DIR=${LOKAHI_METAL_BUILD_DIR:-"$SCRIPT_DIR/build"}
SDK_PATH=$(xcrun --sdk macosx --show-sdk-path)
LOKAHI_BENCH_SDK=$(basename "$SDK_PATH")
LOKAHI_BENCH_REVISION=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || true)
if [ -z "$LOKAHI_BENCH_REVISION" ]; then
  LOKAHI_BENCH_REVISION=unknown
fi
if [ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=normal 2>/dev/null)" ]; then
  LOKAHI_BENCH_DIRTY=true
else
  LOKAHI_BENCH_DIRTY=false
fi

mkdir -p "$BUILD_DIR"

xcrun --sdk macosx metal \
  -std=metal3.1 \
  -c "$SCRIPT_DIR/linear_projection.metal" \
  -o "$BUILD_DIR/linear_projection.air"

xcrun --sdk macosx metallib \
  "$BUILD_DIR/linear_projection.air" \
  -o "$BUILD_DIR/lokahi-linear.metallib"

xcrun --sdk macosx clang++ \
  -std=c++20 \
  -O3 \
  -fobjc-arc \
  -isysroot "$SDK_PATH" \
  -framework Foundation \
  -framework Metal \
  "$SCRIPT_DIR/linear_projection_bench.mm" \
  -o "$BUILD_DIR/lokahi-linear-bench"

LOKAHI_BENCH_REVISION="$LOKAHI_BENCH_REVISION" \
LOKAHI_BENCH_DIRTY="$LOKAHI_BENCH_DIRTY" \
LOKAHI_BENCH_SDK="$LOKAHI_BENCH_SDK" \
  "$BUILD_DIR/lokahi-linear-bench"
