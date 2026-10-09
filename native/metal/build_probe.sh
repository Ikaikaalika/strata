#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BUILD_DIR=${LOKAHI_METAL_BUILD_DIR:-"$SCRIPT_DIR/build"}
SDK_PATH=$(xcrun --sdk macosx --show-sdk-path)

mkdir -p "$BUILD_DIR"

xcrun --sdk macosx metal \
  -std=metal3.1 \
  -c "$SCRIPT_DIR/rmsnorm_residual.metal" \
  -o "$BUILD_DIR/rmsnorm_residual.air"

xcrun --sdk macosx metallib \
  "$BUILD_DIR/rmsnorm_residual.air" \
  -o "$BUILD_DIR/lokahi-metal.metallib"

xcrun --sdk macosx clang++ \
  -std=c++20 \
  -fobjc-arc \
  -isysroot "$SDK_PATH" \
  -framework Foundation \
  -framework Metal \
  "$SCRIPT_DIR/metal_probe.mm" \
  -o "$BUILD_DIR/lokahi-metal-probe"

"$BUILD_DIR/lokahi-metal-probe"
