#!/bin/sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BUILD_DIR=${STRATA_METAL_BUILD_DIR:-"$HOME/Library/Application Support/Strata/build/q4"}
MODE=${1:---build-only}
if [ "$#" -gt 1 ]; then echo 'expected one mode argument' >&2; exit 2; fi
case "$MODE" in --build-only|--check|--benchmark) ;; *) echo 'unknown mode' >&2; exit 2 ;; esac
SDK_PATH=$(xcrun --sdk macosx --show-sdk-path)
mkdir -p "$BUILD_DIR"
xcrun --sdk macosx metal -std=metal3.1 -O3 -fno-fast-math -Wall -Wextra -Werror \
  -c "$SCRIPT_DIR/affine_q4.metal" -o "$BUILD_DIR/affine_q4.air"
xcrun --sdk macosx metallib "$BUILD_DIR/affine_q4.air" -o "$BUILD_DIR/strata-q4.metallib"
xcrun --sdk macosx clang++ -std=c++20 -O3 -Wall -Wextra -Werror -fobjc-arc \
  -isysroot "$SDK_PATH" -framework Foundation -framework Metal \
  "$SCRIPT_DIR/affine_q4_bench.mm" -o "$BUILD_DIR/strata-q4-bench"
if [ "$MODE" = --build-only ]; then exit 0; fi
STRATA_Q4_SOURCE_SHA256=$(shasum -a 256 "$SCRIPT_DIR/affine_q4.metal" | awk '{print $1}')
STRATA_Q4_CONTRACT_SHA256=$(shasum -a 256 "$SCRIPT_DIR/affine_q4_contract.h" | awk '{print $1}')
STRATA_Q4_HOST_SHA256=$(shasum -a 256 "$SCRIPT_DIR/affine_q4_bench.mm" | awk '{print $1}')
STRATA_Q4_BINARY_SHA256=$(shasum -a 256 "$BUILD_DIR/strata-q4-bench" | awk '{print $1}')
STRATA_Q4_METALLIB_SHA256=$(shasum -a 256 "$BUILD_DIR/strata-q4.metallib" | awk '{print $1}')
export STRATA_Q4_SOURCE_SHA256 STRATA_Q4_CONTRACT_SHA256 STRATA_Q4_HOST_SHA256
export STRATA_Q4_BINARY_SHA256 STRATA_Q4_METALLIB_SHA256
"$BUILD_DIR/strata-q4-bench" "$MODE" "$BUILD_DIR/strata-q4.metallib"
