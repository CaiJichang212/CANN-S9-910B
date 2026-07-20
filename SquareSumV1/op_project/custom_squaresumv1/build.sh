#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Build script for SquareSumV1 operator

set -e

SUPPORT_COMPUTE_UNITS=("ascend910b" "ascend910_93")

export BASE_PATH=$(
  cd "$(dirname $0)"
  pwd
)
export BUILD_PATH="${BASE_PATH}/build"
export BUILD_OUT_PATH="${BASE_PATH}/build_out"

CORE_NUMS=$(cat /proc/cpuinfo | grep "processor" | wc -l)
if [ ${CORE_NUMS} -gt 8 ]; then
  CORE_NUMS=8
fi

usage() {
  echo "Build script for squaresumv1 operator"
  echo "Usage: bash build.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  -h, --help              Print this help message"
  echo "  -j[n]                   Compile thread nums, default is ${CORE_NUMS}, eg: -j8"
  echo "  --soc=soc_version       Compile for specified Ascend SoC (ascend910b)"
  echo "  --make_clean            Clean build artifacts"
  echo ""
  echo "Examples:"
  echo "  bash build.sh --soc=ascend910b -j8"
  echo "  bash build.sh --make_clean"
}

check_compute_unit() {
  local unit="$1"
  for support_unit in "${SUPPORT_COMPUTE_UNITS[@]}"; do
    if [[ "$unit" == "$support_unit" ]]; then
      return 0
    fi
  done
  return 1
}

clean_build() {
  if [ -d "${BUILD_PATH}" ]; then
    echo "Cleaning build directory..."
    rm -rf ${BUILD_PATH}/*
  fi
}

clean_build_out() {
  if [ -d "${BUILD_OUT_PATH}" ]; then
    echo "Cleaning build_out directory..."
    rm -rf ${BUILD_OUT_PATH}/*
  fi
}

THREAD_NUM=${CORE_NUMS}
COMPUTE_UNIT=""
ENABLE_CLEAN=FALSE

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -j*)
      THREAD_NUM="${1:2}"
      if [ -z "$THREAD_NUM" ]; then
        THREAD_NUM=${CORE_NUMS}
      fi
      shift
      ;;
    --make_clean)
      ENABLE_CLEAN=TRUE
      shift
      ;;
    --soc=*)
      COMPUTE_UNIT="${1#*=}"
      shift
      ;;
    -*)
      echo "[ERROR] Invalid option: $1"
      usage
      exit 1
      ;;
    *)
      echo "[ERROR] Unexpected argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [ "$ENABLE_CLEAN" = "TRUE" ]; then
  clean_build
  clean_build_out
  exit 0
fi

if [ -z "$COMPUTE_UNIT" ]; then
  COMPUTE_UNIT="ascend910b"
  echo "[INFO] No --soc specified, defaulting to ${COMPUTE_UNIT}"
fi

COMPUTE_UNIT=$(echo "$COMPUTE_UNIT" | tr '[:upper:]' '[:lower:]')
if ! check_compute_unit "$COMPUTE_UNIT"; then
  echo "[ERROR] Invalid SoC version: $COMPUTE_UNIT"
  echo "[INFO] Supported SoC versions: ${SUPPORT_COMPUTE_UNITS[@]}"
  exit 1
fi
echo "[INFO] Compile for SoC: ${COMPUTE_UNIT}"

CMAKE_ARGS=""
CMAKE_ARGS="$CMAKE_ARGS -DASCEND_COMPUTE_UNIT=$COMPUTE_UNIT"

if [ ! -d "${BUILD_PATH}" ]; then
  mkdir -p "${BUILD_PATH}"
fi

[ -f "${BUILD_PATH}/CMakeCache.txt" ] && rm -f ${BUILD_PATH}/CMakeCache.txt

echo "----------------------------------------------------------------"
echo "[INFO] Configuring project..."
echo "[INFO] CMAKE_ARGS: ${CMAKE_ARGS}"
cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} ..

echo "----------------------------------------------------------------"
echo "[INFO] Building project with ${THREAD_NUM} threads..."
cmake --build . --target all binary package -- -j ${THREAD_NUM}

KERNEL_O=$(find ${BUILD_PATH}/op_kernel/ascendc_kernels/binary/${COMPUTE_UNIT} -name "*.o" 2>/dev/null | head -1)
if [ -z "$KERNEL_O" ]; then
    echo "[ERROR] Kernel binary not found"
    exit 1
fi

PKG_PATH=$(ls ${BUILD_PATH}/custom_opp_*.run 2>/dev/null | head -1)
if [ -z "$PKG_PATH" ] || [ ! -s "$PKG_PATH" ]; then
    echo "[ERROR] Package not found or empty"
    exit 1
fi

# Copy to build_out
mkdir -p "${BUILD_OUT_PATH}"
cp "${PKG_PATH}" "${BUILD_OUT_PATH}/"

echo "----------------------------------------------------------------"
echo "[INFO] Build completed successfully!"
echo "[INFO] Kernel binary: ${KERNEL_O}"
echo "[INFO] Package: ${PKG_PATH}"
echo "[INFO] Package (build_out): ${BUILD_OUT_PATH}/custom_opp_openEuler_aarch64.run"
