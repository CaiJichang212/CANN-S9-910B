#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ============================================================================
# SquareSumV1 op_host UT test runner

set -e

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
CLEAN_BUILD=true
VERBOSE=""
RUN_OP_HOST=true

# ============================================================================
# Parse arguments
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--clean)
            if [ "$2" = "false" ]; then
                CLEAN_BUILD=false
                shift 2
            else
                CLEAN_BUILD=true
                shift
            fi
            ;;
        -v|--verbose)
            VERBOSE="VERBOSE=1"
            shift
            ;;
        --ophost)
            RUN_OP_HOST=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "  -c, --clean      Clean build dir (default: true)"
            echo "  -v, --verbose    Verbose output"
            echo "  -h, --help       Show help"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "SquareSumV1 op_host UT Test"
echo "========================================"
echo "Clean build: ${CLEAN_BUILD}"
echo "Work dir: ${SCRIPT_DIR}"
echo "========================================"
echo ""

# ============================================================================
# Check dependencies
# ============================================================================
echo "Checking dependencies..."

if ! command -v cmake &> /dev/null; then
    echo "ERROR: cmake not found"
    exit 1
fi

if ! command -v g++ &> /dev/null; then
    echo "ERROR: g++ not found"
    exit 1
fi

echo "Dependencies OK"
echo ""

# ============================================================================
# Set environment variables
# ============================================================================
if [ -z "$ASCEND_HOME_PATH" ]; then
    echo "WARNING: ASCEND_HOME_PATH not set, using default"
    export ASCEND_HOME_PATH=/usr/local/Ascend/cann-8.5.0
fi

export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}
echo "LD_LIBRARY_PATH: ${ASCEND_HOME_PATH}/lib64"
echo ""

# ============================================================================
# Clean build
# ============================================================================
if [ "$CLEAN_BUILD" = true ]; then
    echo "Cleaning build dir..."
    rm -rf "${BUILD_DIR}"
fi

# ============================================================================
# CMake configure + build
# ============================================================================
echo "Creating build dir..."
mkdir -p "${BUILD_DIR}"

echo ""
echo "CMake configure..."
cd "${BUILD_DIR}"
cmake .. ${VERBOSE}
if [ $? -ne 0 ]; then
    echo "ERROR: CMake configure failed"
    exit 1
fi

echo ""
echo "Building UT..."
make -j$(nproc) ${VERBOSE}
if [ $? -ne 0 ]; then
    echo "ERROR: Build failed"
    exit 1
fi

echo "Build OK"
echo ""

# ============================================================================
# Run tests
# ============================================================================
echo "========================================"
echo "Running op_host UT tests"
echo "========================================"
echo ""

cd "${BUILD_DIR}/op_host"

if [ ! -f "./squaresumv1_op_host_ut" ]; then
    echo "ERROR: UT executable not found"
    exit 1
fi

if ./squaresumv1_op_host_ut; then
    echo ""
    echo "========================================"
    echo "Test Result: PASS"
    echo "========================================"
    exit 0
else
    echo ""
    echo "========================================"
    echo "Test Result: FAIL"
    echo "========================================"
    exit 1
fi
