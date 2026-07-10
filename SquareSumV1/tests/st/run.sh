#!/bin/bash
# ============================================================================
# aclnnSquareSumV1 算子 ST 测试执行脚本
#
# 用法:
#   bash run.sh --mock        # Mock 模式 (CPU golden, 无需 NPU)
#   bash run.sh --real        # Real 模式 (NPU 执行, 默认)
#   bash run.sh --help        # 帮助信息
#
# CSV 用例文件默认: testcases/aclnnSquareSumV1_l0_test_cases.csv
# ============================================================================

set -e

# ============================================================================
# 配置
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USE_MOCK=""
BUILD_DIR=""
CASE_FILE="testcases/aclnnSquareSumV1_l0_test_cases.csv"
EXTRA_CASE_FILE=""

# ============================================================================
# 帮助信息
# ============================================================================
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --mock          使用 Mock 模式 (CPU golden 验证, 无需 NPU)"
    echo "  --real          使用 Real 模式 (NPU 执行, 默认)"
    echo "  --case <file>   执行指定的测试用例 CSV 文件"
    echo "  --l1            同时运行 L0 + L1 sample 用例"
    echo "  --l1-full       同时运行 L0 + L1 全部用例"
    echo "  --help          显示帮助信息"
    echo ""
    echo "Examples:"
    echo "  # Mock 模式 (无 NPU 环境)"
    echo "  $0 --mock"
    echo ""
    echo "  # Mock 模式 + L1 sample"
    echo "  $0 --mock --l1"
    echo ""
    echo "  # Real 模式 (需要 NPU)"
    echo "  $0 --real"
}

# ============================================================================
# 解析参数
# ============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --mock)
            USE_MOCK="-DUSE_MOCK=ON"
            shift
            ;;
        --real)
            USE_MOCK=""
            shift
            ;;
        --l1)
            EXTRA_CASE_FILE="testcases/aclnnSquareSumV1_l1_sample_test_cases.csv"
            shift
            ;;
        --l1-full)
            EXTRA_CASE_FILE="testcases/aclnnSquareSumV1_l1_test_cases.csv"
            shift
            ;;
        --case)
            CASE_FILE="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# ============================================================================
# 根据模式设置构建目录
# ============================================================================
if [ -n "$USE_MOCK" ]; then
    BUILD_DIR="${SCRIPT_DIR}/build-mock"
else
    BUILD_DIR="${SCRIPT_DIR}/build-real"
fi

# ============================================================================
# 显示配置信息
# ============================================================================
echo "========================================"
echo "aclnnSquareSumV1 算子 ST 测试 (C++)"
echo "========================================"
if [ -n "$USE_MOCK" ]; then
    echo "模式: Mock (CPU golden 验证)"
else
    echo "模式: Real (NPU 执行)"
fi
echo "工作目录: ${SCRIPT_DIR}"
echo "CSV 用例: ${CASE_FILE}"
echo "========================================"
echo ""

# ============================================================================
# 检查依赖
# ============================================================================
echo "检查依赖..."

if ! command -v cmake &> /dev/null; then
    echo "错误: 未找到 cmake"
    exit 1
fi

if ! command -v g++ &> /dev/null; then
    echo "错误: 未找到 g++"
    exit 1
fi

# 检查 CSV 文件
if [ ! -f "${SCRIPT_DIR}/${CASE_FILE}" ]; then
    echo "错误: 测试用例 CSV 文件不存在: ${SCRIPT_DIR}/${CASE_FILE}"
    exit 1
fi

# 检查 AscendCL (Real 模式需要)
if [ -z "$USE_MOCK" ]; then
    if [ -z "$ASCEND_HOME_PATH" ]; then
        echo "警告: 未设置 ASCEND_HOME_PATH 环境变量"
        echo "建议设置: source set_env.sh (in CANN install directory)"
    fi
fi

echo "依赖检查完成"
echo ""

# ============================================================================
# 设置环境变量
# ============================================================================
echo "设置环境变量..."

# 设置 LD_LIBRARY_PATH (Real 模式需要 cust_opapi.so)
CUSTOM_OP_LIB_DIR=""
for candidate_dir in \
    "${ASCEND_HOME_PATH}/opp/vendors/squaresumv1_custom/op_api/lib" \
    "/usr/local/Ascend/cann-8.5.0/opp/vendors/squaresumv1_custom/op_api/lib" \
    "/usr/local/Ascend/opp/vendors/squaresumv1_custom/op_api/lib"; do
    if [ -d "$candidate_dir" ]; then
        CUSTOM_OP_LIB_DIR="$candidate_dir"
        break
    fi
done
if [ -n "$CUSTOM_OP_LIB_DIR" ]; then
    export LD_LIBRARY_PATH=${CUSTOM_OP_LIB_DIR}:${LD_LIBRARY_PATH}
    echo "LD_LIBRARY_PATH: ${CUSTOM_OP_LIB_DIR}"
else
    if [ -z "$USE_MOCK" ]; then
        echo "警告: 未找到 squaresumv1_custom op_api lib 目录"
    fi
fi

echo "环境变量设置完成"
echo ""

# ============================================================================
# 创建构建目录
# ============================================================================
echo "创建构建目录..."
mkdir -p "${BUILD_DIR}"

# ============================================================================
# CMake 配置
# ============================================================================
echo ""
echo "CMake 配置..."
cd "${BUILD_DIR}"

if [ -n "$USE_MOCK" ]; then
    cmake .. -DUSE_MOCK=ON
else
    cmake ..
fi

if [ $? -ne 0 ]; then
    echo "错误: CMake 配置失败"
    exit 1
fi

# ============================================================================
# 编译
# ============================================================================
echo ""
echo "编译测试程序..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "错误: 编译失败"
    exit 1
fi

echo "编译成功"
echo ""

# ============================================================================
# 执行测试
# ============================================================================
echo "========================================"
echo "执行测试"
echo "========================================"

cd "${SCRIPT_DIR}"

# 检查附加用例文件是否存在
EXTRA_ARG=""
if [ -n "$EXTRA_CASE_FILE" ]; then
    if [ -f "${SCRIPT_DIR}/${EXTRA_CASE_FILE}" ]; then
        EXTRA_ARG="${EXTRA_CASE_FILE}"
        echo "附加用例: ${EXTRA_CASE_FILE}"
    else
        echo "警告: 附加用例文件不存在: ${SCRIPT_DIR}/${EXTRA_CASE_FILE}"
    fi
fi

"${BUILD_DIR}/test_aclnn_square_sum_v1" "${CASE_FILE}" ${EXTRA_ARG}

TEST_RESULT=$?

# ============================================================================
# 输出结果
# ============================================================================
echo ""
echo "========================================"
if [ $TEST_RESULT -eq 0 ]; then
    echo "测试结果: PASS"
else
    echo "测试结果: FAIL"
fi
echo "========================================"
echo ""

exit $TEST_RESULT
