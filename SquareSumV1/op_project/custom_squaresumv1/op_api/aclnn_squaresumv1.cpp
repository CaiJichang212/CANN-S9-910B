/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * \file aclnn_squaresumv1.cpp
 * \brief ACLNN L2 API implementation for SquareSumV1
 *
 * Two-phase interface:
 * 1. aclnnSquareSumV1GetWorkspaceSize - compute workspace, create executor
 * 2. aclnnSquareSumV1 - execute computation
 */

#include "aclnn_squaresumv1.h"
#include "aclnn_square_sum_v1_custom.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/shape_utils.h"

#include <set>

using namespace op;

#define ACLNN_MAX_SHAPE_RANK 8

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16
};

static bool IsDtypeSupported(DataType dtype)
{
    return CheckType(dtype, AICORE_DTYPE_SUPPORT_LIST);
}

static bool CheckNotNull(const aclTensor* input, const aclIntArray* axis,
                         const aclTensor* result, const uint64_t* workspaceSize,
                         const aclOpExecutor** executor)
{
    OP_CHECK_NULL(input, return false);
    OP_CHECK_NULL(axis, return false);
    OP_CHECK_NULL(result, return false);
    OP_CHECK_NULL(workspaceSize, return false);
    OP_CHECK_NULL(executor, return false);
    return true;
}

static bool CheckAxisValid(const aclIntArray* axis, int64_t rank)
{
    // axis values must be in [-rank, rank-1] and no duplicates
    std::set<int64_t> seen;
    for (uint64_t i = 0; i < axis->Size(); i++) {
        int64_t val = (*axis)[i];
        if (val < -rank || val >= rank) {
            return false;
        }
        int64_t normalized = (val < 0) ? (val + rank) : val;
        if (seen.count(normalized) > 0) {
            return false; // duplicate
        }
        seen.insert(normalized);
    }
    return true;
}

static bool CheckDtypeValid(const aclTensor* input, const aclTensor* result)
{
    OP_CHECK_DTYPE_NOT_MATCH(result, input->GetDataType(), return false);
    OP_CHECK(IsDtypeSupported(input->GetDataType()),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Dtype not supported: dtype=%d. Supported: FLOAT16, BFLOAT16, FLOAT.",
                     static_cast<int>(input->GetDataType())),
             return false);
    return true;
}

static bool CheckFormat(const aclTensor* input, const aclTensor* result)
{
    auto formatIn = input->GetStorageFormat();
    auto formatOut = result->GetStorageFormat();
    OP_CHECK(!(IsPrivateFormat(formatIn) || IsPrivateFormat(formatOut)),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "Private format not supported: input=%d, result=%d",
                     static_cast<int>(formatIn), static_cast<int>(formatOut)),
             return false);
    return true;
}

static bool CheckShape(const aclTensor* input)
{
    OP_CHECK_MAX_DIM(input, ACLNN_MAX_SHAPE_RANK, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* input, const aclIntArray* axis,
                               const aclTensor* result, const uint64_t* workspaceSize,
                               const aclOpExecutor** executor)
{
    CHECK_COND(CheckNotNull(input, axis, result, workspaceSize, executor),
               ACLNN_ERR_PARAM_NULLPTR, "CheckNotNull failed");
    CHECK_COND(CheckDtypeValid(input, result), ACLNN_ERR_PARAM_INVALID,
               "CheckDtypeValid failed: input_dtype=%d, result_dtype=%d",
               static_cast<int>(input->GetDataType()), static_cast<int>(result->GetDataType()));
    CHECK_COND(CheckFormat(input, result), ACLNN_ERR_PARAM_INVALID, "CheckFormat failed");
    CHECK_COND(CheckShape(input), ACLNN_ERR_PARAM_INVALID, "CheckShape failed");
    CHECK_COND(CheckAxisValid(axis, static_cast<int64_t>(input->GetViewShape().GetDimNum())),
               ACLNN_ERR_PARAM_INVALID, "CheckAxisValid failed");
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnSquareSumV1GetWorkspaceSize(
    const aclTensor* input,
    const aclIntArray* axis,
    const bool keepDims,
    aclTensor* result,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSquareSumV1, DFX_IN(input), DFX_OUT(result));

    auto ret = CheckParams(input, axis, result, workspaceSize, const_cast<const aclOpExecutor**>(executor));
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // Use the generated private-L0 wrapper.  It registers the exact
    // SquareSumV1Custom op type with Nnopbase and creates the dynamic-kernel
    // executor directly; our public API remains aclnnSquareSumV1*.
    return aclnnSquareSumV1CustomGetWorkspaceSize(input, axis, keepDims, result,
                                                   workspaceSize, executor);
}

extern "C" aclnnStatus aclnnSquareSumV1(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSquareSumV1);
    return aclnnSquareSumV1Custom(workspace, workspaceSize, executor, stream);
}
