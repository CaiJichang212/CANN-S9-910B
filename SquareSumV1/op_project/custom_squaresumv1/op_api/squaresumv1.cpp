/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * \file squaresumv1.cpp
 * \brief ACLNN L0 API implementation for SquareSumV1
 *
 * L0 API: kernel dispatch for SquareSumV1
 * The result tensor is pre-allocated by the caller (framework side).
 */

#include "squaresumv1.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(SquareSumV1);

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_BF16
};

static bool IsAiCoreSupport(const aclTensor* input)
{
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    OP_CHECK(npuArch == NpuArch::DAV_2201,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "SquareSumV1 not supported on this platform: npuArch=%d.",
                     static_cast<int>(npuArch)),
             return false);
    OP_CHECK(CheckType(input->GetDataType(), AICORE_DTYPE_SUPPORT_LIST),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                     "SquareSumV1 not supported: dtype=%d. Supported: FLOAT16, BFLOAT16, FLOAT.",
                     static_cast<int>(input->GetDataType())),
             return false);
    return true;
}

static const aclTensor* SquareSumV1AiCore(const aclTensor* input,
                                           const aclIntArray* axis,
                                           bool keepDims,
                                           const aclTensor* result,
                                           aclOpExecutor* executor)
{
    L0_DFX(SquareSumV1AiCore, input, result);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(SquareSumV1,
        OP_INPUT(input),
        OP_ATTR(axis, keepDims),
        OP_OUTPUT(result));
    OP_CHECK(
        ret == ACLNN_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SquareSumV1AiCore failed."),
        return nullptr);
    return result;
}

const aclTensor* SquareSumV1(const aclTensor* input, const aclIntArray* axis,
                              bool keepDims, const aclTensor* result, aclOpExecutor* executor)
{
    OP_CHECK(IsAiCoreSupport(input),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "IsAiCoreSupport check failed."),
             return nullptr);

    return SquareSumV1AiCore(input, axis, keepDims, result, executor);
}

} // namespace l0op
