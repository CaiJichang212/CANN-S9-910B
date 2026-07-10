/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * \file squaresumv1.h
 * \brief ACLNN L0 API header for SquareSumV1
 */

#ifndef OP_API_INC_LEVEL0_SQUARESUMV1_H_
#define OP_API_INC_LEVEL0_SQUARESUMV1_H_

#include "opdev/op_executor.h"

namespace l0op {

const aclTensor* SquareSumV1(const aclTensor* input, const aclIntArray* axis,
                              bool keepDims, const aclTensor* result, aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_SQUARESUMV1_H_
