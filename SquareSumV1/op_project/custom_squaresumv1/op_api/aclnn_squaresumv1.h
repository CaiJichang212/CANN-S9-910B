/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * \file aclnn_squaresumv1.h
 * \brief ACLNN L2 API header for SquareSumV1
 */

#ifndef ACLNN_SQUARESUMV1_H_
#define ACLNN_SQUARESUMV1_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

ACLNN_API aclnnStatus aclnnSquareSumV1GetWorkspaceSize(
    const aclTensor *input,
    const aclIntArray *axis,
    const bool keepDims,
    aclTensor *result,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

ACLNN_API aclnnStatus aclnnSquareSumV1(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_SQUARESUMV1_H_
