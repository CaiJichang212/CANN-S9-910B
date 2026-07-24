/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * \file aclnn_square_sum_v1.h
 * \brief Public ACLNN API for SquareSumV1.
 *
 * SquareSumV1Custom is an implementation-only L0 type used to avoid a CANN
 * built-in name collision.  This header deliberately preserves the submitted
 * ACLNN interface.
 */

#ifndef ACLNN_SQUARE_SUM_V1_H_
#define ACLNN_SQUARE_SUM_V1_H_

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

#endif // ACLNN_SQUARE_SUM_V1_H_
