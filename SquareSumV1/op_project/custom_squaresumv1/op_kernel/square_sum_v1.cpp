/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * \file square_sum_v1.cpp
 * \brief SquareSumV1 kernel entry point (arch22 / Ascend910B)
 *
 * Template parameter:
 *   D_T_X: data type (half / float / bfloat16_t)
 */

#include "square_sum_v1.h"

// CANN derives the kernel entry from the private L0 type
// SquareSumV1Custom: square_sum_v1_custom.  It must stay aligned with the
// OP_TYPE in CMakeLists.txt, while the exported ACLNN API keeps its original
// SquareSumV1 name.
template <typename D_T_X>
__global__ __aicore__ void square_sum_v1_custom(GM_ADDR input, GM_ADDR result, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SquareSumV1TilingData);
    GET_TILING_DATA_WITH_STRUCT(SquareSumV1TilingData, tilingData, tiling);
    // Modes 4/5 contain a hard all-AIV phase barrier.  The mix AIV 1:0 task
    // type is required by SyncAll on DAV_2201; the ordinary paths remain
    // vector-only code and do not take the barrier.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    NsSquareSumV1::SquareSumV1<D_T_X> op;
    op.Init(input, result, workspace, &tilingData);
    op.Process();
}
