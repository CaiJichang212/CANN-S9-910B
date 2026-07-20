/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * \file square_sum_v1.cpp
 * \brief SquareSumV1 kernel entry point (arch22 / Ascend910B)
 *
 * Template parameter:
 *   D_T_X: data type (half / float / bfloat16_t)
 */

#include "square_sum_v1.h"

template <typename D_T_X>
__global__ __aicore__ void square_sum_v1(GM_ADDR input, GM_ADDR result, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SquareSumV1TilingData);
    GET_TILING_DATA_WITH_STRUCT(SquareSumV1TilingData, tilingData, tiling);
    NsSquareSumV1::SquareSumV1<D_T_X> op;
    op.Init(input, result, workspace, &tilingData);
    op.Process();
}
