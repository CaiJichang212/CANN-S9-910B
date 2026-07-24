/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * \file square_sum_v1_tiling_key.h
 * \brief SquareSumV1 TilingKey definitions (arch22 / Ascend910B)
 *
 * TilingKey encoding:
 *   D_T_X: input/output data type (half / float / bfloat16_t)
 *
 * For iteration 1, only half (float16) is supported.
 */

#ifndef __SQUARE_SUM_V1_TILING_KEY_H__
#define __SQUARE_SUM_V1_TILING_KEY_H__

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(SquareSumV1Custom,
    ASCENDC_TPL_DATATYPE_DECL(D_T_X, C_DT_FLOAT16, C_DT_FLOAT, C_DT_BF16, ASCENDC_TPL_INPUT(0))
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT16)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_FLOAT)
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(D_T_X, C_DT_BF16)
    ),
);

#endif // __SQUARE_SUM_V1_TILING_KEY_H__
