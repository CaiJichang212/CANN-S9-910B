/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * \file squaresumv1_proto.h
 * \brief GE graph operator registration for SquareSumV1
 */

#ifndef OPS_OP_PROTO_INC_SQUARESUMV1_H_
#define OPS_OP_PROTO_INC_SQUARESUMV1_H_

#include "graph/operator_reg.h"
#include "graph/types.h"

namespace ge {

/**
 * @brief Computes sum(square(x), dim=axis, keepdim=keep_dims)
 * @par Inputs:
 *  input: A ND Tensor of type FLOAT16, BFLOAT16, or FLOAT.
 * @par Attributes:
 *  axis: A list of int64 specifying reduction axes. Supports negative indices.
 *  keep_dims: A bool. If true, retains reduced dimensions with size 1.
 * @par Outputs:
 *  result: A ND Tensor of same type as input.
 */
REG_OP(SquareSumV1)
    .INPUT(input, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(result, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(axis, ListInt)
    .ATTR(keep_dims, Bool, false)
    .OP_END_FACTORY_REG(SquareSumV1)

} // namespace ge

#endif // OPS_OP_PROTO_INC_SQUARESUMV1_H_
