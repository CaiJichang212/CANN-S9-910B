/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * \file squaresumv1_infershape.cpp
 * \brief SquareSumV1 shape inference
 *
 * Output shape is determined by axis + keep_dims.
 * Since result tensor is pre-allocated by the caller, InferShape validates
 * that the output shape matches the expected reduction result.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4SquareSumV1(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    gert::Shape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    // Output shape is pre-allocated by the caller (framework side).
    // The InferShape here just copies from the pre-allocated output shape.
    // In the aclnn path, the output shape is set by the L0 API before calling kernel.
    *outputShape = *inputShape;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SquareSumV1).InferShape(InferShape4SquareSumV1);

} // namespace ops
