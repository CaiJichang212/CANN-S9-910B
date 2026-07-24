/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * \file square_sum_v1_infershape.cpp
 * \brief SquareSumV1 shape inference
 *
 * Output shape is determined by axis + keep_dims.  This is deliberately
 * computed here rather than copied from the pre-allocated result: L0/L2
 * callers use InferShape to validate the result contract before launch.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

#include <vector>

using namespace ge;

namespace ops {

static ge::graphStatus InferShape4SquareSumV1(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    gert::Shape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* axisAttr = attrs->GetListInt(0);
    const auto* keepDimsAttr = attrs->GetBool(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, axisAttr);
    OP_CHECK_NULL_WITH_CONTEXT(context, keepDimsAttr);

    const int64_t rank = static_cast<int64_t>(inputShape->GetDimNum());
    std::vector<bool> reduced(static_cast<size_t>(rank), false);
    for (size_t i = 0; i < axisAttr->GetSize(); ++i) {
        int64_t axis = axisAttr->GetData()[i];
        if (axis < -rank || axis >= rank) {
            OP_LOGE(context, "axis %ld is out of range for rank %ld", axis, rank);
            return ge::GRAPH_FAILED;
        }
        if (axis < 0) {
            axis += rank;
        }
        if (reduced[static_cast<size_t>(axis)]) {
            OP_LOGE(context, "duplicate reduction axis %ld", axis);
            return ge::GRAPH_FAILED;
        }
        reduced[static_cast<size_t>(axis)] = true;
    }

    std::vector<int64_t> dims;
    dims.reserve(static_cast<size_t>(rank));
    for (int64_t i = 0; i < rank; ++i) {
        if (reduced[static_cast<size_t>(i)]) {
            if (*keepDimsAttr) {
                dims.push_back(1);
            }
        } else {
            dims.push_back(inputShape->GetDim(i));
        }
    }
    // A full reduction with keep_dims=false is a scalar, represented by a
    // zero-dimensional shape.  Empty axis intentionally leaves input intact.
    outputShape->SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        outputShape->SetDim(i, dims[i]);
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SquareSumV1).InferShape(InferShape4SquareSumV1);

} // namespace ops
