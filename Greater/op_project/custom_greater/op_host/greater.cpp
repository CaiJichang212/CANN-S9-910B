/**
 * @file greater.cpp
 *
 * Host-side definition, infer-shape/dtype and tiling for the Greater
 * (torch.gt) custom operator. Element-wise x > y with NumPy-style broadcast;
 * output is bool. Target: Ascend 910B (ascend910b).
 */
#include "greater_tiling.h"
#include "register/op_def_registry.h"

#include <algorithm>
#include <cstdint>

namespace optiling {

// Pad a shape to `ndim` dimensions by prepending 1s. Returns the aligned dims
// in `out` (size ndim), index 0 = outermost.
static void AlignShape(const gert::Shape& s, uint32_t ndim, int64_t* out)
{
    uint32_t dn = s.GetDimNum();
    uint32_t pad = (ndim > dn) ? (ndim - dn) : 0;
    uint32_t idx = 0;
    for (uint32_t i = 0; i < pad; ++i) {
        out[idx++] = 1;
    }
    for (uint32_t i = 0; i < dn && idx < ndim; ++i) {
        out[idx++] = s.GetDim(i);
    }
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    GreaterTilingData tiling;

    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* yShape = context->GetInputShape(1);

    uint32_t xNdim = xShape->GetStorageShape().GetDimNum();
    uint32_t yNdim = yShape->GetStorageShape().GetDimNum();
    uint32_t ndim = std::max(xNdim, yNdim);
    if (ndim == 0) {
        ndim = 1; // scalar -> treat as [1]
    }

    int64_t sx[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t sy[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    int64_t sz[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    AlignShape(xShape->GetStorageShape(), ndim, sx);
    AlignShape(yShape->GetStorageShape(), ndim, sy);

    uint64_t totalSize = 1;
    for (uint32_t i = 0; i < ndim; ++i) {
        sz[i] = std::max(sx[i], sy[i]);
        totalSize *= static_cast<uint64_t>(sz[i]);
    }

    // ---- broadcast decomposition ----
    // innerSize = maximal trailing suffix where both operands are non-broadcast
    // (sx==sy), but always at least the innermost dim (which may itself be a
    // broadcast dim, handled as a per-segment scalar).
    uint8_t bcastMode = 0; // 0:both full, 1:x scalar, 2:y scalar
    int last = static_cast<int>(ndim) - 1;
    if (sx[last] != sy[last]) {
        // innermost dim is broadcast for one operand
        if (sx[last] == 1) {
            bcastMode = 1;
        } else {
            bcastMode = 2;
        }
    }

    int64_t innerSize = sz[last];
    int k = last - 1;
    if (bcastMode == 0) {
        // extend upward over trailing non-broadcast dims
        while (k >= 0 && sx[k] == sy[k]) {
            innerSize *= sz[k];
            --k;
        }
    }
    int outerDim = k + 1; // number of outer dims [0..k]
    uint64_t outerSize = 1;
    for (int d = 0; d <= k; ++d) {
        outerSize *= static_cast<uint64_t>(sz[d]);
    }

    // Per-operand memory strides (elements) for the outer dims. Stride is 0 on
    // broadcast dims so the base pointer does not advance there.
    auto memStride = [&](const int64_t* s, int d) -> uint32_t {
        if (s[d] == 1) {
            return 0; // broadcast dim
        }
        int64_t stride = 1;
        for (int j = d + 1; j <= last; ++j) {
            stride *= s[j];
        }
        return static_cast<uint32_t>(stride);
    };

    // Block dim: scale with data size, cap at the 20 AI cores of 910B4.
    uint32_t blockDim = 1;
    if (totalSize > 0) {
        blockDim = static_cast<uint32_t>((totalSize + 255) / 256);
        blockDim = std::min(20u, std::max(1u, blockDim));
    }

    context->SetBlockDim(blockDim);
    tiling.set_totalSize(static_cast<uint32_t>(totalSize));
    tiling.set_blockDim(blockDim);
    tiling.set_innerSize(static_cast<uint32_t>(innerSize));
    tiling.set_outerSize(static_cast<uint32_t>(outerSize));
    tiling.set_bcastMode(static_cast<uint32_t>(bcastMode));
    tiling.set_outerDim(static_cast<uint32_t>(outerDim));

    uint32_t outerShapeArr[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t xStrideArr[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t yStrideArr[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int d = 0; d < 8; ++d) {
        if (d < outerDim) {
            outerShapeArr[d] = static_cast<uint32_t>(sz[d]);
            xStrideArr[d] = memStride(sx, d);
            yStrideArr[d] = memStride(sy, d);
        }
    }
    tiling.set_outerShape(outerShapeArr);
    tiling.set_xStride(xStrideArr);
    tiling.set_yStride(yStrideArr);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
// Broadcast the two input shapes into the output shape.
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    const gert::Shape* yShape = context->GetInputShape(1);
    gert::Shape* zShape = context->GetOutputShape(0);

    uint32_t xNdim = xShape->GetDimNum();
    uint32_t yNdim = yShape->GetDimNum();
    uint32_t ndim = std::max(xNdim, yNdim);
    if (ndim == 0) {
        ndim = 1;
    }
    zShape->SetDimNum(ndim);
    for (uint32_t i = 0; i < ndim; ++i) {
        int64_t dx = (i + xNdim < ndim) ? 1 : xShape->GetDim(i - (ndim - xNdim));
        int64_t dy = (i + yNdim < ndim) ? 1 : yShape->GetDim(i - (ndim - yNdim));
        zShape->SetDim(i, std::max(dx, dy));
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    // Greater always outputs bool, regardless of the input dtype.
    context->SetOutputDataType(0, ge::DT_BOOL);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Greater : public OpDef {
public:
    explicit Greater(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Greater);
} // namespace ops
