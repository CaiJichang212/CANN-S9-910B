/** @file transpose.cpp
 * Host-side validation, shape inference, and data-movement tiling.
 */
#include "transpose_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <limits>

namespace optiling {
namespace {
constexpr uint32_t kMaxDim = 8;
constexpr uint32_t kStridedTileElems = 2048; // <= DataCopyPad blockCount limit
constexpr uint32_t kMaxDmaBlocks = 4095;
constexpr uint32_t kVectorRows = 16;
constexpr uint64_t kTransposeSlotBytes = 32 * 1024;

enum TransposeMode : uint32_t { COPY_CONTIG = 0, ROTATE_2D = 1, STRIDED_ROWS = 2 };

uint32_t DTypeToSize(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT16: return 2;
        case ge::DT_FLOAT:
        case ge::DT_INT32: return 4;
        case ge::DT_INT8: return 1;
        default: return 0;
    }
}

bool MulNoOverflow(uint64_t lhs, uint64_t rhs, uint64_t &out)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    out = lhs * rhs;
    return true;
}

bool NormalizeDims(const gert::RuntimeAttrs *attrs, uint32_t ndim, uint32_t dims[kMaxDim])
{
    if (attrs == nullptr) {
        return false;
    }
    const auto *dimsVec = attrs->GetListInt(0);
    if (dimsVec == nullptr || dimsVec->GetSize() != ndim) {
        return false;
    }
    const int64_t *data = dimsVec->GetData();
    bool seen[kMaxDim] = {false};
    for (uint32_t i = 0; i < ndim; ++i) {
        int64_t axis = data[i];
        if (axis < 0) {
            axis += static_cast<int64_t>(ndim);
        }
        if (axis < 0 || axis >= static_cast<int64_t>(ndim) || seen[axis]) {
            return false;
        }
        dims[i] = static_cast<uint32_t>(axis);
        seen[axis] = true;
    }
    return true;
}

bool IsGroupRotation(const uint32_t dims[kMaxDim], uint32_t ndim, uint32_t &prefix, uint32_t &split)
{
    prefix = 0;
    while (prefix < ndim && dims[prefix] == prefix) {
        ++prefix;
    }
    if (prefix == ndim) {
        return false;
    }
    for (split = prefix + 1; split < ndim; ++split) {
        bool match = true;
        uint32_t out = prefix;
        for (uint32_t axis = split; axis < ndim; ++axis, ++out) {
            match = match && dims[out] == axis;
        }
        for (uint32_t axis = prefix; axis < split; ++axis, ++out) {
            match = match && dims[out] == axis;
        }
        if (match) {
            return true;
        }
    }
    return false;
}

} // namespace

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *xStorageShape = context->GetInputShape(0);
    if (xStorageShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto &xShape = xStorageShape->GetStorageShape();
    const uint32_t ndim = static_cast<uint32_t>(xShape.GetDimNum());
    if (ndim == 0 || ndim > kMaxDim) {
        return ge::GRAPH_FAILED;
    }

    uint32_t dims[kMaxDim] = {0};
    if (!NormalizeDims(context->GetAttrs(), ndim, dims)) {
        return ge::GRAPH_FAILED;
    }

    uint64_t shape[kMaxDim] = {0};
    uint64_t stride[kMaxDim] = {0};
    uint64_t total = 1;
    for (uint32_t i = 0; i < ndim; ++i) {
        const int64_t dim = xShape.GetDim(i);
        if (dim <= 0 || !MulNoOverflow(total, static_cast<uint64_t>(dim), total)) {
            return ge::GRAPH_FAILED;
        }
        shape[i] = static_cast<uint64_t>(dim);
    }
    stride[ndim - 1] = 1;
    for (int32_t i = static_cast<int32_t>(ndim) - 2; i >= 0; --i) {
        if (!MulNoOverflow(stride[i + 1], shape[i + 1], stride[i])) {
            return ge::GRAPH_FAILED;
        }
    }

    const gert::Tensor *input = context->GetInputTensor(0);
    const uint32_t dtypeSize = input == nullptr ? 0 : DTypeToSize(input->GetDataType());
    if (dtypeSize == 0) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t rowWidth = shape[dims[ndim - 1]];
    const uint64_t srcInnerStride = stride[dims[ndim - 1]];
    const uint64_t numRows = total / rowWidth;
    uint64_t outerOutShape[kMaxDim] = {0};
    uint64_t outerSrcStride[kMaxDim] = {0};
    for (uint32_t i = 0; i + 1 < ndim; ++i) {
        outerOutShape[i] = shape[dims[i]];
        outerSrcStride[i] = stride[dims[i]];
    }

    uint32_t prefix = 0;
    uint32_t split = 0;
    const bool isRotation = IsGroupRotation(dims, ndim, prefix, split);
    uint64_t transBatch = 1;
    uint64_t transM = 0;
    uint64_t transN = 0;
    if (isRotation) {
        for (uint32_t i = 0; i < prefix; ++i) {
            MulNoOverflow(transBatch, shape[i], transBatch);
        }
        transM = 1;
        for (uint32_t i = prefix; i < split; ++i) {
            MulNoOverflow(transM, shape[i], transM);
        }
        transN = 1;
        for (uint32_t i = split; i < ndim; ++i) {
            MulNoOverflow(transN, shape[i], transN);
        }
    }

    TransposeMode mode = srcInnerStride == 1 ? COPY_CONTIG : STRIDED_ROWS;
    // int8 still uses the generic DMA path.  It avoids a scalar fallback and
    // is safer than issuing B8 transforms for sub-32B tails.
    if (mode != COPY_CONTIG && isRotation && dtypeSize != 1 && transM > 1 && transN > 1) {
        mode = ROTATE_2D;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreCount = platform.GetCoreNumAiv();
    if (coreCount == 0) {
        coreCount = 1;
    }

    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    const uint64_t safeUbBytes = ubBytes > 4096 ? ubBytes - 4096 : ubBytes;
    const uint64_t copyBudgetBytes = std::max<uint64_t>(32, safeUbBytes / 4);
    uint32_t copyTileElems = static_cast<uint32_t>(std::min<uint64_t>(rowWidth, copyBudgetBytes / dtypeSize));
    copyTileElems = std::max(1u, copyTileElems);

    uint32_t tileN = 0;
    if (mode == ROTATE_2D) {
        const uint32_t granularity = dtypeSize == 2 ? 16 : 8;
        const uint64_t maxTileN = std::max<uint64_t>(granularity, kTransposeSlotBytes / (kVectorRows * dtypeSize));
        tileN = static_cast<uint32_t>(std::min<uint64_t>(transN, maxTileN));
        tileN = std::max(granularity, (tileN / granularity) * granularity);
        // A single vector instruction accepts no more than 255 repeats.
        tileN = std::min(tileN, granularity * 255u);
    }
    uint64_t workUnits = numRows;
    if (mode == ROTATE_2D) {
        const uint64_t mTiles = (transM + kVectorRows - 1) / kVectorRows;
        const uint64_t nTiles = (transN + tileN - 1) / tileN;
        workUnits = transBatch * mTiles * nTiles;
    }
    const uint32_t blockDim = static_cast<uint32_t>(std::min<uint64_t>(coreCount, std::max<uint64_t>(1, workUnits)));

    TransposeTilingData tiling;
    tiling.set_mode(mode);
    tiling.set_dtypeSize(dtypeSize);
    tiling.set_blockDim(blockDim);
    tiling.set_outerCount(ndim - 1);
    tiling.set_total(total);
    tiling.set_rowWidth(rowWidth);
    tiling.set_numRows(numRows);
    tiling.set_srcInnerStride(srcInnerStride);
    tiling.set_outerOutShape(outerOutShape);
    tiling.set_outerSrcStride(outerSrcStride);
    tiling.set_copyTileElems(copyTileElems);
    tiling.set_stridedTileElems(std::min(kStridedTileElems, kMaxDmaBlocks));
    tiling.set_transBatch(transBatch);
    tiling.set_transM(transM);
    tiling.set_transN(transN);
    tiling.set_tileM(mode == ROTATE_2D ? kVectorRows : 0);
    tiling.set_tileN(tileN);

    context->SetBlockDim(blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
namespace {
bool NormalizeInferDims(const gert::RuntimeAttrs *attrs, size_t ndim, uint32_t dims[8])
{
    if (attrs == nullptr) return false;
    const auto *vec = attrs->GetListInt(0);
    if (vec == nullptr || vec->GetSize() != ndim || ndim == 0 || ndim > 8) return false;
    bool seen[8] = {false};
    const int64_t *data = vec->GetData();
    for (size_t i = 0; i < ndim; ++i) {
        int64_t axis = data[i] < 0 ? data[i] + static_cast<int64_t>(ndim) : data[i];
        if (axis < 0 || axis >= static_cast<int64_t>(ndim) || seen[axis]) return false;
        dims[i] = static_cast<uint32_t>(axis);
        seen[axis] = true;
    }
    return true;
}
}

static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x = context->GetInputShape(0);
    gert::Shape *y = context->GetOutputShape(0);
    const size_t ndim = x->GetDimNum();
    uint32_t dims[8] = {0};
    if (!NormalizeInferDims(context->GetAttrs(), ndim, dims)) return GRAPH_FAILED;
    y->SetDimNum(ndim);
    for (size_t i = 0; i < ndim; ++i) y->SetDim(i, x->GetDim(dims[i]));
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Transpose : public OpDef {
public:
    explicit Transpose(const char *name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dims").ListInt();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(Transpose);
} // namespace ops
