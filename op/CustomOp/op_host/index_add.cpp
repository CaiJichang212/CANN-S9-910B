/**
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "index_add_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace optiling {

constexpr uint32_t AICORE_NUM = 20;
constexpr uint32_t COPY_TILE_BYTES = 16U * 1024U;
constexpr uint32_t ATOMIC_TILE_BYTES = 16U * 1024U;
// 320B leaves at least a 256B atomic body even in the general unaligned
// middle-range calculation in the device code.
constexpr uint32_t ATOMIC_THRESHOLD_BYTES = 320U;

enum class IndexAddDtype : uint32_t {
    FLOAT = 0,
    BF16 = 1,
    HALF = 2,
    INT32 = 3,
    INT8 = 4,
};

static uint64_t CeilDiv(uint64_t x, uint64_t y)
{
    return (x + y - 1U) / y;
}

static bool FitsU32(uint64_t value)
{
    return value <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max());
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    IndexAddCustomTilingData tiling;
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) return ge::GRAPH_FAILED;
    const int64_t *dimAttrPtr = attrs->GetInt(0);
    if (dimAttrPtr == nullptr) return ge::GRAPH_FAILED;

    const gert::StorageShape *selfShape = context->GetInputShape(0);
    const gert::StorageShape *indexShape = context->GetInputShape(1);
    const gert::StorageShape *sourceShape = context->GetInputShape(2);
    if (selfShape == nullptr || indexShape == nullptr || sourceShape == nullptr) return ge::GRAPH_FAILED;
    const auto &selfStorage = selfShape->GetStorageShape();
    const uint32_t dimNum = static_cast<uint32_t>(selfStorage.GetDimNum());
    if (dimNum == 0) return ge::GRAPH_FAILED;

    int64_t dim = *dimAttrPtr;
    if (dim < 0) dim += static_cast<int64_t>(dimNum);
    if (dim < 0 || dim >= static_cast<int64_t>(dimNum)) return ge::GRAPH_FAILED;
    const uint32_t udim = static_cast<uint32_t>(dim);

    uint64_t beforeDimSize64 = 1;
    for (uint32_t i = 0; i < udim; ++i) {
        beforeDimSize64 *= static_cast<uint64_t>(selfStorage.GetDim(i));
    }
    const uint64_t dimLen64 = static_cast<uint64_t>(selfStorage.GetDim(udim));
    uint64_t afterDimSize64 = 1;
    for (uint32_t i = udim + 1; i < dimNum; ++i) {
        afterDimSize64 *= static_cast<uint64_t>(selfStorage.GetDim(i));
    }
    const auto &indexStorage = indexShape->GetStorageShape();
    if (indexStorage.GetDimNum() != 1) return ge::GRAPH_FAILED;
    const uint64_t indexLen64 = static_cast<uint64_t>(indexStorage.GetDim(0));

    uint32_t dtypeSize = 0;
    uint32_t dtypeCode = static_cast<uint32_t>(IndexAddDtype::FLOAT);
    const auto selfDescPtr = context->GetInputDesc(0);
    if (selfDescPtr == nullptr) return ge::GRAPH_FAILED;
    switch (selfDescPtr->GetDataType()) {
        case ge::DT_FLOAT:
            dtypeSize = 4; dtypeCode = static_cast<uint32_t>(IndexAddDtype::FLOAT); break;
        case ge::DT_BF16:
            dtypeSize = 2; dtypeCode = static_cast<uint32_t>(IndexAddDtype::BF16); break;
        case ge::DT_FLOAT16:
            dtypeSize = 2; dtypeCode = static_cast<uint32_t>(IndexAddDtype::HALF); break;
        case ge::DT_INT32:
            dtypeSize = 4; dtypeCode = static_cast<uint32_t>(IndexAddDtype::INT32); break;
        case ge::DT_INT8:
            dtypeSize = 1; dtypeCode = static_cast<uint32_t>(IndexAddDtype::INT8); break;
        default:
            return ge::GRAPH_FAILED;
    }

    // The ABI stores extents as uint32_t, but all products/byte counts and
    // device offsets remain uint64_t.
    if (!FitsU32(beforeDimSize64) || !FitsU32(dimLen64) || !FitsU32(afterDimSize64) ||
        !FitsU32(indexLen64) || indexLen64 == 0 || dimLen64 == 0 || afterDimSize64 == 0 ||
        beforeDimSize64 == 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t selfBytes = beforeDimSize64 * dimLen64 * afterDimSize64 * dtypeSize;
    const uint64_t sourceBytes = beforeDimSize64 * indexLen64 * afterDimSize64 * dtypeSize;
    const uint64_t maxBytes = std::max(selfBytes, sourceBytes);
    const uint32_t usedCoreNum = static_cast<uint32_t>(
        std::min<uint64_t>(AICORE_NUM, std::max<uint64_t>(1, CeilDiv(maxBytes, COPY_TILE_BYTES))));

    const uint64_t vectorBytes = afterDimSize64 * dtypeSize;
    // Plain DMA atomic requires a 32B count and aligned source/destination.
    // A vector-size multiple of 32 guarantees that every [row, i] vector has
    // the same aligned start in both source and output.  Other layouts take
    // the ownership path, which is exact for arbitrary remainders/indexes.
    // BF16 index_add is observably order-sensitive because each update is
    // rounded back to BF16.  DMA atomic completion order is intentionally
    // unspecified, so it cannot meet the reference's per-index update
    // semantics for heavily repeated indices.  Keep BF16 on deterministic
    // index ownership; all other native atomic types remain on the fast path.
    const bool atomicEnabled = selfDescPtr->GetDataType() != ge::DT_BF16 &&
        vectorBytes >= ATOMIC_THRESHOLD_BYTES && (vectorBytes % 32U == 0U);
    const uint32_t rmwTileLen = ATOMIC_TILE_BYTES / dtypeSize;

    tiling.set_dim(udim);
    tiling.set_beforeDimSize(static_cast<uint32_t>(beforeDimSize64));
    tiling.set_dimLen(static_cast<uint32_t>(dimLen64));
    tiling.set_afterDimSize(static_cast<uint32_t>(afterDimSize64));
    tiling.set_indexLen(static_cast<uint32_t>(indexLen64));
    tiling.set_dtypeSize(dtypeSize);
    tiling.set_dtype(dtypeCode);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_scatterCoreNum(usedCoreNum);
    tiling.set_atomicEnabled(atomicEnabled ? 1U : 0U);
    tiling.set_copyTileBytes(COPY_TILE_BYTES);
    tiling.set_atomicTileBytes(ATOMIC_TILE_BYTES);
    tiling.set_atomicThresholdBytes(ATOMIC_THRESHOLD_BYTES);
    tiling.set_rmwTileLen(rmwTileLen);

    context->SetBlockDim(usedCoreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *selfShape = context->GetInputShape(0);
    if (selfShape == nullptr) return ge::GRAPH_FAILED;
    *context->GetOutputShape(0) = *selfShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {
class IndexAdd : public OpDef {
public:
    explicit IndexAdd(const char *name) : OpDef(name)
    {
        this->Input("self").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("index").ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("source").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dim").Int();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};

OP_ADD(IndexAdd);
}  // namespace ops
