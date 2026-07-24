/**
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "index_add_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace optiling {

constexpr uint32_t COPY_TILE_BYTES = 16U * 1024U;
constexpr uint32_t ATOMIC_TILE_BYTES = 16U * 1024U;
constexpr uint32_t ATOMIC_THRESHOLD_BYTES = 256U;
constexpr uint32_t INDEX_CHUNK_LEN = 1024U;
constexpr uint32_t POSITION_CHUNK_LEN = 1024U;
constexpr uint32_t DMA_ALIGN_BYTES = 32U;

enum class IndexAddPath : uint32_t {
    ATOMIC = 0,
    OWNER = 1,
};

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

static bool SafeMul(uint64_t a, uint64_t b, uint64_t &result)
{
    if (a != 0U && b > std::numeric_limits<uint64_t>::max() / a) return false;
    result = a * b;
    return true;
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

    // index_add accepts a rank-1 int32 index.  Source must match self on
    // every dimension except dim, where its extent equals index.size(0).
    const auto &sourceStorage = sourceShape->GetStorageShape();
    if (sourceStorage.GetDimNum() != dimNum) return ge::GRAPH_FAILED;
    for (uint32_t i = 0; i < dimNum; ++i) {
        const uint64_t expected = i == udim ? indexLen64 : static_cast<uint64_t>(selfStorage.GetDim(i));
        if (static_cast<uint64_t>(sourceStorage.GetDim(i)) != expected) return ge::GRAPH_FAILED;
    }

    uint32_t dtypeSize = 0;
    uint32_t dtypeCode = static_cast<uint32_t>(IndexAddDtype::FLOAT);
    const auto selfDescPtr = context->GetInputDesc(0);
    const auto indexDescPtr = context->GetInputDesc(1);
    const auto sourceDescPtr = context->GetInputDesc(2);
    if (selfDescPtr == nullptr || indexDescPtr == nullptr || sourceDescPtr == nullptr ||
        indexDescPtr->GetDataType() != ge::DT_INT32 ||
        sourceDescPtr->GetDataType() != selfDescPtr->GetDataType()) return ge::GRAPH_FAILED;
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
    uint64_t selfElems = 0;
    uint64_t sourceElems = 0;
    uint64_t selfBytes = 0;
    uint64_t sourceBytes = 0;
    if (!SafeMul(beforeDimSize64, dimLen64, selfElems) ||
        !SafeMul(selfElems, afterDimSize64, selfElems) ||
        !SafeMul(selfElems, dtypeSize, selfBytes) ||
        !SafeMul(beforeDimSize64, indexLen64, sourceElems) ||
        !SafeMul(sourceElems, afterDimSize64, sourceElems) ||
        !SafeMul(sourceElems, dtypeSize, sourceBytes)) return ge::GRAPH_FAILED;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t platformCoreNum = platform.GetCoreNumAiv();
    if (platformCoreNum == 0U) platformCoreNum = 1U;

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
    const uint32_t kTile = std::max(1U, ATOMIC_TILE_BYTES / dtypeSize);
    const uint32_t indexChunkLen = static_cast<uint32_t>(std::min<uint64_t>(INDEX_CHUNK_LEN, indexLen64));
    const uint32_t positionChunkLen = static_cast<uint32_t>(std::min<uint64_t>(POSITION_CHUNK_LEN, indexLen64));
    // Each source vector is an independent atomic work item.  Index is still
    // DMA-cached in chunks on device, but using chunk count here would launch
    // only a few cores for B=1/M≈2K workloads (for example two 1024-element
    // chunks), defeating the 40-AIV fast path.
    const uint64_t atomicWork = beforeDimSize64 * indexLen64;
    const uint64_t ownerWork = beforeDimSize64 * dimLen64 * CeilDiv(afterDimSize64, kTile);
    const uint64_t work = atomicEnabled ? atomicWork : ownerWork;
    const uint32_t usedCoreNum = static_cast<uint32_t>(
        std::min<uint64_t>(platformCoreNum, std::max<uint64_t>(1U, work)));

    uint64_t workspaceElems = 0;
    uint64_t workspaceBytes = 0;
    if (!atomicEnabled) {
        if (!SafeMul(2U, dimLen64, workspaceElems) ||
            workspaceElems > std::numeric_limits<uint64_t>::max() - indexLen64 - 1U) return ge::GRAPH_FAILED;
        workspaceElems += indexLen64 + 1U;
        if (!SafeMul(workspaceElems, sizeof(int32_t), workspaceBytes) ||
            workspaceBytes > std::numeric_limits<uint64_t>::max() - (DMA_ALIGN_BYTES - 1U)) {
            return ge::GRAPH_FAILED;
        }
        // DataCopyPad may align a GM read down/up to a 32B DMA block.  Keep
        // padding after positions so the final workspace read cannot cross
        // the allocator-visible end of this buffer.
        workspaceBytes = (workspaceBytes + DMA_ALIGN_BYTES - 1U) & ~(static_cast<uint64_t>(DMA_ALIGN_BYTES) - 1U);
    }

    tiling.set_dim(udim);
    tiling.set_beforeDimSize(static_cast<uint32_t>(beforeDimSize64));
    tiling.set_dimLen(static_cast<uint32_t>(dimLen64));
    tiling.set_afterDimSize(static_cast<uint32_t>(afterDimSize64));
    tiling.set_indexLen(static_cast<uint32_t>(indexLen64));
    tiling.set_dtypeSize(dtypeSize);
    tiling.set_dtype(dtypeCode);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_scatterCoreNum(usedCoreNum);
    tiling.set_path(static_cast<uint32_t>(atomicEnabled ? IndexAddPath::ATOMIC : IndexAddPath::OWNER));
    tiling.set_atomicEnabled(atomicEnabled ? 1U : 0U);
    tiling.set_copyTileBytes(COPY_TILE_BYTES);
    tiling.set_atomicTileBytes(ATOMIC_TILE_BYTES);
    tiling.set_kTile(kTile);
    tiling.set_targetGroupSize(1U);
    tiling.set_indexChunkLen(indexChunkLen);
    tiling.set_positionChunkLen(positionChunkLen);
    tiling.set_workspaceBytes(workspaceBytes);

    context->SetBlockDim(usedCoreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(workspaceBytes);
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
