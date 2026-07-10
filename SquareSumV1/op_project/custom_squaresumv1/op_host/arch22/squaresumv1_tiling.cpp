/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * \file squaresumv1_tiling.cpp
 * \brief SquareSumV1 tiling implementation (arch22 / Ascend910B)
 *
 * Iteration 2: Multi-TilingKey integration
 *   Key=0 AR_FULLLOAD:  tail-axis reduce, full load (existing, refined)
 *   Key=1 AR_COLSPLIT:  tail-axis reduce, column chunk + fp32 accumulator
 *   Key=2 ARA_FULLLOAD: non-tail-axis reduce, Pattern::Reduce::RA full load
 *   Key=3 ARA_ROWSPLIT: non-tail-axis reduce, R-chunk + cross-chunk accumulation
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch22/squaresumv1_tiling_data.h"
#include "../../op_kernel/arch22/squaresumv1_tiling_key.h"

#include <algorithm>
#include <vector>
#include <set>

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr size_t WORKSPACE_NUM = 1;
constexpr uint32_t UB_SIZE_910B = 192 * 1024; // 192KB total, use 184KB as safe limit
constexpr uint32_t UB_SAFE_LIMIT = 184 * 1024;

// Get platform info
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t* ubSize, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        *ubSize = UB_SIZE_910B;
        *coreNum = 20;
        return ge::GRAPH_SUCCESS;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    if (*coreNum == 0) {
        *coreNum = 20;
    }
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, *ubSize);
    if (*ubSize == 0) {
        *ubSize = UB_SIZE_910B;
    }
    return ge::GRAPH_SUCCESS;
}

// Axis preprocessing: normalize negative indices and sort
static std::vector<int64_t> NormalizeAxis(const std::vector<int64_t>& axis, int64_t rank)
{
    std::vector<int64_t> result;
    for (auto a : axis) {
        if (a < 0) {
            a += rank;
        }
        result.push_back(a);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

// Coalesced shape after axis merging.
// Determines if reduction axis is tail (innermost contiguous) or non-tail.
struct CoalescedShape {
    int64_t totalRows;   // A1 = product of non-reduce outer dims
    int64_t rLength;     // R = product of reduce dims
    int64_t a0Length;    // A0 = product of non-reduce dims after R (0 if tail reduce)
    bool isTailReduce;   // true = AR mode (tail reduce), false = ARA mode
};

static CoalescedShape CoalesceAxis(const gert::Shape& inputShape, const std::vector<int64_t>& axisList)
{
    CoalescedShape result{1, 1, 1, false};
    int64_t rank = static_cast<int64_t>(inputShape.GetDimNum());

    if (rank == 0) {
        result.totalRows = 1;
        result.rLength = 1;
        result.a0Length = 0;
        result.isTailReduce = true;
        return result;
    }

    // Build is_reduction flag per dim
    std::vector<bool> isReduce(rank, false);
    for (auto a : axisList) {
        isReduce[a] = true;
    }

    // Check if reduction axes form a contiguous block at the end (tail reduce)
    // Find the position where reduction starts from the end
    int64_t firstReduceFromEnd = rank;
    for (int64_t i = rank - 1; i >= 0; i--) {
        if (isReduce[i]) {
            firstReduceFromEnd = i;
        } else {
            break;
        }
    }

    // Check if all reduce dims are contiguous from firstReduceFromEnd to end
    bool allReduceContiguous = true;
    for (int64_t i = firstReduceFromEnd; i < rank; i++) {
        if (!isReduce[i]) {
            allReduceContiguous = false;
            break;
        }
    }

    // Check if there are any non-reduce dims after first reduce dim (indicates non-tail reduce)
    int64_t firstReduceDim = rank;
    for (int64_t i = 0; i < rank; i++) {
        if (isReduce[i]) {
            firstReduceDim = i;
            break;
        }
    }

    bool hasNonReduceAfterReduce = false;
    if (firstReduceDim < rank) {
        for (int64_t i = firstReduceDim + 1; i < rank; i++) {
            if (!isReduce[i]) {
                hasNonReduceAfterReduce = true;
                break;
            }
        }
    }

    if (!hasNonReduceAfterReduce) {
        // AR mode: all reduce dims at the end (tail reduce)
        // totalRows = product of non-reduce dims, rLength = product of reduce dims
        result.isTailReduce = true;
        result.a0Length = 0; // no tail non-reduce axis
        for (int64_t i = 0; i < rank; i++) {
            if (!isReduce[i]) {
                result.totalRows *= inputShape.GetDim(i);
            } else {
                result.rLength *= inputShape.GetDim(i);
            }
        }
    } else {
        // ARA mode: reduction is non-tail, followed by non-reduce dims
        // Structure: [outer_dims, R, a0_dims]
        // totalRows = product of dims before R
        // rLength = product of reduce dims
        // a0Length = product of dims after reduce block
        result.isTailReduce = false;

        // Find contiguous reduce block
        // Assume reduce dims are contiguous (for simple ARA case like [4,3,1000] axis=[1])
        int64_t reduceStart = firstReduceDim;
        int64_t reduceEnd = firstReduceDim;
        for (int64_t i = firstReduceDim; i < rank; i++) {
            if (isReduce[i]) {
                reduceEnd = i;
            } else {
                break;
            }
        }

        // outer dims (before reduce)
        for (int64_t i = 0; i < reduceStart; i++) {
            result.totalRows *= inputShape.GetDim(i);
        }
        // reduce dims
        for (int64_t i = reduceStart; i <= reduceEnd; i++) {
            result.rLength *= inputShape.GetDim(i);
        }
        // a0 dims (after reduce)
        for (int64_t i = reduceEnd + 1; i < rank; i++) {
            result.a0Length *= inputShape.GetDim(i);
        }
        // If no a0 dims, fall back to AR mode
        if (result.a0Length == 1 && (reduceEnd + 1 >= rank)) {
            result.a0Length = 0;
            result.isTailReduce = true;
        }
    }

    // Handle empty input
    if (result.totalRows == 0) {
        result.rLength = 0;
    }

    return result;
}

// Compute tmpBuf size for ReduceSum (first-n version)
static uint32_t ComputeTmpBufSize(uint32_t count, uint32_t typeSize)
{
    uint32_t epr = 256 / typeSize; // 64 for float
    uint32_t epb = 32 / typeSize;  // 8 for float
    uint32_t firstMaxRep = (count + epr - 1) / epr;
    if (firstMaxRep == 0) firstMaxRep = 1;
    uint32_t iter1Out = firstMaxRep;
    uint32_t finalNeed = ((iter1Out + epb - 1) / epb) * epb;
    if (finalNeed < epb) finalNeed = epb;
    return finalNeed * typeSize;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SquareSumV1TilingFunc(gert::TilingContext* context)
{
    // 1. Get platform info
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, &ubSize, &coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. Get input shape and attrs
    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    auto inputShape = inputShapePtr->GetStorageShape();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto axisVec = attrs->GetListInt(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, axisVec);

    const auto* axisData = axisVec->GetData();
    size_t axisSize = axisVec->GetSize();
    std::vector<int64_t> axisList(axisData, axisData + axisSize);

    int64_t rank = static_cast<int64_t>(inputShape.GetDimNum());
    auto normalizedAxis = NormalizeAxis(axisList, rank);

    // 3. Coalesce axis
    auto coalesced = CoalesceAxis(inputShape, normalizedAxis);
    int64_t totalRows = coalesced.totalRows;
    int64_t rLength = coalesced.rLength;
    int64_t a0Length = coalesced.a0Length;
    bool isTailReduce = coalesced.isTailReduce;

    // 4. Handle empty tensor
    if (totalRows == 0 || rLength == 0) {
        SquareSumV1TilingData* tiling = context->GetTilingData<SquareSumV1TilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        memset_s(tiling, sizeof(SquareSumV1TilingData), 0, sizeof(SquareSumV1TilingData));
        tiling->totalRows = 0;
        tiling->rLength = 0;
        tiling->usedCoreNum = 1;
        tiling->tilingMode = 0;
        context->SetBlockDim(1);
        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dataType));
        GetWorkspaceSize(context);
        return ge::GRAPH_SUCCESS;
    }

    // 5. Compute dtype parameters
    uint32_t typeSize = 0;
    switch (dataType) {
        case ge::DT_FLOAT16: typeSize = 2; break;
        case ge::DT_BF16: typeSize = 2; break;
        case ge::DT_FLOAT: typeSize = 4; break;
        default: typeSize = 2; break;
    }

    uint32_t inputElementsPerBlock = 32 / typeSize;  // 16 for half, 8 for float
    uint32_t fp32ElementsPerBlock = 32 / sizeof(float); // 8
    uint32_t fp32ElementsPerRepeat = 256 / sizeof(float); // 64

    // Alignment for AR mode (input/fp32 path)
    int64_t rLengthAlignInput = CeilAlign(rLength, static_cast<int64_t>(inputElementsPerBlock));
    int64_t rLengthAlignFp32 = CeilAlign(rLength, static_cast<int64_t>(fp32ElementsPerBlock));
    int64_t rLengthAlign = std::max(rLengthAlignInput, rLengthAlignFp32);

    uint32_t isAlign32B = (rLength * typeSize % 32 == 0) ? 1 : 0;

    // 6. Determine tilingMode and compute parameters
    uint32_t tilingMode = 0;
    int64_t chunkCols = 0;
    int64_t numChunks = 0;
    int64_t a0LengthAlign = 0;
    int64_t tileA0Len = 0;
    int64_t tileA0Align = 0;
    int64_t numA0Tiles = 1;
    int64_t rChunkSize = 0;
    int64_t numRChunks = 0;

    if (isTailReduce) {
        // === AR mode ===
        // Compute fullload threshold
        uint32_t tmpBufBytes = ComputeTmpBufSize(static_cast<uint32_t>(rLengthAlign), sizeof(float));

        uint64_t ubNeededFullLoad;
        if (dataType == ge::DT_FLOAT) {
            ubNeededFullLoad = 2 * rLengthAlign * sizeof(float) + tmpBufBytes + 2 * 32;
        } else {
            ubNeededFullLoad = 2 * rLengthAlign * typeSize + rLengthAlignFp32 * sizeof(float) + tmpBufBytes + 2 * 32;
        }

        bool canFullLoad = (ubNeededFullLoad <= ubSize);

        if (canFullLoad) {
            tilingMode = 0; // AR_FULLLOAD
        } else {
            tilingMode = 1; // AR_COLSPLIT
            // Compute chunkCols: fit in UB with single buffer
            // inQueueX(T): chunkCols * typeSize
            // computeBuf(float): chunkCols * 4 (for non-float)
            // tmpBuf: computed
            // accBuf: 32
            // outQueueY: 32
            uint32_t chunkTmpBuf = ComputeTmpBufSize(
                static_cast<uint32_t>(255 * fp32ElementsPerRepeat), sizeof(float));

            if (dataType == ge::DT_FLOAT) {
                uint64_t overhead = chunkTmpBuf + 2 * 32;
                int64_t maxCols = static_cast<int64_t>((ubSize - overhead) / sizeof(float));
                chunkCols = std::min(maxCols, static_cast<int64_t>(255 * fp32ElementsPerRepeat));
            } else {
                uint64_t overhead = chunkTmpBuf + 2 * 32;
                // Need both input buffer (typeSize) and compute buffer (4 bytes) per element
                int64_t maxCols = static_cast<int64_t>((ubSize - overhead) / (typeSize + sizeof(float)));
                chunkCols = std::min(maxCols, static_cast<int64_t>(255 * fp32ElementsPerRepeat));
            }
            chunkCols = std::max(chunkCols, static_cast<int64_t>(1));
            // Align chunkCols to fp32 block for ReduceSum
            chunkCols = CeilAlign(chunkCols, static_cast<int64_t>(fp32ElementsPerBlock));
            chunkCols = std::min(chunkCols, rLengthAlign);

            numChunks = CeilDiv(rLength, chunkCols);
        }
    } else {
        // === ARA mode ===
        // a0Length must be >= 1 (we set it to 0 only for tail reduce)
        if (a0Length == 0) a0Length = 1;

        // A0 alignment in fp32 terms
        a0LengthAlign = CeilAlign(a0Length, static_cast<int64_t>(fp32ElementsPerBlock));

        // Determine tileA0: we want to fit [R, tileA0Align] in UB
        // For ARA_FULLLOAD (Key=2): R * tileA0Align * sizeof(float) must fit
        // For ARA_ROWSPLIT (Key=3): rChunkSize * tileA0Align * sizeof(float) must fit

        // Compute UB budget for ARA_FULLLOAD:
        // inQueueX(T): R * tileA0Align * typeSize (single buffer)
        // computeBuf(float): R * tileA0Align * sizeof(float) (for non-float)
        // accBuf: tileA0Align * sizeof(float)
        // outQueueY: tileA0Align * sizeof(T)
        // tmpBuf: tileA0Align * sizeof(float)

        // Start with tileA0 = a0Length (try full)
        tileA0Len = a0Length;
        tileA0Align = a0LengthAlign;

        // Check if R * tileA0Align fits in UB
        auto computeAraUbNeeded = [&](int64_t rRows, int64_t cols) -> uint64_t {
            uint64_t inBytes = rRows * cols * typeSize;
            uint64_t computeBytes = 0;
            if (dataType != ge::DT_FLOAT) {
                computeBytes = rRows * cols * sizeof(float);
            }
            uint64_t accBytes = cols * sizeof(float);
            uint64_t outBytes = cols * typeSize;
            uint64_t tmpBytes = static_cast<uint64_t>(cols * sizeof(float));
            if (tmpBytes < 32) tmpBytes = 32;
            return inBytes + computeBytes + accBytes + outBytes + tmpBytes;
        };

        uint64_t ubNeededAraFull = computeAraUbNeeded(rLength, tileA0Align);

        if (ubNeededAraFull <= ubSize) {
            // ARA_FULLLOAD
            tilingMode = 2;
            numA0Tiles = 1;
        } else {
            // Need to split. Try reducing tileA0 first
            // Binary search for max tileA0Align that fits with R=rLength
            int64_t maxTileA0 = a0LengthAlign;
            int64_t minTileA0 = static_cast<int64_t>(fp32ElementsPerBlock);
            int64_t bestTileA0 = 0; // 0 means not found yet

            while (minTileA0 <= maxTileA0) {
                int64_t mid = (minTileA0 + maxTileA0) / 2;
                mid = CeilAlign(mid, static_cast<int64_t>(fp32ElementsPerBlock));
                if (mid < static_cast<int64_t>(fp32ElementsPerBlock)) {
                    mid = fp32ElementsPerBlock;
                }
                uint64_t ubNeeded = computeAraUbNeeded(rLength, mid);
                if (ubNeeded <= ubSize) {
                    bestTileA0 = mid;
                    minTileA0 = mid + fp32ElementsPerBlock;
                } else {
                    maxTileA0 = mid - fp32ElementsPerBlock;
                }
            }

            if (bestTileA0 >= static_cast<int64_t>(fp32ElementsPerBlock)) {
                // Can fit with reduced tileA0, still ARA_FULLLOAD but with multiple A0 tiles
                tilingMode = 2;
                tileA0Align = bestTileA0;
                tileA0Len = std::min(tileA0Align, a0Length);
                numA0Tiles = CeilDiv(a0Length, tileA0Len);
            } else {
                // ARA_ROWSPLIT: need to split R as well
                tilingMode = 3;

                // Use a reasonable tileA0 (e.g., 64 elements aligned)
                tileA0Align = static_cast<int64_t>(fp32ElementsPerBlock * 8); // 64
                tileA0Align = std::min(tileA0Align, a0LengthAlign);
                tileA0Len = std::min(tileA0Align, a0Length);
                numA0Tiles = CeilDiv(a0Length, tileA0Len);

                // Find max rChunkSize that fits UB
                int64_t maxRChunk = rLength;
                int64_t minRChunk = 1;
                int64_t bestRChunk = 1;

                while (minRChunk <= maxRChunk) {
                    int64_t mid = (minRChunk + maxRChunk) / 2;
                    uint64_t ubNeeded = computeAraUbNeeded(mid, tileA0Align);
                    if (ubNeeded <= ubSize) {
                        bestRChunk = mid;
                        minRChunk = mid + 1;
                    } else {
                        maxRChunk = mid - 1;
                    }
                }

                rChunkSize = std::max(bestRChunk, static_cast<int64_t>(1));
                numRChunks = CeilDiv(rLength, rChunkSize);
            }
        }
    }

    // 7. Multi-core splitting: split by totalRows
    int64_t minRowsPerCore = 1;
    int64_t usedCoreNum = std::min(static_cast<int64_t>(coreNum), CeilDiv(totalRows, minRowsPerCore));
    if (usedCoreNum < 1) usedCoreNum = 1;

    int64_t rowsPerCore = CeilDiv(totalRows, usedCoreNum);
    int64_t tailRows = totalRows - rowsPerCore * (usedCoreNum - 1);
    if (tailRows < 0) tailRows = rowsPerCore;

    // 8. Set TilingData
    SquareSumV1TilingData* tiling = context->GetTilingData<SquareSumV1TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(SquareSumV1TilingData), 0, sizeof(SquareSumV1TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->totalRows = totalRows;
    tiling->rowsPerCore = rowsPerCore;
    tiling->tailRows = tailRows;
    tiling->usedCoreNum = usedCoreNum;
    tiling->rLength = rLength;
    tiling->rLengthAlign = rLengthAlign;
    tiling->chunkCols = chunkCols;
    tiling->numChunks = numChunks;
    tiling->a0Length = a0Length;
    tiling->a0LengthAlign = a0LengthAlign;
    tiling->tileA0Len = tileA0Len;
    tiling->tileA0Align = tileA0Align;
    tiling->numA0Tiles = numA0Tiles;
    tiling->rChunkSize = rChunkSize;
    tiling->numRChunks = numRChunks;
    tiling->tilingMode = tilingMode;
    tiling->inputDtype = static_cast<uint32_t>(dataType);
    tiling->isAlign32B = isAlign32B;

    context->SetBlockDim(usedCoreNum);

    // 9. Set TilingKey (template parameter selection)
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dataType));

    // 10. Workspace
    GetWorkspaceSize(context);

    OP_LOGD(context, "SquareSumV1 tiling: mode=%u, totalRows=%ld, rLength=%ld, a0Length=%ld, "
            "rowsPerCore=%ld, usedCoreNum=%ld, isAlign32B=%u, "
            "chunkCols=%ld, numChunks=%ld, tileA0Len=%ld, tileA0Align=%ld, "
            "numA0Tiles=%ld, rChunkSize=%ld, numRChunks=%ld",
            tilingMode, totalRows, rLength, a0Length,
            rowsPerCore, usedCoreNum, isAlign32B,
            chunkCols, numChunks, tileA0Len, tileA0Align,
            numA0Tiles, rChunkSize, numRChunks);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSquareSumV1([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SquareSumV1CompileInfo {};

IMPL_OP_OPTILING(SquareSumV1)
    .Tiling(SquareSumV1TilingFunc)
    .TilingParse<SquareSumV1CompileInfo>(TilingParseForSquareSumV1);

} // namespace optiling
