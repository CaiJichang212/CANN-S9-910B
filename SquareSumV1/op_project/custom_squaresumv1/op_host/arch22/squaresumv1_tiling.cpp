/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * \file squaresumv1_tiling.cpp
 * \brief SquareSumV1 tiling implementation (arch22 / Ascend910B)
 *
 * Iteration 3: Multi-TilingKey integration + MULTI_AXIS
 *   Key=0 AR_FULLLOAD:  tail-axis reduce, full load (existing, refined)
 *   Key=1 AR_COLSPLIT:  tail-axis reduce, column chunk + fp32 accumulator
 *   Key=2 ARA_FULLLOAD: non-tail-axis reduce, Pattern::Reduce::RA full load
 *   Key=3 ARA_ROWSPLIT: non-tail-axis reduce, R-chunk + cross-chunk accumulation
 *   Key=4 MULTI_AXIS:   non-contiguous multi-axis, layer-by-layer reduce (innermost first)
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
#include <cstring>

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

    // Find contiguous reduce block starting from first reduce dim
    int64_t firstReduceDim = rank;
    for (int64_t i = 0; i < rank; i++) {
        if (isReduce[i]) {
            firstReduceDim = i;
            break;
        }
    }

    if (firstReduceDim == rank) {
        // No reduction axis at all
        result.totalRows = 1;
        result.rLength = 1;
        for (int64_t i = 0; i < rank; i++) {
            result.totalRows *= inputShape.GetDim(i);
        }
        result.a0Length = 0;
        result.isTailReduce = true;
        return result;
    }

    // Find contiguous reduce block extent
    int64_t reduceEnd = firstReduceDim;
    for (int64_t i = firstReduceDim; i < rank; i++) {
        if (isReduce[i]) {
            reduceEnd = i;
        } else {
            break;
        }
    }

    // Check if there are non-reduce dims after the reduce block
    bool hasNonReduceAfterReduce = false;
    for (int64_t i = reduceEnd + 1; i < rank; i++) {
        if (!isReduce[i]) {
            hasNonReduceAfterReduce = true;
            break;
        }
    }

    // Check if reduce axes are contiguous (no gaps)
    // If any non-reduce dim exists between two reduce dims, it's non-contiguous
    // Also check: are there reduce dims after the contiguous block that we already found?
    bool reduceAfterNonReduce = false;
    for (int64_t i = reduceEnd + 1; i < rank; i++) {
        if (isReduce[i]) {
            reduceAfterNonReduce = true;
            break;
        }
    }

    if (reduceAfterNonReduce) {
        // Non-contiguous multi-axis: this will be handled by MULTI_AXIS mode
        // Mark with sentinel values
        result.totalRows = -1;
        result.rLength = -1;
        result.a0Length = -1;
        result.isTailReduce = false;
        return result;
    }

    if (!hasNonReduceAfterReduce) {
        // AR mode: all reduce dims at the end (tail reduce)
        result.isTailReduce = true;
        result.a0Length = 0;
        for (int64_t i = 0; i < rank; i++) {
            if (!isReduce[i]) {
                result.totalRows *= inputShape.GetDim(i);
            } else {
                result.rLength *= inputShape.GetDim(i);
            }
        }
    } else {
        // ARA mode: reduction is non-tail, followed by non-reduce dims
        result.isTailReduce = false;
        for (int64_t i = 0; i < firstReduceDim; i++) {
            result.totalRows *= inputShape.GetDim(i);
        }
        for (int64_t i = firstReduceDim; i <= reduceEnd; i++) {
            result.rLength *= inputShape.GetDim(i);
        }
        for (int64_t i = reduceEnd + 1; i < rank; i++) {
            result.a0Length *= inputShape.GetDim(i);
        }
        if (result.a0Length == 1 && (reduceEnd + 1 >= rank)) {
            result.a0Length = 0;
            result.isTailReduce = true;
        }
    }

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

// ============================================================
// MULTI_AXIS: Compute per-layer tiling parameters
// ============================================================

struct LayerInfo {
    int32_t axisIdx;              // original axis index
    std::vector<int64_t> shapeBefore; // shape before this layer's reduce
    int64_t reduceAxisInShape;    // position of reduce axis within current shape
    int64_t rLength;              // reduce axis length
    int64_t a0Length;             // non-reduce tail (0 if tail reduce)
    bool isTailReduce;
    int64_t inputElemCount;       // total elements at layer input
    int64_t outputElemCount;      // total elements at layer output
    int64_t workspaceOffset;      // byte offset in workspace
    int64_t subMode;              // 0=AR_FULLLOAD, 1=AR_COLSPLIT, 2=ARA_FULLLOAD, 3=ARA_ROWSPLIT
    int64_t chunkCols;
    int64_t numChunks;
    int64_t tileA0Align;
    int64_t tileA0Len;
    int64_t numA0Tiles;
    int64_t rChunkSize;
    int64_t numRChunks;
};

static void ComputeLayerSubTiling(
    LayerInfo& layer,
    uint64_t ubSize,
    uint32_t typeSize,
    uint32_t fp32Epb,
    uint32_t fp32Epr)
{
    int64_t rLength = layer.rLength;
    int64_t a0Length = layer.a0Length;
    bool isTailReduce = layer.isTailReduce;

    int64_t rLengthAlignInput = CeilAlign(rLength, static_cast<int64_t>(32 / typeSize));
    int64_t rLengthAlignFp32 = CeilAlign(rLength, static_cast<int64_t>(fp32Epb));
    int64_t rLengthAlign = std::max(rLengthAlignInput, rLengthAlignFp32);

    if (isTailReduce || a0Length == 0) {
        // AR mode
        uint32_t tmpBufBytes = ComputeTmpBufSize(static_cast<uint32_t>(rLengthAlign), sizeof(float));

        uint64_t ubNeededFullLoad;
        if (typeSize == 4) {
            ubNeededFullLoad = 2 * rLengthAlign * sizeof(float) + tmpBufBytes + 2 * 32;
        } else {
            ubNeededFullLoad = 2 * rLengthAlign * typeSize + rLengthAlignFp32 * sizeof(float) + tmpBufBytes + 2 * 32;
        }

        if (ubNeededFullLoad <= ubSize) {
            layer.subMode = 0; // AR_FULLLOAD
        } else {
            layer.subMode = 1; // AR_COLSPLIT
            uint32_t chunkTmpBuf = ComputeTmpBufSize(255 * fp32Epr, sizeof(float));
            int64_t maxCols;
            if (typeSize == 4) {
                uint64_t overhead = chunkTmpBuf + 2 * 32;
                maxCols = static_cast<int64_t>((ubSize - overhead) / sizeof(float));
            } else {
                uint64_t overhead = chunkTmpBuf + 2 * 32;
                maxCols = static_cast<int64_t>((ubSize - overhead) / (typeSize + sizeof(float)));
            }
            layer.chunkCols = std::min(maxCols, static_cast<int64_t>(255 * fp32Epr));
            layer.chunkCols = std::max(layer.chunkCols, static_cast<int64_t>(1));
            layer.chunkCols = CeilAlign(layer.chunkCols, static_cast<int64_t>(fp32Epb));
            layer.chunkCols = std::min(layer.chunkCols, rLengthAlign);
            layer.numChunks = CeilDiv(rLength, layer.chunkCols);
        }
    } else {
        // ARA mode
        int64_t a0Align = CeilAlign(a0Length, static_cast<int64_t>(fp32Epb));

        auto computeAraUbNeeded = [&](int64_t rRows, int64_t cols) -> uint64_t {
            uint64_t inBytes = rRows * cols * typeSize;
            uint64_t computeBytes = 0;
            if (typeSize != 4) {
                computeBytes = rRows * cols * sizeof(float);
            }
            uint64_t accBytes = cols * sizeof(float);
            uint64_t outBytes = cols * typeSize;
            uint64_t tmpBytes = static_cast<uint64_t>(cols * sizeof(float));
            if (tmpBytes < 32) tmpBytes = 32;
            return inBytes + computeBytes + accBytes + outBytes + tmpBytes;
        };

        int64_t tileA0Align = a0Align;
        int64_t tileA0Len = a0Length;
        int64_t numA0Tiles = 1;

        uint64_t ubNeeded = computeAraUbNeeded(rLength, tileA0Align);

        if (ubNeeded <= ubSize) {
            layer.subMode = 2; // ARA_FULLLOAD
        } else {
            // Binary search for max tileA0
            int64_t maxA0 = a0Align;
            int64_t minA0 = static_cast<int64_t>(fp32Epb);
            int64_t bestA0 = 0;

            while (minA0 <= maxA0) {
                int64_t mid = CeilAlign((minA0 + maxA0) / 2, static_cast<int64_t>(fp32Epb));
                if (mid < static_cast<int64_t>(fp32Epb)) mid = fp32Epb;
                if (computeAraUbNeeded(rLength, mid) <= ubSize) {
                    bestA0 = mid;
                    minA0 = mid + fp32Epb;
                } else {
                    maxA0 = mid - fp32Epb;
                }
            }

            if (bestA0 >= static_cast<int64_t>(fp32Epb)) {
                layer.subMode = 2;
                tileA0Align = bestA0;
                tileA0Len = std::min(tileA0Align, a0Length);
                numA0Tiles = CeilDiv(a0Length, tileA0Len);
            } else {
                // ARA_ROWSPLIT
                layer.subMode = 3;
                tileA0Align = std::min(static_cast<int64_t>(fp32Epb * 8), a0Align);
                tileA0Len = std::min(tileA0Align, a0Length);
                numA0Tiles = CeilDiv(a0Length, tileA0Len);

                int64_t maxR = rLength;
                int64_t minR = 1;
                int64_t bestR = 1;
                while (minR <= maxR) {
                    int64_t mid = (minR + maxR) / 2;
                    if (computeAraUbNeeded(mid, tileA0Align) <= ubSize) {
                        bestR = mid;
                        minR = mid + 1;
                    } else {
                        maxR = mid - 1;
                    }
                }
                layer.rChunkSize = std::max(bestR, static_cast<int64_t>(1));
                layer.numRChunks = CeilDiv(rLength, layer.rChunkSize);
            }
        }

        layer.tileA0Align = tileA0Align;
        layer.tileA0Len = tileA0Len;
        layer.numA0Tiles = numA0Tiles;
    }
}

static std::vector<LayerInfo> ComputeMultiAxisLayers(
    const gert::Shape& inputShape,
    const std::vector<int64_t>& sortedAxis,
    uint64_t ubSize,
    uint32_t typeSize,
    uint32_t fp32Epb,
    uint32_t fp32Epr)
{
    int64_t rank = static_cast<int64_t>(inputShape.GetDimNum());
    std::vector<int64_t> currentShape;
    for (int64_t i = 0; i < rank; i++) {
        currentShape.push_back(inputShape.GetDim(i));
    }

    // Sort axis ascending (innermost = highest index first when processing)
    // We process from innermost (last) to outermost (first)
    // sortedAxis is already ascending, so reverse for processing order
    std::vector<int64_t> processOrder(sortedAxis.rbegin(), sortedAxis.rend());

    std::vector<LayerInfo> layers;

    for (size_t li = 0; li < processOrder.size(); li++) {
        int64_t targetAxis = processOrder[li];

        // Find position of targetAxis in current shape
        // currentShape may have shrunk from previous layers, but axes maintain their relative positions
        // Actually, we need to track which original axis indices remain and their positions
        // After each layer reduce, the reduced axis is removed from currentShape.
        // The "targetAxis" is the original axis index, which maps to a position in currentShape
        // after accounting for previously removed axes.

        // Simpler approach: we track the current shape with axis markers
        // Let's just use the original shape and track which axes have been reduced
        // For layer li, we know all axes processed before li (higher index axes in processOrder)
        // have been reduced. The position of targetAxis in the current shape = targetAxis - (number of
        // already-reduced axes with index < targetAxis)

        int64_t alreadyReducedBefore = 0;
        for (size_t lj = li + 1; lj < processOrder.size(); lj++) {
            // processOrder is reversed, so lj > li means these were processed AFTER current
            // Wait - processOrder is [innermost, ..., outermost]
            // li=0 is innermost (processed first), li=1 is next, etc.
            // So axes with lj < li were already reduced before current layer
            // alreadyReduced axes with original index < targetAxis reduce the position
            if (processOrder[lj] < targetAxis) {
                alreadyReducedBefore++;
            }
        }

        // Wait, let me reconsider. processOrder[0] is the innermost (highest index), processed first.
        // processOrder[1] is next, etc.
        // For layer li, axes processOrder[0..li-1] have already been reduced.
        // The position of targetAxis = processOrder[li] in the current (reduced) shape is:
        //   targetAxis - count(already-reduced axes with original index < targetAxis)

        int64_t reducedBefore = 0;
        for (size_t lj = 0; lj < li; lj++) {
            if (processOrder[lj] < targetAxis) {
                reducedBefore++;
            }
        }
        int64_t posInShape = targetAxis - reducedBefore;

        // Build current shape (after previous reductions)
        std::vector<int64_t> shape;
        for (int64_t i = 0; i < rank; i++) {
            bool alreadyReduced = false;
            for (size_t lj = 0; lj < li; lj++) {
                if (processOrder[lj] == i) {
                    alreadyReduced = true;
                    break;
                }
            }
            if (!alreadyReduced) {
                shape.push_back(currentShape[static_cast<size_t>(i)]);
            }
        }

        int64_t rLength = shape[static_cast<size_t>(posInShape)];
        int64_t nDims = static_cast<int64_t>(shape.size());

        // Compute totalRows (product of dims before reduce axis)
        int64_t totalRows = 1;
        for (int64_t i = 0; i < posInShape; i++) {
            totalRows *= shape[static_cast<size_t>(i)];
        }

        // Compute a0Length (product of dims after reduce axis)
        int64_t a0Length = 1;
        for (int64_t i = posInShape + 1; i < nDims; i++) {
            a0Length *= shape[static_cast<size_t>(i)];
        }

        bool isTailReduce = (posInShape == nDims - 1);
        if (isTailReduce) a0Length = 0;

        // Total elements
        int64_t inputElemCount = 1;
        for (auto d : shape) inputElemCount *= d;
        int64_t outputElemCount = inputElemCount / rLength;

        LayerInfo layer;
        layer.axisIdx = static_cast<int32_t>(targetAxis);
        layer.shapeBefore = shape;
        layer.reduceAxisInShape = posInShape;
        layer.rLength = rLength;
        layer.a0Length = isTailReduce ? 0 : a0Length;
        layer.isTailReduce = isTailReduce;
        layer.inputElemCount = inputElemCount;
        layer.outputElemCount = outputElemCount;
        layer.workspaceOffset = 0; // will be set later
        layer.subMode = 0;
        layer.chunkCols = 0;
        layer.numChunks = 0;
        layer.tileA0Align = 0;
        layer.tileA0Len = 0;
        layer.numA0Tiles = 0;
        layer.rChunkSize = 0;
        layer.numRChunks = 0;

        ComputeLayerSubTiling(layer, ubSize, typeSize, fp32Epb, fp32Epr);

        layers.push_back(layer);
    }

    return layers;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context, size_t wsSize)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = wsSize;
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
        GetWorkspaceSize(context, WS_SYS_SIZE);
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

    // MULTI_AXIS detection: coalesced.totalRows == -1 signals non-contiguous multi-axis
    if (totalRows == -1) {
        // === MULTI_AXIS (Key=4) ===
        tilingMode = 4;

        // Compute per-layer info
        auto layers = ComputeMultiAxisLayers(
            inputShape, normalizedAxis, ubSize,
            typeSize, fp32ElementsPerBlock, fp32ElementsPerRepeat);

        // Compute workspace offsets (in float32 element count, including padding)
        // Each intermediate element is stored as a 32-byte (8 fp32) block for
        // reliable 32B-aligned DataCopyPad transfers.
        // Convention: layerWorkspaceOffset[k] = element offset where layer k reads from.
        //   Layer 0: reads from inputGM, so layerWorkspaceOffset[0] is unused.
        //   Layer k (k>0): reads from workspace[layerWorkspaceOffset[k]].
        //   Layer k writes to layerWorkspaceOffset[k+1].
        constexpr int64_t WS_PAD = 8; // 8 fp32 = 32 bytes per element
        int64_t wsElemOffset = 0;
        for (size_t li = 0; li < layers.size(); li++) {
            if (li == 0) {
                layers[li].workspaceOffset = 0; // unused (reads from inputGM)
            } else {
                layers[li].workspaceOffset = wsElemOffset;
                wsElemOffset += layers[li - 1].outputElemCount * WS_PAD;
            }
        }

        // For layer 0: reads from input, writes intermediate to workspace[0..outputElemCount*4]
        // For layer k (k>0, k<N-1): reads from workspace, writes to workspace at next offset
        // For layer N-1: reads from workspace (or input if N=1), writes to result GM

        // Set up multi-core: split by the "outermost" dimension (rows)
        // For MULTI_AXIS, we split by totalRows of the FIRST layer (product of dims before first reduce axis)
        int64_t firstLayerRows = 1;
        {
            // totalRows for layer 0 = product of dims before reduce axis in original shape
            int64_t firstReducePos = layers[0].reduceAxisInShape;
            for (int64_t i = 0; i < firstReducePos; i++) {
                firstLayerRows *= layers[0].shapeBefore[static_cast<size_t>(i)];
            }
        }

        // Force single core for MULTI_AXIS: each layer has different totalRows,
        // and cross-core partitioning across layers is complex.
        int64_t usedCoreNum = 1;
        int64_t rowsPerCore = firstLayerRows;

        // Set TilingData
        SquareSumV1TilingData* tiling = context->GetTilingData<SquareSumV1TilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        OP_CHECK_IF(
            memset_s(tiling, sizeof(SquareSumV1TilingData), 0, sizeof(SquareSumV1TilingData)) != EOK,
            OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

        tiling->totalRows = firstLayerRows;
        tiling->rowsPerCore = rowsPerCore;
        tiling->tailRows = firstLayerRows - rowsPerCore * (usedCoreNum - 1);
        if (tiling->tailRows < 0) tiling->tailRows = rowsPerCore;
        tiling->usedCoreNum = usedCoreNum;
        tiling->tilingMode = 4;
        tiling->inputDtype = static_cast<uint32_t>(dataType);
        tiling->isAlign32B = 0;

        // Fill MULTI_AXIS layer info
        int32_t numLayers = static_cast<int32_t>(std::min(layers.size(), static_cast<size_t>(SS_MAX_LAYERS)));
        tiling->numLayers = numLayers;

        for (int32_t li = 0; li < numLayers; li++) {
            const auto& lyr = layers[static_cast<size_t>(li)];
            tiling->layerAxis[li] = lyr.axisIdx;
            tiling->layerNDims[li] = static_cast<int32_t>(std::min(lyr.shapeBefore.size(),
                                                                   static_cast<size_t>(SS_MAX_LAYERS + 1)));
            for (int32_t di = 0; di < tiling->layerNDims[li] && di < SS_MAX_LAYERS + 1; di++) {
                tiling->layerShapeBefore[li][di] = lyr.shapeBefore[static_cast<size_t>(di)];
            }
            tiling->layerReduceAxisIdx[li] = lyr.reduceAxisInShape;
            tiling->layerRLength[li] = lyr.rLength;
            tiling->layerA0Length[li] = lyr.a0Length;
            tiling->layerInputElemCount[li] = lyr.inputElemCount;
            tiling->layerOutputElemCount[li] = lyr.outputElemCount;
            tiling->layerIsTailReduce[li] = lyr.isTailReduce ? 1 : 0;
            tiling->layerWorkspaceOffset[li] = lyr.workspaceOffset;
            tiling->layerChunkCols[li] = lyr.chunkCols;
            tiling->layerNumChunks[li] = lyr.numChunks;
            tiling->layerTileA0Align[li] = lyr.tileA0Align;
            tiling->layerTileA0Len[li] = lyr.tileA0Len;
            tiling->layerNumA0Tiles[li] = lyr.numA0Tiles;
            tiling->layerRChunkSize[li] = lyr.rChunkSize;
            tiling->layerNumRChunks[li] = lyr.numRChunks;
            tiling->layerMode[li] = lyr.subMode;
        }

        // Workspace = sum of all intermediate output sizes, in bytes (stored as float32)
        // wsElemOffset now holds the total element count needed
        size_t wsSize = static_cast<size_t>(wsElemOffset) * sizeof(float);
        // Ensure at least one page and 4K alignment
        if (wsSize == 0) wsSize = 4096;
        wsSize = (wsSize + 4095) & ~static_cast<size_t>(4095);

        context->SetBlockDim(static_cast<int32_t>(usedCoreNum));
        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dataType));
        GetWorkspaceSize(context, wsSize);

        OP_LOGD(context, "SquareSumV1 MULTI_AXIS tiling: numLayers=%d, firstLayerRows=%ld, "
                "usedCoreNum=%ld, wsSize=%zu",
                numLayers, firstLayerRows, usedCoreNum, wsSize);

        return ge::GRAPH_SUCCESS;
    }

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
            uint32_t chunkTmpBuf = ComputeTmpBufSize(
                static_cast<uint32_t>(255 * fp32ElementsPerRepeat), sizeof(float));

            if (dataType == ge::DT_FLOAT) {
                uint64_t overhead = chunkTmpBuf + 2 * 32;
                int64_t maxCols = static_cast<int64_t>((ubSize - overhead) / sizeof(float));
                chunkCols = std::min(maxCols, static_cast<int64_t>(255 * fp32ElementsPerRepeat));
            } else {
                uint64_t overhead = chunkTmpBuf + 2 * 32;
                int64_t maxCols = static_cast<int64_t>((ubSize - overhead) / (typeSize + sizeof(float)));
                chunkCols = std::min(maxCols, static_cast<int64_t>(255 * fp32ElementsPerRepeat));
            }
            chunkCols = std::max(chunkCols, static_cast<int64_t>(1));
            chunkCols = CeilAlign(chunkCols, static_cast<int64_t>(fp32ElementsPerBlock));
            chunkCols = std::min(chunkCols, rLengthAlign);

            numChunks = CeilDiv(rLength, chunkCols);
        }
    } else {
        // === ARA mode ===
        if (a0Length == 0) a0Length = 1;

        a0LengthAlign = CeilAlign(a0Length, static_cast<int64_t>(fp32ElementsPerBlock));

        tileA0Len = a0Length;
        tileA0Align = a0LengthAlign;

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
            tilingMode = 2;
            numA0Tiles = 1;
        } else {
            int64_t maxTileA0 = a0LengthAlign;
            int64_t minTileA0 = static_cast<int64_t>(fp32ElementsPerBlock);
            int64_t bestTileA0 = 0;

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
                tilingMode = 2;
                tileA0Align = bestTileA0;
                tileA0Len = std::min(tileA0Align, a0Length);
                numA0Tiles = CeilDiv(a0Length, tileA0Len);
            } else {
                tilingMode = 3;
                tileA0Align = static_cast<int64_t>(fp32ElementsPerBlock * 8);
                tileA0Align = std::min(tileA0Align, a0LengthAlign);
                tileA0Len = std::min(tileA0Align, a0Length);
                numA0Tiles = CeilDiv(a0Length, tileA0Len);

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
    GetWorkspaceSize(context, WS_SYS_SIZE);

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
