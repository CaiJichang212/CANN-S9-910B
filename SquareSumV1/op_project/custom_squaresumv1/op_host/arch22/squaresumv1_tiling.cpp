/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * \file squaresumv1_tiling.cpp
 * \brief SquareSumV1 tiling implementation (arch22 / Ascend910B)
 *
 * Iteration 1: AR_FULLLOAD (TilingKey=0)
 *   - axis=-1 (innermost continuous reduction)
 *   - single dtype: fp16
 *   - full-load: entire reduction row fits in UB
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
        *coreNum = 20; // default for 910B
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
    // Remove duplicates
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

// Simple axis coalescing for AR mode:
// After normalization, determine if we have a single contiguous reduction at the end.
// For iteration 1, we only support axis=-1 (single innermost axis).
// This produces: totalRows = product of non-reduction dims, rLength = product of reduction dims.
struct CoalescedShape {
    int64_t totalRows;   // A1 (product of non-reduction outer dims)
    int64_t rLength;     // R (product of reduction dims)
};

static CoalescedShape CoalesceAxis(const gert::Shape& inputShape, const std::vector<int64_t>& axisList)
{
    CoalescedShape result{1, 1};
    int64_t rank = static_cast<int64_t>(inputShape.GetDimNum());

    // Build is_reduction flag per dim
    std::vector<bool> isReduce(rank, false);
    for (auto a : axisList) {
        isReduce[a] = true;
    }

    // For AR mode (axis=-1 only supported in iter1):
    // Check if the reduction axes form a contiguous block at the end
    bool tailReduce = true;
    for (int64_t i = rank - 1; i >= 0; i--) {
        if (!isReduce[i]) {
            // Found first non-reduction from the end; all after it must be reduction
            for (int64_t j = i + 1; j < rank; j++) {
                if (!isReduce[j]) {
                    tailReduce = false;
                    break;
                }
            }
            break;
        }
    }

    if (tailReduce) {
        // AR mode: split into (A1, R)
        for (int64_t i = 0; i < rank; i++) {
            if (!isReduce[i]) {
                result.totalRows *= inputShape.GetDim(i);
            } else {
                result.rLength *= inputShape.GetDim(i);
            }
        }
    } else {
        // Non-tail reduction: fall back to treating everything as rows + R
        // For iteration 1, this shouldn't happen (axis=-1 only)
        // But handle gracefully: compute totalRows as product of non-R dims, R as product of R dims
        for (int64_t i = 0; i < rank; i++) {
            if (!isReduce[i]) {
                result.totalRows *= inputShape.GetDim(i);
            } else {
                result.rLength *= inputShape.GetDim(i);
            }
        }
    }

    // Handle scalar input (rank=0)
    if (rank == 0) {
        result.totalRows = 1;
        result.rLength = 1;
    }

    // Handle empty input
    if (result.totalRows == 0) {
        result.rLength = 0;
    }

    return result;
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

    // Get dtype
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();

    // Get axis attribute
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    auto axisVec = attrs->GetListInt(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, axisVec);

    const auto* axisData = axisVec->GetData();
    size_t axisSize = axisVec->GetSize();
    std::vector<int64_t> axisList(axisData, axisData + axisSize);

    int64_t rank = static_cast<int64_t>(inputShape.GetDimNum());
    auto normalizedAxis = NormalizeAxis(axisList, rank);

    // 3. Coalesce axis to get (totalRows, rLength)
    auto coalesced = CoalesceAxis(inputShape, normalizedAxis);
    int64_t totalRows = coalesced.totalRows;
    int64_t rLength = coalesced.rLength;

    // 4. Handle empty tensor
    if (totalRows == 0 || rLength == 0) {
        SquareSumV1TilingData* tiling = context->GetTilingData<SquareSumV1TilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
        memset_s(tiling, sizeof(SquareSumV1TilingData), 0, sizeof(SquareSumV1TilingData));
        tiling->totalRows = 0;
        tiling->rLength = 0;
        tiling->usedCoreNum = 1;
        context->SetBlockDim(1);
        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dataType));
        GetWorkspaceSize(context);
        return ge::GRAPH_SUCCESS;
    }

    // 5. Compute alignment
    uint32_t typeSize = 0;
    switch (dataType) {
        case ge::DT_FLOAT16: typeSize = 2; break;
        case ge::DT_BF16: typeSize = 2; break;
        case ge::DT_FLOAT: typeSize = 4; break;
        default: typeSize = 2; break;
    }

    // For input buffer: align to 32B in terms of input dtype elements
    uint32_t inputElementsPerBlock = 32 / typeSize; // 16 for half, 8 for float
    int64_t rLengthAlignInput = CeilAlign(rLength, static_cast<int64_t>(inputElementsPerBlock));

    // For fp32 compute: alignment in terms of float elements
    uint32_t fp32ElementsPerBlock = 32 / sizeof(float); // 8
    int64_t rLengthAlignFp32 = CeilAlign(rLength, static_cast<int64_t>(fp32ElementsPerBlock));

    // Use the larger alignment (fp32 path requires more elements per block)
    int64_t rLengthAlign = std::max(rLengthAlignInput, rLengthAlignFp32);

    // Check 32B alignment of input data
    uint32_t isAlign32B = (rLength * typeSize % 32 == 0) ? 1 : 0;

    // 6. UB budget check for AR_FULLLOAD
    // For fp16/bf16 input with fp32 compute:
    //   inQueueX(half): 2 * rLengthAlign * typeSize  (Double Buffer)
    //   computeBuf(float): rLengthAlignFp32 * 4
    //   tmpBuf: computed
    //   outQueueY: 2 * 32 (Double Buffer)
    // For fp32 input:
    //   inQueueX(float): 2 * rLengthAlign * 4
    //   tmpBuf: computed
    //   outQueueY: 2 * 32

    // Compute tmpBuf size for ReduceSum
    uint32_t elementsPerRepeatFp32 = 256 / sizeof(float); // 64
    uint32_t elementsPerBlockFp32 = 32 / sizeof(float);   // 8
    uint32_t firstMaxRepeat =
        (static_cast<uint32_t>(rLengthAlign) + elementsPerRepeatFp32 - 1) / elementsPerRepeatFp32;
    if (firstMaxRepeat == 0) firstMaxRepeat = 1;
    uint32_t tmpBufElements =
        ((firstMaxRepeat + elementsPerBlockFp32 - 1) / elementsPerBlockFp32) * elementsPerBlockFp32;
    if (tmpBufElements < elementsPerBlockFp32) tmpBufElements = elementsPerBlockFp32;
    uint32_t tmpBufBytes = tmpBufElements * sizeof(float);

    uint64_t totalUbNeeded;
    if (dataType == ge::DT_FLOAT) {
        totalUbNeeded = 2 * rLengthAlign * sizeof(float) + tmpBufBytes + 2 * 32;
    } else {
        // half or bf16
        totalUbNeeded = 2 * rLengthAlign * typeSize + rLengthAlignFp32 * sizeof(float) + tmpBufBytes + 2 * 32;
    }

    // AR_FULLLOAD threshold check
    bool canFullLoad = (totalUbNeeded <= ubSize);

    // For iteration 1, we implement AR_FULLLOAD only.
    // If data doesn't fit, we still try with a warning (will be addressed in iteration 2).
    if (!canFullLoad) {
        OP_LOGW(context, "SquareSumV1: data may not fit in UB for fullload (need=%lu, ubSize=%lu). Proceeding anyway.",
                totalUbNeeded, ubSize);
    }

    // 7. Multi-core splitting: split by totalRows
    int64_t minRowsPerCore = 1;
    int64_t usedCoreNum = std::min(static_cast<int64_t>(coreNum), CeilDiv(totalRows, minRowsPerCore));
    if (usedCoreNum < 1) usedCoreNum = 1;

    int64_t rowsPerCore = CeilDiv(totalRows, usedCoreNum);

    // 8. Set TilingData
    SquareSumV1TilingData* tiling = context->GetTilingData<SquareSumV1TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(SquareSumV1TilingData), 0, sizeof(SquareSumV1TilingData)) != EOK,
        OP_LOGE(context, "set tiling data error"), return ge::GRAPH_FAILED);

    tiling->totalRows = totalRows;
    tiling->rowsPerCore = rowsPerCore;
    tiling->tailRows = totalRows - rowsPerCore * (usedCoreNum - 1);
    tiling->usedCoreNum = usedCoreNum;
    tiling->rLength = rLength;
    tiling->rLengthAlign = rLengthAlign;
    tiling->inputDtype = static_cast<uint32_t>(dataType);
    tiling->isAlign32B = isAlign32B;

    context->SetBlockDim(usedCoreNum);

    // 9. Set TilingKey (template parameter selection)
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dataType));

    // 10. Workspace
    GetWorkspaceSize(context);

    OP_LOGD(context, "SquareSumV1 tiling: totalRows=%ld, rLength=%ld, rLengthAlign=%ld, "
            "rowsPerCore=%ld, usedCoreNum=%ld, isAlign32B=%u",
            totalRows, rLength, rLengthAlign, rowsPerCore, usedCoreNum, isAlign32B);

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
