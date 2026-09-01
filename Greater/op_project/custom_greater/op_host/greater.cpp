/**
 * @file greater.cpp
 *
 * Host-side definition, infer-shape/dtype and tiling for the Greater
 * (torch.gt) custom operator. Element-wise x > y with NumPy-style broadcast;
 * output is bool. Target: Ascend 910B (ascend910b).
 */
#include "greater_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace optiling {

constexpr uint32_t MAX_DIMS = 8;
constexpr uint64_t MAX_TILING_VALUE = std::numeric_limits<uint32_t>::max();

static bool CheckedMulU32(uint64_t lhs, uint64_t rhs, uint64_t& result)
{
    if (rhs != 0 && lhs > MAX_TILING_VALUE / rhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

static bool IsSupportedType(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT || dtype == ge::DT_BF16 ||
           dtype == ge::DT_INT32 || dtype == ge::DT_INT8;
}

// Keep these core grains in sync with the per-dtype Kernel TILE constants.
// Each launched AIV initializes TILE-sized constant buffers before doing work,
// so launching a core for only a 256-element output block is counterproductive.
static uint32_t GetCoreGrain(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_INT32: return 4096;
        case ge::DT_BF16: return 6144;
        case ge::DT_FLOAT: return 5120;
        case ge::DT_INT8: return 10240;
        case ge::DT_FLOAT16: return 9216;
        default: return 256;
    }
}

static uint32_t GetInputBytes(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_FLOAT:
        case ge::DT_INT32: return 4;
        case ge::DT_FLOAT16:
        case ge::DT_BF16: return 2;
        case ge::DT_INT8: return 1;
        default: return 1;
    }
}

// P2 buffer planning leaves the DAV_2201 basic-API temporary area above the
// 184KiB user UB boundary untouched. The Kernel conditionally allocates only
// the compute buffers used by each dtype; keep these batch caps in sync.
static uint32_t GetP2BatchLimitBytes(ge::DataType dtype)
{
    switch (dtype) {
        case ge::DT_BF16: return 48 * 1024;
        case ge::DT_INT8: return 60 * 1024;
        default: return 64 * 1024;
    }
}

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
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    GreaterTilingData tiling;

    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* yShape = context->GetInputShape(1);
    if (xShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint32_t xNdim = xShape->GetStorageShape().GetDimNum();
    uint32_t yNdim = yShape->GetStorageShape().GetDimNum();
    if (xNdim > MAX_DIMS || yNdim > MAX_DIMS) {
        return ge::GRAPH_FAILED;
    }
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
        if (sx[i] < 0 || sy[i] < 0 ||
            static_cast<uint64_t>(sx[i]) > MAX_TILING_VALUE ||
            static_cast<uint64_t>(sy[i]) > MAX_TILING_VALUE ||
            (sx[i] != sy[i] && sx[i] != 1 && sy[i] != 1)) {
            return ge::GRAPH_FAILED;
        }
        sz[i] = (sx[i] == 1) ? sy[i] : sx[i];
        uint64_t nextTotal = 0;
        if (!CheckedMulU32(totalSize, static_cast<uint64_t>(sz[i]), nextTotal)) {
            return ge::GRAPH_FAILED;
        }
        totalSize = nextTotal;
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

    uint64_t innerSize = static_cast<uint64_t>(sz[last]);
    int k = last - 1;
    if (bcastMode == 0) {
        // extend upward over trailing non-broadcast dims
        while (k >= 0 && sx[k] == sy[k]) {
            uint64_t nextInner = 0;
            if (!CheckedMulU32(innerSize, static_cast<uint64_t>(sz[k]), nextInner)) {
                return ge::GRAPH_FAILED;
            }
            innerSize = nextInner;
            --k;
        }
    }
    int outerDim = k + 1; // number of outer dims [0..k]
    uint64_t outerSize = 1;
    for (int d = 0; d <= k; ++d) {
        uint64_t nextOuter = 0;
        if (!CheckedMulU32(outerSize, static_cast<uint64_t>(sz[d]), nextOuter)) {
            return ge::GRAPH_FAILED;
        }
        outerSize = nextOuter;
    }

    // Per-operand memory strides (elements) for the outer dims. Stride is 0 on
    // broadcast dims so the base pointer does not advance there.
    auto memStride = [&](const int64_t* s, int d, uint32_t& result) -> bool {
        if (s[d] == 1) {
            result = 0;
            return true;
        }
        uint64_t stride = 1;
        for (int j = d + 1; j <= last; ++j) {
            uint64_t nextStride = 0;
            if (!CheckedMulU32(stride, static_cast<uint64_t>(s[j]), nextStride)) {
                return false;
            }
            stride = nextStride;
        }
        result = static_cast<uint32_t>(stride);
        return true;
    };

    uint32_t outerShapeArr[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t xStrideArr[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t yStrideArr[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int d = 0; d < 8; ++d) {
        if (d < outerDim) {
            outerShapeArr[d] = static_cast<uint32_t>(sz[d]);
            if (!memStride(sx, d, xStrideArr[d]) || !memStride(sy, d, yStrideArr[d])) {
                return ge::GRAPH_FAILED;
            }
        }
    }

    // Keep the generic fallback on the proven 256-element work policy. Only
    // routes that satisfy the Kernel's P1/P2 predicates use AIV/tile-aware
    // scaling, with their real group or segment count as the useful work.
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    const gert::CompileTimeTensorDesc* xDesc = context->GetInputDesc(0);
    const gert::CompileTimeTensorDesc* yDesc = context->GetInputDesc(1);
    if (platformInfo == nullptr || xDesc == nullptr || yDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType dtype = xDesc->GetDataType();
    if (!IsSupportedType(dtype) || dtype != yDesc->GetDataType()) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t aivCoreNum = platform.GetCoreNumAiv();
    if (aivCoreNum == 0) {
        aivCoreNum = 1;
    }
    uint32_t genericCoreLimit = platform.GetCoreNumAic();
    if (genericCoreLimit == 0) {
        genericCoreLimit = aivCoreNum;
    } else {
        genericCoreLimit = std::min(genericCoreLimit, aivCoreNum);
    }

    auto ceilDiv = [](uint64_t value, uint64_t divisor) -> uint64_t {
        return value == 0 ? 0 : (value - 1) / divisor + 1;
    };
    auto clampCoreCount = [](uint64_t units, uint32_t limit) -> uint32_t {
        if (units == 0) {
            return 1;
        }
        return static_cast<uint32_t>(std::min<uint64_t>(units, limit));
    };

    uint32_t tileElems = GetCoreGrain(dtype);
    uint32_t inputBytes = GetInputBytes(dtype);
    uint32_t innerElems = static_cast<uint32_t>(innerSize);
    uint64_t rowElems64 = ceilDiv(innerSize, 256) * 256;
    uint32_t rowElems = rowElems64 <= MAX_TILING_VALUE ? static_cast<uint32_t>(rowElems64) : 0;
    bool rowPadded = (innerElems % 256) != 0 && innerElems <= tileElems &&
                     rowElems != 0 && rowElems <= tileElems;
    bool vectorRowEligible = ((innerElems % 256) == 0 || rowPadded) && innerElems <= tileElems;

    uint32_t blockDim = 1;
    if (totalSize > 0) {
        blockDim = clampCoreCount(ceilDiv(totalSize, 256), genericCoreLimit);
    }

    auto fastCoreCount = [&](uint64_t usefulUnits) -> uint32_t {
        uint64_t units = std::min(usefulUnits, ceilDiv(totalSize, tileElems));
        return clampCoreCount(units, aivCoreNum);
    };
    auto residentGroups = [&](const uint32_t* residentStride,
                              const uint32_t* streamStride) -> uint32_t {
        uint64_t groupSegs = 1;
        uint64_t expectedStride = innerElems;
        for (int d = outerDim - 1; d >= 0; --d) {
            if (residentStride[d] != 0 || streamStride[d] != expectedStride) {
                break;
            }
            uint64_t nextGroupSegs = 0;
            uint64_t nextExpectedStride = 0;
            if (!CheckedMulU32(groupSegs, outerShapeArr[d], nextGroupSegs) ||
                !CheckedMulU32(expectedStride, outerShapeArr[d], nextExpectedStride)) {
                return 1;
            }
            groupSegs = nextGroupSegs;
            expectedStride = nextExpectedStride;
        }
        return static_cast<uint32_t>(groupSegs);
    };

    bool fastRouteSelected = false;
    if (totalSize > 0 && bcastMode == 0 && outerDim > 0 && vectorRowEligible) {
        uint32_t residentElems = rowPadded ? rowElems : innerElems + 256;
        uint64_t residentBytes = static_cast<uint64_t>(residentElems) * inputBytes;
        if (residentBytes <= 96 * 1024) {
            uint32_t xGroups = residentGroups(xStrideArr, yStrideArr);
            uint32_t yGroups = residentGroups(yStrideArr, xStrideArr);
            uint32_t groupSegs = 1;
            if (xGroups > yGroups) {
                groupSegs = xGroups;
            } else if (yGroups > 1) {
                groupSegs = yGroups;
            }
            if (groupSegs > 1) {
                uint64_t usefulUnits = (groupSegs == outerSize)
                    ? outerSize : outerSize / groupSegs;
                blockDim = fastCoreCount(usefulUnits);
                fastRouteSelected = true;
            }
        }
    }

    // Large complete-outer P1 keeps one inner slice resident per core. Only
    // exceed the proven Generic core cap when there is at least one dtype TILE
    // of output for every available AIV; low-work 2D partitions otherwise pay
    // more per-core initialization than the extra parallelism can hide.
    if (totalSize > 0 && !fastRouteSelected && bcastMode == 0 && outerDim > 0 &&
        outerSize > 1 && innerElems > tileElems) {
        uint32_t xGroups = residentGroups(xStrideArr, yStrideArr);
        uint32_t yGroups = residentGroups(yStrideArr, xStrideArr);
        bool fullResident = xGroups == outerSize || yGroups == outerSize;
        if (fullResident) {
            uint64_t usefulTiles = ceilDiv(totalSize, tileElems);
            uint32_t largeCoreLimit = usefulTiles >= aivCoreNum
                ? aivCoreNum : genericCoreLimit;
            uint64_t maxUsefulCores = std::min<uint64_t>(largeCoreLimit, usefulTiles);
            uint64_t innerTiles = ceilDiv(innerSize, tileElems);
            uint64_t innerWorkers = std::min(innerTiles, maxUsefulCores);
            if (innerWorkers > 0) {
                uint64_t outerWorkers = std::min<uint64_t>(
                    outerSize, maxUsefulCores / innerWorkers);
                if (outerWorkers > 0) {
                    blockDim = static_cast<uint32_t>(innerWorkers * outerWorkers);
                    fastRouteSelected = true;
                }
            }
        }
    }

    auto streamIsContinuous = [&](const uint32_t* streamStride) -> bool {
        uint64_t expected = innerElems;
        for (int d = outerDim - 1; d >= 0; --d) {
            uint32_t dim = outerShapeArr[d];
            if (dim <= 1) {
                continue;
            }
            if (expected > UINT32_MAX || streamStride[d] != expected) {
                return false;
            }
            uint64_t nextExpected = 0;
            if (!CheckedMulU32(expected, dim, nextExpected)) {
                return false;
            }
            expected = nextExpected;
        }
        return true;
    };
    auto scalarIsContinuous = [&](const uint32_t* scalarStride) -> bool {
        uint64_t expected = 1;
        for (int d = outerDim - 1; d >= 0; --d) {
            if (scalarStride[d] != expected) {
                return false;
            }
            uint64_t nextExpected = 0;
            if (!CheckedMulU32(expected, outerShapeArr[d], nextExpected)) {
                return false;
            }
            expected = nextExpected;
        }
        return true;
    };

    if (totalSize > 0 && !fastRouteSelected &&
        (bcastMode == 1 || bcastMode == 2) && vectorRowEligible) {
        const uint32_t* scalarStride = (bcastMode == 1) ? xStrideArr : yStrideArr;
        const uint32_t* streamStride = (bcastMode == 1) ? yStrideArr : xStrideArr;
        if (streamIsContinuous(streamStride)) {
            uint64_t maxScalarOffset = 0;
            bool scalarRangeValid = true;
            for (int d = 0; d < outerDim; ++d) {
                uint64_t term = static_cast<uint64_t>(outerShapeArr[d] - 1) * scalarStride[d];
                if (term > MAX_TILING_VALUE - maxScalarOffset) {
                    scalarRangeValid = false;
                    break;
                }
                maxScalarOffset += term;
            }
            if (!scalarRangeValid) {
                return ge::GRAPH_FAILED;
            }
            uint64_t batchCount = maxScalarOffset + 1;
            uint32_t fastBlockDim = fastCoreCount(outerSize);
            bool scalarContinuous = scalarIsContinuous(scalarStride);
            uint64_t allocCount = scalarContinuous ? ceilDiv(outerSize, fastBlockDim) : batchCount;
            uint64_t batchBytes = (allocCount + 256) * inputBytes;
            bool blockedScalarRows = scalarContinuous && rowPadded && rowElems == 256 &&
                                     (dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT);
            if (allocCount <= UINT32_MAX &&
                (batchBytes <= GetP2BatchLimitBytes(dtype) || blockedScalarRows)) {
                blockDim = fastBlockDim;
            }
        }
    }

    // Large contiguous same-shape tensors amortize the per-AIV initialization
    // cost and benefit from all vector cores. Keep smaller generic work on the
    // measured 20-core policy; the threshold is based on total IO traffic, not
    // any individual shape.
    constexpr uint64_t largeFlatIoThreshold = 64ULL * 1024 * 1024;
    if (!fastRouteSelected && bcastMode == 0 && outerDim == 0) {
        uint64_t estimatedIoBytes = totalSize * (2ULL * inputBytes + 1);
        if (estimatedIoBytes >= largeFlatIoThreshold) {
            blockDim = fastCoreCount(totalSize);
        }
    }

    gert::TilingData* rawTiling = context->GetRawTilingData();
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (rawTiling == nullptr || rawTiling->GetData() == nullptr || currentWorkspace == nullptr ||
        rawTiling->GetCapacity() < tiling.GetDataSize()) {
        return ge::GRAPH_FAILED;
    }
    ge::graphStatus blockStatus = context->SetBlockDim(blockDim);
    if (blockStatus != ge::GRAPH_SUCCESS) {
        return blockStatus;
    }
    tiling.set_totalSize(static_cast<uint32_t>(totalSize));
    tiling.set_blockDim(blockDim);
    tiling.set_innerSize(static_cast<uint32_t>(innerSize));
    tiling.set_outerSize(static_cast<uint32_t>(outerSize));
    tiling.set_bcastMode(static_cast<uint32_t>(bcastMode));
    tiling.set_outerDim(static_cast<uint32_t>(outerDim));

    tiling.set_outerShape(outerShapeArr);
    tiling.set_xStride(xStrideArr);
    tiling.set_yStride(yStrideArr);

    tiling.SaveToBuffer(rawTiling->GetData(), rawTiling->GetCapacity());
    rawTiling->SetDataSize(tiling.GetDataSize());

    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
// Broadcast the two input shapes into the output shape.
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::Shape* xShape = context->GetInputShape(0);
    const gert::Shape* yShape = context->GetInputShape(1);
    gert::Shape* zShape = context->GetOutputShape(0);
    if (xShape == nullptr || yShape == nullptr || zShape == nullptr) {
        return GRAPH_FAILED;
    }

    uint32_t xNdim = xShape->GetDimNum();
    uint32_t yNdim = yShape->GetDimNum();
    if (xNdim > optiling::MAX_DIMS || yNdim > optiling::MAX_DIMS) {
        return GRAPH_FAILED;
    }
    uint32_t ndim = std::max(xNdim, yNdim);
    if (ndim == 0) {
        zShape->SetDimNum(0);
        return GRAPH_SUCCESS;
    }

    int64_t outputDims[optiling::MAX_DIMS] = {0};
    uint64_t totalSize = 1;
    for (uint32_t i = 0; i < ndim; ++i) {
        int64_t dx = (i + xNdim < ndim) ? 1 : xShape->GetDim(i - (ndim - xNdim));
        int64_t dy = (i + yNdim < ndim) ? 1 : yShape->GetDim(i - (ndim - yNdim));
        if (dx < 0 || dy < 0 || (dx != dy && dx != 1 && dy != 1)) {
            return GRAPH_FAILED;
        }
        outputDims[i] = (dx == 1) ? dy : dx;
        uint64_t nextTotal = 0;
        if (!optiling::CheckedMulU32(totalSize, static_cast<uint64_t>(outputDims[i]), nextTotal)) {
            return GRAPH_FAILED;
        }
        totalSize = nextTotal;
    }

    zShape->SetDimNum(ndim);
    for (uint32_t i = 0; i < ndim; ++i) {
        zShape->SetDim(i, outputDims[i]);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    // Greater always outputs bool, regardless of the input dtype.
    return context->SetOutputDataType(0, ge::DT_BOOL);
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
