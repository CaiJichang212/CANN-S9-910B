/**
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Concat 算子 host 侧实现：支持任意维度拼接，多输入（tensor_list）。
 *
 * 数据视图：
 *   每个输入 i 的内存可视为 [beforeDimSize, inputCatLen[i], afterDimSize]
 *   输出为                   [beforeDimSize, totalCatLen,     afterDimSize]
 *   afterDimSize 维在内存上连续。
 *
 * 核切分：优先枚举行×输出列的二维方案。一个核只遍历自己列区间相交
 * 的输入；输出行不能安全列分时保留整行切分。
 */
#include "concat_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace optiling {

constexpr uint32_t DATA_BLOCK_BYTES = 32;
constexpr uint32_t PREFERRED_COL_BYTES = 512;
constexpr uint32_t TILE_BYTES = 64 * 1024;
constexpr uint32_t DMA_SETUP_COST = 4096;
constexpr uint64_t IDENTITY_MIN_BYTES_PER_CORE = 128 * 1024;
constexpr uint64_t IDENTITY_TILING_KEY = 2;

static uint32_t Gcd(uint32_t lhs, uint32_t rhs)
{
    while (rhs != 0) {
        uint32_t next = lhs % rhs;
        lhs = rhs;
        rhs = next;
    }
    return lhs;
}

static uint64_t AlignUp(uint64_t value, uint32_t alignment)
{
    if (value == 0) return 0;
    return ((value - 1) / alignment + 1) * alignment;
}

static uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return value / divisor + (value % divisor != 0);
}

static uint64_t CeilDiv64(uint64_t value, uint64_t divisor)
{
    return value / divisor + (value % divisor != 0);
}

static bool CheckedMul(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) return false;
    result = lhs * rhs;
    return true;
}

static bool CheckedAdd(uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) return false;
    result = lhs + rhs;
    return true;
}

static uint32_t SelectIdentityCoreNum(uint64_t totalBytes, uint32_t availableCores)
{
    if (totalBytes == 0) return 1;
    const uint64_t workCores = CeilDiv64(totalBytes, IDENTITY_MIN_BYTES_PER_CORE);
    const uint64_t ownershipUnits = CeilDiv64(totalBytes, DATA_BLOCK_BYTES);
    const uint64_t used = std::min<uint64_t>(availableCores,
                                             std::min<uint64_t>(workCores, ownershipUnits));
    return static_cast<uint32_t>(std::max<uint64_t>(1, used));
}

struct SplitChoice {
    uint32_t usedCoreNum = 1;
    uint32_t splitMode = 0;
    uint32_t rowPeriod = 1;
    uint32_t rowSliceNum = 1;
    uint32_t colCoreNum = 1;
    uint32_t colBlockBytes = 0;
    uint64_t worstCost = ~0ULL;
};

static uint64_t EstimateColumnCost(uint32_t rows, uint64_t colBegin, uint64_t colEnd,
                                   uint32_t inputNum, const uint32_t *inputCatLen,
                                   const uint32_t *inputCatOffset, uint64_t catUnitBytes)
{
    uint64_t cost = 0;
    for (uint32_t input = 0; input < inputNum; ++input) {
        const uint64_t inputBegin = static_cast<uint64_t>(inputCatOffset[input]) * catUnitBytes;
        const uint64_t inputEnd = inputBegin + static_cast<uint64_t>(inputCatLen[input]) * catUnitBytes;
        const uint64_t begin = std::max<uint64_t>(colBegin, inputBegin);
        const uint64_t end = std::min<uint64_t>(colEnd, inputEnd);
        if (begin >= end) continue;
        const uint64_t pieceBytes = end - begin;
        const uint64_t alignedPieceBytes = AlignUp(pieceBytes, DATA_BLOCK_BYTES);
        // 超出 UB 的窄列会逐行再切块；模型保守估计其 DMA 数。
        const uint64_t rowsPerCopy = std::max<uint64_t>(1, TILE_BYTES / alignedPieceBytes);
        const uint64_t copyCount = (static_cast<uint64_t>(rows) + rowsPerCopy - 1) / rowsPerCopy;
        cost += copyCount * DMA_SETUP_COST + static_cast<uint64_t>(rows) * alignedPieceBytes;
    }
    return cost;
}

static SplitChoice ChooseSplit(uint32_t availableCores, uint32_t beforeDimSize, uint64_t rowBytes,
                               uint32_t inputNum, const uint32_t *inputCatLen,
                               const uint32_t *inputCatOffset, uint64_t catUnitBytes)
{
    SplitChoice best;
    if (beforeDimSize == 0 || rowBytes == 0) return best;

    const uint32_t rowPeriod = DATA_BLOCK_BYTES /
                               Gcd(static_cast<uint32_t>(rowBytes % DATA_BLOCK_BYTES), DATA_BLOCK_BYTES);
    const uint32_t rowCores = std::max(1U, std::min(availableCores, beforeDimSize));
    const uint32_t rowsPerCore = CeilDiv(beforeDimSize, rowCores);
    best.usedCoreNum = rowCores;
    best.rowPeriod = rowPeriod;
    best.colBlockBytes = rowBytes > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(rowBytes);
    best.worstCost = EstimateColumnCost(rowsPerCore, 0, rowBytes, inputNum,
                                        inputCatLen, inputCatOffset, catUnitBytes);

    // A fixed logical column boundary cannot be 32B aligned for every row when
    // rowBytes is not 32B aligned.  Keep those layouts on the race-free row path.
    if ((rowBytes % DATA_BLOCK_BYTES) != 0 || rowBytes > UINT32_MAX) return best;

    const uint32_t rowBytes32 = static_cast<uint32_t>(rowBytes);
    auto consider = [&](uint32_t columnBytes) {
        if (columnBytes == 0) return;
        const uint32_t colCores = CeilDiv(rowBytes32, columnBytes);
        if (colCores < 2 || colCores > availableCores) return;
        for (uint32_t rowSlices = 1; rowSlices <= availableCores / colCores; ++rowSlices) {
            const uint32_t used = rowSlices * colCores;
            const uint32_t rows = CeilDiv(beforeDimSize, rowSlices);
            uint64_t worst = 0;
            for (uint32_t col = 0; col < colCores; ++col) {
                const uint32_t begin = col * columnBytes;
                const uint32_t end = std::min(rowBytes32, begin + columnBytes);
                worst = std::max(worst, EstimateColumnCost(rows, begin, end, inputNum,
                                                           inputCatLen, inputCatOffset, catUnitBytes));
            }
            // Prefer 512B boundaries for indistinguishable estimates, while
            // allowing the 32B candidate to use every AIV when it models faster.
            if (worst < best.worstCost ||
                (worst == best.worstCost && columnBytes == PREFERRED_COL_BYTES && best.splitMode != 1)) {
                best.usedCoreNum = used;
                best.splitMode = 1;
                best.rowSliceNum = rowSlices;
                best.colCoreNum = colCores;
                best.colBlockBytes = columnBytes;
                best.worstCost = worst;
            }
        }
    };

    if (rowBytes32 >= PREFERRED_COL_BYTES) consider(PREFERRED_COL_BYTES);
    const uint32_t maxColumnParts = std::min(availableCores, CeilDiv(rowBytes32, DATA_BLOCK_BYTES));
    for (uint32_t parts = 2; parts <= maxColumnParts; ++parts) {
        const uint32_t columnBytes = static_cast<uint32_t>(AlignUp(CeilDiv(rowBytes32, parts), DATA_BLOCK_BYTES));
        consider(columnBytes);
    }
    return best;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    // 1. 读取属性 dim
    auto attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t *dimAttrPtr = attrs->GetInt(0);
    if (dimAttrPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t dimAttr = *dimAttrPtr;

    // 2. 获取动态输入张量个数
    auto computeNodeInfoPtr = context->GetComputeNodeInfo();
    if (computeNodeInfoPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto idxInstanceInfoPtr = computeNodeInfoPtr->GetInputInstanceInfo(0);  // 输入 IR index = 0
    if (idxInstanceInfoPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint64_t tensorNum = idxInstanceInfoPtr->GetInstanceNum();
    if (tensorNum == 0 || tensorNum > MAX_CONCAT_INPUT_NUM) {
        return ge::GRAPH_FAILED;
    }

    // 3. 取第 0 个非零维输入的 shape/dtype 作为基准
    auto x0ShapePtr = context->GetDynamicInputShape(0, 0);
    if (x0ShapePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto &x0Storage = x0ShapePtr->GetStorageShape();
    uint32_t dimNum = static_cast<uint32_t>(x0Storage.GetDimNum());
    if (dimNum == 0) {
        return ge::GRAPH_FAILED;
    }

    // dim 转非负
    int64_t dim = dimAttr;
    if (dim < 0) {
        dim += static_cast<int64_t>(dimNum);
    }
    if (dim < 0 || dim >= static_cast<int64_t>(dimNum)) {
        return ge::GRAPH_FAILED;
    }
    uint32_t udim = static_cast<uint32_t>(dim);

    // 4. 计算 beforeDimSize / afterDimSize / 各输入 catLen / totalCatLen / offset
    uint64_t beforeDimSize64 = 1;
    for (uint32_t i = 0; i < udim; i++) {
        const int64_t shapeDim = x0Storage.GetDim(i);
        if (shapeDim < 0 ||
            !CheckedMul(beforeDimSize64, static_cast<uint64_t>(shapeDim), beforeDimSize64)) {
            return ge::GRAPH_FAILED;
        }
    }
    uint64_t afterDimSize64 = 1;
    for (uint32_t i = udim + 1; i < dimNum; i++) {
        const int64_t shapeDim = x0Storage.GetDim(i);
        if (shapeDim < 0 ||
            !CheckedMul(afterDimSize64, static_cast<uint64_t>(shapeDim), afterDimSize64)) {
            return ge::GRAPH_FAILED;
        }
    }

    uint32_t inputCatLenArr[MAX_CONCAT_INPUT_NUM] = {0};
    uint32_t inputCatOffsetArr[MAX_CONCAT_INPUT_NUM] = {0};
    uint64_t totalCatLen64 = 0;
    for (uint64_t i = 0; i < tensorNum; i++) {
        auto xiShapePtr = context->GetDynamicInputShape(0, i);
        if (xiShapePtr == nullptr) {
            inputCatLenArr[i] = 0;
            if (totalCatLen64 > UINT32_MAX) return ge::GRAPH_FAILED;
            inputCatOffsetArr[i] = static_cast<uint32_t>(totalCatLen64);
            continue;
        }
        const auto &xiStorage = xiShapePtr->GetStorageShape();
        uint64_t catLen64 = 0;
        if (xiStorage.GetDimNum() > udim) {
            const int64_t shapeDim = xiStorage.GetDim(udim);
            if (shapeDim < 0) return ge::GRAPH_FAILED;
            catLen64 = static_cast<uint64_t>(shapeDim);
        }
        if (catLen64 > UINT32_MAX || totalCatLen64 > UINT32_MAX) {
            return ge::GRAPH_FAILED;
        }
        inputCatLenArr[i] = static_cast<uint32_t>(catLen64);
        inputCatOffsetArr[i] = static_cast<uint32_t>(totalCatLen64);
        if (!CheckedAdd(totalCatLen64, catLen64, totalCatLen64) || totalCatLen64 > UINT32_MAX) {
            return ge::GRAPH_FAILED;
        }
    }

    // 5. 数据类型字节数
    auto x0DescPtr = context->GetDynamicInputDesc(0, 0);
    if (x0DescPtr == nullptr || beforeDimSize64 > UINT32_MAX || afterDimSize64 > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType dtype = x0DescPtr->GetDataType();
    uint32_t dtypeSize = 1;
    switch (dtype) {
        case ge::DT_FLOAT:   dtypeSize = 4; break;
        case ge::DT_FLOAT16: dtypeSize = 2; break;
        case ge::DT_BF16:    dtypeSize = 2; break;
        case ge::DT_INT32:   dtypeSize = 4; break;
        case ge::DT_INT16:   dtypeSize = 2; break;
        case ge::DT_INT8:    dtypeSize = 1; break;
        case ge::DT_UINT8:   dtypeSize = 1; break;
        default: return ge::GRAPH_FAILED;
    }

    const uint32_t beforeDimSize = static_cast<uint32_t>(beforeDimSize64);
    const uint32_t afterDimSize = static_cast<uint32_t>(afterDimSize64);
    const uint32_t totalCatLen = static_cast<uint32_t>(totalCatLen64);

    // 6. 用运行时平台核数建模，而不是把具体卡型的核数写死在算子里。
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t platformAivCores = platform.GetCoreNumAiv();
    uint32_t availableCores = platformAivCores > 0 ? static_cast<uint32_t>(platformAivCores) : 1U;
    uint64_t catUnitBytes = 0;
    uint64_t outputRowBytes = 0;
    uint64_t totalOutputBytes = 0;
    if (!CheckedMul(afterDimSize64, dtypeSize, catUnitBytes) ||
        !CheckedMul(totalCatLen64, catUnitBytes, outputRowBytes) ||
        !CheckedMul(beforeDimSize64, outputRowBytes, totalOutputBytes)) {
        return ge::GRAPH_FAILED;
    }

    if (tensorNum == 1) {
        const uint32_t usedCoreNum = SelectIdentityCoreNum(totalOutputBytes, availableCores);
        ConcatIdentityTilingData identityTiling;
        identityTiling.set_totalBytes(totalOutputBytes);
        identityTiling.set_usedCoreNum(usedCoreNum);
        identityTiling.set_tileBytes(TILE_BYTES);
        context->SetTilingKey(IDENTITY_TILING_KEY);
        context->SetBlockDim(usedCoreNum);
        identityTiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                                    context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(identityTiling.GetDataSize());
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    ConcatCustomTilingData tiling;
    SplitChoice split = ChooseSplit(availableCores, beforeDimSize, outputRowBytes,
                                    static_cast<uint32_t>(tensorNum), inputCatLenArr,
                                    inputCatOffsetArr, catUnitBytes);

    // 7. 写 tiling
    tiling.set_inputNum(static_cast<uint32_t>(tensorNum));
    tiling.set_dim(udim);
    tiling.set_dimNum(dimNum);
    tiling.set_dtypeSize(dtypeSize);
    tiling.set_beforeDimSize(beforeDimSize);
    tiling.set_afterDimSize(afterDimSize);
    tiling.set_totalCatLen(totalCatLen);
    tiling.set_usedCoreNum(split.usedCoreNum);
    tiling.set_splitMode(split.splitMode);
    tiling.set_rowPeriod(split.rowPeriod);
    tiling.set_rowSliceNum(split.rowSliceNum);
    tiling.set_colCoreNum(split.colCoreNum);
    tiling.set_colBlockBytes(split.colBlockBytes);

    // 数组字段整体写入（set 方法接收指针，内部 memcpy）
    tiling.set_inputCatLen(inputCatLenArr);
    tiling.set_inputCatOffset(inputCatOffsetArr);

    context->SetTilingKey(0);
    context->SetBlockDim(split.usedCoreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling


namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    // 输出 shape = 第 0 个输入 shape，但 dim 维替换为所有输入该维之和
    const gert::Shape *x0Shape = context->GetDynamicInputShape(0, 0);
    if (x0Shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape *yShape = context->GetOutputShape(0);

    auto attrs = context->GetAttrs();
    int64_t dimAttr = 0;
    if (attrs != nullptr) {
        const int64_t *dimAttrPtr = attrs->GetInt(0);
        if (dimAttrPtr != nullptr) {
            dimAttr = *dimAttrPtr;
        }
    }

    int64_t dimNum = static_cast<int64_t>(x0Shape->GetDimNum());
    int64_t dim = dimAttr;
    if (dim < 0) {
        dim += dimNum;
    }

    yShape->SetDimNum(dimNum);
    for (int64_t i = 0; i < dimNum; i++) {
        yShape->SetDim(i, x0Shape->GetDim(i));
    }

    // 累加所有输入沿 dim 维的长度
    int64_t totalDim = 0;
    for (int64_t i = 0;; i++) {
        const gert::Shape *xiShape = context->GetDynamicInputShape(0, i);
        if (xiShape == nullptr) {
            break;
        }
        if (static_cast<int64_t>(xiShape->GetDimNum()) > dim) {
            totalDim += xiShape->GetDim(dim);
        }
    }
    yShape->SetDim(dim, totalDim);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetDynamicInputDataType(0, 0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge


namespace ops {
class Concat : public OpDef {
public:
    explicit Concat(const char *name) : OpDef(name)
    {
        this->Input("srcList")
            .ParamType(DYNAMIC)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("dst")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("dim").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Concat);
}  // namespace ops
