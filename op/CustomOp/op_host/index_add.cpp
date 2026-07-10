/**
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * IndexAdd 算子 host 侧实现。
 * 参考 torch.index_add：output = copy(self)，再沿 dim 维做 scatter-add：
 *   对每个 i ∈ [0, M)，output[..., index[i], ...] += source[..., i, ...]。
 *
 * 数据视图：[beforeDimSize, dimLen, afterDimSize]，afterDimSize 维内存连续。
 *   self/output: [beforeDimSize, dimLen, afterDimSize]
 *   source:      [beforeDimSize, M,       afterDimSize]
 *   index:       [M] (int32)
 *
 * 核切分（两模式，各核输出区域互不重叠 → 无 WAW、无需原子，适配任意对齐）：
 *   ROW   (mode=0)：按 beforeDim 行切分。beforeDimSize 较大时优先。
 *   AFTER (mode=1)：按 afterDim 轴切分。beforeDimSize 较小（含 dim=0）时优先。
 *   host 取 rowCores=min(20,beforeDimSize) 与 afterCores=min(20,afterDimSize) 中较大者对应模式。
 */
#include "index_add_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

#include <cstdint>
#include <algorithm>

namespace optiling {

// 910B4-1 单卡 AICore 数量
constexpr uint32_t AICORE_NUM = 20;

// dtype 枚举（与 kernel 侧 dispatch 一致）
enum class IndexAddDtype : uint32_t {
    FLOAT = 0,
    BF16 = 1,
    HALF = 2,
    INT32 = 3,
    INT8 = 4,
};

// 核切分模式
constexpr uint32_t MODE_ROW = 0;
constexpr uint32_t MODE_AFTER = 1;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    IndexAddCustomTilingData tiling;

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

    // 2. 获取 self/index/source 的 shape
    const gert::StorageShape *selfShape = context->GetInputShape(0);   // self
    const gert::StorageShape *indexShape = context->GetInputShape(1);  // index
    const gert::StorageShape *sourceShape = context->GetInputShape(2);  // source
    if (selfShape == nullptr || indexShape == nullptr || sourceShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto &selfStorage = selfShape->GetStorageShape();
    uint32_t dimNum = static_cast<uint32_t>(selfStorage.GetDimNum());
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

    // 3. 计算 beforeDimSize / dimLen / afterDimSize
    uint32_t beforeDimSize = 1;
    for (uint32_t i = 0; i < udim; i++) {
        beforeDimSize *= static_cast<uint32_t>(selfStorage.GetDim(i));
    }
    uint32_t dimLen = static_cast<uint32_t>(selfStorage.GetDim(udim));
    uint32_t afterDimSize = 1;
    for (uint32_t i = udim + 1; i < dimNum; i++) {
        afterDimSize *= static_cast<uint32_t>(selfStorage.GetDim(i));
    }

    // 4. index 长度 M（1D）
    const auto &indexStorage = indexShape->GetStorageShape();
    uint32_t indexLen = static_cast<uint32_t>(indexStorage.GetDim(0));

    // 5. 数据类型字节数 + dtype 枚举
    auto selfDescPtr = context->GetInputDesc(0);
    ge::DataType dtype = selfDescPtr->GetDataType();
    uint32_t dtypeSize = 0;
    uint32_t dtypeCode = static_cast<uint32_t>(IndexAddDtype::FLOAT);
    switch (dtype) {
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

    // 6. 核切分模式选择：取能给出更多核的模式
    uint32_t rowCores = (beforeDimSize == 0) ? 0 : std::min(AICORE_NUM, beforeDimSize);
    uint32_t afterCores = (afterDimSize == 0) ? 0 : std::min(AICORE_NUM, afterDimSize);
    uint32_t mode;
    uint32_t usedCoreNum;
    if (rowCores >= afterCores && rowCores > 0) {
        mode = MODE_ROW;
        usedCoreNum = rowCores;
    } else if (afterCores > 0) {
        mode = MODE_AFTER;
        usedCoreNum = afterCores;
    } else {
        mode = MODE_ROW;
        usedCoreNum = 1;
    }

    // 7. scatter 阶段连续向量内部分块（按 dtype 精算 UB，双缓冲下留余量）。
    //    每个分块需要：源读 buf + 输出读(RMW) buf（均为 InputT），非原生 dtype 还需 2 个 ComputeT buf。
    //    这里取保守值，确保 2×(bufTotal) ≤ ~160KB。
    uint32_t scatterTileLen = 0;
    switch (dtype) {
        case ge::DT_FLOAT:   scatterTileLen = 8192;  break;  // 2 buf × 4B × 2 dbl = 128KB
        case ge::DT_INT32:   scatterTileLen = 8192;  break;
        case ge::DT_FLOAT16:  scatterTileLen = 16384; break;  // 2 buf × 2B × 2 dbl = 128KB
        case ge::DT_BF16:     scatterTileLen = 4096;  break;  // 4 buf (2×InputT2B + 2×float4B)=12B × 2 dbl = 96KB
        case ge::DT_INT8:     scatterTileLen = 16384; break;  // 4 buf (2×1B + 2×half2B)=6B × 2 dbl = 192KB→收紧
        default:              scatterTileLen = 4096;  break;
    }
    // int8 实际 6B×2dbl×16384=192KB 超 UB，收紧到 12288（=144KB）
    if (dtype == ge::DT_INT8) {
        scatterTileLen = 12288;
    }

    // 8. 写 tiling
    tiling.set_dim(udim);
    tiling.set_beforeDimSize(beforeDimSize);
    tiling.set_dimLen(dimLen);
    tiling.set_afterDimSize(afterDimSize);
    tiling.set_indexLen(indexLen);
    tiling.set_dtypeSize(dtypeSize);
    tiling.set_dtype(dtypeCode);
    tiling.set_usedCoreNum(usedCoreNum);
    tiling.set_mode(mode);
    tiling.set_scatterTileLen(scatterTileLen);

    context->SetBlockDim(usedCoreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // 所有 dtype 均直接在 output 上逐次 RMW（与 torch 的逐次 RNE 舍入一致），无需 workspace。
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling


namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    // output shape == self shape
    const gert::Shape *selfShape = context->GetInputShape(0);
    if (selfShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape *yShape = context->GetOutputShape(0);
    *yShape = *selfShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge


namespace ops {
// 注册名必须是 IndexAdd，build 后生成 aclnnIndexAdd 覆盖 torch_npu 内置同名算子。
class IndexAdd : public OpDef {
public:
    explicit IndexAdd(const char *name) : OpDef(name)
    {
        this->Input("self")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("index")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("source")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->Attr("dim").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(IndexAdd);
}  // namespace ops
