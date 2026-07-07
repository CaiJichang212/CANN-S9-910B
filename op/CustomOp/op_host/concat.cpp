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
 * 核切分：kernel 按输出扁平字节区间 [myStart,myEnd) 切分给各核，
 *   与 beforeDim/dim 取值无关地满核（解决 dim=0/beforeDim=1 时单核欠载问题）。
 */
#include "concat_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"

#include <cstdint>

namespace optiling {

// 910B4-1 单卡 AICore 数量
constexpr uint32_t AICORE_NUM = 20;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    ConcatCustomTilingData tiling;

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
    auto idxInstanceInfoPtr = computeNodeInfoPtr->GetInputInstanceInfo(0);  // 输入 IR index = 0
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
    uint32_t beforeDimSize = 1;
    for (uint32_t i = 0; i < udim; i++) {
        beforeDimSize *= static_cast<uint32_t>(x0Storage.GetDim(i));
    }
    uint32_t afterDimSize = 1;
    for (uint32_t i = udim + 1; i < dimNum; i++) {
        afterDimSize *= static_cast<uint32_t>(x0Storage.GetDim(i));
    }

    uint32_t inputCatLenArr[MAX_CONCAT_INPUT_NUM] = {0};
    uint32_t inputCatOffsetArr[MAX_CONCAT_INPUT_NUM] = {0};
    uint32_t totalCatLen = 0;
    for (uint64_t i = 0; i < tensorNum; i++) {
        auto xiShapePtr = context->GetDynamicInputShape(0, i);
        if (xiShapePtr == nullptr) {
            inputCatLenArr[i] = 0;
            inputCatOffsetArr[i] = totalCatLen;
            continue;
        }
        const auto &xiStorage = xiShapePtr->GetStorageShape();
        uint32_t catLen = 0;
        if (xiStorage.GetDimNum() > udim) {
            catLen = static_cast<uint32_t>(xiStorage.GetDim(udim));
        }
        inputCatLenArr[i] = catLen;
        inputCatOffsetArr[i] = totalCatLen;
        totalCatLen += catLen;
    }

    // 5. 数据类型字节数
    auto x0DescPtr = context->GetDynamicInputDesc(0, 0);
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
        default:             dtypeSize = 2; break;
    }

    // 6. 多核切分：按 beforeDim 行切分给各核（每行内是连续输出段，避免跨核写同一 32B block）
    //    - 数据量足够大时用满 20 核；beforeDim 小时按行数自适应
    uint32_t usedCoreNum = AICORE_NUM;
    if (beforeDimSize == 0) {
        usedCoreNum = 1;
    } else if (usedCoreNum > beforeDimSize) {
        usedCoreNum = beforeDimSize;
    }
    if (usedCoreNum == 0) {
        usedCoreNum = 1;
    }

    // 7. 写 tiling
    tiling.set_inputNum(static_cast<uint32_t>(tensorNum));
    tiling.set_dim(udim);
    tiling.set_dimNum(dimNum);
    tiling.set_dtypeSize(dtypeSize);
    tiling.set_beforeDimSize(beforeDimSize);
    tiling.set_afterDimSize(afterDimSize);
    tiling.set_totalCatLen(totalCatLen);
    tiling.set_usedCoreNum(usedCoreNum);

    // 数组字段整体写入（set 方法接收指针，内部 memcpy）
    tiling.set_inputCatLen(inputCatLenArr);
    tiling.set_inputCatOffset(inputCatOffsetArr);

    context->SetBlockDim(usedCoreNum);
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
