/**
 * @file transpose.cpp
 * @brief Transpose (torch.permute) host: TilingFunc / InferShape / InferDataType / OpDef.
 *
 * host 侧完成全部几何计算：读 input shape + dims → 算 outShape / inStride /
 * 末维源步长 S / numRows / 外层映射；并按 dims 选路径：
 *   - 末维源端连续 (S==1) → COPY (mode=0)
 *   - 末两维相邻交换且前缀 identity → TRANSPOSE (mode=1)
 */
#include "transpose_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>

namespace optiling {
static constexpr uint32_t MAX_BLOCK_DIM = 20;
static constexpr uint32_t MAX_DIM = 8;
static constexpr uint32_t ALIGN = 16; // 向量 stride 重排建议 16 倍数
// TransposeGeneric performs scalar LocalTensor GetValue/SetValue accesses.
// Keep each source/destination queue buffer below one 64 KiB local address
// window.  With two queues and double buffering, the generic path uses at
// most 128 KiB after the mandatory 16-element tile alignment.
static constexpr uint32_t GENERIC_TRANSPOSE_UB_BUFFER_BYTES = 32 * 1024;

static uint32_t DTypeToSize(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT16: return 2;
        case ge::DT_FLOAT:
        case ge::DT_INT32:   return 4;
        case ge::DT_INT8:    return 1;
        default:             return 4;
    }
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    TransposeTilingData tiling;

    const gert::StorageShape *xShape = context->GetInputShape(0);
    const auto &inShape = xShape->GetStorageShape();
    uint32_t ndim = static_cast<uint32_t>(inShape.GetDimNum());
    if (ndim == 0 || ndim > MAX_DIM) {
        return ge::GRAPH_FAILED;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const auto *dimsVec = attrs->GetListInt(0);
    if (dimsVec == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t dimsN = static_cast<uint32_t>(dimsVec->GetSize());
    const int64_t *dimsData = dimsVec->GetData();
    if (dimsN != ndim) {
        return ge::GRAPH_FAILED;
    }

    // 规整 dims (负值 +ndim)
    uint32_t dimsArr[MAX_DIM] = {0};
    for (uint32_t i = 0; i < ndim; i++) {
        int64_t d = dimsData[i];
        if (d < 0) {
            d += ndim;
        }
        dimsArr[i] = static_cast<uint32_t>(d);
    }

    // input shape + 元素总数 + 行主序 inStride (元素)
    uint32_t inSh[MAX_DIM] = {0};
    uint32_t inStride[MAX_DIM] = {0};
    uint64_t total = 1;
    for (uint32_t i = 0; i < ndim; i++) {
        inSh[i] = static_cast<uint32_t>(inShape.GetDim(i));
        total *= inSh[i];
    }
    inStride[ndim - 1] = 1;
    for (int32_t i = static_cast<int32_t>(ndim) - 2; i >= 0; i--) {
        inStride[i] = inStride[i + 1] * inSh[i + 1];
    }

    const gert::Tensor *inTensor = context->GetInputTensor(0);
    ge::DataType inDt = (inTensor != nullptr) ? inTensor->GetDataType() : ge::DT_FLOAT;
    uint32_t dtypeSize = DTypeToSize(inDt);

    // 末输出维 W = inSh[dims[ndim-1]]，源步长 S = inStride[dims[ndim-1]]
    uint32_t srcLastDim = dimsArr[ndim - 1];
    uint32_t W = inSh[srcLastDim];
    uint32_t S = inStride[srcLastDim];

    // 检测 TRANSPOSE 路径：末两维相邻交换 + 前缀 identity
    //   dims = [0,1,...,ndim-3, ndim-1, ndim-2]  (ndim>=2)
    bool isTranspose = false;
    if (ndim >= 2 && dimsArr[ndim - 1] == ndim - 2 && dimsArr[ndim - 2] == ndim - 1) {
        isTranspose = true;
        for (uint32_t i = 0; i + 2 < ndim; i++) {
            if (dimsArr[i] != i) {
                isTranspose = false;
                break;
            }
        }
    }

    // A contiguous source inner dimension never needs UB transpose.  In
    // particular, (M,1)->(1,M) has identical linear storage and must take the
    // DMA COPY path before the 2D-transpose special case is considered.
    // For int8 rows shorter than one 32B block, use the established strided
    // COPY fallback instead of issuing many narrow DataCopyPad rows on the
    // generic transpose path.
    bool narrowInt8Transpose = isTranspose && dtypeSize == 1 && inSh[ndim - 1] < 32;
    uint32_t mode;
    if (S == 1) {
        mode = 0;
    } else if (isTranspose && !narrowInt8Transpose) {
        mode = 1;
    } else {
        mode = 0;
    }

    // 外层 (输出维 0..ndim-2) 映射
    uint32_t outerCount = (ndim >= 1) ? (ndim - 1) : 0;
    uint32_t outerOutShape[MAX_DIM] = {0};
    uint32_t outerSrcStride[MAX_DIM] = {0};
    for (uint32_t i = 0; i < outerCount; i++) {
        outerOutShape[i] = inSh[dimsArr[i]];
        outerSrcStride[i] = inStride[dimsArr[i]];
    }

    uint32_t numRows = (W > 0) ? static_cast<uint32_t>(std::min<uint64_t>(total / W, 0xFFFFFFFFull)) : 0;
    // COPY work is distributed by output rows.  TRANSPOSE recalculates this
    // from its tile count after tile geometry is known.
    uint32_t blockDim = std::min(MAX_BLOCK_DIM, std::max(1u, numRows));

    // COPY tileLen：贴满 UB (双缓冲，单 buffer ~80KB)
    uint32_t ubTileBytes = 80 * 1024;
    uint32_t maxTile = ubTileBytes / std::max(1u, dtypeSize);
    uint32_t copyTileLen = std::min(W, maxTile);
    if (copyTileLen == 0) {
        copyTileLen = 1;
    }

    // TRANSPOSE 参数
    uint32_t transBatch = 1;
    uint32_t transM = 0;
    uint32_t transN = 0;
    uint32_t tileM = 0;
    uint32_t tileN = 0;
    if (mode == 1) {
        transM = inSh[ndim - 2]; // A
        transN = inSh[ndim - 1]; // B
        for (uint32_t i = 0; i + 2 < ndim; i++) {
            transBatch *= inSh[i];
        }
        // tile 选择：
        //  - half: 固定 16×16（配合 vtranspose 硬件指令，UB 内紧凑 16×16）。
        //  - 其余 dtype: 16 倍数贴满 UB（走通用逐元素转置，tile 大可摊薄逐元素开销）。
        uint32_t tn;
        uint32_t tm;
        if (dtypeSize == 2) { // half
            tn = 16;
            tm = 16;
        } else {
            uint32_t budget = GENERIC_TRANSPOSE_UB_BUFFER_BYTES;
            tn = std::min(transN, static_cast<uint32_t>(256));
            tn = std::max(ALIGN, (tn / ALIGN) * ALIGN);
            if (tn == 0) {
                tn = ALIGN;
            }
            // The input tile has one 32B-aligned UB row per M row.  Bound the
            // row count by that padded footprint (rather than raw elements),
            // keeping each of the two single-buffer queues below the scalar
            // LocalTensor address window even for int8 narrow-N matrices.
            // DataCopyPad blockCount is also uint16_t and accepts at most
            // 4095 rows.
            uint32_t paddedRowBytes = ((tn * dtypeSize + 31) / 32) * 32;
            uint32_t maxTileM = std::min<uint32_t>(budget / paddedRowBytes, 4095);
            tm = std::min(transM, maxTileM);
            tm = (tm / ALIGN) * ALIGN;
            if (tm == 0) {
                tm = ALIGN;
            }
        }
        tileM = tm;
        tileN = tn;

        // TRANSPOSE work units are tiles, not rows.  Basing blockDim on rows
        // can leave a large matrix with a narrow N dimension on one core.
        uint64_t mTiles = (static_cast<uint64_t>(transM) + tileM - 1) / tileM;
        uint64_t nTiles = (static_cast<uint64_t>(transN) + tileN - 1) / tileN;
        uint64_t totalTiles = static_cast<uint64_t>(transBatch) * mTiles * nTiles;
        blockDim = static_cast<uint32_t>(std::min<uint64_t>(MAX_BLOCK_DIM,
            std::max<uint64_t>(1, totalTiles)));
    }

    tiling.set_mode(mode);
    tiling.set_total(static_cast<uint32_t>(std::min<uint64_t>(total, 0xFFFFFFFFull)));
    tiling.set_ndim(ndim);
    tiling.set_dtypeSize(dtypeSize);
    tiling.set_blockDim(blockDim);
    tiling.set_W(W);
    tiling.set_numRows(numRows);
    tiling.set_srcStrideInner(S);
    tiling.set_outerCount(outerCount);
    tiling.set_outerOutShape(outerOutShape);
    tiling.set_outerSrcStride(outerSrcStride);
    tiling.set_copyTileLen(copyTileLen);
    tiling.set_transBatch(transBatch);
    tiling.set_transM(transM);
    tiling.set_transN(transN);
    tiling.set_tileM(tileM);
    tiling.set_tileN(tileN);

    context->SetBlockDim(blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *yShape = context->GetOutputShape(0);
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    const auto *dimsVec = attrs->GetListInt(0);
    if (dimsVec == nullptr) {
        *yShape = *xShape;
        return GRAPH_SUCCESS;
    }
    size_t n = dimsVec->GetSize();
    const int64_t *dimsData = dimsVec->GetData();
    size_t xDim = xShape->GetDimNum();
    yShape->SetDimNum(n);
    for (size_t i = 0; i < n; i++) {
        int64_t d = dimsData[i];
        if (d < 0) {
            d += static_cast<int64_t>(xDim);
        }
        yShape->SetDim(i, xShape->GetDim(static_cast<size_t>(d)));
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Transpose : public OpDef {
public:
    explicit Transpose(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT8})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("dims").ListInt();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(Transpose);
} // namespace ops
