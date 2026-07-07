/**
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Concat 算子 kernel 侧实现（通用版本，支持任意 shape / dim / dtype）。
 *
 * 数据视图：
 *   每个输入 i 的内存可视为 [beforeDimSize, inputCatLen[i], afterDimSize]
 *   输出为                   [beforeDimSize, totalCatLen,     afterDimSize]
 *   afterDimSize 维在内存上连续。
 *
 * 并行/流水策略：
 *   - 按 beforeDim 行数切分给各核（每行内是一个连续的输出段）
 *   - 每行按输入依次拷贝；每个 chunk 内部按固定 TILE_BYTES 分批
 *
 * DataCopyPad（CANN 8.5 Ascend C）：
 *   - DataCopyExtParams(blockCount, blockLen, srcStride, dstStride, rsv)
 *       blockLen 单位为「字节」，uint32_t
 *   - 使用 uint8_t 视角搬运，统一处理所有 dtype
 */
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

using namespace AscendC;

constexpr uint32_t MAX_INPUT_NUM = 256;  // 与 host 端 MAX_CONCAT_INPUT_NUM 保持一致（ACLNN 框架上限 256）
// 单个 UB tile 字节数：贴满 UB（910B 192KB，可用 ~184KB），双队列 2+2 buffer 共 4×TILE ≤ 184KB。
constexpr uint32_t TILE_BYTES = 32768;   // 32KB，4×32KB=128KB 充裕（原 8KB 太小）

class KernelConcat {
public:
    __aicore__ inline KernelConcat() {}

    template <typename TilingT>
    __aicore__ inline void Init(GM_ADDR srcList, GM_ADDR dst, const TilingT &tiling)
    {
        inputNum_      = tiling.inputNum;
        afterDimSize_  = tiling.afterDimSize;
        dtypeSize_     = tiling.dtypeSize;
        totalCatLen_   = tiling.totalCatLen;
        usedCoreNum_   = tiling.usedCoreNum;
        beforeDimSize_ = tiling.beforeDimSize;

        for (uint32_t i = 0; i < inputNum_; i++) {
            inputCatLen_[i]    = tiling.inputCatLen[i];
            inputCatOffset_[i] = tiling.inputCatOffset[i];
        }

        AscendC::ListTensorDesc listDesc(reinterpret_cast<__gm__ void*>(srcList));
        for (uint32_t i = 0; i < inputNum_; i++) {
            __gm__ uint8_t *ptr = listDesc.GetDataPtr<__gm__ uint8_t>(i);
            inputGm_[i].SetGlobalBuffer(ptr);
        }

        yGm_.SetGlobalBuffer((__gm__ uint8_t *)dst);

        // 使用两个队列：VECIN 用于 GM->UB，VECOUT 用于 UB->GM
        // 中间通过 EnQue/DeQue 完成 MTE2→VECTOR/MTE3 同步
        pipe.InitBuffer(inQueue_,  2, TILE_BYTES);
        pipe.InitBuffer(outQueue_, 2, TILE_BYTES);
    }

    __aicore__ inline void Process()
    {
        uint32_t coreId = GetBlockIdx();
        if (coreId >= usedCoreNum_ || inputNum_ == 0 || beforeDimSize_ == 0) {
            return;
        }

        // 按行切分，每行整段（totalCatLen * afterDimSize 字节）由单核处理
        uint32_t perCoreRows = (beforeDimSize_ + usedCoreNum_ - 1) / usedCoreNum_;
        uint32_t startRow = coreId * perCoreRows;
        uint32_t endRow = startRow + perCoreRows;
        if (endRow > beforeDimSize_) endRow = beforeDimSize_;
        if (startRow >= endRow) return;
        uint32_t numRows = endRow - startRow;

        for (uint32_t i = 0; i < inputNum_; i++) {
            uint32_t catLen = inputCatLen_[i];
            if (catLen == 0) continue;
            CopyInputRows(i, startRow, numRows, catLen);
        }
    }

private:
    // 搬运输入 inputIdx 的 [startRow, startRow+numRows) 行。
    // 该输入内存布局连续 [beforeDim, catLen, afterDim]，各行无 gap。
    // 多行合并路径：当每行字节数 32B 对齐时，用 DataCopyExtParams 一次搬多行，
    //   把每输入每行的 Scalar 地址计算开销从 numRows 次降到 ~numRows/TileRows 次。
    // 非对齐回退：逐行 + TILE 分批（保证任意 shape 正确）。
    //
    // stride 语义（910B, DataCopyPad）：
    //   - srcStride（UB 侧）单位 32B，是相邻块「尾→头」的 gap（不含 blockLen）。
    //   - dstStride（GM 侧）单位 字节，同样是 gap。
    //   - 源 GM 连续 → srcStride=0；目标行间 gap = (totalCatLen-catLen)*afterDim*dtypeSize 字节。
    __aicore__ inline void CopyInputRows(uint32_t inputIdx, uint32_t startRow,
                                         uint32_t numRows, uint32_t catLen)
    {
        uint64_t rowBytes = static_cast<uint64_t>(catLen) * afterDimSize_ * dtypeSize_;
        uint64_t srcStart = static_cast<uint64_t>(startRow) * rowBytes;   // 源每行 rowBytes，连续
        uint64_t dstStart = static_cast<uint64_t>(startRow) * totalCatLen_ * afterDimSize_ * dtypeSize_
                          + static_cast<uint64_t>(inputCatOffset_[inputIdx]) * afterDimSize_ * dtypeSize_;
        // 目标行间 gap（字节，GM 侧 dstStride）：每写 rowBytes 后跳过其他输入占的 (totalCatLen-catLen)*afterDim*dtypeSize
        uint64_t dstGapBytes = static_cast<uint64_t>(totalCatLen_ - catLen) * afterDimSize_ * dtypeSize_;

        // 多行快路径：每行 32B 对齐（GM→UB 与 UB→GM 多块 stride 路径要求 blockLen 32B 对齐）
        bool canMultiRow = (rowBytes % 32 == 0) && (rowBytes <= TILE_BYTES);
        if (canMultiRow) {
            CopyInputMultiRow(inputIdx, srcStart, dstStart, numRows,
                              static_cast<uint32_t>(rowBytes), dstGapBytes);
        } else {
            // 逐行回退
            for (uint32_t r = 0; r < numRows; r++) {
                CopyOneRange(inputIdx, srcStart + r * rowBytes,
                             dstStart + r * (rowBytes + dstGapBytes), rowBytes);
            }
        }
    }

    __aicore__ inline void CopyInputMultiRow(uint32_t inputIdx, uint64_t srcStart, uint64_t dstStart,
                                             uint32_t numRows, uint32_t rowBytes, uint64_t dstGapBytes)
    {
        // 一次最多搬的行数：受 TILE_BYTES / rowBytes 与 blockCount 上限(4095) 限制
        uint32_t maxRowsByTile = TILE_BYTES / rowBytes;
        if (maxRowsByTile == 0) maxRowsByTile = 1;
        uint32_t maxRows = maxRowsByTile < 4095 ? maxRowsByTile : 4095;
        // dstStride（GM 侧，字节）；srcStride（UB 侧，32B 单位）= 0（UB 内块紧挨）
        uint32_t dstStride = static_cast<uint32_t>(dstGapBytes);

        uint32_t remainingRows = numRows;
        uint64_t srcCur = srcStart;
        uint64_t dstCur = dstStart;
        while (remainingRows > 0) {
            uint32_t rows = (remainingRows > maxRows) ? maxRows : remainingRows;
            uint32_t batchBytes = rowBytes * rows;
            // GM -> UB：一次搬 rows 行（blockCount=rows, blockLen=rowBytes, srcStride=0 源连续）
            LocalTensor<uint8_t> xLocal = inQueue_.AllocTensor<uint8_t>();
            DataCopyExtParams copyInParams{static_cast<uint16_t>(rows), rowBytes, 0, 0, 0};
            DataCopyPadExtParams<uint8_t> padIn{false, 0, 0, 0};
            DataCopyPad(xLocal, inputGm_[inputIdx][srcCur], copyInParams, padIn);
            inQueue_.EnQue(xLocal);

            LocalTensor<uint8_t> xLocalDeq = inQueue_.DeQue<uint8_t>();
            LocalTensor<uint8_t> yLocal = outQueue_.AllocTensor<uint8_t>();
            // UB->UB：按 32B 对齐字节数连续拷贝（rowBytes 32B 对齐，故 UB 内多行数据连续紧挨）
            uint32_t alignedBytes = (batchBytes + 31) & ~31u;
            AscendC::DataCopy(yLocal, xLocalDeq, alignedBytes);
            outQueue_.EnQue(yLocal);
            inQueue_.FreeTensor(xLocalDeq);

            // UB -> GM：多行写出，dstStride（字节）= 目标行间 gap
            LocalTensor<uint8_t> yLocalDeq = outQueue_.DeQue<uint8_t>();
            DataCopyExtParams copyOutParams{static_cast<uint16_t>(rows), rowBytes, 0, dstStride, 0};
            DataCopyPad(yGm_[dstCur], yLocalDeq, copyOutParams);
            outQueue_.FreeTensor(yLocalDeq);

            srcCur += static_cast<uint64_t>(rows) * rowBytes;             // 源连续
            dstCur += static_cast<uint64_t>(rows) * (rowBytes + dstGapBytes);
            remainingRows -= rows;
        }
    }

    __aicore__ inline void CopyOneRange(uint32_t inputIdx, uint64_t srcOff, uint64_t dstOff, uint64_t byteLen)
    {
        uint64_t remaining = byteLen;
        uint64_t src = srcOff;
        uint64_t dst = dstOff;
        while (remaining > 0) {
            uint32_t batchBytes = (remaining > TILE_BYTES) ? TILE_BYTES : static_cast<uint32_t>(remaining);
            CopyBatch(inputIdx, src, dst, batchBytes);
            src += batchBytes;
            dst += batchBytes;
            remaining -= batchBytes;
        }
    }

    __aicore__ inline void CopyBatch(uint32_t inputIdx, uint64_t srcOffsetBytes,
                                     uint64_t dstOffsetBytes, uint32_t bytes)
    {
        // GM -> UB (VECIN)
        LocalTensor<uint8_t> xLocal = inQueue_.AllocTensor<uint8_t>();
        DataCopyExtParams copyInParams{1, bytes, 0, 0, 0};
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, inputGm_[inputIdx][srcOffsetBytes], copyInParams, padParams);
        inQueue_.EnQue(xLocal);

        // 从 VECIN DeQue（保证 GM→UB 完成），拷贝到 VECOUT buffer
        LocalTensor<uint8_t> xLocalDeq = inQueue_.DeQue<uint8_t>();
        LocalTensor<uint8_t> yLocal = outQueue_.AllocTensor<uint8_t>();
        // UB->UB 拷贝：把已就绪的输入数据搬到输出队列 buffer
        // 使用 DataCopy 需要 32B 对齐，元素级拷贝更通用
        uint32_t alignedBytes = (bytes + 31) & ~31u;
        AscendC::DataCopy(yLocal, xLocalDeq, alignedBytes);
        outQueue_.EnQue(yLocal);
        inQueue_.FreeTensor(xLocalDeq);

        // UB -> GM (VECOUT)
        LocalTensor<uint8_t> yLocalDeq = outQueue_.DeQue<uint8_t>();
        DataCopyExtParams copyOutParams{1, bytes, 0, 0, 0};
        DataCopyPad(yGm_[dstOffsetBytes], yLocalDeq, copyOutParams);
        outQueue_.FreeTensor(yLocalDeq);
    }

private:
    uint32_t inputNum_;
    uint32_t afterDimSize_;
    uint32_t dtypeSize_;
    uint32_t totalCatLen_;
    uint32_t usedCoreNum_;
    uint32_t beforeDimSize_;

    uint32_t inputCatLen_[MAX_INPUT_NUM];
    uint32_t inputCatOffset_[MAX_INPUT_NUM];

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,  2> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQueue_;
    AscendC::GlobalTensor<uint8_t> inputGm_[MAX_INPUT_NUM];
    AscendC::GlobalTensor<uint8_t> yGm_;
};

extern "C" __global__ __aicore__ void concat(GM_ADDR srcList, GM_ADDR dst,
                                             GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    KernelConcat op;
    op.Init(srcList, dst, tiling_data);
    op.Process();
}
