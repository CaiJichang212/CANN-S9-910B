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
 *   - 每个输入段按固定 TILE_BYTES 批量搬运；使用绑定队列直接完成 GM→UB→GM
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
// 单个 UB tile 固定 64KiB。TQueBind 双缓冲共 128KiB，低于 910B 的可用 UB 预算。
constexpr uint32_t TILE_BYTES = 64 * 1024;
constexpr uint32_t DATA_BLOCK_BYTES = 32;
constexpr uint32_t MAX_COPY_BLOCK_COUNT = 4095;

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

        // 同一 UB 双缓冲由绑定队列在 VECIN 和 VECOUT 间流转；EnQue/DeQue 建立
        // MTE2→MTE3 同步，无需也不能再作 UB→UB 中转。
        pipe.InitBuffer(copyQueue_, 2, TILE_BYTES);
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
    // 当一行不超过 TILE_BYTES 时，统一用 DataCopyPad 多行搬运；DataCopyPad 在 UB
    // 中自动保留每行的 32B dummy 空间，并在写回 GM 时丢弃。超过 TILE_BYTES 的单行
    // 段退化为精确字节块搬运，仍使用同一个绑定队列。
    //
    // stride 语义（910B, DataCopyPad）：
    //   - srcStride（UB 侧）单位 32B，是相邻块「尾→头」的 gap（不含 blockLen）。
    //   - dstStride（GM 侧）单位 字节，同样是 gap。
    //   - 源 GM 连续 → srcStride=0；目标行间 gap = (totalCatLen-catLen)*afterDim*dtypeSize 字节。
    __aicore__ inline void CopyInputRows(uint32_t inputIdx, uint32_t startRow,
                                         uint32_t numRows, uint32_t catLen)
    {
        uint64_t rowBytes = static_cast<uint64_t>(catLen) * afterDimSize_ * dtypeSize_;
        if (rowBytes == 0) {
            return;
        }
        uint64_t srcStart = static_cast<uint64_t>(startRow) * rowBytes;   // 源每行 rowBytes，连续
        uint64_t dstStart = static_cast<uint64_t>(startRow) * totalCatLen_ * afterDimSize_ * dtypeSize_
                          + static_cast<uint64_t>(inputCatOffset_[inputIdx]) * afterDimSize_ * dtypeSize_;
        // 目标行间 gap（字节，GM 侧 dstStride）：每写 rowBytes 后跳过其他输入占的 (totalCatLen-catLen)*afterDim*dtypeSize
        uint64_t dstGapBytes = static_cast<uint64_t>(totalCatLen_ - catLen) * afterDimSize_ * dtypeSize_;

        // DataCopyExtParams 的 GM stride 字段为 uint32_t。极端 gap 超出表达范围时
        // 逐行写回，避免截断；正常路径（包括全部非对齐行）均批量处理。
        bool canMultiRow = rowBytes <= TILE_BYTES && dstGapBytes <= 0xFFFFFFFFULL;
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
        // 每行在 UB 中占 AlignUp(rowBytes, 32)，以保证每个 block 的 UB 起址对齐。
        uint32_t alignedRowBytes = AlignUp32(rowBytes);
        uint32_t maxRowsByTile = TILE_BYTES / alignedRowBytes;
        uint32_t maxRows = maxRowsByTile < MAX_COPY_BLOCK_COUNT ? maxRowsByTile : MAX_COPY_BLOCK_COUNT;

        uint32_t remainingRows = numRows;
        uint64_t srcCur = srcStart;
        uint64_t dstCur = dstStart;
        bool hasPending = false;
        uint64_t pendingDst = 0;
        uint32_t pendingRows = 0;
        while (remainingRows > 0) {
            uint32_t rows = (remainingRows > maxRows) ? maxRows : remainingRows;
            SubmitCopyIn(inputIdx, srcCur, rows, rowBytes);
            // 在 MTE2 填充下一批的同时，MTE3 写回前一批。
            if (hasPending) {
                CopyOutRows(pendingDst, pendingRows, rowBytes, static_cast<uint32_t>(dstGapBytes));
            }
            hasPending = true;
            pendingDst = dstCur;
            pendingRows = rows;

            srcCur += static_cast<uint64_t>(rows) * rowBytes;             // 源连续
            dstCur += static_cast<uint64_t>(rows) * (rowBytes + dstGapBytes);
            remainingRows -= rows;
        }
        if (hasPending) {
            CopyOutRows(pendingDst, pendingRows, rowBytes, static_cast<uint32_t>(dstGapBytes));
        }
    }

    __aicore__ inline void CopyOneRange(uint32_t inputIdx, uint64_t srcOff, uint64_t dstOff, uint64_t byteLen)
    {
        uint64_t remaining = byteLen;
        uint64_t src = srcOff;
        uint64_t dst = dstOff;
        bool hasPending = false;
        uint64_t pendingDst = 0;
        uint32_t pendingBytes = 0;
        while (remaining > 0) {
            uint32_t batchBytes = (remaining > TILE_BYTES) ? TILE_BYTES : static_cast<uint32_t>(remaining);
            SubmitCopyIn(inputIdx, src, 1, batchBytes);
            if (hasPending) {
                CopyOutRows(pendingDst, 1, pendingBytes, 0);
            }
            hasPending = true;
            pendingDst = dst;
            pendingBytes = batchBytes;
            src += batchBytes;
            dst += batchBytes;
            remaining -= batchBytes;
        }
        if (hasPending) {
            CopyOutRows(pendingDst, 1, pendingBytes, 0);
        }
    }

    __aicore__ inline void SubmitCopyIn(uint32_t inputIdx, uint64_t srcOffsetBytes,
                                        uint32_t rows, uint32_t rowBytes)
    {
        LocalTensor<uint8_t> local = copyQueue_.AllocTensor<uint8_t>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = rows;
        copyInParams.blockLen = rowBytes;
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        DataCopyPad(local, inputGm_[inputIdx][srcOffsetBytes], copyInParams, padParams);
        copyQueue_.EnQue(local);
    }

    __aicore__ inline void CopyOutRows(uint64_t dstOffsetBytes, uint32_t rows,
                                       uint32_t rowBytes, uint32_t dstGapBytes)
    {
        // DeQue 将同一块 UB 切换为 VECOUT，等待对应的 MTE2 完成后才允许 MTE3 读取。
        LocalTensor<uint8_t> local = copyQueue_.DeQue<uint8_t>();
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = rows;
        copyOutParams.blockLen = rowBytes;
        // DataCopyPad 自动跳过每行的 dummy 数据，因此 UB 端保持紧凑 stride=0。
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = dstGapBytes;
        copyOutParams.rsv = 0;
        DataCopyPad(yGm_[dstOffsetBytes], local, copyOutParams);
        copyQueue_.FreeTensor(local);
    }

    static __aicore__ inline uint32_t AlignUp32(uint32_t bytes)
    {
        return (bytes + DATA_BLOCK_BYTES - 1) & ~(DATA_BLOCK_BYTES - 1);
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
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 2> copyQueue_;
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
