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
 *   - 输出行 32B 对齐时按「行切片 × 输出列」切分；一个核仅处理和自己列
 *     区间相交的输入。其余布局回退为整行切分，避免核间写同一 32B block。
 *   - 每个相交区域用 DataCopyPad 的二维 stride 搬运。绑定 VECIN→VECOUT
 *     队列以 buffer 级事件衔接 MTE2/MTE3；双缓冲让下一 tile 的搬入能与
 *     上一 tile 的搬出重叠，而无需停顿整个流水线。
 *
 * DataCopyPad（CANN 8.5 Ascend C）：
 *   - DataCopyExtParams(blockCount, blockLen, srcStride, dstStride, rsv)
 *       blockLen 单位为「字节」，uint32_t
 *   - 使用 uint8_t 视角搬运，统一处理所有 dtype
 */
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

using namespace AscendC;

// 绑定队列的两个 64KiB slot，共 128KiB，低于 DAV_2201 的可用 UB 预算。
constexpr uint32_t TILE_BYTES = 64 * 1024;
constexpr uint32_t DATA_BLOCK_BYTES = 32;
constexpr uint32_t MAX_COPY_BLOCK_COUNT = 4095;
constexpr uint32_t SPLIT_COLUMNS = 1;
constexpr uint32_t SPLIT_TINY = 2;
constexpr uint32_t SPLIT_FLAT_SPAN = 3;
constexpr uint32_t SPLIT_IDENTITY = 4;
constexpr uint64_t FLAT_SPAN_UNIT_BYTES = 512;

class KernelConcat {
public:
    __aicore__ inline KernelConcat(TPipe *pipe) : pipe_(pipe) {}

    template <typename TilingT>
    __aicore__ inline void Init(GM_ADDR srcList, GM_ADDR dst, const TilingT &tiling)
    {
        inputNum_      = tiling.inputNum;
        afterDimSize_  = tiling.afterDimSize;
        dtypeSize_     = tiling.dtypeSize;
        totalCatLen_   = tiling.totalCatLen;
        beforeDimSize_ = tiling.beforeDimSize;
        listDesc_.Init(reinterpret_cast<__gm__ void*>(srcList));
        yGm_.SetGlobalBuffer((__gm__ uint8_t *)dst);
        const uint64_t totalBytes = static_cast<uint64_t>(beforeDimSize_) *
                                    static_cast<uint64_t>(totalCatLen_) * afterDimSize_ * dtypeSize_;
        // Tiny and a <=64KiB Identity have one ordered tile, so a second slot
        // only consumes UB.  All other paths preserve P0's ping-pong queue.
        const bool oneSlot = tiling.splitMode == SPLIT_TINY ||
                             (tiling.splitMode == SPLIT_IDENTITY && totalBytes <= TILE_BYTES);
        pipe_->InitBuffer(copyQueue_, oneSlot ? 1 : 2, TILE_BYTES);
    }

    template <typename TilingT>
    __aicore__ inline void Process(const TilingT &tiling)
    {
        const uint32_t coreId = GetBlockIdx();
        const uint32_t blockNum = GetBlockNum();
        if (coreId >= blockNum || blockNum == 0 || inputNum_ == 0 || beforeDimSize_ == 0) {
            return;
        }
        const uint64_t catUnitBytes = static_cast<uint64_t>(afterDimSize_) * dtypeSize_;
        const uint64_t outputRowBytes = static_cast<uint64_t>(totalCatLen_) * catUnitBytes;
        if (outputRowBytes == 0) return;

        if (tiling.splitMode == SPLIT_IDENTITY) {
            CopyIdentity(blockNum, coreId, outputRowBytes);
            return;
        }
        if (tiling.splitMode == SPLIT_FLAT_SPAN) {
            CopyFlatSpan(tiling, blockNum, coreId, catUnitBytes, outputRowBytes);
            return;
        }

        uint32_t startRow = 0;
        uint32_t endRow = 0;
        uint64_t colBegin = 0;
        uint64_t colEnd = outputRowBytes;

        if (tiling.splitMode == SPLIT_COLUMNS) {
            const uint32_t rowSlice = coreId / tiling.colCoreNum;
            const uint32_t col = coreId - rowSlice * tiling.colCoreNum;
            startRow = static_cast<uint64_t>(beforeDimSize_) * rowSlice / tiling.rowSliceNum;
            endRow = static_cast<uint64_t>(beforeDimSize_) * (rowSlice + 1) / tiling.rowSliceNum;
            colBegin = static_cast<uint64_t>(col) * tiling.colBlockBytes;
            colEnd = colBegin + tiling.colBlockBytes;
            if (colBegin >= outputRowBytes) return;
            if (colEnd > outputRowBytes) colEnd = outputRowBytes;
        } else {
            startRow = static_cast<uint64_t>(beforeDimSize_) * coreId / blockNum;
            endRow = static_cast<uint64_t>(beforeDimSize_) * (coreId + 1) / blockNum;
        }
        if (startRow >= endRow || colBegin >= colEnd) return;

        CopyColumn(tiling, startRow, endRow - startRow, colBegin, colEnd, catUnitBytes, outputRowBytes);
    }

private:
    __aicore__ inline uint64_t SpanOffset(uint64_t totalBytes, uint32_t coreId, uint32_t blockNum)
    {
        const uint64_t units = totalBytes / FLAT_SPAN_UNIT_BYTES +
                               (totalBytes % FLAT_SPAN_UNIT_BYTES != 0);
        const uint64_t unit = units * coreId / blockNum;
        const uint64_t offset = unit * FLAT_SPAN_UNIT_BYTES;
        return offset < totalBytes ? offset : totalBytes;
    }

    __aicore__ inline void CopyIdentity(uint32_t blockNum, uint32_t coreId, uint64_t outputRowBytes)
    {
        const uint64_t totalBytes = static_cast<uint64_t>(beforeDimSize_) * outputRowBytes;
        if (totalBytes == 0) return;
        const uint64_t begin = SpanOffset(totalBytes, coreId, blockNum);
        const uint64_t end = SpanOffset(totalBytes, coreId + 1, blockNum);
        if (begin >= end) return;
        GlobalTensor<uint8_t> inputGm;
        inputGm.SetGlobalBuffer(listDesc_.GetDataPtr<__gm__ uint8_t>(0));
        SubmitLinearRange(inputGm, begin, begin, end - begin);
    }

    template <typename TilingT>
    __aicore__ inline void CopyFlatSpan(const TilingT &tiling, uint32_t blockNum, uint32_t coreId,
                                        uint64_t catUnitBytes, uint64_t outputRowBytes)
    {
        const uint64_t totalBytes = static_cast<uint64_t>(beforeDimSize_) * outputRowBytes;
        if (totalBytes == 0 || outputRowBytes == 0) return;
        const uint64_t spanBegin = SpanOffset(totalBytes, coreId, blockNum);
        const uint64_t spanEnd = SpanOffset(totalBytes, coreId + 1, blockNum);
        if (spanBegin >= spanEnd) return;

        uint64_t absolute = spanBegin;
        uint64_t row = absolute / outputRowBytes;
        uint64_t column = absolute - row * outputRowBytes;
        uint32_t input = 0;
        uint64_t inputBegin = 0;
        // Locate only the first fragment.  At a row boundary the logical
        // prefix restarts at input zero; otherwise input advances monotonically.
        while (input < inputNum_) {
            const uint64_t inputBytes = static_cast<uint64_t>(tiling.inputCatLen[input]) * catUnitBytes;
            if (inputBytes != 0 && column < inputBegin + inputBytes) break;
            inputBegin += inputBytes;
            ++input;
        }
        if (input >= inputNum_) return;

        while (absolute < spanEnd && row < beforeDimSize_) {
            if (input >= inputNum_) {
                ++row;
                column = 0;
                input = 0;
                inputBegin = 0;
                while (input < inputNum_ && tiling.inputCatLen[input] == 0) ++input;
                if (input >= inputNum_) return;
            }
            const uint64_t inputBytes = static_cast<uint64_t>(tiling.inputCatLen[input]) * catUnitBytes;
            if (inputBytes == 0) {
                ++input;
                continue;
            }
            const uint64_t inputEnd = inputBegin + inputBytes;
            if (column >= inputEnd) {
                inputBegin = inputEnd;
                ++input;
                continue;
            }
            const uint64_t rowRemaining = outputRowBytes - column;
            const uint64_t inputRemaining = inputEnd - column;
            const uint64_t spanRemaining = spanEnd - absolute;
            uint64_t bytes = rowRemaining < inputRemaining ? rowRemaining : inputRemaining;
            bytes = bytes < spanRemaining ? bytes : spanRemaining;
            if (bytes == 0) return;
            GlobalTensor<uint8_t> inputGm;
            inputGm.SetGlobalBuffer(listDesc_.GetDataPtr<__gm__ uint8_t>(input));
            const uint64_t srcOffset = row * inputBytes + column - inputBegin;
            SubmitLinearRange(inputGm, srcOffset, absolute, bytes);
            absolute += bytes;
            column += bytes;
            if (column == inputEnd) {
                inputBegin = inputEnd;
                ++input;
            }
            if (column == outputRowBytes && absolute < spanEnd) {
                ++row;
                column = 0;
                input = 0;
                inputBegin = 0;
                while (input < inputNum_ && tiling.inputCatLen[input] == 0) ++input;
            }
        }
    }

    template <typename TilingT>
    __aicore__ inline void CopyColumn(const TilingT &tiling, uint32_t startRow, uint32_t numRows,
                                      uint64_t colBegin, uint64_t colEnd, uint64_t catUnitBytes,
                                      uint64_t outputRowBytes)
    {
        uint64_t inputBegin = 0;
        for (uint32_t input = 0; input < inputNum_; ++input) {
            if (inputBegin >= colEnd) break;
            const uint64_t inputRowBytes = static_cast<uint64_t>(tiling.inputCatLen[input]) * catUnitBytes;
            const uint64_t inputEnd = inputBegin + inputRowBytes;
            const uint64_t begin = inputBegin > colBegin ? inputBegin : colBegin;
            const uint64_t end = inputEnd < colEnd ? inputEnd : colEnd;
            // Advance before a possible empty/non-intersecting continue.
            const uint64_t nextInputBegin = inputEnd;
            if (begin >= end || inputRowBytes == 0) {
                inputBegin = nextInputBegin;
                continue;
            }

            GlobalTensor<uint8_t> inputGm;
            inputGm.SetGlobalBuffer(listDesc_.GetDataPtr<__gm__ uint8_t>(input));
            const uint64_t pieceBytes = end - begin;
            const uint64_t srcOffset = static_cast<uint64_t>(startRow) * inputRowBytes + begin - inputBegin;
            const uint64_t dstOffset = static_cast<uint64_t>(startRow) * outputRowBytes + begin;
            SubmitStridedPiece(inputGm, srcOffset, dstOffset, numRows, pieceBytes,
                               inputRowBytes - pieceBytes, outputRowBytes - pieceBytes);
            inputBegin = nextInputBegin;
        }
    }

    __aicore__ inline void SubmitStridedPiece(const GlobalTensor<uint8_t> &inputGm, uint64_t srcStart,
                                              uint64_t dstStart, uint32_t numRows, uint64_t pieceBytes,
                                              uint64_t srcGapBytes, uint64_t dstGapBytes)
    {
        if (pieceBytes == 0) return;
        if (pieceBytes > TILE_BYTES || pieceBytes > 0xFFFFFFFFULL ||
            srcGapBytes > 0xFFFFFFFFULL || dstGapBytes > 0xFFFFFFFFULL) {
            for (uint32_t row = 0; row < numRows; ++row) {
                SubmitLinearRange(inputGm, srcStart + static_cast<uint64_t>(row) * (pieceBytes + srcGapBytes),
                                  dstStart + static_cast<uint64_t>(row) * (pieceBytes + dstGapBytes), pieceBytes);
            }
            return;
        }

        const uint32_t rowBytes = static_cast<uint32_t>(pieceBytes);
        const uint32_t maxRowsByTile = TILE_BYTES / AlignUp32(rowBytes);
        const uint32_t maxRows = maxRowsByTile < MAX_COPY_BLOCK_COUNT ? maxRowsByTile : MAX_COPY_BLOCK_COUNT;
        uint32_t remainingRows = numRows;
        uint64_t srcCur = srcStart;
        uint64_t dstCur = dstStart;
        while (remainingRows > 0) {
            const uint32_t rows = remainingRows > maxRows ? maxRows : remainingRows;
            SubmitTile(inputGm, srcCur, dstCur, rows, rowBytes,
                       static_cast<uint32_t>(srcGapBytes), static_cast<uint32_t>(dstGapBytes));
            srcCur += static_cast<uint64_t>(rows) * (rowBytes + srcGapBytes);
            dstCur += static_cast<uint64_t>(rows) * (rowBytes + dstGapBytes);
            remainingRows -= rows;
        }
    }

    __aicore__ inline void SubmitLinearRange(const GlobalTensor<uint8_t> &inputGm, uint64_t srcOff,
                                             uint64_t dstOff, uint64_t byteLen)
    {
        uint64_t remaining = byteLen;
        uint64_t src = srcOff;
        uint64_t dst = dstOff;
        while (remaining > 0) {
            const uint32_t batchBytes = remaining > TILE_BYTES ? TILE_BYTES : static_cast<uint32_t>(remaining);
            SubmitTile(inputGm, src, dst, 1, batchBytes, 0, 0);
            src += batchBytes;
            dst += batchBytes;
            remaining -= batchBytes;
        }
    }

    __aicore__ inline void SubmitTile(const GlobalTensor<uint8_t> &inputGm, uint64_t srcOffsetBytes,
                                      uint64_t dstOffsetBytes, uint32_t rows, uint32_t rowBytes,
                                      uint32_t srcGapBytes, uint32_t dstGapBytes)
    {
        LocalTensor<uint8_t> local = copyQueue_.AllocTensor<uint8_t>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = rows;
        copyInParams.blockLen = rowBytes;
        copyInParams.srcStride = srcGapBytes;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        DataCopyPad(local, inputGm[srcOffsetBytes], copyInParams, padParams);
        copyQueue_.EnQue(local);
        LocalTensor<uint8_t> out = copyQueue_.DeQue<uint8_t>();
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = rows;
        copyOutParams.blockLen = rowBytes;
        // DataCopyPad 自动跳过每行的 dummy 数据，因此 UB 端保持 stride=0。
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = dstGapBytes;
        copyOutParams.rsv = 0;
        DataCopyPad(yGm_[dstOffsetBytes], out, copyOutParams);
        copyQueue_.FreeTensor(out);
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
    uint32_t beforeDimSize_;

    AscendC::TPipe *pipe_;
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 1> copyQueue_;
    AscendC::ListTensorDesc listDesc_;
    AscendC::GlobalTensor<uint8_t> yGm_;
};

extern "C" __global__ __aicore__ void concat(GM_ADDR srcList, GM_ADDR dst,
                                              GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA(tiling_data, tiling);

    AscendC::TPipe pipe;
    KernelConcat op(&pipe);
    op.Init(srcList, dst, tiling_data);
    op.Process(tiling_data);
}
