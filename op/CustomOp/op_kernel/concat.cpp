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
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

using namespace AscendC;

// 绑定队列的两个 64KiB slot，共 128KiB，低于 DAV_2201 的可用 UB 预算。
constexpr uint32_t TILE_BYTES = 64 * 1024;
constexpr uint32_t DATA_BLOCK_BYTES = 32;
constexpr uint32_t MAX_COPY_BLOCK_COUNT = 4095;

class KernelConcatIdentity {
public:
    __aicore__ inline KernelConcatIdentity() {}

    template <typename TilingT>
    __aicore__ inline void Init(GM_ADDR srcList, GM_ADDR dst, const TilingT &tiling)
    {
        totalBytes_ = tiling.totalBytes;
        usedCoreNum_ = tiling.usedCoreNum;
        tileBytes_ = tiling.tileBytes;
        listDesc_.Init(reinterpret_cast<__gm__ void *>(srcList));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(dst));
        pipe.InitBuffer(copyQueue_, 2, tileBytes_);
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreId = GetBlockIdx();
        if (coreId >= usedCoreNum_ || usedCoreNum_ == 0 || totalBytes_ == 0) return;

        const uint64_t dstBase = reinterpret_cast<uint64_t>(yGm_.GetPhyAddr(0));
        uint64_t headBytes = (DATA_BLOCK_BYTES - (dstBase & (DATA_BLOCK_BYTES - 1))) &
                             (DATA_BLOCK_BYTES - 1);
        if (headBytes > totalBytes_) headBytes = totalBytes_;
        const uint64_t bodyUnits = (totalBytes_ - headBytes) / DATA_BLOCK_BYTES;
        const uint64_t unitBegin = SplitUnits(bodyUnits, coreId, usedCoreNum_);
        const uint64_t unitEnd = SplitUnits(bodyUnits, coreId + 1, usedCoreNum_);

        uint64_t begin = headBytes + unitBegin * DATA_BLOCK_BYTES;
        uint64_t end = headBytes + unitEnd * DATA_BLOCK_BYTES;
        if (coreId == 0) begin = 0;
        if (coreId + 1 == usedCoreNum_) end = totalBytes_;
        if (begin >= end) return;

        GlobalTensor<uint8_t> inputGm;
        inputGm.SetGlobalBuffer(listDesc_.GetDataPtr<__gm__ uint8_t>(0));
        SubmitLinearRange(inputGm, begin, begin, end - begin);
    }

private:
    static __aicore__ inline uint64_t SplitUnits(uint64_t units, uint32_t part, uint32_t parts)
    {
        const uint64_t quotient = units / parts;
        const uint64_t remainder = units % parts;
        return quotient * part + remainder * part / parts;
    }

    __aicore__ inline void SubmitLinearRange(const GlobalTensor<uint8_t> &inputGm, uint64_t srcOff,
                                             uint64_t dstOff, uint64_t byteLen)
    {
        uint64_t remaining = byteLen;
        while (remaining > 0) {
            const uint32_t batchBytes = remaining > tileBytes_ ? tileBytes_ : static_cast<uint32_t>(remaining);
            LocalTensor<uint8_t> local = copyQueue_.AllocTensor<uint8_t>();
            DataCopyExtParams copyInParams{1, batchBytes, 0, 0, 0};
            DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
            DataCopyPad(local, inputGm[srcOff], copyInParams, padParams);
            copyQueue_.EnQue(local);
            LocalTensor<uint8_t> out = copyQueue_.DeQue<uint8_t>();
            DataCopyExtParams copyOutParams{1, batchBytes, 0, 0, 0};
            DataCopyPad(yGm_[dstOff], out, copyOutParams);
            copyQueue_.FreeTensor(out);
            srcOff += batchBytes;
            dstOff += batchBytes;
            remaining -= batchBytes;
        }
    }

private:
    uint64_t totalBytes_;
    uint32_t usedCoreNum_;
    uint32_t tileBytes_;
    AscendC::TPipe pipe;
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 1> copyQueue_;
    AscendC::ListTensorDesc listDesc_;
    AscendC::GlobalTensor<uint8_t> yGm_;
};

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
        listDesc_.Init(reinterpret_cast<__gm__ void*>(srcList));
        yGm_.SetGlobalBuffer((__gm__ uint8_t *)dst);
        // num=2 开启 ping-pong。TQueBind 的 EnQue/DeQue 负责 MTE2→MTE3
        // 依赖，而 FreeTensor 在 MTE3 完成后才允许 slot 被下一 tile 复用。
        pipe.InitBuffer(copyQueue_, 2, TILE_BYTES);
    }

    template <typename TilingT>
    __aicore__ inline void Process(const TilingT &tiling)
    {
        const uint32_t coreId = GetBlockIdx();
        if (coreId >= usedCoreNum_ || inputNum_ == 0 || beforeDimSize_ == 0) {
            return;
        }
        const uint64_t catUnitBytes = static_cast<uint64_t>(afterDimSize_) * dtypeSize_;
        const uint64_t outputRowBytes = static_cast<uint64_t>(totalCatLen_) * catUnitBytes;
        uint32_t startRow = 0;
        uint32_t endRow = 0;
        uint64_t colBegin = 0;
        uint64_t colEnd = outputRowBytes;

        if (tiling.splitMode == 1) {
            const uint32_t rowSlice = coreId / tiling.colCoreNum;
            const uint32_t col = coreId - rowSlice * tiling.colCoreNum;
            startRow = beforeDimSize_ * rowSlice / tiling.rowSliceNum;
            endRow = beforeDimSize_ * (rowSlice + 1) / tiling.rowSliceNum;
            colBegin = static_cast<uint64_t>(col) * tiling.colBlockBytes;
            colEnd = colBegin + tiling.colBlockBytes;
            if (colBegin >= outputRowBytes) return;
            if (colEnd > outputRowBytes) colEnd = outputRowBytes;
        } else {
            startRow = beforeDimSize_ * coreId / usedCoreNum_;
            endRow = beforeDimSize_ * (coreId + 1) / usedCoreNum_;
        }
        if (startRow >= endRow || colBegin >= colEnd) return;

        CopyColumn(tiling, startRow, endRow - startRow, colBegin, colEnd, catUnitBytes, outputRowBytes);
    }

private:
    template <typename TilingT>
    __aicore__ inline void CopyColumn(const TilingT &tiling, uint32_t startRow, uint32_t numRows,
                                      uint64_t colBegin, uint64_t colEnd, uint64_t catUnitBytes,
                                      uint64_t outputRowBytes)
    {
        // Prefix offsets are sorted, so binary-search the first possibly
        // intersecting input and stop once the next input starts beyond this column.
        uint32_t low = 0;
        uint32_t high = inputNum_;
        while (low < high) {
            const uint32_t mid = low + (high - low) / 2;
            const uint64_t end = static_cast<uint64_t>(tiling.inputCatOffset[mid] +
                                                       tiling.inputCatLen[mid]) * catUnitBytes;
            if (end <= colBegin) low = mid + 1;
            else high = mid;
        }

        for (uint32_t input = low; input < inputNum_; ++input) {
            const uint64_t inputBegin = static_cast<uint64_t>(tiling.inputCatOffset[input]) * catUnitBytes;
            if (inputBegin >= colEnd) break;
            const uint64_t inputRowBytes = static_cast<uint64_t>(tiling.inputCatLen[input]) * catUnitBytes;
            const uint64_t inputEnd = inputBegin + inputRowBytes;
            const uint64_t begin = inputBegin > colBegin ? inputBegin : colBegin;
            const uint64_t end = inputEnd < colEnd ? inputEnd : colEnd;
            if (begin >= end || inputRowBytes == 0) continue;

            GlobalTensor<uint8_t> inputGm;
            inputGm.SetGlobalBuffer(listDesc_.GetDataPtr<__gm__ uint8_t>(input));
            const uint64_t pieceBytes = end - begin;
            const uint64_t srcOffset = static_cast<uint64_t>(startRow) * inputRowBytes + begin - inputBegin;
            const uint64_t dstOffset = static_cast<uint64_t>(startRow) * outputRowBytes + begin;
            SubmitStridedPiece(inputGm, srcOffset, dstOffset, numRows, pieceBytes,
                               inputRowBytes - pieceBytes, outputRowBytes - pieceBytes);
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
    uint32_t usedCoreNum_;
    uint32_t beforeDimSize_;

    AscendC::TPipe pipe;
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, 1> copyQueue_;
    AscendC::ListTensorDesc listDesc_;
    AscendC::GlobalTensor<uint8_t> yGm_;
};

extern "C" __global__ __aicore__ void concat(GM_ADDR srcList, GM_ADDR dst,
                                              GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(2)) {
        GET_TILING_DATA_WITH_STRUCT(ConcatIdentityTilingData, identity_tiling, tiling);
        KernelConcatIdentity op;
        op.Init(srcList, dst, identity_tiling);
        op.Process();
        return;
    } else if (TILING_KEY_IS(0)) {
        GET_TILING_DATA(tiling_data, tiling);
        KernelConcat op;
        op.Init(srcList, dst, tiling_data);
        op.Process(tiling_data);
        return;
    }
}
