/**
 * @file transpose.cpp
 * @brief 910B transpose kernel: DMA rows plus vnchwconv for 2-D rotations.
 *
 * There are deliberately no GlobalTensor GetValue/SetValue accesses here.
 * The generic path uses one batched strided DataCopyPad in each direction.
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kBufferNum = 2;
constexpr uint32_t kMaxDim = 8;
constexpr uint32_t kBlockBytes = 32;
constexpr uint32_t kVectorRows = 16;
constexpr uint32_t kMaxDmaBlocks = 4095;
constexpr uint32_t kTransposeSlotBytes = 32 * 1024;

enum TransposeMode : uint32_t { COPY_CONTIG = 0, ROTATE_2D = 1, STRIDED_ROWS = 2 };

__aicore__ inline uint32_t Align32(uint32_t value)
{
    return (value + kBlockBytes - 1) / kBlockBytes * kBlockBytes;
}
} // namespace

class KernelTranspose {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TransposeTilingData &t)
    {
        mode_ = t.mode;
        dtypeSize_ = t.dtypeSize;
        blockDim_ = t.blockDim;
        outerCount_ = t.outerCount;
        rowWidth_ = t.rowWidth;
        numRows_ = t.numRows;
        srcInnerStride_ = t.srcInnerStride;
        copyTileElems_ = t.copyTileElems;
        stridedTileElems_ = t.stridedTileElems;
        transBatch_ = t.transBatch;
        transM_ = t.transM;
        transN_ = t.transN;
        tileM_ = t.tileM;
        tileN_ = t.tileN;
        for (uint32_t i = 0; i < outerCount_; ++i) {
            outerOutShape_[i] = t.outerOutShape[i];
            outerSrcStride_[i] = t.outerSrcStride[i];
        }
        xGm_.SetGlobalBuffer((__gm__ DTYPE_X *)x);
        yGm_.SetGlobalBuffer((__gm__ DTYPE_Y *)y);
    }

    __aicore__ inline void Process()
    {
        if (mode_ == ROTATE_2D) {
            ProcessRotate2D();
        } else if (mode_ == COPY_CONTIG) {
            ProcessRows(true);
        } else {
            ProcessRows(false);
        }
    }

private:
    __aicore__ inline void ProcessRows(bool contiguous)
    {
        if (blockDim_ == 0) return;
        const uint64_t perCore = (numRows_ + blockDim_ - 1) / blockDim_;
        const uint64_t begin = static_cast<uint64_t>(GetBlockIdx()) * perCore;
        const uint64_t end = begin + perCore < numRows_ ? begin + perCore : numRows_;
        if (begin >= end) return;

        // Strided rows reserve one 32B UB block for each element, while
        // contiguous rows use compact double-buffered DMA slots.
        const uint32_t compactBytes = copyTileElems_ * dtypeSize_;
        const uint32_t stridedBytes = stridedTileElems_ * kBlockBytes;
        const uint32_t queueBytes = contiguous
            ? (compactBytes == 0 ? 1 : compactBytes)
            : (stridedBytes == 0 ? 1 : stridedBytes);
        pipe_.InitBuffer(rowQueue_, kBufferNum, queueBytes);
        for (uint64_t row = begin; row < end; ++row) {
            const uint64_t srcBase = DecodeRow(row);
            const uint64_t dstBase = row * rowWidth_;
            if (contiguous) {
                CopyContiguousRow(srcBase, dstBase);
            } else {
                CopyStridedRow(srcBase, dstBase);
            }
        }
    }

    __aicore__ inline uint64_t DecodeRow(uint64_t row) const
    {
        uint64_t rem = row;
        uint64_t offset = 0;
        for (int32_t i = static_cast<int32_t>(outerCount_) - 1; i >= 0; --i) {
            const uint64_t extent = outerOutShape_[i];
            const uint64_t digit = rem % extent;
            rem /= extent;
            offset += digit * outerSrcStride_[i];
        }
        return offset;
    }

    __aicore__ inline void CopyContiguousRow(uint64_t srcBase, uint64_t dstBase)
    {
        for (uint64_t offset = 0; offset < rowWidth_;) {
            const uint32_t count = static_cast<uint32_t>(
                (rowWidth_ - offset) < copyTileElems_ ? (rowWidth_ - offset) : copyTileElems_);
            LocalTensor<DTYPE_X> ub = rowQueue_.AllocTensor<DTYPE_X>();
            const uint32_t bytes = count * dtypeSize_;
            if ((bytes % kBlockBytes) == 0) {
                // This overload takes an element count (not a number of 32B blocks).
                DataCopy(ub, xGm_[srcBase + offset], count);
                rowQueue_.EnQue(ub);
                LocalTensor<DTYPE_X> out = rowQueue_.DeQue<DTYPE_X>();
                DataCopy(yGm_[dstBase + offset], out, count);
                rowQueue_.FreeTensor(out);
            } else {
                DataCopyExtParams in;
                in.blockCount = 1;
                in.blockLen = bytes;
                in.srcStride = 0;
                in.dstStride = 0;
                in.rsv = 0;
                DataCopyPadExtParams<DTYPE_X> pad;
                pad.isPad = false;
                pad.leftPadding = 0;
                pad.rightPadding = 0;
                pad.paddingValue = static_cast<DTYPE_X>(0);
                DataCopyPad(ub, xGm_[srcBase + offset], in, pad);
                rowQueue_.EnQue(ub);
                LocalTensor<DTYPE_X> out = rowQueue_.DeQue<DTYPE_X>();
                DataCopyExtParams outParams;
                outParams.blockCount = 1;
                outParams.blockLen = bytes;
                outParams.srcStride = 0;
                outParams.dstStride = 0;
                outParams.rsv = 0;
                DataCopyPad(yGm_[dstBase + offset], out, outParams);
                rowQueue_.FreeTensor(out);
            }
            offset += count;
        }
    }

    __aicore__ inline void CopyStridedRow(uint64_t srcBase, uint64_t dstBase)
    {
        for (uint64_t offset = 0; offset < rowWidth_;) {
            const uint64_t dmaLimit = stridedTileElems_ < kMaxDmaBlocks ? stridedTileElems_ : kMaxDmaBlocks;
            const uint64_t remaining = rowWidth_ - offset;
            const uint32_t count = static_cast<uint32_t>(remaining < dmaLimit ? remaining : dmaLimit);
            const uint64_t strideBytes64 = (srcInnerStride_ - 1) * dtypeSize_;
            // The ext stride is uint32_t.  A tensor with a larger pitch is
            // handled as individually-addressed one-burst transfers; it is
            // still DMA-only and never falls back to GM scalar reads.
            if (strideBytes64 > 0xFFFFFFFFull) {
                CopyWideStrideBursts(srcBase, dstBase, offset, count);
                offset += count;
                continue;
            }
            LocalTensor<DTYPE_X> ub = rowQueue_.AllocTensor<DTYPE_X>();
            DataCopyExtParams in;
            in.blockCount = static_cast<uint16_t>(count);
            in.blockLen = dtypeSize_;
            in.srcStride = static_cast<uint32_t>(strideBytes64);
            in.dstStride = 0; // DataCopyPad gives every narrow burst its 32B UB slot.
            in.rsv = 0;
            DataCopyPadExtParams<DTYPE_X> pad;
            pad.isPad = false;
            pad.leftPadding = 0;
            pad.rightPadding = 0;
            pad.paddingValue = static_cast<DTYPE_X>(0);
            DataCopyPad(ub, xGm_[srcBase + offset * srcInnerStride_], in, pad);
            rowQueue_.EnQue(ub);

            LocalTensor<DTYPE_X> out = rowQueue_.DeQue<DTYPE_X>();
            DataCopyExtParams outParams;
            outParams.blockCount = static_cast<uint16_t>(count);
            outParams.blockLen = dtypeSize_;
            outParams.srcStride = 0;
            outParams.dstStride = 0;
            outParams.rsv = 0;
            DataCopyPad(yGm_[dstBase + offset], out, outParams);
            rowQueue_.FreeTensor(out);
            offset += count;
        }
    }

    __aicore__ inline void CopyWideStrideBursts(uint64_t srcBase, uint64_t dstBase, uint64_t offset, uint32_t count)
    {
        // Each burst has its own 32B slot.  This branch only applies to an
        // address pitch outside the DMA stride encoding range.
        for (uint32_t i = 0; i < count; ++i) {
            LocalTensor<DTYPE_X> ub = rowQueue_.AllocTensor<DTYPE_X>();
            DataCopyExtParams in;
            in.blockCount = 1;
            in.blockLen = dtypeSize_;
            in.srcStride = 0;
            in.dstStride = 0;
            in.rsv = 0;
            DataCopyPadExtParams<DTYPE_X> pad;
            pad.isPad = false;
            pad.leftPadding = 0;
            pad.rightPadding = 0;
            pad.paddingValue = static_cast<DTYPE_X>(0);
            DataCopyPad(ub, xGm_[srcBase + (offset + i) * srcInnerStride_], in, pad);
            rowQueue_.EnQue(ub);
            LocalTensor<DTYPE_X> out = rowQueue_.DeQue<DTYPE_X>();
            DataCopyPad(yGm_[dstBase + offset + i], out, in);
            rowQueue_.FreeTensor(out);
        }
    }

    __aicore__ inline void ProcessRotate2D()
    {
        if (blockDim_ == 0 || tileN_ == 0 || tileM_ != kVectorRows) return;
        const uint64_t mTiles = (transM_ + kVectorRows - 1) / kVectorRows;
        const uint64_t totalTiles = transBatch_ * mTiles * ((transN_ + tileN_ - 1) / tileN_);
        const uint64_t perCore = (totalTiles + blockDim_ - 1) / blockDim_;
        const uint64_t begin = static_cast<uint64_t>(GetBlockIdx()) * perCore;
        const uint64_t end = begin + perCore < totalTiles ? begin + perCore : totalTiles;
        if (begin >= end) return;

        pipe_.InitBuffer(transInQueue_, kBufferNum, kTransposeSlotBytes);
        pipe_.InitBuffer(transOutQueue_, kBufferNum, kTransposeSlotBytes);
        const uint64_t nTiles = (transN_ + tileN_ - 1) / tileN_;
        const uint64_t tilesPerBatch = mTiles * nTiles;
        const uint64_t matrixElems = transM_ * transN_;
        for (uint64_t linear = begin; linear < end; ++linear) {
            const uint64_t batch = linear / tilesPerBatch;
            const uint64_t withinBatch = linear % tilesPerBatch;
            const uint64_t mi = (withinBatch / nTiles) * kVectorRows;
            const uint64_t nj = (withinBatch % nTiles) * tileN_;
            const uint64_t remainM = transM_ - mi;
            const uint64_t remainN = transN_ - nj;
            const uint32_t mh = static_cast<uint32_t>(remainM < kVectorRows ? remainM : kVectorRows);
            const uint32_t nw = static_cast<uint32_t>(remainN < tileN_ ? remainN : tileN_);
            RotateTile(batch * matrixElems, mi, nj, mh, nw);
        }
    }

    __aicore__ inline void RotateTile(uint64_t matrixBase, uint64_t mi, uint64_t nj, uint32_t mh, uint32_t nw)
    {
        LocalTensor<DTYPE_X> inUb = transInQueue_.AllocTensor<DTYPE_X>();
        const uint32_t inputRowBytes = tileN_ * dtypeSize_;
        const uint32_t copiedRowBytes = Align32(nw * dtypeSize_);
        DataCopyExtParams in;
        in.blockCount = static_cast<uint16_t>(mh);
        in.blockLen = nw * dtypeSize_;
        in.srcStride = static_cast<uint32_t>((transN_ - nw) * dtypeSize_);
        in.dstStride = inputRowBytes / kBlockBytes - copiedRowBytes / kBlockBytes;
        in.rsv = 0;
        DataCopyPadExtParams<DTYPE_X> pad;
        pad.isPad = false;
        pad.leftPadding = 0;
        pad.rightPadding = 0;
        pad.paddingValue = static_cast<DTYPE_X>(0);
        DataCopyPad(inUb, xGm_[matrixBase + mi * transN_ + nj], in, pad);
        transInQueue_.EnQue(inUb);
        LocalTensor<DTYPE_X> src = transInQueue_.DeQue<DTYPE_X>();

        LocalTensor<DTYPE_X> outUb = transOutQueue_.AllocTensor<DTYPE_X>();
        Transpose16xN(outUb, src, mh);
        PipeBarrier<PIPE_V>();
        transOutQueue_.EnQue(outUb);
        transInQueue_.FreeTensor(src);
        LocalTensor<DTYPE_X> dst = transOutQueue_.DeQue<DTYPE_X>();

        DataCopyExtParams out;
        out.blockCount = static_cast<uint16_t>(nw);
        out.blockLen = mh * dtypeSize_;
        out.srcStride = Align32(kVectorRows * dtypeSize_) / kBlockBytes - Align32(mh * dtypeSize_) / kBlockBytes;
        out.dstStride = static_cast<uint32_t>((transM_ - mh) * dtypeSize_);
        out.rsv = 0;
        DataCopyPad(yGm_[matrixBase + nj * transM_ + mi], dst, out);
        transOutQueue_.FreeTensor(dst);
    }

    __aicore__ inline void Transpose16xN(LocalTensor<DTYPE_X> &dst, const LocalTensor<DTYPE_X> &src, uint32_t mh)
    {
        LocalTensor<DTYPE_X> srcList[kVectorRows];
        for (uint32_t row = 0; row < kVectorRows; ++row) {
            srcList[row] = src[(row < mh ? row : 0) * tileN_];
        }
        if constexpr (sizeof(DTYPE_X) == 2) {
            LocalTensor<DTYPE_X> dstList[kVectorRows];
            for (uint32_t i = 0; i < kVectorRows; ++i) dstList[i] = dst[i * kVectorRows];
            const uint8_t repeats = static_cast<uint8_t>(tileN_ / 16);
            TransDataTo5HDParams params(false, false, repeats, repeats == 1 ? 0 : 16, repeats == 1 ? 0 : 1);
            TransDataTo5HD<DTYPE_X>(dstList, srcList, params);
        } else if constexpr (sizeof(DTYPE_X) == 4) {
            LocalTensor<DTYPE_X> dstList[kVectorRows];
            for (uint32_t i = 0; i < 8; ++i) {
                dstList[2 * i] = dst[i * kVectorRows];
                dstList[2 * i + 1] = dst[i * kVectorRows + 8];
            }
            const uint8_t repeats = static_cast<uint8_t>(tileN_ / 8);
            TransDataTo5HDParams params(false, false, repeats, repeats == 1 ? 0 : 16, repeats == 1 ? 0 : 1);
            TransDataTo5HD<DTYPE_X>(dstList, srcList, params);
        }
    }

private:
    TPipe pipe_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, kBufferNum> rowQueue_;
    TQue<TPosition::VECIN, kBufferNum> transInQueue_;
    TQue<TPosition::VECOUT, kBufferNum> transOutQueue_;
    GlobalTensor<DTYPE_X> xGm_;
    GlobalTensor<DTYPE_Y> yGm_;

    uint32_t mode_ = 0;
    uint32_t dtypeSize_ = 0;
    uint32_t blockDim_ = 0;
    uint32_t outerCount_ = 0;
    uint64_t rowWidth_ = 0;
    uint64_t numRows_ = 0;
    uint64_t srcInnerStride_ = 0;
    uint64_t outerOutShape_[kMaxDim] = {0};
    uint64_t outerSrcStride_[kMaxDim] = {0};
    uint32_t copyTileElems_ = 0;
    uint32_t stridedTileElems_ = 0;
    uint64_t transBatch_ = 0;
    uint64_t transM_ = 0;
    uint64_t transN_ = 0;
    uint32_t tileM_ = 0;
    uint32_t tileN_ = 0;
};

extern "C" __global__ __aicore__ void transpose(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(t, tiling);
    KernelTranspose op;
    op.Init(x, y, t);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void transpose_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x, uint8_t *y,
    uint8_t *workspace, uint8_t *tiling)
{
    transpose<<<blockDim, l2ctrl, stream>>>(x, y, workspace, tiling);
}
#endif
