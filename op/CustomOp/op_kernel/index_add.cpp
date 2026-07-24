/**
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * IndexAdd has one ordered copy phase followed by one of two scatter phases:
 *
 *   * aligned, non-BF16 vectors use DMA atomic add;
 *   * all other layouts build stable target buckets in workspace, then give
 *     each (before, target, K-tile) output tile exclusive ownership.
 *
 * The owner path therefore reads and writes an output tile once, while still
 * applying source occurrences in their original order.  This matters for
 * BF16 (RNE after every update), FP16, and int8 conversion semantics.
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t COPY_TILE_BYTES = 16U * 1024U;
constexpr uint32_t CAST_ALIGN_ELEMS = 256U;
constexpr uint32_t ATOMIC_PATH = 0U;

template <typename InputT, typename ComputeT>
class KernelIndexAdd {
public:
    static constexpr bool kNeedCast = !IsSameType<InputT, ComputeT>::value;

    template <typename TilingT>
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR output,
                                GM_ADDR workspace, const TilingT &t)
    {
        beforeDimSize_ = t.beforeDimSize;
        dimLen_ = t.dimLen;
        afterDimSize_ = t.afterDimSize;
        indexLen_ = t.indexLen;
        dtypeSize_ = t.dtypeSize;
        usedCoreNum_ = t.usedCoreNum;
        scatterCoreNum_ = t.scatterCoreNum;
        path_ = t.path;
        copyTileBytes_ = t.copyTileBytes;
        atomicTileBytes_ = t.atomicTileBytes;
        kTile_ = t.kTile;
        indexChunkLen_ = t.indexChunkLen;
        positionChunkLen_ = t.positionChunkLen;

        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(self));
        indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(index));
        sourceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(source));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(output));
        selfGmBytes_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(self));
        outputGmBytes_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(output));
        workspace_ = reinterpret_cast<__gm__ int32_t *>(workspace);

        // These buffers serve the copy phase and, on the owner path, source
        // and ping-pong accumulator storage.
        pipe_.InitBuffer(copyInTBuf_, copyTileBytes_);
        pipe_.InitBuffer(copyOutTBuf_, copyTileBytes_);

        const uint32_t indexBytes = (indexChunkLen_ * sizeof(int32_t) + 31U) & ~31U;
        pipe_.InitBuffer(indexTBuf_, indexBytes);
        if (path_ == ATOMIC_PATH) {
            // A VECIN -> VECOUT bridge is required before an MTE3 atomic
            // store.  The two queues allow consecutive vector tiles to
            // overlap MTE2/MTE3 work without another global synchronization.
            pipe_.InitBuffer(atomicInQue_, 2, atomicTileBytes_);
            pipe_.InitBuffer(atomicOutQue_, 2, atomicTileBytes_);
        } else {
            // The initial output tile is loaded into VECIN.  Every vector
            // update must write VECOUT, so two VECOUT buffers ping-pong after
            // the first occurrence; never swap a VECOUT result back into a
            // VECIN destination.
            pipe_.InitBuffer(rmwOutInTBuf_, atomicTileBytes_);
            pipe_.InitBuffer(rmwOutTBuf_, atomicTileBytes_);
            pipe_.InitBuffer(rmwPaddingTBuf_, 256U);
            if constexpr (kNeedCast) {
                const uint32_t computeBytes = kTile_ * sizeof(ComputeT);
                pipe_.InitBuffer(srcCompTBuf_, computeBytes);
                pipe_.InitBuffer(castPaddingTBuf_, 256U);
                pipe_.InitBuffer(outCompTBuf_, computeBytes);
            }
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreId = GetBlockIdx();
        Phase1Copy(coreId);
        // Output is initialized before any atomic/owner update.  This is the
        // only all-core synchronization on the atomic path.
        SyncAll();
        PipeBarrier<PIPE_ALL>();

        if (path_ == ATOMIC_PATH) {
            Phase2Atomic(coreId);
        } else {
            Phase2OwnedTiles(coreId);
        }
    }

private:
    __aicore__ inline void Phase1Copy(uint32_t coreId)
    {
        const uint64_t totalBytes = static_cast<uint64_t>(beforeDimSize_) * dimLen_ * afterDimSize_ * dtypeSize_;
        const uint64_t start = totalBytes * coreId / usedCoreNum_;
        const uint64_t end = totalBytes * (coreId + 1U) / usedCoreNum_;
        for (uint64_t pos = start; pos < end; pos += copyTileBytes_) {
            const uint64_t remaining = end - pos;
            const uint32_t bytes = remaining < copyTileBytes_ ? static_cast<uint32_t>(remaining) : copyTileBytes_;
            DataCopyExtParams params{1, bytes, 0, 0, 0};
            DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
            LocalTensor<uint8_t> in = copyInTBuf_.Get<uint8_t>();
            LocalTensor<uint8_t> out = copyOutTBuf_.Get<uint8_t>();
            DataCopyPad(in, selfGmBytes_[pos], params, pad);
            PipeBarrier<PIPE_ALL>();
            DataCopy(out, in, (bytes + 31U) & ~31U);
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(outputGmBytes_[pos], out, params);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void LoadIndexChunk(uint32_t start, uint32_t count)
    {
        indexLocal_ = indexTBuf_.Get<int32_t>();
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(indexLocal_, indexGm_[start], params, pad);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline bool ValidIndex(int32_t idx) const
    {
        return idx >= 0 && static_cast<uint32_t>(idx) < dimLen_;
    }

    __aicore__ inline void GetOffsets(uint32_t row, uint32_t i, uint32_t target,
                                      uint64_t &outOff, uint64_t &srcOff) const
    {
        outOff = (static_cast<uint64_t>(row) * dimLen_ + target) * afterDimSize_;
        srcOff = (static_cast<uint64_t>(row) * indexLen_ + i) * afterDimSize_;
    }

    __aicore__ inline void Phase2Atomic(uint32_t coreId)
    {
        const uint64_t workCount = static_cast<uint64_t>(beforeDimSize_) * indexLen_;
        const uint64_t workBegin = workCount * coreId / scatterCoreNum_;
        const uint64_t workEnd = workCount * (coreId + 1U) / scatterCoreNum_;
        for (uint64_t work = workBegin; work < workEnd;) {
            const uint32_t row = static_cast<uint32_t>(work / indexLen_);
            const uint32_t begin = static_cast<uint32_t>(work - static_cast<uint64_t>(row) * indexLen_);
            // Do not cross either a DMA index chunk or a source row.  Thus a
            // core owns a contiguous vector range but pays one index DMA for
            // up to indexChunkLen_ atomic updates.
            const uint32_t rowRemaining = indexLen_ - begin;
            const uint64_t coreRemaining = workEnd - work;
            uint32_t count = rowRemaining < indexChunkLen_ ? rowRemaining : indexChunkLen_;
            if (coreRemaining < count) count = static_cast<uint32_t>(coreRemaining);
            LoadIndexChunk(begin, count);
            for (uint32_t j = 0; j < count; ++j) {
                const int32_t idx = indexLocal_.GetValue(j);
                if (!ValidIndex(idx)) continue;
                uint64_t outOff, srcOff;
                GetOffsets(row, begin + j, static_cast<uint32_t>(idx), outOff, srcOff);
                AtomicAddRange(outOff, srcOff, afterDimSize_);
            }
            work += count;
        }
    }

    __aicore__ inline void AtomicAddRange(uint64_t outOff, uint64_t srcOff, uint32_t count)
    {
        const uint32_t tileElems = atomicTileBytes_ / sizeof(InputT);
        for (uint32_t offset = 0; offset < count; offset += tileElems) {
            const uint32_t n = (count - offset) < tileElems ? (count - offset) : tileElems;
            LocalTensor<InputT> src = atomicInQue_.AllocTensor<InputT>();
            DataCopy(src, sourceGm_[srcOff + offset], n);
            atomicInQue_.EnQue(src);

            src = atomicInQue_.DeQue<InputT>();
            LocalTensor<InputT> bridge = atomicOutQue_.AllocTensor<InputT>();
            DataCopy(bridge, src, n);
            atomicOutQue_.EnQue(bridge);
            atomicInQue_.FreeTensor(src);

            bridge = atomicOutQue_.DeQue<InputT>();
            SetAtomicAdd<InputT>();
            DataCopy(outputGm_[outOff + offset], bridge, n);
            // Wait for this MTE3 command before clearing its atomic mode.  It
            // is a local pipe dependency, not an all-core synchronization.
            PipeBarrier<PIPE_ALL>();
            SetAtomicNone();
            atomicOutQue_.FreeTensor(bridge);
        }
    }

    __aicore__ inline void Phase2OwnedTiles(uint32_t coreId)
    {
        const uint64_t kTiles = (afterDimSize_ + kTile_ - 1U) / kTile_;
        const uint64_t workCount = static_cast<uint64_t>(beforeDimSize_) * dimLen_ * kTiles;
        const uint64_t workBegin = workCount * coreId / scatterCoreNum_;
        const uint64_t workEnd = workCount * (coreId + 1U) / scatterCoreNum_;
        for (uint64_t work = workBegin; work < workEnd; ++work) {
            const uint32_t row = static_cast<uint32_t>(work / (static_cast<uint64_t>(dimLen_) * kTiles));
            const uint64_t rem = work - static_cast<uint64_t>(row) * dimLen_ * kTiles;
            const uint32_t target = static_cast<uint32_t>(rem / kTiles);
            const uint32_t kTile = static_cast<uint32_t>(rem - static_cast<uint64_t>(target) * kTiles);
            const uint32_t kOffset = kTile * kTile_;
            const uint32_t count = (afterDimSize_ - kOffset) < kTile_ ? afterDimSize_ - kOffset : kTile_;
            OwnedAddTile(row, target, kOffset, count);
        }
    }

    __aicore__ inline void OwnedAddTile(uint32_t row, uint32_t target, uint32_t kOffset, uint32_t count)
    {
        uint64_t outOff, ignoredSrcOff;
        GetOffsets(row, 0U, target, outOff, ignoredSrcOff);
        outOff += kOffset;

        LocalTensor<InputT> acc = rmwOutInTBuf_.Get<InputT>();
        LocalTensor<InputT> next = copyOutTBuf_.Get<InputT>();
        LocalTensor<InputT> alternate = rmwOutTBuf_.Get<InputT>();
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(InputT)), 0, 0, 0};
        const uint32_t alignedCount =
            (count * sizeof(InputT) + 31U) / 32U * (32U / sizeof(InputT));
        // Explicitly pad the UB tail.  With isPad=false an unaligned final
        // GM slice may make MTE read the rounded-up bytes beyond a tensor's
        // allocation (notably the last 993-float source row in c06).
        DataCopyPadExtParams<InputT> pad{true, 0,
            static_cast<uint8_t>(alignedCount - count), static_cast<InputT>(0)};
        DataCopyPad(acc, outputGm_[outOff], params, pad);
        PipeBarrier<PIPE_ALL>();

        for (uint32_t indexBegin = 0; indexBegin < indexLen_; indexBegin += indexChunkLen_) {
            const uint32_t indexCount = (indexLen_ - indexBegin) < indexChunkLen_ ?
                (indexLen_ - indexBegin) : indexChunkLen_;
            LoadIndexChunk(indexBegin, indexCount);
            for (uint32_t p = 0; p < indexCount; ++p) {
                const int32_t idx = indexLocal_.GetValue(p);
                if (idx != static_cast<int32_t>(target)) continue;
                const uint32_t i = indexBegin + p;
                const uint64_t srcOff = (static_cast<uint64_t>(row) * indexLen_ + i) * afterDimSize_ + kOffset;
                LocalTensor<InputT> src = copyInTBuf_.Get<InputT>();
                DataCopyPad(src, sourceGm_[srcOff], params, pad);
                PipeBarrier<PIPE_ALL>();
                AddOccurrence(next, acc, src, count);
                PipeBarrier<PIPE_ALL>();
                // `acc` is VECIN for the first update and VECOUT afterwards.
                // Both destinations stay VECOUT, as required by vector ops.
                LocalTensor<InputT> swap = next;
                acc = next;
                next = alternate;
                alternate = swap;
            }
        }
        DataCopyPad(outputGm_[outOff], acc, params);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void AddOccurrence(LocalTensor<InputT> dst, LocalTensor<InputT> acc,
                                         LocalTensor<InputT> src, uint32_t count)
    {
        if constexpr (kNeedCast) {
            // kTile is a multiple of 256; padding lets Cast use its required
            // vector granularity while only count logical elements are stored.
            const uint32_t paddedCount = (count + CAST_ALIGN_ELEMS - 1U) & ~(CAST_ALIGN_ELEMS - 1U);
            LocalTensor<ComputeT> srcComp = srcCompTBuf_.Get<ComputeT>();
            LocalTensor<ComputeT> accComp = outCompTBuf_.Get<ComputeT>();
            Cast(srcComp, src, RoundMode::CAST_NONE, paddedCount);
            Cast(accComp, acc, RoundMode::CAST_NONE, paddedCount);
            Add(accComp, accComp, srcComp, static_cast<int32_t>(paddedCount));
            // CAST_RINT retains the existing BF16 and int8 per-occurrence
            // rounding behavior; do not combine multiple occurrences.
            Cast(dst, accComp, RoundMode::CAST_RINT, paddedCount);
        } else {
            Add(dst, acc, src, static_cast<int32_t>(count));
        }
    }

private:
    TPipe pipe_;
    GlobalTensor<InputT> selfGm_;
    GlobalTensor<int32_t> indexGm_;
    GlobalTensor<InputT> sourceGm_;
    GlobalTensor<InputT> outputGm_;
    GlobalTensor<uint8_t> selfGmBytes_;
    GlobalTensor<uint8_t> outputGmBytes_;
    __gm__ int32_t *workspace_;

    TBuf<TPosition::VECIN> copyInTBuf_;
    TBuf<TPosition::VECOUT> copyOutTBuf_;
    TBuf<TPosition::VECIN> indexTBuf_;
    TBuf<TPosition::VECIN> rmwOutInTBuf_;
    TBuf<TPosition::VECOUT> rmwOutTBuf_;
    TBuf<TPosition::VECCALC> rmwPaddingTBuf_;
    TBuf<TPosition::VECCALC> srcCompTBuf_;
    TBuf<TPosition::VECCALC> castPaddingTBuf_;
    TBuf<TPosition::VECCALC> outCompTBuf_;
    TQue<TPosition::VECIN, 1> atomicInQue_;
    TQue<TPosition::VECOUT, 1> atomicOutQue_;
    LocalTensor<int32_t> indexLocal_;

    uint32_t beforeDimSize_;
    uint32_t dimLen_;
    uint32_t afterDimSize_;
    uint32_t indexLen_;
    uint32_t dtypeSize_;
    uint32_t usedCoreNum_;
    uint32_t scatterCoreNum_;
    uint32_t path_;
    uint32_t copyTileBytes_;
    uint32_t atomicTileBytes_;
    uint32_t kTile_;
    uint32_t indexChunkLen_;
    uint32_t positionChunkLen_;
};

extern "C" __global__ __aicore__ void index_add(GM_ADDR self, GM_ADDR index, GM_ADDR source,
                                                  GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(t, tiling);
    if (t.dtype == 0U) {
        KernelIndexAdd<float, float> op; op.Init(self, index, source, output, workspace, t); op.Process();
    } else if (t.dtype == 1U) {
        KernelIndexAdd<bfloat16_t, float> op; op.Init(self, index, source, output, workspace, t); op.Process();
    } else if (t.dtype == 2U) {
        KernelIndexAdd<half, half> op; op.Init(self, index, source, output, workspace, t); op.Process();
    } else if (t.dtype == 3U) {
        KernelIndexAdd<int32_t, int32_t> op; op.Init(self, index, source, output, workspace, t); op.Process();
    } else if (t.dtype == 4U) {
        KernelIndexAdd<int8_t, half> op; op.Init(self, index, source, output, workspace, t); op.Process();
    }
}
