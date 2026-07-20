/**
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * IndexAdd has two globally ordered phases:
 *   1. split and copy self to output;
 *   2. scatter source vectors.  Aligned vectors use native DMA atomic add in
 *      parallel.  Any unaligned prefix/suffix is applied by core 0 after an
 *      all-core barrier, preserving source order.  Layouts without a useful
 *      aligned body use deterministic index ownership instead: every update
 *      for index j belongs to core j % scatterCoreNum.
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t COPY_TILE_BYTES = 16U * 1024U;
constexpr uint32_t CAST_ALIGN_ELEMS = 256U;
constexpr uint32_t MIN_ATOMIC_BODY_BYTES = 256U;

template <typename InputT, typename ComputeT>
class KernelIndexAdd {
public:
    static constexpr bool kNeedCast = !IsSameType<InputT, ComputeT>::value;

    template <typename TilingT>
    __aicore__ inline void Init(GM_ADDR self, GM_ADDR index, GM_ADDR source, GM_ADDR output,
                                const TilingT &t)
    {
        beforeDimSize_ = t.beforeDimSize;
        dimLen_ = t.dimLen;
        afterDimSize_ = t.afterDimSize;
        indexLen_ = t.indexLen;
        dtypeSize_ = t.dtypeSize;
        usedCoreNum_ = t.usedCoreNum;
        scatterCoreNum_ = t.scatterCoreNum;
        atomicEnabled_ = t.atomicEnabled;
        copyTileBytes_ = t.copyTileBytes;
        atomicTileBytes_ = t.atomicTileBytes;
        rmwTileLen_ = t.rmwTileLen;

        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(self));
        indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(index));
        sourceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(source));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(output));
        selfGmBytes_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(self));
        outputGmBytes_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(output));

        // copyIn/copyOut are reused as the atomic source/bridge after phase 1.
        pipe_.InitBuffer(copyInTBuf_, copyTileBytes_);
        pipe_.InitBuffer(copyOutTBuf_, copyTileBytes_);
        const uint32_t indexBytes = (indexLen_ * sizeof(int32_t) + 31U) & ~31U;
        pipe_.InitBuffer(indexTBuf_, indexBytes);
        pipe_.InitBuffer(rmwOutInTBuf_, atomicTileBytes_);
        // Keep the Cast/Add working buffers separated by one 256B allocation
        // to avoid the DAV_2201 UB bank-conflict pattern.
        pipe_.InitBuffer(rmwPaddingTBuf_, 256U);
        if constexpr (kNeedCast) {
            const uint32_t computeBytes = rmwTileLen_ * sizeof(ComputeT);
            pipe_.InitBuffer(srcCompTBuf_, computeBytes);
            pipe_.InitBuffer(castPaddingTBuf_, 256U);
            pipe_.InitBuffer(outCompTBuf_, computeBytes);
        }
    }

    __aicore__ inline void Process()
    {
        const uint32_t coreId = GetBlockIdx();
        Phase1Copy(coreId);
        SyncAll();
        PipeBarrier<PIPE_ALL>();

        LoadIndex();
        if (atomicEnabled_ != 0U) {
            Phase2Atomic(coreId);
            // Atomic MTE3 writes must all be visible before the one serial
            // owner writes non-atomic tails.
            SyncAll();
            PipeBarrier<PIPE_ALL>();
            if (coreId == 0U) {
                Phase3SerialTails();
            }
        } else {
            Phase2OwnedRmw(coreId);
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
            DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = bytes;
            params.srcStride = 0;
            params.dstStride = 0;
            params.rsv = 0;
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

    __aicore__ inline void LoadIndex()
    {
        indexLocal_ = indexTBuf_.Get<int32_t>();
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = indexLen_ * sizeof(int32_t);
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(indexLocal_, indexGm_, params, pad);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline bool ValidIndex(int32_t idx) const
    {
        return idx >= 0 && static_cast<uint32_t>(idx) < dimLen_;
    }

    __aicore__ inline void GetOffsets(uint32_t row, uint32_t i, int32_t idx,
                                      uint64_t &outOff, uint64_t &srcOff) const
    {
        outOff = (static_cast<uint64_t>(row) * dimLen_ + static_cast<uint32_t>(idx)) * afterDimSize_;
        srcOff = (static_cast<uint64_t>(row) * indexLen_ + i) * afterDimSize_;
    }

    // Return an atomic middle [middleOffset, middleOffset + middleCount).
    // Tensors are 32B aligned at their base; offsets establish whether source
    // and output can be aligned at the same logical element.  The host only
    // enables this mode for full-vector alignment, but retaining this general
    // calculation makes tail handling safe for future threshold tuning.
    __aicore__ inline void GetAtomicRange(uint64_t outOff, uint64_t srcOff,
                                          uint32_t &middleOffset, uint32_t &middleCount) const
    {
        const uint64_t vectorBytes = static_cast<uint64_t>(afterDimSize_) * sizeof(InputT);
        const uint64_t outBytes = outOff * sizeof(InputT);
        const uint64_t srcBytes = srcOff * sizeof(InputT);
        middleOffset = 0;
        middleCount = 0;
        if ((outBytes & 31U) != (srcBytes & 31U)) return;
        const uint32_t leadBytes = static_cast<uint32_t>((32U - (srcBytes & 31U)) & 31U);
        if (leadBytes >= vectorBytes) return;
        const uint64_t bodyBytes = (vectorBytes - leadBytes) & ~static_cast<uint64_t>(31U);
        if (bodyBytes < MIN_ATOMIC_BODY_BYTES) return;
        middleOffset = leadBytes / sizeof(InputT);
        middleCount = static_cast<uint32_t>(bodyBytes / sizeof(InputT));
    }

    __aicore__ inline void Phase2Atomic(uint32_t coreId)
    {
        const uint64_t taskCount = static_cast<uint64_t>(beforeDimSize_) * indexLen_;
        const uint64_t taskBegin = taskCount * coreId / scatterCoreNum_;
        const uint64_t taskEnd = taskCount * (coreId + 1U) / scatterCoreNum_;
        for (uint64_t task = taskBegin; task < taskEnd; ++task) {
            const uint32_t row = static_cast<uint32_t>(task / indexLen_);
            const uint32_t i = static_cast<uint32_t>(task - static_cast<uint64_t>(row) * indexLen_);
            const int32_t idx = indexLocal_.GetValue(i);
            if (!ValidIndex(idx)) continue;
            uint64_t outOff, srcOff;
            GetOffsets(row, i, idx, outOff, srcOff);
            uint32_t middleOffset, middleCount;
            GetAtomicRange(outOff, srcOff, middleOffset, middleCount);
            AtomicAddRange(outOff + middleOffset, srcOff + middleOffset, middleCount);
        }
    }

    __aicore__ inline void AtomicAddRange(uint64_t outOff, uint64_t srcOff, uint32_t count)
    {
        const uint32_t tileElems = atomicTileBytes_ / sizeof(InputT);
        for (uint32_t offset = 0; offset < count; offset += tileElems) {
            const uint32_t n = (count - offset) < tileElems ? (count - offset) : tileElems;
            LocalTensor<InputT> src = copyInTBuf_.Get<InputT>();
            LocalTensor<InputT> bridge = copyOutTBuf_.Get<InputT>();
            DataCopy(src, sourceGm_[srcOff + offset], n);
            PipeBarrier<PIPE_ALL>();
            DataCopy(bridge, src, n);
            PipeBarrier<PIPE_ALL>();
            SetAtomicAdd<InputT>();
            DataCopy(outputGm_[outOff + offset], bridge, n);
            PipeBarrier<PIPE_ALL>();
            SetAtomicNone();
        }
    }

    __aicore__ inline void Phase3SerialTails()
    {
        for (uint32_t row = 0; row < beforeDimSize_; ++row) {
            for (uint32_t i = 0; i < indexLen_; ++i) {
                const int32_t idx = indexLocal_.GetValue(i);
                if (!ValidIndex(idx)) continue;
                uint64_t outOff, srcOff;
                GetOffsets(row, i, idx, outOff, srcOff);
                uint32_t middleOffset, middleCount;
                GetAtomicRange(outOff, srcOff, middleOffset, middleCount);
                if (middleOffset != 0U) ScatterRange(outOff, srcOff, middleOffset);
                const uint32_t suffix = afterDimSize_ - middleOffset - middleCount;
                if (suffix != 0U) ScatterRange(outOff + middleOffset + middleCount,
                                                srcOff + middleOffset + middleCount, suffix);
            }
        }
    }

    __aicore__ inline void Phase2OwnedRmw(uint32_t coreId)
    {
        // All occurrences of a target index are handled by one core, so this
        // path retains exact sequential RMW semantics even for repeated index.
        for (uint32_t row = 0; row < beforeDimSize_; ++row) {
            for (uint32_t i = 0; i < indexLen_; ++i) {
                const int32_t idx = indexLocal_.GetValue(i);
                if (!ValidIndex(idx) || (static_cast<uint32_t>(idx) % scatterCoreNum_) != coreId) continue;
                uint64_t outOff, srcOff;
                GetOffsets(row, i, idx, outOff, srcOff);
                ScatterRange(outOff, srcOff, afterDimSize_);
            }
        }
    }

    __aicore__ inline void ScatterRange(uint64_t outOff, uint64_t srcOff, uint32_t count)
    {
        for (uint32_t offset = 0; offset < count; offset += rmwTileLen_) {
            const uint32_t n = (count - offset) < rmwTileLen_ ? (count - offset) : rmwTileLen_;
            ScatterAddTile(outOff + offset, srcOff + offset, n);
        }
    }

    __aicore__ inline void ScatterAddTile(uint64_t outOff, uint64_t srcOff, uint32_t n)
    {
        LocalTensor<InputT> src = copyInTBuf_.Get<InputT>();
        LocalTensor<InputT> outIn = rmwOutInTBuf_.Get<InputT>();
        LocalTensor<InputT> out = copyOutTBuf_.Get<InputT>();
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = n * sizeof(InputT);
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;
        DataCopyPadExtParams<InputT> pad{false, 0, 0, 0};
        DataCopyPad(src, sourceGm_[srcOff], params, pad);
        DataCopyPad(outIn, outputGm_[outOff], params, pad);
        PipeBarrier<PIPE_ALL>();

        if constexpr (kNeedCast) {
            const uint32_t paddedN = (n + CAST_ALIGN_ELEMS - 1U) & ~(CAST_ALIGN_ELEMS - 1U);
            LocalTensor<ComputeT> srcComp = srcCompTBuf_.Get<ComputeT>();
            LocalTensor<ComputeT> outComp = outCompTBuf_.Get<ComputeT>();
            Cast(srcComp, src, RoundMode::CAST_NONE, paddedN);
            Cast(outComp, outIn, RoundMode::CAST_NONE, paddedN);
            Add(outComp, outComp, srcComp, static_cast<int32_t>(paddedN));
            Cast(out, outComp, RoundMode::CAST_RINT, paddedN);
        } else {
            Add(out, outIn, src, static_cast<int32_t>(n));
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(outputGm_[outOff], out, params);
        PipeBarrier<PIPE_ALL>();
    }

private:
    TPipe pipe_;
    GlobalTensor<InputT> selfGm_;
    GlobalTensor<int32_t> indexGm_;
    GlobalTensor<InputT> sourceGm_;
    GlobalTensor<InputT> outputGm_;
    GlobalTensor<uint8_t> selfGmBytes_;
    GlobalTensor<uint8_t> outputGmBytes_;

    TBuf<TPosition::VECIN> copyInTBuf_;
    TBuf<TPosition::VECOUT> copyOutTBuf_;
    TBuf<TPosition::VECIN> indexTBuf_;
    TBuf<TPosition::VECIN> rmwOutInTBuf_;
    TBuf<TPosition::VECCALC> rmwPaddingTBuf_;
    TBuf<TPosition::VECCALC> srcCompTBuf_;
    TBuf<TPosition::VECCALC> castPaddingTBuf_;
    TBuf<TPosition::VECCALC> outCompTBuf_;
    LocalTensor<int32_t> indexLocal_;

    uint32_t beforeDimSize_;
    uint32_t dimLen_;
    uint32_t afterDimSize_;
    uint32_t indexLen_;
    uint32_t dtypeSize_;
    uint32_t usedCoreNum_;
    uint32_t scatterCoreNum_;
    uint32_t atomicEnabled_;
    uint32_t copyTileBytes_;
    uint32_t atomicTileBytes_;
    uint32_t rmwTileLen_;
};

extern "C" __global__ __aicore__ void index_add(GM_ADDR self, GM_ADDR index, GM_ADDR source,
                                                  GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(t, tiling);
    if (t.dtype == 0U) {
        KernelIndexAdd<float, float> op; op.Init(self, index, source, output, t); op.Process();
    } else if (t.dtype == 1U) {
        KernelIndexAdd<bfloat16_t, float> op; op.Init(self, index, source, output, t); op.Process();
    } else if (t.dtype == 2U) {
        KernelIndexAdd<half, half> op; op.Init(self, index, source, output, t); op.Process();
    } else if (t.dtype == 3U) {
        KernelIndexAdd<int32_t, int32_t> op; op.Init(self, index, source, output, t); op.Process();
    } else if (t.dtype == 4U) {
        KernelIndexAdd<int8_t, half> op; op.Init(self, index, source, output, t); op.Process();
    }
}
