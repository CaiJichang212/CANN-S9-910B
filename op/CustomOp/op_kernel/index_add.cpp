/**
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * IndexAdd 算子 kernel 侧实现。
 *
 * 语义：output = copy(self)，再对每个 i ∈ [0, M)：
 *   output[..., index[i], ...] += source[..., i, ...]
 *
 * 数据视图：[beforeDimSize, dimLen, afterDimSize]，afterDimSize 维内存连续。
 *
 * 两阶段（所有 dtype 统一）：
 *   阶段 1 bulk copy self→output：扁平字节均匀切给 usedCoreNum 核，每核写互不重叠的 32B
 *            区间。读入 VECIN，UB→UB 拷到 VECOUT 再 MTE3 写回（MTE3 须从 VECOUT 读）。
 *            完成后 SyncAll() 全核同步。
 *   阶段 2 scatter-add：按 (rowRange×afterSlice) 二维切分，各核输出区域互不重叠 → 无 WAW、
 *            无需原子。每核串行 RMW（读 output→加 source→写 output），重复 index 同核内自然累加。
 *
 * dtype 分派（template<InputT,ComputeT>）：
 *   fp32 / fp16 / int32 : InputT==ComputeT，直接 Add。
 *   int8  : InputT=int8, ComputeT=half。int8→half→Add→half→int8（half 精确表示 int8，无舍入损失）。
 *   bf16  : InputT=bfloat16_t, ComputeT=float。bf16→float→Add→float→bf16。逐次 RMW 舍入为 RNE，
 *           与 torch.bfloat16 原生 bf16 add（逐次 RNE）一致。
 *
 * Cast count 须 256B 对齐：DataCopyPad 读 n 真实元素，Cast 用 round_up(n,256) 对齐计数；
 *   padding 区读到 UB 残留垃圾但不写回（仅写真实 n 字节），故结果正确。
 *   上行转换（int8→half / bf16→float）只支持 CAST_NONE；下行转换用 CAST_RINT（RNE，与 torch 一致；
 *   CAST_ROUND 是 ties-away，会在精确 tie 处与 torch 差 1 ULP）。
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr uint32_t AICORE_NUM = 20;
constexpr uint32_t COPY_TILE_BYTES = 16384;    // 阶段 1 单块字节数
constexpr uint32_t MODE_ROW = 0;
constexpr uint32_t MODE_AFTER = 1;
constexpr uint32_t CAST_ALIGN_ELEMS = 256;    // Cast count 对齐粒度

template <typename InputT, typename ComputeT>
class KernelIndexAdd {
public:
    static constexpr bool kNeedCast = !IsSameType<InputT, ComputeT>::value;

    __aicore__ inline KernelIndexAdd() {}

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
        mode_ = t.mode;
        scatterTileLen_ = t.scatterTileLen;

        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(self));
        indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(index));
        sourceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(source));
        outputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT *>(output));
        selfGmBytes_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(self));
        outGmBytes_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(output));

        // 阶段 1 bulk copy：VECIN 读 + VECOUT 写
        pipe.InitBuffer(copyInTBuf_, COPY_TILE_BYTES);
        pipe.InitBuffer(copyOutTBuf_, COPY_TILE_BYTES);

        // index UB 缓冲（读一次，循环复用）
        uint32_t indexBytes = ((indexLen_ * sizeof(int32_t)) + 31U) & ~31U;
        pipe.InitBuffer(indexTBuf_, indexBytes);

        // 阶段 2 scatter 缓冲
        uint32_t scBytes = scatterTileLen_ * static_cast<uint32_t>(sizeof(InputT));
        pipe.InitBuffer(srcTBuf_, scBytes);       // VECIN: source 读
        pipe.InitBuffer(outInTBuf_, scBytes);     // VECIN: output 读 (RMW)
        pipe.InitBuffer(outOutTBuf_, scBytes);    // VECOUT: output 写
        if constexpr (kNeedCast) {
            uint32_t ccBytes = scatterTileLen_ * static_cast<uint32_t>(sizeof(ComputeT));
            pipe.InitBuffer(srcCompTBuf_, ccBytes);  // VECCALC: Cast src
            pipe.InitBuffer(outCompTBuf_, ccBytes);  // VECCALC: Cast out + Add
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t coreId = GetBlockIdx();
        bool active = (coreId < usedCoreNum_) && (beforeDimSize_ != 0) && (dimLen_ != 0) &&
                      (afterDimSize_ != 0) && (indexLen_ != 0);

        if (active) {
            Phase1Copy(coreId);
        }
        SyncAll();
        PipeBarrier<PIPE_ALL>();

        if (!active) {
            return;
        }
        LoadIndex();
        Phase2Scatter(coreId);
    }

private:
    // ===== 阶段 1：bulk copy self→output（本核负责的扁平字节区间）=====
    __aicore__ inline void Phase1Copy(uint32_t coreId)
    {
        uint64_t totalBytes = static_cast<uint64_t>(beforeDimSize_) * dimLen_ * afterDimSize_ * dtypeSize_;
        if (totalBytes == 0) return;
        uint64_t start = totalBytes * coreId / usedCoreNum_;
        uint64_t end = totalBytes * (coreId + 1) / usedCoreNum_;
        if (start >= end) return;
        uint64_t total = end - start;

        for (uint64_t off = 0; off < total; off += COPY_TILE_BYTES) {
            uint64_t remain = total - off;
            uint32_t chunk = (remain < COPY_TILE_BYTES) ? static_cast<uint32_t>(remain) : COPY_TILE_BYTES;

            DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = chunk;
            params.srcStride = 0;
            params.dstStride = 0;
            params.rsv = 0;
            DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};

            LocalTensor<uint8_t> inBuf = copyInTBuf_.Get<uint8_t>();
            DataCopyPad(inBuf, selfGmBytes_[start + off], params, pad);
            PipeBarrier<PIPE_ALL>();
            LocalTensor<uint8_t> outBuf = copyOutTBuf_.Get<uint8_t>();
            uint32_t alignedBytes = (chunk + 31U) & ~31U;
            DataCopy(outBuf, inBuf, alignedBytes);
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(outGmBytes_[start + off], outBuf, params);
            PipeBarrier<PIPE_ALL>();
        }
    }

    // ===== 加载 index 到 UB =====
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

    // ===== 阶段 2：scatter-add =====
    __aicore__ inline void Phase2Scatter(uint32_t coreId)
    {
        uint32_t rowStart, rowEnd, afterOff, afterCnt;
        if (mode_ == MODE_ROW) {
            rowStart = beforeDimSize_ * coreId / usedCoreNum_;
            rowEnd = beforeDimSize_ * (coreId + 1) / usedCoreNum_;
            afterOff = 0;
            afterCnt = afterDimSize_;
        } else {  // MODE_AFTER
            rowStart = 0;
            rowEnd = beforeDimSize_;
            afterOff = afterDimSize_ * coreId / usedCoreNum_;
            afterCnt = afterDimSize_ * (coreId + 1) / usedCoreNum_ - afterOff;
        }
        if (rowEnd <= rowStart || afterCnt == 0) return;

        for (uint32_t row = rowStart; row < rowEnd; row++) {
            uint64_t rowOutBase = static_cast<uint64_t>(row) * dimLen_ * afterDimSize_ + afterOff;
            uint64_t rowSrcBase = static_cast<uint64_t>(row) * indexLen_ * afterDimSize_ + afterOff;
            for (uint32_t i = 0; i < indexLen_; i++) {
                int32_t idx = indexLocal_.GetValue(i);
                if (idx < 0 || static_cast<uint32_t>(idx) >= dimLen_) {
                    continue;
                }
                uint64_t outOff = rowOutBase + static_cast<uint64_t>(idx) * afterDimSize_;
                uint64_t srcOff = rowSrcBase + static_cast<uint64_t>(i) * afterDimSize_;
                ScatterRange(outputGm_[outOff], sourceGm_[srcOff], afterCnt);
            }
        }
    }

    __aicore__ inline void ScatterRange(const GlobalTensor<InputT> &dstGm,
                                        const GlobalTensor<InputT> &srcGm, uint32_t cnt)
    {
        for (uint32_t t0 = 0; t0 < cnt; t0 += scatterTileLen_) {
            uint32_t n = cnt - t0;
            if (n > scatterTileLen_) {
                n = scatterTileLen_;
            }
            ScatterAddTile(dstGm[t0], srcGm[t0], n);
        }
    }

    // 单段 n 元素的 RMW：output[outOff] += source[srcOff]
    __aicore__ inline void ScatterAddTile(const GlobalTensor<InputT> &dstGm,
                                          const GlobalTensor<InputT> &srcGm, uint32_t n)
    {
        LocalTensor<InputT> srcLocal = srcTBuf_.Get<InputT>();      // VECIN
        LocalTensor<InputT> outInLocal = outInTBuf_.Get<InputT>();  // VECIN
        LocalTensor<InputT> outOutLocal = outOutTBuf_.Get<InputT>(); // VECOUT

        if constexpr (kNeedCast) {
            uint32_t paddedN = (n + CAST_ALIGN_ELEMS - 1) & ~(CAST_ALIGN_ELEMS - 1);
            DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = n * sizeof(InputT);
            params.srcStride = 0;
            params.dstStride = 0;
            params.rsv = 0;
            DataCopyPadExtParams<InputT> pad{false, 0, 0, 0};
            DataCopyPad(srcLocal, srcGm, params, pad);
            DataCopyPad(outInLocal, dstGm, params, pad);
            PipeBarrier<PIPE_ALL>();

            LocalTensor<ComputeT> srcComp = srcCompTBuf_.Get<ComputeT>();
            LocalTensor<ComputeT> outComp = outCompTBuf_.Get<ComputeT>();
            // 上行 CAST_NONE（int8→half/bf16→float 唯一支持模式）；下行 CAST_RINT = RNE（与 torch 一致）。
            Cast(srcComp, srcLocal, RoundMode::CAST_NONE, paddedN);
            Cast(outComp, outInLocal, RoundMode::CAST_NONE, paddedN);
            Add(outComp, outComp, srcComp, static_cast<int32_t>(paddedN));
            Cast(outOutLocal, outComp, RoundMode::CAST_RINT, paddedN);
            PipeBarrier<PIPE_ALL>();

            DataCopyExtParams outParams;
            outParams.blockCount = 1;
            outParams.blockLen = n * sizeof(InputT);
            outParams.srcStride = 0;
            outParams.dstStride = 0;
            outParams.rsv = 0;
            DataCopyPad(dstGm, outOutLocal, outParams);
            PipeBarrier<PIPE_ALL>();
        } else {
            DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = n * sizeof(InputT);
            params.srcStride = 0;
            params.dstStride = 0;
            params.rsv = 0;
            DataCopyPadExtParams<InputT> pad{false, 0, 0, 0};
            DataCopyPad(srcLocal, srcGm, params, pad);
            DataCopyPad(outInLocal, dstGm, params, pad);
            PipeBarrier<PIPE_ALL>();
            Add(outOutLocal, outInLocal, srcLocal, static_cast<int32_t>(n));
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(dstGm, outOutLocal, params);
            PipeBarrier<PIPE_ALL>();
        }
    }

private:
    TPipe pipe;
    GlobalTensor<InputT> selfGm_;
    GlobalTensor<int32_t> indexGm_;
    GlobalTensor<InputT> sourceGm_;
    GlobalTensor<InputT> outputGm_;
    GlobalTensor<uint8_t> selfGmBytes_;
    GlobalTensor<uint8_t> outGmBytes_;

    TBuf<TPosition::VECIN> copyInTBuf_;
    TBuf<TPosition::VECOUT> copyOutTBuf_;
    TBuf<TPosition::VECIN> indexTBuf_;
    TBuf<TPosition::VECIN> srcTBuf_;
    TBuf<TPosition::VECIN> outInTBuf_;
    TBuf<TPosition::VECOUT> outOutTBuf_;
    TBuf<TPosition::VECCALC> srcCompTBuf_;
    TBuf<TPosition::VECCALC> outCompTBuf_;
    LocalTensor<int32_t> indexLocal_;

    uint32_t beforeDimSize_;
    uint32_t dimLen_;
    uint32_t afterDimSize_;
    uint32_t indexLen_;
    uint32_t dtypeSize_;
    uint32_t usedCoreNum_;
    uint32_t mode_;
    uint32_t scatterTileLen_;
};

extern "C" __global__ __aicore__ void index_add(GM_ADDR self, GM_ADDR index, GM_ADDR source,
                                                GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(t, tiling);
    constexpr uint32_t DTYPE_FLOAT = 0;
    constexpr uint32_t DTYPE_BF16 = 1;
    constexpr uint32_t DTYPE_HALF = 2;
    constexpr uint32_t DTYPE_INT32 = 3;
    constexpr uint32_t DTYPE_INT8 = 4;

    if (t.dtype == DTYPE_FLOAT) {
        KernelIndexAdd<float, float> op;
        op.Init(self, index, source, output, t);
        op.Process();
    } else if (t.dtype == DTYPE_BF16) {
        KernelIndexAdd<bfloat16_t, float> op;
        op.Init(self, index, source, output, t);
        op.Process();
    } else if (t.dtype == DTYPE_HALF) {
        KernelIndexAdd<half, half> op;
        op.Init(self, index, source, output, t);
        op.Process();
    } else if (t.dtype == DTYPE_INT32) {
        KernelIndexAdd<int32_t, int32_t> op;
        op.Init(self, index, source, output, t);
        op.Process();
    } else if (t.dtype == DTYPE_INT8) {
        KernelIndexAdd<int8_t, half> op;
        op.Init(self, index, source, output, t);
        op.Process();
    }
}
