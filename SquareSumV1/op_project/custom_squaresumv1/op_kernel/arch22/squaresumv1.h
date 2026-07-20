/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * \file squaresumv1.h
 * \brief SquareSumV1 kernel class (arch22 / Ascend910B)
 *
 * Multi-TilingKey support:
 *   Key=0 AR_FULLLOAD:  tail-axis reduce, entire row fits in UB
 *   Key=1 AR_COLSPLIT:  tail-axis reduce, column chunk + fp32 accumulator
 *   Key=2 ARA_FULLLOAD: non-tail-axis reduce, Pattern::Reduce::RA full load
 *   Key=3 ARA_ROWSPLIT: non-tail-axis reduce, R-chunk + cross-chunk accumulation
 *   Key=4 MULTI_AXIS:   non-contiguous multi-axis, layer-by-layer reduce
 *
 * Data flow (fp16/bf16 input):
 *   DataCopyPad -> Cast(half->float) -> Mul(x,x) -> ReduceSum -> Cast(float->half) -> DataCopyPad
 * fp32 input skips all Cast operations.
 *
 * MULTI_AXIS (Key=4) data flow:
 *   Layer 0: inputGM -> UB -> square+reduce -> workspaceGM (float32)
 *   Layer k: workspaceGM -> UB -> reduce -> workspaceGM or resultGM (float32 or T)
 *   Last layer: -> Cast(float->T) -> resultGM
 */

#ifndef SQUARESUMV1_H
#define SQUARESUMV1_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "squaresumv1_tiling_data.h"
#include "squaresumv1_tiling_key.h"

namespace NsSquareSumV1 {

using namespace AscendC;

template <typename T>
class SquareSumV1 {
    static constexpr int32_t BUFFER_NUM = 2;
    using ComputeT = float;

public:
    __aicore__ inline SquareSumV1() {};
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR result, GM_ADDR workspace, const SquareSumV1TilingData* tilingData);
    __aicore__ inline void Process();

private:
    // AR_FULLLOAD (Key=0) - uses double-buffer queues
    __aicore__ inline void ProcessArFullLoad();
    __aicore__ inline void ArFullLoadCopyIn(int64_t rowIdx);
    __aicore__ inline void ArFullLoadCompute(int64_t rowIdx);
    __aicore__ inline void ArFullLoadCopyOut(int64_t rowIdx);

    // AR_COLSPLIT (Key=1)
    __aicore__ inline void ProcessArColSplit();

    // ARA_FULLLOAD (Key=2)
    __aicore__ inline void ProcessAraFullLoad();

    // ARA_ROWSPLIT (Key=3)
    __aicore__ inline void ProcessAraRowSplit();

    // MULTI_AXIS (Key=4)
    __aicore__ inline void ProcessMultiAxis();
    __aicore__ inline void ProcessMultiAxisLayer(int32_t layerIdx);

private:
    TPipe pipe;

    // === Buffers for AR_FULLLOAD (Key=0) ===
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;     // double-buffer input
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;    // double-buffer output
    TBuf<TPosition::VECCALC> computeBuf;                // fp32 work area for Cast/Mul
    TBuf<TPosition::VECCALC> tmpBuf;                    // ReduceSum work area

    // === Buffers for AR_COLSPLIT / ARA modes (Key=1,2,3) ===
    TBuf<TPosition::VECCALC> inQueueXSingle;            // single-buffer input for chunk modes
    TBuf<TPosition::VECCALC> outQueueYSingle;           // single-buffer output for chunk modes
    TBuf<TPosition::VECCALC> accBuf;                    // fp32 accumulator

    // === Buffers for MULTI_AXIS (Key=4) ===
    TBuf<TPosition::VECCALC> multiInBuf;               // input buffer for layer read
    TBuf<TPosition::VECCALC> multiComputeBuf;           // compute buffer (Cast/Mul)
    TBuf<TPosition::VECCALC> multiOutBuf;               // output buffer for layer result
    TBuf<TPosition::VECCALC> multiAccBuf;               // accumulator for cross-chunk
    TBuf<TPosition::VECCALC> multiTmpBuf;               // tmpBuf for ReduceSum

    // === GM tensors ===
    GlobalTensor<T> inputGM;
    GlobalTensor<T> resultGM;
    GlobalTensor<float> workspaceGM;

    // === Parameters ===
    uint32_t tilingMode_ = 0;
    int64_t totalRows_ = 0;
    int64_t rowsPerCore_ = 0;
    int64_t rLength_ = 0;
    int64_t rLengthAlign_ = 0;
    int64_t myRowOffset_ = 0;
    int64_t myRows_ = 0;

    // AR_COLSPLIT params (Key=1)
    int64_t chunkCols_ = 0;
    int64_t numChunks_ = 0;

    // ARA params (Key=2,3)
    int64_t a0Length_ = 0;
    int64_t a0LengthAlign_ = 0;
    int64_t tileA0Len_ = 0;
    int64_t tileA0Align_ = 0;
    int64_t numA0Tiles_ = 0;
    int64_t rChunkSize_ = 0;
    int64_t numRChunks_ = 0;

    // MULTI_AXIS params (Key=4)
    int32_t numLayers_ = 0;
    const SquareSumV1TilingData* tilingData_ = nullptr;

    uint32_t isAlign32B_ = 0;

    static constexpr bool isFloatInput = std::is_same_v<T, float>;
};

// ============================================================
// Init
// ============================================================

template <typename T>
__aicore__ inline void SquareSumV1<T>::Init(GM_ADDR input, GM_ADDR result, GM_ADDR workspace, const SquareSumV1TilingData* tilingData)
{
    tilingMode_ = tilingData->tilingMode;
    totalRows_ = tilingData->totalRows;
    rowsPerCore_ = tilingData->rowsPerCore;
    rLength_ = tilingData->rLength;
    rLengthAlign_ = tilingData->rLengthAlign;
    chunkCols_ = tilingData->chunkCols;
    numChunks_ = tilingData->numChunks;
    a0Length_ = tilingData->a0Length;
    a0LengthAlign_ = tilingData->a0LengthAlign;
    tileA0Len_ = tilingData->tileA0Len;
    tileA0Align_ = tilingData->tileA0Align;
    numA0Tiles_ = tilingData->numA0Tiles;
    rChunkSize_ = tilingData->rChunkSize;
    numRChunks_ = tilingData->numRChunks;
    isAlign32B_ = tilingData->isAlign32B;

    numLayers_ = tilingData->numLayers;
    tilingData_ = tilingData;

    int64_t blockIdx = GetBlockIdx();
    myRowOffset_ = blockIdx * rowsPerCore_;
    if (blockIdx == static_cast<int64_t>(tilingData->usedCoreNum) - 1) {
        myRows_ = totalRows_ - myRowOffset_;
    } else {
        myRows_ = rowsPerCore_;
    }
    if (myRows_ < 0) myRows_ = 0;

    inputGM.SetGlobalBuffer((__gm__ T*)input);
    resultGM.SetGlobalBuffer((__gm__ T*)result);

    if (tilingMode_ == 0) {
        // AR_FULLLOAD: double-buffer queue
        pipe.InitBuffer(inQueueX, BUFFER_NUM, rLengthAlign_ * sizeof(T));
        if constexpr (!isFloatInput) {
            pipe.InitBuffer(computeBuf, rLengthAlign_ * sizeof(float));
        }
        {
            uint32_t typeSize = sizeof(float);
            uint32_t epr = 256 / typeSize;
            uint32_t epb = 32 / typeSize;
            uint32_t firstMaxRep = (static_cast<uint32_t>(rLengthAlign_) + epr - 1) / epr;
            if (firstMaxRep == 0) firstMaxRep = 1;
            uint32_t iter1Out = firstMaxRep;
            uint32_t finalNeed = ((iter1Out + epb - 1) / epb) * epb;
            if (finalNeed < epb) finalNeed = epb;
            pipe.InitBuffer(tmpBuf, finalNeed * typeSize);
        }
        pipe.InitBuffer(outQueueY, BUFFER_NUM, 32);
    } else if (tilingMode_ == 1) {
        // AR_COLSPLIT: chunk-based, single buffer
        pipe.InitBuffer(inQueueXSingle, chunkCols_ * sizeof(T));
        if constexpr (!isFloatInput) {
            pipe.InitBuffer(computeBuf, chunkCols_ * sizeof(float));
        }
        pipe.InitBuffer(accBuf, 32);
        {
            uint32_t typeSize = sizeof(float);
            uint32_t epr = 256 / typeSize;
            uint32_t epb = 32 / typeSize;
            uint32_t firstMaxRep = (static_cast<uint32_t>(chunkCols_) + epr - 1) / epr;
            if (firstMaxRep == 0) firstMaxRep = 1;
            uint32_t iter1Out = firstMaxRep;
            uint32_t finalNeed = ((iter1Out + epb - 1) / epb) * epb;
            if (finalNeed < epb) finalNeed = epb;
            pipe.InitBuffer(tmpBuf, finalNeed * typeSize);
        }
        pipe.InitBuffer(outQueueYSingle, 32);
    } else if (tilingMode_ == 2 || tilingMode_ == 3) {
        // ARA mode: [R rows, alignedCols] 2D block
        int64_t rRows = (tilingMode_ == 2) ? rLength_ : rChunkSize_;
        int64_t totalCols = tileA0Align_;

        pipe.InitBuffer(inQueueXSingle, rRows * totalCols * sizeof(T));
        if constexpr (!isFloatInput) {
            pipe.InitBuffer(computeBuf, rRows * totalCols * sizeof(float));
        }
        pipe.InitBuffer(accBuf, totalCols * sizeof(float));
        pipe.InitBuffer(outQueueYSingle, totalCols * sizeof(T));

        {
            uint32_t tmpBufBytes = static_cast<uint32_t>(totalCols * sizeof(float));
            if (tmpBufBytes < 32) tmpBufBytes = 32;
            pipe.InitBuffer(tmpBuf, tmpBufBytes);
        }
    } else if (tilingMode_ == 4) {
        // MULTI_AXIS: allocate workspace GM and UB buffers
        workspaceGM.SetGlobalBuffer((__gm__ float*)workspace);

        // Compute maximum buffer sizes needed across all layers.
        // New approach: all layers use element-wise accumulate (no Pattern::Reduce::RA).
        //   For ARA (non-tail reduce): per-row read -> Cast/Cast(skip) -> Mul(first only) -> Add accumulator
        //   For AR (tail reduce): read row -> Cast/Mul -> ReduceSum scalar
        // So we need:
        //   - inputBuf: max(rLen) elements of T  (for reading one r-row)
        //   - computeBuf: max(rLen) elements of float (for Cast/Mul)
        //   - accBuf: max(tileA0Align) elements of float (ARA accumulator)
        //   - outBuf: max(8, tileA0Align) elements of float (ReduceSum scalar or temp)
        //   - tmpBuf: small, for ReduceSum scalar helper

        int64_t maxRLen = 1;
        int64_t maxA0Align = 8; // minimum 8 fp32 elements (32 bytes)

        for (int32_t li = 0; li < numLayers_; li++) {
            int64_t rLen = tilingData->layerRLength[li];
            int64_t tileA0Align = tilingData->layerTileA0Align[li];
            int64_t a0Len = tilingData->layerA0Length[li];
            int64_t isTail = tilingData->layerIsTailReduce[li];

            if (rLen > maxRLen) maxRLen = rLen;

            // For tail reduce layers, a0Align not needed (scalar output)
            if (!isTail && a0Len > 0) {
                int64_t a0Align = (a0Len + 7) / 8 * 8; // align to 8 for fp32
                if (a0Align > maxA0Align) maxA0Align = a0Align;
            }
        }

        // inputBuf: needs max(maxRLen * sizeof(T), maxA0Align * sizeof(T)) for reuse
        int64_t maxInputElements = maxRLen > maxA0Align ? maxRLen : maxA0Align;
        uint32_t inputBufBytes = static_cast<uint32_t>(maxInputElements * sizeof(T));
        if (inputBufBytes < 32) inputBufBytes = 32;
        pipe.InitBuffer(multiInBuf, inputBufBytes);

        // computeBuf: same size in float
        uint32_t computeBufBytes = static_cast<uint32_t>(maxInputElements * sizeof(float));
        if (computeBufBytes < 32) computeBufBytes = 32;
        pipe.InitBuffer(multiComputeBuf, computeBufBytes);

        // accBuf: maxA0Align fp32 elements (for ARA accumulate)
        uint32_t accBufBytes = static_cast<uint32_t>(maxA0Align * sizeof(float));
        if (accBufBytes < 32) accBufBytes = 32;
        pipe.InitBuffer(multiAccBuf, accBufBytes);

        // outBuf: same as accBuf (also used for scalar ReduceSum result)
        pipe.InitBuffer(multiOutBuf, accBufBytes);

        // tmpBuf: for ReduceSum helper (tail reduce layers). Only need 32 bytes.
        pipe.InitBuffer(multiTmpBuf, 4096);
    }
}

// ============================================================
// Process - dispatch to mode handler
// ============================================================

template <typename T>
__aicore__ inline void SquareSumV1<T>::Process()
{
    if (myRows_ == 0) return;

    switch (tilingMode_) {
        case 0: ProcessArFullLoad(); break;
        case 1: ProcessArColSplit(); break;
        case 2: ProcessAraFullLoad(); break;
        case 3: ProcessAraRowSplit(); break;
        case 4: ProcessMultiAxis(); break;
        default: ProcessArFullLoad(); break;
    }
}

// ============================================================
// AR_FULLLOAD (Key=0)
// ============================================================

template <typename T>
__aicore__ inline void SquareSumV1<T>::ArFullLoadCopyIn(int64_t rowIdx)
{
    LocalTensor<T> xLocal = inQueueX.template AllocTensor<T>();

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = rLength_ * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(xLocal, inputGM[rowIdx * rLength_], copyParams, padParams);

    inQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void SquareSumV1<T>::ArFullLoadCompute(int64_t rowIdx)
{
    LocalTensor<T> xLocal = inQueueX.template DeQue<T>();
    LocalTensor<T> yLocal = outQueueY.template AllocTensor<T>();
    LocalTensor<float> tmpLocal = tmpBuf.Get<float>();

    if constexpr (isFloatInput) {
        Mul(xLocal, xLocal, xLocal, rLength_);
        ReduceSum<float>(yLocal.template ReinterpretCast<float>(), xLocal, tmpLocal,
                         static_cast<int32_t>(rLength_));
    } else {
        LocalTensor<float> xFp32 = computeBuf.Get<float>();
        Cast(xFp32, xLocal, RoundMode::CAST_NONE, rLength_);
        Mul(xFp32, xFp32, xFp32, rLength_);
        ReduceSum<float>(yLocal.template ReinterpretCast<float>(), xFp32, tmpLocal,
                         static_cast<int32_t>(rLength_));
        PipeBarrier<PIPE_V>();
        Cast(yLocal, yLocal.template ReinterpretCast<float>(), RoundMode::CAST_NONE, 8);
    }

    outQueueY.EnQue(yLocal);
    inQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void SquareSumV1<T>::ArFullLoadCopyOut(int64_t rowIdx)
{
    LocalTensor<T> yLocal = outQueueY.template DeQue<T>();

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    DataCopyPad(resultGM[rowIdx], yLocal, copyParams);
    outQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void SquareSumV1<T>::ProcessArFullLoad()
{
    for (int64_t i = 0; i < myRows_; i++) {
        int64_t globalRowIdx = myRowOffset_ + i;
        ArFullLoadCopyIn(globalRowIdx);
        ArFullLoadCompute(globalRowIdx);
        ArFullLoadCopyOut(globalRowIdx);
    }
}

// ============================================================
// AR_COLSPLIT (Key=1) - chunk-based tail reduce
// ============================================================

template <typename T>
__aicore__ inline void SquareSumV1<T>::ProcessArColSplit()
{
    LocalTensor<float> tmpLocal = tmpBuf.Get<float>();

    for (int64_t i = 0; i < myRows_; i++) {
        int64_t globalRowIdx = myRowOffset_ + i;

        float accVal = 0.0f;

        for (int64_t chunkIdx = 0; chunkIdx < numChunks_; chunkIdx++) {
            int64_t chunkStart = chunkIdx * chunkCols_;
            int64_t chunkSize = chunkCols_;
            if (chunkStart + chunkSize > rLength_) {
                chunkSize = rLength_ - chunkStart;
            }
            if (chunkSize <= 0) break;

            LocalTensor<T> xLocal = inQueueXSingle.Get<T>();

            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = chunkSize * sizeof(T);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.rsv = 0;

            DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
            DataCopyPad(xLocal, inputGM[globalRowIdx * rLength_ + chunkStart], copyParams, padParams);
            PipeBarrier<PIPE_V>();

            LocalTensor<float> reduceDst = outQueueYSingle.Get<float>();

            if constexpr (isFloatInput) {
                Mul(xLocal, xLocal, xLocal, chunkSize);
                PipeBarrier<PIPE_V>();
                ReduceSum<float>(reduceDst, xLocal, tmpLocal, static_cast<int32_t>(chunkSize));
            } else {
                LocalTensor<float> xFp32 = computeBuf.Get<float>();
                Cast(xFp32, xLocal, RoundMode::CAST_NONE, chunkSize);
                PipeBarrier<PIPE_V>();
                Mul(xFp32, xFp32, xFp32, chunkSize);
                PipeBarrier<PIPE_V>();
                ReduceSum<float>(reduceDst, xFp32, tmpLocal, static_cast<int32_t>(chunkSize));
            }

            float partial = reduceDst.GetValue(0);
            accVal += partial;
        }

        LocalTensor<T> yLocal = outQueueYSingle.Get<T>();
        LocalTensor<float> yFp32 = outQueueYSingle.Get<float>();
        yFp32.SetValue(0, accVal);
        PipeBarrier<PIPE_V>();

        if constexpr (!isFloatInput) {
            Cast(yLocal, yFp32, RoundMode::CAST_NONE, 8);
            PipeBarrier<PIPE_V>();
        }

        DataCopyExtParams copyParamsOut;
        copyParamsOut.blockCount = 1;
        copyParamsOut.blockLen = sizeof(T);
        copyParamsOut.srcStride = 0;
        copyParamsOut.dstStride = 0;
        copyParamsOut.rsv = 0;
        DataCopyPad(resultGM[globalRowIdx], yLocal, copyParamsOut);
    }
}

// ============================================================
// ARA_FULLLOAD (Key=2) - Pattern::Reduce::RA on non-tail axis
// ============================================================

template <typename T>
__aicore__ inline void SquareSumV1<T>::ProcessAraFullLoad()
{
    LocalTensor<uint8_t> tmpLocal = tmpBuf.Get<uint8_t>();

    for (int64_t i = 0; i < myRows_; i++) {
        int64_t globalRowIdx = myRowOffset_ + i;

        for (int64_t a0TileIdx = 0; a0TileIdx < numA0Tiles_; a0TileIdx++) {
            int64_t a0Start = a0TileIdx * tileA0Len_;
            int64_t a0Len = tileA0Len_;
            if (a0Start + a0Len > a0Length_) {
                a0Len = a0Length_ - a0Start;
            }
            if (a0Len <= 0) break;

            int64_t alignedCols = tileA0Align_;
            int64_t gmOffset = globalRowIdx * rLength_ * a0Length_ + a0Start;

            LocalTensor<T> xLocal = inQueueXSingle.Get<T>();
            // 清零 xLocal：最后 a0 tile 的 a0Len 可能 < alignedCols，避免未 Copy 的 padding 垃圾参与 reduce
            Duplicate(xLocal, static_cast<T>(0), static_cast<int32_t>(rLength_ * alignedCols));
            PipeBarrier<PIPE_V>();

            DataCopyExtParams copyParams;
            copyParams.blockCount = static_cast<uint16_t>(rLength_);
            copyParams.blockLen = a0Len * sizeof(T);
            copyParams.srcStride = static_cast<uint16_t>((a0Length_ - a0Len) * sizeof(T) / 32);
            copyParams.dstStride = 0;
            copyParams.rsv = 0;
            DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
            DataCopyPad(xLocal, inputGM[gmOffset], copyParams, padParams);
            PipeBarrier<PIPE_ALL>();

            // 用 Add 循环沿 R 累加（替代 Pattern::Reduce::RA，避免小 R 的 NPU 行为差异）
            LocalTensor<float> accLocal = accBuf.Get<float>();
            Duplicate(accLocal, static_cast<float>(0), static_cast<int32_t>(alignedCols));
            PipeBarrier<PIPE_V>();

            if constexpr (isFloatInput) {
                Mul(xLocal, xLocal, xLocal, rLength_ * alignedCols);
                PipeBarrier<PIPE_V>();
                for (int64_t rIdx = 0; rIdx < rLength_; rIdx++) {
                    Add(accLocal, accLocal,
                        xLocal.template ReinterpretCast<float>()[static_cast<uint32_t>(rIdx * alignedCols)],
                        static_cast<int32_t>(alignedCols));
                    PipeBarrier<PIPE_V>();
                }
            } else {
                LocalTensor<float> xFp32 = computeBuf.Get<float>();
                uint32_t castCount = static_cast<uint32_t>(rLength_ * alignedCols);
                Cast(xFp32, xLocal, RoundMode::CAST_NONE, castCount);
                PipeBarrier<PIPE_V>();
                Mul(xFp32, xFp32, xFp32, rLength_ * alignedCols);
                PipeBarrier<PIPE_V>();
                for (int64_t rIdx = 0; rIdx < rLength_; rIdx++) {
                    Add(accLocal, accLocal, xFp32[static_cast<uint32_t>(rIdx * alignedCols)],
                        static_cast<int32_t>(alignedCols));
                    PipeBarrier<PIPE_V>();
                }
            }

            LocalTensor<T> yLocal = outQueueYSingle.Get<T>();
            if constexpr (!isFloatInput) {
                Cast(yLocal, accLocal, RoundMode::CAST_NONE, alignedCols);
                PipeBarrier<PIPE_V>();
            }

            int64_t resultGmOffset = globalRowIdx * a0Length_ + a0Start;
            DataCopyExtParams copyParamsOut;
            copyParamsOut.blockCount = 1;
            copyParamsOut.blockLen = a0Len * sizeof(T);
            copyParamsOut.srcStride = 0;
            copyParamsOut.dstStride = 0;
            copyParamsOut.rsv = 0;

            if constexpr (isFloatInput) {
                DataCopyPad(resultGM[resultGmOffset], accLocal.template ReinterpretCast<T>(), copyParamsOut);
            } else {
                DataCopyPad(resultGM[resultGmOffset], yLocal, copyParamsOut);
            }
        }
    }
}

// ============================================================
// ARA_ROWSPLIT (Key=3) - R-chunk + cross-chunk accumulation
// ============================================================

template <typename T>
__aicore__ inline void SquareSumV1<T>::ProcessAraRowSplit()
{
    LocalTensor<uint8_t> tmpLocal = tmpBuf.Get<uint8_t>();
    LocalTensor<float> accLocal = accBuf.Get<float>();

    for (int64_t i = 0; i < myRows_; i++) {
        int64_t globalRowIdx = myRowOffset_ + i;

        for (int64_t a0TileIdx = 0; a0TileIdx < numA0Tiles_; a0TileIdx++) {
            int64_t a0Start = a0TileIdx * tileA0Len_;
            int64_t a0Len = tileA0Len_;
            if (a0Start + a0Len > a0Length_) {
                a0Len = a0Length_ - a0Start;
            }
            if (a0Len <= 0) break;

            int64_t alignedCols = tileA0Align_;

            Duplicate(accLocal, static_cast<float>(0), alignedCols);
            PipeBarrier<PIPE_V>();

            for (int64_t rChunkIdx = 0; rChunkIdx < numRChunks_; rChunkIdx++) {
                int64_t rStart = rChunkIdx * rChunkSize_;
                int64_t rSize = rChunkSize_;
                if (rStart + rSize > rLength_) {
                    rSize = rLength_ - rStart;
                }
                if (rSize <= 0) break;

                int64_t gmOffset = globalRowIdx * rLength_ * a0Length_
                                   + rStart * a0Length_ + a0Start;

                LocalTensor<T> xLocal = inQueueXSingle.Get<T>();

                DataCopyExtParams copyParams;
                copyParams.blockCount = static_cast<uint16_t>(rSize);
                copyParams.blockLen = a0Len * sizeof(T);
                copyParams.srcStride = static_cast<uint16_t>((a0Length_ - a0Len) * sizeof(T) / 32);
                copyParams.dstStride = 0;
                copyParams.rsv = 0;

                DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
                DataCopyPad(xLocal, inputGM[gmOffset], copyParams, padParams);
                PipeBarrier<PIPE_V>();

                LocalTensor<float> chunkResult = outQueueYSingle.Get<float>();

                if constexpr (isFloatInput) {
                    Mul(xLocal, xLocal, xLocal, rSize * alignedCols);
                    PipeBarrier<PIPE_V>();
                    uint32_t srcShape[2] = {static_cast<uint32_t>(rSize),
                                            static_cast<uint32_t>(alignedCols)};
                    ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(
                        chunkResult, xLocal, tmpLocal, srcShape, true);
                } else {
                    LocalTensor<float> xFp32 = computeBuf.Get<float>();
                    uint32_t castCount = static_cast<uint32_t>(rSize * alignedCols);
                    uint32_t castAlign = 256 / sizeof(float);
                    castCount = ((castCount + castAlign - 1) / castAlign) * castAlign;
                    Cast(xFp32, xLocal, RoundMode::CAST_NONE, castCount);
                    PipeBarrier<PIPE_V>();
                    Mul(xFp32, xFp32, xFp32, rSize * alignedCols);
                    PipeBarrier<PIPE_V>();
                    uint32_t srcShape[2] = {static_cast<uint32_t>(rSize),
                                            static_cast<uint32_t>(alignedCols)};
                    ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(
                        chunkResult, xFp32, tmpLocal, srcShape, true);
                }

                PipeBarrier<PIPE_V>();
                Add(accLocal, accLocal, chunkResult, alignedCols);
                PipeBarrier<PIPE_V>();
            }

            LocalTensor<T> yLocal = outQueueYSingle.Get<T>();
            if constexpr (!isFloatInput) {
                Cast(yLocal, accLocal, RoundMode::CAST_NONE, alignedCols);
                PipeBarrier<PIPE_V>();
            }

            int64_t resultGmOffset = globalRowIdx * a0Length_ + a0Start;
            DataCopyExtParams copyParamsOut;
            copyParamsOut.blockCount = 1;
            copyParamsOut.blockLen = a0Len * sizeof(T);
            copyParamsOut.srcStride = 0;
            copyParamsOut.dstStride = 0;
            copyParamsOut.rsv = 0;

            if constexpr (isFloatInput) {
                DataCopyPad(resultGM[resultGmOffset], accLocal.template ReinterpretCast<T>(), copyParamsOut);
            } else {
                DataCopyPad(resultGM[resultGmOffset], yLocal, copyParamsOut);
            }
        }
    }
}

// ============================================================
// MULTI_AXIS (Key=4) - layer-by-layer reduce
//
// Workspace I/O convention: every scalar/vector element is stored as
// a full 32-byte (8 fp32) block. This ensures all DataCopyPad GM
// transfers are 32B-aligned and deterministic.
//
//   Tail reduce layer output: each scalar at offset rowIdx * 8
//   Non-tail reduce layer output: each element at offset (rowIdx * a0Len + ei) * 8
// ============================================================

template <typename T>
__aicore__ inline void SquareSumV1<T>::ProcessMultiAxisLayer(int32_t layerIdx)
{
    int64_t rLen = tilingData_->layerRLength[layerIdx];
    int64_t a0Len = tilingData_->layerA0Length[layerIdx];
    bool isTailReduce = tilingData_->layerIsTailReduce[layerIdx] != 0;
    bool isLastLayer = (layerIdx == numLayers_ - 1);
    bool isFirstLayer = (layerIdx == 0);

    LocalTensor<float> tmpLocal = multiTmpBuf.Get<float>();
    constexpr int64_t PAD = 8; // 8 fp32 elements = 32 bytes

    // Compute per-layer totalRows
    int64_t layerTotalRows;
    if (isTailReduce || a0Len == 0) {
        layerTotalRows = tilingData_->layerOutputElemCount[layerIdx];
    } else {
        layerTotalRows = tilingData_->layerOutputElemCount[layerIdx] / a0Len;
    }

    for (int64_t i = 0; i < layerTotalRows; i++) {
        int64_t rowIdx = i;

        if (isTailReduce || a0Len == 0) {
            // === Tail reduce: sum rLen scalars into 1 scalar ===
            float accVal = 0.0f;

            if (isFirstLayer) {
                // Read rLen elements from inputGM, square, ReduceSum
                LocalTensor<T> xLocal = multiInBuf.Get<T>();
                int64_t rLenAlign = (rLen + (32 / sizeof(T)) - 1) / (32 / sizeof(T)) * (32 / sizeof(T));
                if (rLenAlign < (32 / sizeof(T))) rLenAlign = (32 / sizeof(T));
                DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = rLenAlign * sizeof(T);
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                copyParams.rsv = 0;
                DataCopyPadExtParams<T> padParams{true, 0, 0, static_cast<T>(0)};
                DataCopyPad(xLocal, inputGM[rowIdx * rLen], copyParams, padParams);
                PipeBarrier<PIPE_ALL>();

                LocalTensor<float> reduceDst = multiOutBuf.Get<float>();
                if constexpr (isFloatInput) {
                    Mul(xLocal, xLocal, xLocal, rLen);
                    PipeBarrier<PIPE_V>();
                    ReduceSum<float>(reduceDst, xLocal, tmpLocal, static_cast<int32_t>(rLen));
                } else {
                    LocalTensor<float> xFp32 = multiComputeBuf.Get<float>();
                    Cast(xFp32, xLocal, RoundMode::CAST_NONE, rLen);
                    PipeBarrier<PIPE_V>();
                    Mul(xFp32, xFp32, xFp32, rLen);
                    PipeBarrier<PIPE_V>();
                    ReduceSum<float>(reduceDst, xFp32, tmpLocal, static_cast<int32_t>(rLen));
                }
                PipeBarrier<PIPE_V>();
                accVal = reduceDst.GetValue(0);
            } else {
                // Read rLen padded scalars from workspace, manual sum
                LocalTensor<float> xFp32 = multiComputeBuf.Get<float>();
                int64_t wsReadBase = tilingData_->layerWorkspaceOffset[layerIdx] + rowIdx * rLen * PAD;
                // Read each scalar individually as 32B block
                for (int64_t rIdx = 0; rIdx < rLen; rIdx++) {
                    DataCopyExtParams cp;
                    cp.blockCount = 1;
                    cp.blockLen = 32;
                    cp.srcStride = 0;
                    cp.dstStride = 0;
                    cp.rsv = 0;
                    DataCopyPadExtParams<float> pp{false, 0, 0, 0.0f};
                    DataCopyPad(xFp32, workspaceGM[wsReadBase + rIdx * PAD], cp, pp);
                    PipeBarrier<PIPE_ALL>();
                    accVal += xFp32.GetValue(0);
                }
            }

            // Write result
            if (isLastLayer) {
                LocalTensor<T> yLocal = multiInBuf.Get<T>();
                LocalTensor<float> yFp32 = multiOutBuf.Get<float>();
                yFp32.SetValue(0, accVal);
                PipeBarrier<PIPE_V>();
                if constexpr (!isFloatInput) {
                    Cast(yLocal, yFp32, RoundMode::CAST_RINT, 8);
                    PipeBarrier<PIPE_V>();
                }
                DataCopyExtParams copyParamsOut;
                copyParamsOut.blockCount = 1;
                copyParamsOut.blockLen = sizeof(T);
                copyParamsOut.srcStride = 0;
                copyParamsOut.dstStride = 0;
                copyParamsOut.rsv = 0;
                if constexpr (isFloatInput) {
                    DataCopyPad(resultGM[rowIdx], yFp32.template ReinterpretCast<T>(), copyParamsOut);
                } else {
                    DataCopyPad(resultGM[rowIdx], yLocal, copyParamsOut);
                }
                PipeBarrier<PIPE_ALL>();
            } else {
                // Write padded scalar to workspace
                int64_t wsOutOffset = tilingData_->layerWorkspaceOffset[layerIdx + 1] + rowIdx * PAD;
                LocalTensor<float> wsOut = multiOutBuf.Get<float>();
                wsOut.SetValue(0, accVal);
                PipeBarrier<PIPE_V>();
                DataCopyExtParams cp;
                cp.blockCount = 1;
                cp.blockLen = 32;
                cp.srcStride = 0;
                cp.dstStride = 0;
                cp.rsv = 0;
                DataCopyPad(workspaceGM[wsOutOffset], wsOut, cp);
                PipeBarrier<PIPE_ALL>();
            }
        } else {
            // === Non-tail reduce: reduce along rLen, keep a0Len elements ===
            int64_t a0Align = (a0Len + 7) / 8 * 8;
            LocalTensor<float> accLocal = multiAccBuf.Get<float>();
            Duplicate(accLocal, static_cast<float>(0), a0Align);
            PipeBarrier<PIPE_V>();

            for (int64_t rIdx = 0; rIdx < rLen; rIdx++) {
                LocalTensor<float> xFp32 = multiComputeBuf.Get<float>();

                if (isFirstLayer) {
                    // Read a0Len elements from inputGM
                    LocalTensor<T> xLocal = multiInBuf.Get<T>();
                    int64_t gmOffset = rowIdx * rLen * a0Len + rIdx * a0Len;
                    int64_t a0LenAlign = (a0Len + (32 / sizeof(T)) - 1) / (32 / sizeof(T)) * (32 / sizeof(T));
                    if (a0LenAlign < (32 / sizeof(T))) a0LenAlign = (32 / sizeof(T));
                    DataCopyExtParams cp;
                    cp.blockCount = 1;
                    cp.blockLen = a0LenAlign * sizeof(T);
                    cp.srcStride = 0;
                    cp.dstStride = 0;
                    cp.rsv = 0;
                    DataCopyPadExtParams<T> pp{true, 0, 0, static_cast<T>(0)};
                    DataCopyPad(xLocal, inputGM[gmOffset], cp, pp);
                    PipeBarrier<PIPE_ALL>();

                    if constexpr (isFloatInput) {
                        Mul(xLocal, xLocal, xLocal, a0Len);
                        PipeBarrier<PIPE_V>();
                        Add(accLocal, accLocal, xLocal.template ReinterpretCast<float>(), a0Len);
                        PipeBarrier<PIPE_V>();
                    } else {
                        Cast(xFp32, xLocal, RoundMode::CAST_NONE, a0Len);
                        PipeBarrier<PIPE_V>();
                        Mul(xFp32, xFp32, xFp32, a0Len);
                        PipeBarrier<PIPE_V>();
                        Add(accLocal, accLocal, xFp32, a0Len);
                        PipeBarrier<PIPE_V>();
                    }
                } else {
                    // Read a0Len padded elements from workspace
                    int64_t wsReadBase = tilingData_->layerWorkspaceOffset[layerIdx]
                                        + (rowIdx * rLen + rIdx) * a0Len * PAD;
                    LocalTensor<float> tmpRead = multiOutBuf.Get<float>();
                    for (int64_t ei = 0; ei < a0Len; ei++) {
                        DataCopyExtParams cp;
                        cp.blockCount = 1;
                        cp.blockLen = 32;
                        cp.srcStride = 0;
                        cp.dstStride = 0;
                        cp.rsv = 0;
                        DataCopyPadExtParams<float> pp{false, 0, 0, 0.0f};
                        DataCopyPad(tmpRead, workspaceGM[wsReadBase + ei * PAD], cp, pp);
                        PipeBarrier<PIPE_ALL>();
                        float val = tmpRead.GetValue(0);
                        accLocal.SetValue(ei, accLocal.GetValue(ei) + val);
                    }
                    PipeBarrier<PIPE_V>();
                }
            }

            // Write result
            if (isLastLayer) {
                LocalTensor<T> yLocal = multiInBuf.Get<T>();
                if constexpr (!isFloatInput) {
                    Cast(yLocal, accLocal, RoundMode::CAST_RINT, a0Len);
                    PipeBarrier<PIPE_V>();
                }
                int64_t resultGmOffset = rowIdx * a0Len;
                DataCopyExtParams copyParamsOut;
                copyParamsOut.blockCount = 1;
                copyParamsOut.blockLen = a0Len * sizeof(T);
                copyParamsOut.srcStride = 0;
                copyParamsOut.dstStride = 0;
                copyParamsOut.rsv = 0;
                if constexpr (isFloatInput) {
                    DataCopyPad(resultGM[resultGmOffset], accLocal.template ReinterpretCast<T>(), copyParamsOut);
                } else {
                    DataCopyPad(resultGM[resultGmOffset], yLocal, copyParamsOut);
                }
                PipeBarrier<PIPE_ALL>();
            } else {
                // Write padded a0Len elements to workspace
                int64_t wsOutBase = tilingData_->layerWorkspaceOffset[layerIdx + 1]
                                   + rowIdx * a0Len * PAD;
                LocalTensor<float> wsOut = multiOutBuf.Get<float>();
                for (int64_t ei = 0; ei < a0Len; ei++) {
                    wsOut.SetValue(0, accLocal.GetValue(ei));
                    PipeBarrier<PIPE_V>();
                    DataCopyExtParams cp;
                    cp.blockCount = 1;
                    cp.blockLen = 32;
                    cp.srcStride = 0;
                    cp.dstStride = 0;
                    cp.rsv = 0;
                    DataCopyPad(workspaceGM[wsOutBase + ei * PAD], wsOut, cp);
                    PipeBarrier<PIPE_ALL>();
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void SquareSumV1<T>::ProcessMultiAxis()
{
    for (int32_t li = 0; li < numLayers_; li++) {
        ProcessMultiAxisLayer(li);
        PipeBarrier<PIPE_ALL>();
    }
}

} // namespace NsSquareSumV1
#endif // SQUARESUMV1_H
