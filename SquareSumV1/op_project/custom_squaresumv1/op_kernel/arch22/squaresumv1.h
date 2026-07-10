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

    // Helper: reduce one row (AR_FULLLOAD sub-mode for MULTI_AXIS layers)
    __aicore__ inline void MultiAxisArFullLoadRow(
        int64_t rowIdx, int64_t rLen, int64_t rLenAlign,
        LocalTensor<float>& accScalar, LocalTensor<float>& tmpLocal,
        bool isFirstLayer);
    __aicore__ inline void MultiAxisArColSplitRow(
        int64_t rowIdx, int64_t rLen, int64_t chunkCols, int64_t numChunks,
        LocalTensor<float>& accScalar, LocalTensor<float>& tmpLocal,
        bool isFirstLayer);
    __aicore__ inline void MultiAxisAraFullLoad(
        int64_t layerIdx, int64_t rowIdx,
        LocalTensor<float>& tmpLocal);
    __aicore__ inline void MultiAxisAraRowSplit(
        int64_t layerIdx, int64_t rowIdx,
        LocalTensor<float>& tmpLocal);

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

        // Allocate UB buffers for MULTI_AXIS processing
        // We need flexible buffers that can handle the largest layer
        // For layer 0: may need to read T (half/float/bf16) and compute in float
        // For layer k>0: read float32, compute in float32

        // Find the maximum rLength across all layers
        int64_t maxRLen = 0;
        int64_t maxA0Align = 32 / sizeof(float); // minimum 8
        int64_t maxBufSize = 0; // max(rLength_align, rRows * alignedCols)
        for (int32_t li = 0; li < numLayers_; li++) {
            int64_t rLen = tilingData->layerRLength[li];
            int64_t a0Len = tilingData->layerA0Length[li];
            int64_t rChunkSz = tilingData->layerRChunkSize[li];
            int64_t tileA0Align = tilingData->layerTileA0Align[li];
            int64_t chunkCols = tilingData->layerChunkCols[li];
            int64_t mode = tilingData->layerMode[li];

            int64_t maxChunkR;
            if (mode == 0) {
                // AR_FULLLOAD
                int64_t rLenAlignInput = (rLen + (32 / sizeof(T)) - 1) / (32 / sizeof(T)) * (32 / sizeof(T));
                int64_t rLenAlignFp32 = (rLen + 7) / 8 * 8;
                maxChunkR = rLenAlignInput > rLenAlignFp32 ? rLenAlignInput : rLenAlignFp32;
            } else if (mode == 1) {
                // AR_COLSPLIT
                maxChunkR = chunkCols;
            } else if (mode == 2) {
                // ARA_FULLLOAD
                int64_t rRows = rLen;
                int64_t cols = tileA0Align;
                maxChunkR = rRows * cols;
                if (tileA0Align > maxA0Align) maxA0Align = tileA0Align;
            } else {
                // ARA_ROWSPLIT
                int64_t rRows = rChunkSz > 0 ? rChunkSz : 1;
                int64_t cols = tileA0Align;
                maxChunkR = rRows * cols;
                if (tileA0Align > maxA0Align) maxA0Align = tileA0Align;
            }
            if (maxChunkR > maxBufSize) maxBufSize = maxChunkR;
            if (rLen > maxRLen) maxRLen = rLen;
        }

        // For layer 0, we may need to read T elements and have a compute buffer
        // For layers > 0, we read float32 elements
        // Allocate for the worst case: max(maxBufSize elements of T, maxBufSize elements of float)
        // Plus compute buffer, accumulator, output buffer, tmpBuf

        uint32_t inputBufBytes = static_cast<uint32_t>(maxBufSize * sizeof(T));
        if (inputBufBytes < 32) inputBufBytes = 32;
        pipe.InitBuffer(multiInBuf, inputBufBytes);

        uint32_t computeBufBytes = static_cast<uint32_t>(maxBufSize * sizeof(float));
        if (computeBufBytes < 32) computeBufBytes = 32;
        pipe.InitBuffer(multiComputeBuf, computeBufBytes);

        // Output buffer: max aligned cols
        uint32_t outBufBytes = static_cast<uint32_t>(maxA0Align * sizeof(float));
        if (outBufBytes < 32) outBufBytes = 32;
        pipe.InitBuffer(multiOutBuf, outBufBytes);

        uint32_t accBufBytes = outBufBytes;
        pipe.InitBuffer(multiAccBuf, accBufBytes);

        // tmpBuf for ReduceSum
        uint32_t maxTmpBuf = 4096;
        // For AR_FULLLOAD sub-mode
        for (int32_t li = 0; li < numLayers_; li++) {
            int64_t rLen = tilingData->layerRLength[li];
            int64_t mode = tilingData->layerMode[li];
            int64_t chunkCols = tilingData->layerChunkCols[li];
            int64_t actualR;
            if (mode == 0) actualR = rLen;
            else if (mode == 1) actualR = chunkCols;
            else if (mode == 2) actualR = rLen;
            else actualR = tilingData->layerRChunkSize[li];
            if (actualR <= 0) actualR = 1;

            uint32_t epr = 256 / sizeof(float);
            uint32_t epb = 32 / sizeof(float);
            uint32_t firstMaxRep = (static_cast<uint32_t>(actualR) + epr - 1) / epr;
            if (firstMaxRep == 0) firstMaxRep = 1;
            uint32_t finalNeed = ((firstMaxRep + epb - 1) / epb) * epb;
            if (finalNeed < epb) finalNeed = epb;
            uint32_t need = finalNeed * sizeof(float);
            if (need > maxTmpBuf) maxTmpBuf = need;
        }
        pipe.InitBuffer(multiTmpBuf, maxTmpBuf);
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

            DataCopyExtParams copyParams;
            copyParams.blockCount = static_cast<uint16_t>(rLength_);
            copyParams.blockLen = a0Len * sizeof(T);
            copyParams.srcStride = static_cast<uint16_t>((a0Length_ - a0Len) * sizeof(T) / 32);
            copyParams.dstStride = 0;
            copyParams.rsv = 0;

            DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
            DataCopyPad(xLocal, inputGM[gmOffset], copyParams, padParams);
            PipeBarrier<PIPE_V>();

            LocalTensor<float> reduceDst = outQueueYSingle.Get<float>();

            if constexpr (isFloatInput) {
                Mul(xLocal, xLocal, xLocal, rLength_ * alignedCols);
                PipeBarrier<PIPE_V>();
                uint32_t srcShape[2] = {static_cast<uint32_t>(rLength_),
                                        static_cast<uint32_t>(alignedCols)};
                ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(
                    reduceDst, xLocal, tmpLocal, srcShape, true);
            } else {
                LocalTensor<float> xFp32 = computeBuf.Get<float>();
                uint32_t castCount = static_cast<uint32_t>(rLength_ * alignedCols);
                uint32_t castAlign = 256 / sizeof(float);
                castCount = ((castCount + castAlign - 1) / castAlign) * castAlign;
                Cast(xFp32, xLocal, RoundMode::CAST_NONE, castCount);
                PipeBarrier<PIPE_V>();
                Mul(xFp32, xFp32, xFp32, rLength_ * alignedCols);
                PipeBarrier<PIPE_V>();
                uint32_t srcShape[2] = {static_cast<uint32_t>(rLength_),
                                        static_cast<uint32_t>(alignedCols)};
                ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(
                    reduceDst, xFp32, tmpLocal, srcShape, true);
            }

            PipeBarrier<PIPE_V>();

            LocalTensor<T> yLocal = outQueueYSingle.Get<T>();
            if constexpr (!isFloatInput) {
                Cast(yLocal, reduceDst, RoundMode::CAST_NONE, alignedCols);
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
                DataCopyPad(resultGM[resultGmOffset], reduceDst.template ReinterpretCast<T>(), copyParamsOut);
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
// ============================================================

// Process one row for a tail-reduce layer in MULTI_AXIS mode
// Uses AR_FULLLOAD or AR_COLSPLIT sub-mode
template <typename T>
__aicore__ inline void SquareSumV1<T>::MultiAxisArFullLoadRow(
    int64_t rowIdx, int64_t rLen, int64_t rLenAlign,
    LocalTensor<float>& accScalar, LocalTensor<float>& tmpLocal,
    bool isFirstLayer)
{
    // Read rLen elements from source GM (input or workspace)
    LocalTensor<T> xLocal = multiInBuf.Get<T>();

    if (isFirstLayer) {
        // Read from inputGM in T format
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = rLen * sizeof(T);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(xLocal, inputGM[rowIdx * rLen], copyParams, padParams);
        PipeBarrier<PIPE_V>();

        // Cast to float and square
        LocalTensor<float> xFp32 = multiComputeBuf.Get<float>();
        if constexpr (isFloatInput) {
            Mul(xLocal, xLocal, xLocal, rLen);
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(accScalar, xLocal, tmpLocal, static_cast<int32_t>(rLen));
        } else {
            Cast(xFp32, xLocal, RoundMode::CAST_NONE, rLen);
            PipeBarrier<PIPE_V>();
            Mul(xFp32, xFp32, xFp32, rLen);
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(accScalar, xFp32, tmpLocal, static_cast<int32_t>(rLen));
        }
    } else {
        // Read from workspaceGM in float32 format
        LocalTensor<float> xFp32 = multiComputeBuf.Get<float>();
        // Copy float32 data from workspace
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = rLen * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        DataCopyPad(xFp32, workspaceGM[rowIdx * rLen], copyParams, padParams);
        PipeBarrier<PIPE_V>();

        // No square needed - just reduce
        ReduceSum<float>(accScalar, xFp32, tmpLocal, static_cast<int32_t>(rLen));
    }
}

template <typename T>
__aicore__ inline void SquareSumV1<T>::MultiAxisArColSplitRow(
    int64_t rowIdx, int64_t rLen, int64_t chunkCols, int64_t numChunks,
    LocalTensor<float>& accScalar, LocalTensor<float>& tmpLocal,
    bool isFirstLayer)
{
    float accVal = 0.0f;

    for (int64_t chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
        int64_t chunkStart = chunkIdx * chunkCols;
        int64_t chunkSize = chunkCols;
        if (chunkStart + chunkSize > rLen) {
            chunkSize = rLen - chunkStart;
        }
        if (chunkSize <= 0) break;

        if (isFirstLayer) {
            LocalTensor<T> xLocal = multiInBuf.Get<T>();
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = chunkSize * sizeof(T);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.rsv = 0;
            DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
            DataCopyPad(xLocal, inputGM[rowIdx * rLen + chunkStart], copyParams, padParams);
            PipeBarrier<PIPE_V>();

            LocalTensor<float> xFp32 = multiComputeBuf.Get<float>();
            LocalTensor<float> reduceDst = multiOutBuf.Get<float>();

            if constexpr (isFloatInput) {
                Mul(xLocal, xLocal, xLocal, chunkSize);
                PipeBarrier<PIPE_V>();
                ReduceSum<float>(reduceDst, xLocal, tmpLocal, static_cast<int32_t>(chunkSize));
            } else {
                Cast(xFp32, xLocal, RoundMode::CAST_NONE, chunkSize);
                PipeBarrier<PIPE_V>();
                Mul(xFp32, xFp32, xFp32, chunkSize);
                PipeBarrier<PIPE_V>();
                ReduceSum<float>(reduceDst, xFp32, tmpLocal, static_cast<int32_t>(chunkSize));
            }
            accVal += reduceDst.GetValue(0);
        } else {
            LocalTensor<float> xFp32 = multiComputeBuf.Get<float>();
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = chunkSize * sizeof(float);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            copyParams.rsv = 0;
            DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
            DataCopyPad(xFp32, workspaceGM[rowIdx * rLen + chunkStart], copyParams, padParams);
            PipeBarrier<PIPE_V>();

            LocalTensor<float> reduceDst = multiOutBuf.Get<float>();
            ReduceSum<float>(reduceDst, xFp32, tmpLocal, static_cast<int32_t>(chunkSize));
            accVal += reduceDst.GetValue(0);
        }
    }

    accScalar.SetValue(0, accVal);
}

template <typename T>
__aicore__ inline void SquareSumV1<T>::MultiAxisAraFullLoad(
    int64_t layerIdx, int64_t rowIdx,
    LocalTensor<float>& tmpLocal)
{
    int64_t rLen = tilingData_->layerRLength[layerIdx];
    int64_t a0Len = tilingData_->layerA0Length[layerIdx];
    int64_t tileA0Align = tilingData_->layerTileA0Align[layerIdx];
    int64_t numA0Tiles = tilingData_->layerNumA0Tiles[layerIdx];
    int64_t tileA0Len = tilingData_->layerTileA0Len[layerIdx];
    bool isFirstLayer = (layerIdx == 0);

    LocalTensor<uint8_t> tmpU8 = multiTmpBuf.Get<uint8_t>();
    LocalTensor<float> accLocal = multiAccBuf.Get<float>();

    for (int64_t a0TileIdx = 0; a0TileIdx < numA0Tiles; a0TileIdx++) {
        int64_t a0Start = a0TileIdx * tileA0Len;
        int64_t curA0Len = tileA0Len;
        if (a0Start + curA0Len > a0Len) {
            curA0Len = a0Len - a0Start;
        }
        if (curA0Len <= 0) break;

        int64_t alignedCols = tileA0Align;

        if (isFirstLayer) {
            // Read from inputGM in T format: [R, a0Len] block
            LocalTensor<T> xLocal = multiInBuf.Get<T>();
            int64_t gmOffset = rowIdx * rLen * a0Len + a0Start;

            DataCopyExtParams copyParams;
            copyParams.blockCount = static_cast<uint16_t>(rLen);
            copyParams.blockLen = curA0Len * sizeof(T);
            copyParams.srcStride = static_cast<uint16_t>((a0Len - curA0Len) * sizeof(T) / 32);
            copyParams.dstStride = 0;
            copyParams.rsv = 0;
            DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
            DataCopyPad(xLocal, inputGM[gmOffset], copyParams, padParams);
            PipeBarrier<PIPE_V>();

            // Square + reduce
            LocalTensor<float> xFp32 = multiComputeBuf.Get<float>();
            LocalTensor<float> reduceDst = multiOutBuf.Get<float>();

            if constexpr (isFloatInput) {
                Mul(xLocal, xLocal, xLocal, rLen * alignedCols);
                PipeBarrier<PIPE_V>();
                uint32_t srcShape[2] = {static_cast<uint32_t>(rLen),
                                        static_cast<uint32_t>(alignedCols)};
                ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(
                    reduceDst, xLocal, tmpU8, srcShape, true);
            } else {
                uint32_t castCount = static_cast<uint32_t>(rLen * alignedCols);
                uint32_t castAlign = 256 / sizeof(float);
                castCount = ((castCount + castAlign - 1) / castAlign) * castAlign;
                Cast(xFp32, xLocal, RoundMode::CAST_NONE, castCount);
                PipeBarrier<PIPE_V>();
                Mul(xFp32, xFp32, xFp32, rLen * alignedCols);
                PipeBarrier<PIPE_V>();
                uint32_t srcShape[2] = {static_cast<uint32_t>(rLen),
                                        static_cast<uint32_t>(alignedCols)};
                ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(
                    reduceDst, xFp32, tmpU8, srcShape, true);
            }
            PipeBarrier<PIPE_V>();

            // Write result to workspace (float32)
            int64_t wsOffset = rowIdx * a0Len + a0Start;
            DataCopyExtParams copyParamsOut;
            copyParamsOut.blockCount = 1;
            copyParamsOut.blockLen = curA0Len * sizeof(float);
            copyParamsOut.srcStride = 0;
            copyParamsOut.dstStride = 0;
            copyParamsOut.rsv = 0;
            DataCopyPad(workspaceGM[wsOffset], reduceDst, copyParamsOut);
        } else {
            // Read from workspaceGM in float32 format: [R, a0Len] block
            LocalTensor<float> xFp32 = multiComputeBuf.Get<float>();
            int64_t wsOffset = rowIdx * rLen * a0Len + a0Start;

            DataCopyExtParams copyParams;
            copyParams.blockCount = static_cast<uint16_t>(rLen);
            copyParams.blockLen = curA0Len * sizeof(float);
            copyParams.srcStride = static_cast<uint16_t>((a0Len - curA0Len) * sizeof(float) / 32);
            copyParams.dstStride = 0;
            copyParams.rsv = 0;
            DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
            DataCopyPad(xFp32, workspaceGM[wsOffset], copyParams, padParams);
            PipeBarrier<PIPE_V>();

            // No square - just reduce
            LocalTensor<float> reduceDst = multiOutBuf.Get<float>();
            uint32_t srcShape[2] = {static_cast<uint32_t>(rLen),
                                    static_cast<uint32_t>(alignedCols)};
            ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(
                reduceDst, xFp32, tmpU8, srcShape, true);
            PipeBarrier<PIPE_V>();

            // Write result to workspace (float32) or resultGM
            bool isLastLayer = (layerIdx == numLayers_ - 1);
            if (isLastLayer) {
                // Cast to T and write to resultGM
                LocalTensor<T> yLocal = multiInBuf.Get<T>();
                if constexpr (!isFloatInput) {
                    Cast(yLocal, reduceDst, RoundMode::CAST_NONE, alignedCols);
                    PipeBarrier<PIPE_V>();
                }
                int64_t resultGmOffset = rowIdx * a0Len + a0Start;
                DataCopyExtParams copyParamsOut;
                copyParamsOut.blockCount = 1;
                copyParamsOut.blockLen = curA0Len * sizeof(T);
                copyParamsOut.srcStride = 0;
                copyParamsOut.dstStride = 0;
                copyParamsOut.rsv = 0;
                if constexpr (isFloatInput) {
                    DataCopyPad(resultGM[resultGmOffset], reduceDst.template ReinterpretCast<T>(), copyParamsOut);
                } else {
                    DataCopyPad(resultGM[resultGmOffset], yLocal, copyParamsOut);
                }
            } else {
                // Write to workspace at next layer's offset
                // Use the same workspace region (ping-pong: alternate between two halves)
                int64_t wsOutOffset = tilingData_->layerWorkspaceOffset[layerIdx + 1]
                                     + rowIdx * a0Len + a0Start;
                DataCopyExtParams copyParamsOut;
                copyParamsOut.blockCount = 1;
                copyParamsOut.blockLen = curA0Len * sizeof(float);
                copyParamsOut.srcStride = 0;
                copyParamsOut.dstStride = 0;
                copyParamsOut.rsv = 0;
                DataCopyPad(workspaceGM[wsOutOffset], reduceDst, copyParamsOut);
            }
        }
    }
}

template <typename T>
__aicore__ inline void SquareSumV1<T>::MultiAxisAraRowSplit(
    int64_t layerIdx, int64_t rowIdx,
    LocalTensor<float>& tmpLocal)
{
    int64_t rLen = tilingData_->layerRLength[layerIdx];
    int64_t a0Len = tilingData_->layerA0Length[layerIdx];
    int64_t tileA0Align = tilingData_->layerTileA0Align[layerIdx];
    int64_t numA0Tiles = tilingData_->layerNumA0Tiles[layerIdx];
    int64_t tileA0Len = tilingData_->layerTileA0Len[layerIdx];
    int64_t rChunkSize = tilingData_->layerRChunkSize[layerIdx];
    int64_t numRChunks = tilingData_->layerNumRChunks[layerIdx];
    bool isFirstLayer = (layerIdx == 0);
    bool isLastLayer = (layerIdx == numLayers_ - 1);

    LocalTensor<uint8_t> tmpU8 = multiTmpBuf.Get<uint8_t>();
    LocalTensor<float> accLocal = multiAccBuf.Get<float>();

    for (int64_t a0TileIdx = 0; a0TileIdx < numA0Tiles; a0TileIdx++) {
        int64_t a0Start = a0TileIdx * tileA0Len;
        int64_t curA0Len = tileA0Len;
        if (a0Start + curA0Len > a0Len) {
            curA0Len = a0Len - a0Start;
        }
        if (curA0Len <= 0) break;

        int64_t alignedCols = tileA0Align;

        // Initialize accumulator
        Duplicate(accLocal, static_cast<float>(0), alignedCols);
        PipeBarrier<PIPE_V>();

        for (int64_t rChunkIdx = 0; rChunkIdx < numRChunks; rChunkIdx++) {
            int64_t rStart = rChunkIdx * rChunkSize;
            int64_t rSize = rChunkSize;
            if (rStart + rSize > rLen) {
                rSize = rLen - rStart;
            }
            if (rSize <= 0) break;

            if (isFirstLayer) {
                // Read from inputGM in T format: [rSize, curA0Len] block
                LocalTensor<T> xLocal = multiInBuf.Get<T>();
                int64_t gmOffset = rowIdx * rLen * a0Len + rStart * a0Len + a0Start;

                DataCopyExtParams copyParams;
                copyParams.blockCount = static_cast<uint16_t>(rSize);
                copyParams.blockLen = curA0Len * sizeof(T);
                copyParams.srcStride = static_cast<uint16_t>((a0Len - curA0Len) * sizeof(T) / 32);
                copyParams.dstStride = 0;
                copyParams.rsv = 0;
                DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
                DataCopyPad(xLocal, inputGM[gmOffset], copyParams, padParams);
                PipeBarrier<PIPE_V>();

                LocalTensor<float> xFp32 = multiComputeBuf.Get<float>();
                LocalTensor<float> chunkResult = multiOutBuf.Get<float>();

                if constexpr (isFloatInput) {
                    Mul(xLocal, xLocal, xLocal, rSize * alignedCols);
                    PipeBarrier<PIPE_V>();
                    uint32_t srcShape[2] = {static_cast<uint32_t>(rSize),
                                            static_cast<uint32_t>(alignedCols)};
                    ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(
                        chunkResult, xLocal, tmpU8, srcShape, true);
                } else {
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
                        chunkResult, xFp32, tmpU8, srcShape, true);
                }
                PipeBarrier<PIPE_V>();
                Add(accLocal, accLocal, chunkResult, alignedCols);
                PipeBarrier<PIPE_V>();
            } else {
                // Read from workspaceGM in float32 format: [rSize, curA0Len] block
                LocalTensor<float> xFp32 = multiComputeBuf.Get<float>();
                int64_t wsOffset = rowIdx * rLen * a0Len + rStart * a0Len + a0Start;

                DataCopyExtParams copyParams;
                copyParams.blockCount = static_cast<uint16_t>(rSize);
                copyParams.blockLen = curA0Len * sizeof(float);
                copyParams.srcStride = static_cast<uint16_t>((a0Len - curA0Len) * sizeof(float) / 32);
                copyParams.dstStride = 0;
                copyParams.rsv = 0;
                DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
                DataCopyPad(xFp32, workspaceGM[wsOffset], copyParams, padParams);
                PipeBarrier<PIPE_V>();

                LocalTensor<float> chunkResult = multiOutBuf.Get<float>();
                uint32_t srcShape[2] = {static_cast<uint32_t>(rSize),
                                        static_cast<uint32_t>(alignedCols)};
                ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(
                    chunkResult, xFp32, tmpU8, srcShape, true);
                PipeBarrier<PIPE_V>();
                Add(accLocal, accLocal, chunkResult, alignedCols);
                PipeBarrier<PIPE_V>();
            }
        }

        // Write result
        if (isLastLayer) {
            // Cast to T and write to resultGM
            LocalTensor<T> yLocal = multiInBuf.Get<T>();
            if constexpr (!isFloatInput) {
                Cast(yLocal, accLocal, RoundMode::CAST_NONE, alignedCols);
                PipeBarrier<PIPE_V>();
            }
            int64_t resultGmOffset = rowIdx * a0Len + a0Start;
            DataCopyExtParams copyParamsOut;
            copyParamsOut.blockCount = 1;
            copyParamsOut.blockLen = curA0Len * sizeof(T);
            copyParamsOut.srcStride = 0;
            copyParamsOut.dstStride = 0;
            copyParamsOut.rsv = 0;
            if constexpr (isFloatInput) {
                DataCopyPad(resultGM[resultGmOffset], accLocal.template ReinterpretCast<T>(), copyParamsOut);
            } else {
                DataCopyPad(resultGM[resultGmOffset], yLocal, copyParamsOut);
            }
        } else {
            // Write to workspace
            int64_t wsOutOffset = tilingData_->layerWorkspaceOffset[layerIdx + 1]
                                 + rowIdx * a0Len + a0Start;
            DataCopyExtParams copyParamsOut;
            copyParamsOut.blockCount = 1;
            copyParamsOut.blockLen = curA0Len * sizeof(float);
            copyParamsOut.srcStride = 0;
            copyParamsOut.dstStride = 0;
            copyParamsOut.rsv = 0;
            DataCopyPad(workspaceGM[wsOutOffset], accLocal, copyParamsOut);
        }
    }
}

template <typename T>
__aicore__ inline void SquareSumV1<T>::ProcessMultiAxisLayer(int32_t layerIdx)
{
    int64_t subMode = tilingData_->layerMode[layerIdx];
    int64_t rLen = tilingData_->layerRLength[layerIdx];
    int64_t a0Len = tilingData_->layerA0Length[layerIdx];
    bool isTailReduce = tilingData_->layerIsTailReduce[layerIdx] != 0;
    bool isLastLayer = (layerIdx == numLayers_ - 1);
    bool isFirstLayer = (layerIdx == 0);

    LocalTensor<float> tmpLocal = multiTmpBuf.Get<float>();
    LocalTensor<float> accScalar = multiOutBuf.Get<float>();

    // Compute totalRows for this layer
    // For MULTI_AXIS, rows are the product of dims before the reduce axis in the current shape
    // We use the same row partitioning as layer 0 (myRows_, myRowOffset_)
    // For subsequent layers, each row produces exactly 1 output row (tail) or a0Len outputs

    for (int64_t i = 0; i < myRows_; i++) {
        int64_t rowIdx = myRowOffset_ + i;

        if (isTailReduce || a0Len == 0) {
            // Tail reduce sub-layer (AR_FULLLOAD or AR_COLSPLIT)
            if (subMode == 0) {
                // AR_FULLLOAD
                int64_t rLenAlign = (rLen + 7) / 8 * 8; // align to 8 for fp32
                MultiAxisArFullLoadRow(rowIdx, rLen, rLenAlign, accScalar, tmpLocal, isFirstLayer);
            } else {
                // AR_COLSPLIT
                int64_t chunkCols = tilingData_->layerChunkCols[layerIdx];
                int64_t numChunks = tilingData_->layerNumChunks[layerIdx];
                MultiAxisArColSplitRow(rowIdx, rLen, chunkCols, numChunks, accScalar, tmpLocal, isFirstLayer);
            }

            PipeBarrier<PIPE_V>();

            // Write result
            if (isLastLayer) {
                // Cast to T and write to resultGM
                LocalTensor<T> yLocal = multiInBuf.Get<T>();
                if constexpr (!isFloatInput) {
                    Cast(yLocal, accScalar, RoundMode::CAST_NONE, 8);
                    PipeBarrier<PIPE_V>();
                }
                if constexpr (isFloatInput) {
                    DataCopyExtParams copyParamsOut;
                    copyParamsOut.blockCount = 1;
                    copyParamsOut.blockLen = sizeof(T);
                    copyParamsOut.srcStride = 0;
                    copyParamsOut.dstStride = 0;
                    copyParamsOut.rsv = 0;
                    DataCopyPad(resultGM[rowIdx], accScalar.template ReinterpretCast<T>(), copyParamsOut);
                } else {
                    DataCopyExtParams copyParamsOut;
                    copyParamsOut.blockCount = 1;
                    copyParamsOut.blockLen = sizeof(T);
                    copyParamsOut.srcStride = 0;
                    copyParamsOut.dstStride = 0;
                    copyParamsOut.rsv = 0;
                    DataCopyPad(resultGM[rowIdx], yLocal, copyParamsOut);
                }
            } else {
                // Write float32 scalar to workspace
                int64_t wsOutOffset = tilingData_->layerWorkspaceOffset[layerIdx + 1] + rowIdx;
                DataCopyExtParams copyParamsOut;
                copyParamsOut.blockCount = 1;
                copyParamsOut.blockLen = sizeof(float);
                copyParamsOut.srcStride = 0;
                copyParamsOut.dstStride = 0;
                copyParamsOut.rsv = 0;
                DataCopyPad(workspaceGM[wsOutOffset], accScalar, copyParamsOut);
            }
        } else {
            // Non-tail reduce sub-layer (ARA_FULLLOAD or ARA_ROWSPLIT)
            if (subMode == 2) {
                MultiAxisAraFullLoad(layerIdx, rowIdx, tmpLocal);
            } else {
                MultiAxisAraRowSplit(layerIdx, rowIdx, tmpLocal);
            }
        }
    }
}

template <typename T>
__aicore__ inline void SquareSumV1<T>::ProcessMultiAxis()
{
    // Process layers from innermost (last in sorted axis) to outermost (first)
    // tilingData layers are ordered: layer[0] = innermost reduce axis, layer[N-1] = outermost
    // (because host sorts ascending and reverses for processing)
    for (int32_t li = 0; li < numLayers_; li++) {
        ProcessMultiAxisLayer(li);
    }
}

} // namespace NsSquareSumV1
#endif // SQUARESUMV1_H
