/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * \file square_sum_v1.h
 * \brief SquareSumV1 kernel class (arch22 / Ascend910B)
 *
 * Multi-TilingKey support:
 *   Key=0 AR_FULLLOAD:  tail-axis reduce, entire row fits in UB
 *   Key=1 AR_COLSPLIT:  tail-axis reduce, column chunk + fp32 accumulator
 *   Key=2 ARA_FULLLOAD: non-tail-axis reduce, Pattern::Reduce::RA full load
 *   Key=3 ARA_ROWSPLIT: non-tail-axis reduce, R-chunk + cross-chunk accumulation
 *   Key=4 MULTI_AXIS_COMPACT: non-contiguous multi-axis, compact fp32 ping-pong stages
 *   Key=5 REDUCE_ALL_COOPERATIVE: large all-reduce, per-core fp32 partials + merge
 *
 * Data flow (fp16/bf16 input):
 *   DataCopyPad -> Cast(half->float) -> Mul(x,x) -> ReduceSum -> Cast(float->half) -> DataCopyPad
 * fp32 input skips all Cast operations.
 *
 * MULTI_AXIS_COMPACT (mode=4) data flow:
 *   Layer 0: inputGM -> UB -> square+reduce -> compact fp32 stage
 *   Layer k: compact fp32 stage -> UB -> reduce -> alternate compact stage
 *   Last layer: -> Cast(float->T) -> resultGM
 */

#ifndef SQUARE_SUM_V1_H
#define SQUARE_SUM_V1_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "square_sum_v1_tiling_data.h"
#include "square_sum_v1_tiling_key.h"

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
    __aicore__ inline void ProcessReduceAllCooperative();

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
    TBuf<TPosition::VECCALC> reduceBuf;                 // fp32 Pattern::Reduce destination

    // === Buffers for MULTI_AXIS (Key=4) ===
    TBuf<TPosition::VECCALC> multiInBuf;               // input buffer for layer read
    TBuf<TPosition::VECCALC> multiComputeBuf;           // compute buffer (Cast/Mul)
    TBuf<TPosition::VECCALC> multiOutBuf;               // output buffer for layer result
    TBuf<TPosition::VECCALC> multiAccBuf;               // accumulator for cross-chunk
    TBuf<TPosition::VECCALC> multiTmpBuf;               // tmpBuf for ReduceSum
    TBuf<TPosition::VECCALC> cooperativeBuf;            // compact fp32 partial vector

    // === GM tensors ===
    GlobalTensor<T> inputGM;
    GlobalTensor<T> resultGM;
    GlobalTensor<float> workspaceGM;

    // === Parameters ===
    uint32_t tilingMode_ = 0;
    int64_t totalRows_ = 0;
    int64_t totalWorkItems_ = 0;
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
    uint32_t reduceTmpBytes_ = 0;

    // MULTI_AXIS params (Key=4)
    int32_t numLayers_ = 0;
    const SquareSumV1TilingData* tilingData_ = nullptr;
    int64_t cooperativeChunkCols_ = 0;
    int64_t cooperativeCoreNum_ = 0;

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
    totalWorkItems_ = tilingData->totalWorkItems;
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
    reduceTmpBytes_ = tilingData->reduceTmpBytes;
    cooperativeChunkCols_ = tilingData->cooperativeChunkCols;
    cooperativeCoreNum_ = tilingData->cooperativeCoreNum;
    isAlign32B_ = tilingData->isAlign32B;

    numLayers_ = tilingData->numLayers;
    tilingData_ = tilingData;

    int64_t blockIdx = GetBlockIdx();
    myRowOffset_ = blockIdx * rowsPerCore_;
    if (blockIdx == static_cast<int64_t>(tilingData->usedCoreNum) - 1) {
        myRows_ = totalWorkItems_ - myRowOffset_;
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
        // The low precision path must not use the fp32 reduce destination as
        // its fp16/bf16 output buffer in-place.
        pipe.InitBuffer(accBuf, 32);
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
        pipe.InitBuffer(reduceBuf, totalCols * sizeof(float));
        pipe.InitBuffer(outQueueYSingle, totalCols * sizeof(T));

        {
            uint32_t tmpBufBytes = reduceTmpBytes_;
            if (tmpBufBytes < 32) tmpBufBytes = 32;
            pipe.InitBuffer(tmpBuf, tmpBufBytes);
        }
    } else if (tilingMode_ == 4) {
        // MULTI_AXIS_COMPACT: both workspace stages are dense fp32 arrays.
        // Size UB from the largest *actual chunk*, never from full R.
        workspaceGM.SetGlobalBuffer((__gm__ float*)workspace);
        int64_t maxMatrixElems = 8;
        int64_t maxCols = 8;
        uint32_t maxTmpBytes = 32;
        for (int32_t li = 0; li < numLayers_; ++li) {
            const int64_t cols = tilingData->layerIsTailReduce[li] ? 1 : tilingData->layerTileA0Align[li];
            const int64_t rows = tilingData->layerRChunkSizeCompact[li];
            const int64_t matrixElems = rows * cols;
            if (matrixElems > maxMatrixElems) maxMatrixElems = matrixElems;
            if (cols > maxCols) maxCols = cols;
            if (tilingData->layerReduceTmpBytes[li] > maxTmpBytes) {
                maxTmpBytes = tilingData->layerReduceTmpBytes[li];
            }
        }
        const uint32_t matrixBytes = static_cast<uint32_t>(maxMatrixElems * sizeof(float));
        const uint32_t colsBytes = static_cast<uint32_t>(maxCols * sizeof(float));
        pipe.InitBuffer(multiInBuf, matrixBytes);
        pipe.InitBuffer(multiComputeBuf, matrixBytes);
        pipe.InitBuffer(multiAccBuf, colsBytes);
        pipe.InitBuffer(multiOutBuf, colsBytes);
        pipe.InitBuffer(multiTmpBuf, maxTmpBytes);
    } else if (tilingMode_ == 5) {
        workspaceGM.SetGlobalBuffer((__gm__ float*)workspace);
        pipe.InitBuffer(inQueueXSingle, cooperativeChunkCols_ * sizeof(T));
        if constexpr (!isFloatInput) {
            pipe.InitBuffer(computeBuf, cooperativeChunkCols_ * sizeof(float));
        }
        pipe.InitBuffer(accBuf, 32);
        pipe.InitBuffer(outQueueYSingle, 32);
        pipe.InitBuffer(cooperativeBuf,
            ((cooperativeCoreNum_ + 7) / 8) * 8 * static_cast<int64_t>(sizeof(float)));
        // 4 KiB covers the Level-2 fp32 ReduceSum scratch for the bounded
        // 255-repeat cooperative chunk and keeps this device-only header
        // independent from the host tiling helper.
        pipe.InitBuffer(tmpBuf, 4096);
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
        case 5: ProcessReduceAllCooperative(); break;
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
        LocalTensor<float> reduceDst = accBuf.Get<float>();
        Cast(xFp32, xLocal, RoundMode::CAST_NONE, rLength_);
        Mul(xFp32, xFp32, xFp32, rLength_);
        ReduceSum<float>(reduceDst, xFp32, tmpLocal,
                         static_cast<int32_t>(rLength_));
        PipeBarrier<PIPE_V>();
        Cast(yLocal, reduceDst, RoundMode::CAST_RINT, 8);
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
    LocalTensor<float> accLocal = accBuf.Get<float>();
    LocalTensor<T> resultLocal = outQueueYSingle.Get<T>();

    for (int64_t i = 0; i < myRows_; i++) {
        int64_t globalRowIdx = myRowOffset_ + i;
        Duplicate(accLocal, 0.0f, 8);
        PipeBarrier<PIPE_V>();

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
            // DataCopyPad is issued by MTE.  PIPE_V alone does not establish
            // an MTE2 -> Vector dependency for a raw TBuf.
            PipeBarrier<PIPE_ALL>();

            LocalTensor<float> reduceDst = resultLocal.template ReinterpretCast<float>();
            Duplicate(reduceDst, 0.0f, 8);

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

            PipeBarrier<PIPE_V>();
            Add(accLocal, accLocal, reduceDst, 8);
            PipeBarrier<PIPE_V>();
        }

        if constexpr (!isFloatInput) {
            Cast(resultLocal, accLocal, RoundMode::CAST_RINT, 8);
            PipeBarrier<PIPE_V>();
        } else {
            LocalTensor<float> resultFp32 = resultLocal.template ReinterpretCast<float>();
            Duplicate(resultFp32, 0.0f, 8);
            PipeBarrier<PIPE_V>();
            Add(resultFp32, resultFp32, accLocal, 8);
            PipeBarrier<PIPE_V>();
        }

        PipeBarrier<PIPE_ALL>();

        DataCopyExtParams copyParamsOut;
        copyParamsOut.blockCount = 1;
        copyParamsOut.blockLen = sizeof(T);
        copyParamsOut.srcStride = 0;
        copyParamsOut.dstStride = 0;
        copyParamsOut.rsv = 0;
        DataCopyPad(resultGM[globalRowIdx], resultLocal, copyParamsOut);
        // outQueueYSingle is reused by the next row; wait for MTE3 before
        // overwriting the raw TBuf source of this non-queued DMA.
        PipeBarrier<PIPE_ALL>();
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
        const int64_t globalWorkIdx = myRowOffset_ + i;
        int64_t globalRowIdx = globalWorkIdx / numA0Tiles_;
        const int64_t a0TileIdx = globalWorkIdx % numA0Tiles_;
        {
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
            PipeBarrier<PIPE_ALL>();

            DataCopyExtParams copyParams;
            copyParams.blockCount = static_cast<uint16_t>(rLength_);
            copyParams.blockLen = a0Len * sizeof(T);
            // Source is GM: srcStride is measured in bytes (not 32B blocks).
            copyParams.srcStride = static_cast<uint32_t>((a0Length_ - a0Len) * sizeof(T));
            // Destination is UB: dstStride is in 32B datablocks.  A partial
            // final A0 tile must still land at the alignedCols row pitch.
            const int64_t ubRowBlocks = alignedCols * sizeof(T) / 32;
            const int64_t copiedBlocks = (a0Len * sizeof(T) + 31) / 32;
            copyParams.dstStride = static_cast<uint32_t>(ubRowBlocks - copiedBlocks);
            copyParams.rsv = 0;
            DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
            DataCopyPad(xLocal, inputGM[gmOffset], copyParams, padParams);
            PipeBarrier<PIPE_ALL>();

            LocalTensor<float> reduceDst = reduceBuf.Get<float>();

            if constexpr (isFloatInput) {
                Mul(xLocal, xLocal, xLocal, rLength_ * alignedCols);
                PipeBarrier<PIPE_V>();
                uint32_t srcShape[] = {static_cast<uint32_t>(rLength_),
                                       static_cast<uint32_t>(alignedCols)};
                ReduceSum<float, Pattern::Reduce::RA, true>(
                    reduceDst, xLocal.template ReinterpretCast<float>(), tmpLocal, srcShape, true);
            } else {
                LocalTensor<float> xFp32 = computeBuf.Get<float>();
                uint32_t castCount = static_cast<uint32_t>(rLength_ * alignedCols);
                Cast(xFp32, xLocal, RoundMode::CAST_NONE, castCount);
                PipeBarrier<PIPE_V>();
                Mul(xFp32, xFp32, xFp32, rLength_ * alignedCols);
                PipeBarrier<PIPE_V>();
                uint32_t srcShape[] = {static_cast<uint32_t>(rLength_),
                                       static_cast<uint32_t>(alignedCols)};
                ReduceSum<float, Pattern::Reduce::RA, true>(
                    reduceDst, xFp32, tmpLocal, srcShape, true);
            }

            LocalTensor<T> yLocal = outQueueYSingle.Get<T>();
            if constexpr (!isFloatInput) {
                Cast(yLocal, reduceDst, RoundMode::CAST_RINT, alignedCols);
                PipeBarrier<PIPE_V>();
            }

            PipeBarrier<PIPE_ALL>();

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
            // The next A0 tile reuses accLocal/yLocal.  Without an MTE3
            // dependency the following Duplicate can turn this tile into zero.
            PipeBarrier<PIPE_ALL>();
        }
    }
}

// ============================================================
// ARA_ROWSPLIT (Key=3) - R-chunk + cross-chunk accumulation
// ============================================================

template <typename T>
__aicore__ inline void SquareSumV1<T>::ProcessAraRowSplit()
{
    LocalTensor<float> accLocal = accBuf.Get<float>();
    LocalTensor<float> reduceDst = reduceBuf.Get<float>();
    LocalTensor<uint8_t> tmpLocal = tmpBuf.Get<uint8_t>();

    for (int64_t i = 0; i < myRows_; i++) {
        const int64_t globalWorkIdx = myRowOffset_ + i;
        int64_t globalRowIdx = globalWorkIdx / numA0Tiles_;
        const int64_t a0TileIdx = globalWorkIdx % numA0Tiles_;
        {
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

                // DataCopyPad advances each UB block to a 32B boundary.  The
                // host guarantees alignedCols is an input-type 32B multiple;
                // clear the final partial A0 tile before copying its rows.
                Duplicate(xLocal, static_cast<T>(0), static_cast<int32_t>(rSize * alignedCols));
                PipeBarrier<PIPE_ALL>();

                DataCopyExtParams copyParams;
                copyParams.blockCount = static_cast<uint16_t>(rSize);
                copyParams.blockLen = a0Len * sizeof(T);
                // GM-side stride is in bytes; rSize is capped to 4095 by host tiling.
                copyParams.srcStride = static_cast<uint32_t>((a0Length_ - a0Len) * sizeof(T));
                // Keep each copied UB row at alignedCols, including the last
                // partial A0 tile (UB stride unit is one 32B datablock).
                const int64_t ubRowBlocks = alignedCols * sizeof(T) / 32;
                const int64_t copiedBlocks = (a0Len * sizeof(T) + 31) / 32;
                copyParams.dstStride = static_cast<uint32_t>(ubRowBlocks - copiedBlocks);
                copyParams.rsv = 0;

                DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
                DataCopyPad(xLocal, inputGM[gmOffset], copyParams, padParams);
                PipeBarrier<PIPE_ALL>();

                if constexpr (isFloatInput) {
                    Mul(xLocal, xLocal, xLocal, rSize * alignedCols);
                    PipeBarrier<PIPE_V>();
                    uint32_t srcShape[] = {static_cast<uint32_t>(rSize),
                                           static_cast<uint32_t>(alignedCols)};
                    ReduceSum<float, Pattern::Reduce::RA, true>(
                        reduceDst, xLocal.template ReinterpretCast<float>(), tmpLocal, srcShape, true);
                } else {
                    LocalTensor<float> xFp32 = computeBuf.Get<float>();
                    // Do not round castCount up: xFp32 is allocated exactly
                    // rSize * alignedCols elements and the old round-up read
                    // past its end on partial chunks.
                    uint32_t castCount = static_cast<uint32_t>(rSize * alignedCols);
                    Cast(xFp32, xLocal, RoundMode::CAST_NONE, castCount);
                    PipeBarrier<PIPE_V>();
                    Mul(xFp32, xFp32, xFp32, rSize * alignedCols);
                    PipeBarrier<PIPE_V>();
                    uint32_t srcShape[] = {static_cast<uint32_t>(rSize),
                                           static_cast<uint32_t>(alignedCols)};
                    ReduceSum<float, Pattern::Reduce::RA, true>(
                        reduceDst, xFp32, tmpLocal, srcShape, true);
                }
                PipeBarrier<PIPE_V>();
                Add(accLocal, accLocal, reduceDst, alignedCols);
                PipeBarrier<PIPE_V>();
            }

            LocalTensor<T> yLocal = outQueueYSingle.Get<T>();
            if constexpr (!isFloatInput) {
                Cast(yLocal, accLocal, RoundMode::CAST_RINT, alignedCols);
                PipeBarrier<PIPE_V>();
            }

            PipeBarrier<PIPE_ALL>();

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
            // Raw TBuf storage is immediately reused by the next A0 tile.
            PipeBarrier<PIPE_ALL>();
        }
    }
}

// ============================================================
// REDUCE_ALL_COOPERATIVE (mode=5)
//
// The first stage has one owner per contiguous R range and stores one fp32
// partial per core.  After the hard barrier core 0 merges those partials.
// No atomic writes are used, so the result is stable across launches.
// ============================================================

template <typename T>
__aicore__ inline void SquareSumV1<T>::ProcessReduceAllCooperative()
{
    const int64_t blockIdx = GetBlockIdx();
    const int64_t range = (rLength_ + cooperativeCoreNum_ - 1) / cooperativeCoreNum_;
    const int64_t begin = blockIdx * range;
    const int64_t end = (begin + range < rLength_) ? begin + range : rLength_;
    LocalTensor<float> acc = accBuf.Get<float>();
    LocalTensor<float> partial = cooperativeBuf.Get<float>();
    LocalTensor<float> tmp = tmpBuf.Get<float>();
    Duplicate(acc, 0.0f, 8);
    PipeBarrier<PIPE_V>();

    for (int64_t offset = begin; offset < end; offset += cooperativeChunkCols_) {
        const int64_t count = ((offset + cooperativeChunkCols_) < end) ? cooperativeChunkCols_ : (end - offset);
        LocalTensor<T> x = inQueueXSingle.Get<T>();
        DataCopyExtParams cp{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pp{false, 0, 0, static_cast<T>(0)};
        DataCopyPad(x, inputGM[offset], cp, pp);
        PipeBarrier<PIPE_ALL>();
        if constexpr (isFloatInput) {
            Mul(x, x, x, static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(partial, x, tmp, static_cast<int32_t>(count));
        } else {
            LocalTensor<float> fp32 = computeBuf.Get<float>();
            Cast(fp32, x, RoundMode::CAST_NONE, static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
            Mul(fp32, fp32, fp32, static_cast<int32_t>(count));
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(partial, fp32, tmp, static_cast<int32_t>(count));
        }
        PipeBarrier<PIPE_V>();
        Add(acc, acc, partial, 8);
        PipeBarrier<PIPE_V>();
    }

    DataCopyExtParams partialOut{1, sizeof(float), 0, 0, 0};
    DataCopyPad(workspaceGM[blockIdx], acc, partialOut);
    PipeBarrier<PIPE_ALL>();
    SyncAll();

    if (blockIdx != 0) return;
    DataCopyExtParams partialIn{1, static_cast<uint32_t>(cooperativeCoreNum_ * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> partialPad{false, 0, 0, 0.0f};
    DataCopyPad(partial, workspaceGM[0], partialIn, partialPad);
    PipeBarrier<PIPE_ALL>();
    ReduceSum<float>(acc, partial, tmp, static_cast<int32_t>(cooperativeCoreNum_));
    PipeBarrier<PIPE_V>();
    DataCopyExtParams out{1, sizeof(T), 0, 0, 0};
    if constexpr (isFloatInput) {
        DataCopyPad(resultGM[0], acc.template ReinterpretCast<T>(), out);
    } else {
        LocalTensor<T> y = outQueueYSingle.Get<T>();
        Cast(y, acc, RoundMode::CAST_RINT, 8);
        PipeBarrier<PIPE_V>();
        DataCopyPad(resultGM[0], y, out);
    }
}

// ============================================================
// MULTI_AXIS (Key=4) - legacy layer routine retained for source compatibility.
// ProcessMultiAxis below uses the compact implementation and never calls it.
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
                DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                // blockLen describes valid GM data.  Passing an aligned length
                // reads beyond the final row when rLen is not 32B aligned.
                copyParams.blockLen = rLen * sizeof(T);
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
                    DataCopyExtParams cp;
                    cp.blockCount = 1;
                    // Keep GM transfer length equal to the valid tail.  UB
                    // padding is handled by DataCopyPad, without an OOB read.
                    cp.blockLen = a0Len * sizeof(T);
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
    // Each layer owns disjoint (outer, A0-tile) output ranges.  Intermediate
    // tensors are dense fp32 arrays, so later layers use one 2D DMA per
    // chunk instead of the former 32B-per-scalar staging protocol.
    const int64_t stageSpan = tilingData_->layerWorkspaceOffset[numLayers_ - 1] >= 0
        ? tilingData_->layerWorkspaceOffset[numLayers_ - 1] : 0;
    (void)stageSpan; // offsets are supplied per destination layer below.
    const int64_t blockIdx = GetBlockIdx();
    const int64_t usedCores = tilingData_->usedCoreNum;

    for (int32_t li = 0; li < numLayers_; ++li) {
        const bool isFirst = (li == 0);
        const bool isLast = (li == numLayers_ - 1);
        const bool isTail = tilingData_->layerIsTailReduce[li] != 0;
        const int64_t outer = tilingData_->layerOuterLength[li];
        const int64_t rLen = tilingData_->layerRLength[li];
        const int64_t inner = isTail ? 1 : tilingData_->layerA0Length[li];
        const int64_t tileLen = isTail ? 1 : tilingData_->layerTileA0Len[li];
        const int64_t tileAlign = isTail ? 8 : tilingData_->layerTileA0Align[li];
        const int64_t rChunk = tilingData_->layerRChunkSizeCompact[li];
        const int64_t tileCount = (inner + tileLen - 1) / tileLen;
        const int64_t workCount = outer * tileCount;
        const int64_t workPerCore = (workCount + usedCores - 1) / usedCores;
        const int64_t workBegin = blockIdx * workPerCore;
        const int64_t workEnd = (workBegin + workPerCore < workCount) ? workBegin + workPerCore : workCount;

        LocalTensor<float> acc = multiAccBuf.Get<float>();
        LocalTensor<float> reduced = multiOutBuf.Get<float>();
        LocalTensor<uint8_t> tmp = multiTmpBuf.Get<uint8_t>();
        LocalTensor<float> tmpForAr = multiTmpBuf.Get<float>();

        for (int64_t work = workBegin; work < workEnd; ++work) {
            const int64_t outerIdx = work / tileCount;
            const int64_t tileIdx = work % tileCount;
            const int64_t a0Start = tileIdx * tileLen;
            const int64_t validCols = (a0Start + tileLen <= inner) ? tileLen : (inner - a0Start);
            Duplicate(acc, 0.0f, static_cast<int32_t>(tileAlign));
            PipeBarrier<PIPE_V>();

            for (int64_t rStart = 0; rStart < rLen; rStart += rChunk) {
                const int64_t validRows = (rStart + rChunk <= rLen) ? rChunk : (rLen - rStart);
                if (isTail) {
                    if (isFirst) {
                        LocalTensor<T> x = multiInBuf.Get<T>();
                        DataCopyExtParams cp{1, static_cast<uint32_t>(validRows * sizeof(T)), 0, 0, 0};
                        DataCopyPadExtParams<T> pp{false, 0, 0, static_cast<T>(0)};
                        DataCopyPad(x, inputGM[outerIdx * rLen + rStart], cp, pp);
                        PipeBarrier<PIPE_ALL>();
                        if constexpr (isFloatInput) {
                            Mul(x, x, x, static_cast<int32_t>(validRows));
                            PipeBarrier<PIPE_V>();
                            ReduceSum<float>(reduced, x, tmpForAr, static_cast<int32_t>(validRows));
                        } else {
                            LocalTensor<float> fp32 = multiComputeBuf.Get<float>();
                            Cast(fp32, x, RoundMode::CAST_NONE, static_cast<int32_t>(validRows));
                            PipeBarrier<PIPE_V>();
                            Mul(fp32, fp32, fp32, static_cast<int32_t>(validRows));
                            PipeBarrier<PIPE_V>();
                            ReduceSum<float>(reduced, fp32, tmpForAr, static_cast<int32_t>(validRows));
                        }
                    } else {
                        LocalTensor<float> x = multiInBuf.Get<float>();
                        DataCopyExtParams cp{1, static_cast<uint32_t>(validRows * sizeof(float)), 0, 0, 0};
                        DataCopyPadExtParams<float> pp{false, 0, 0, 0.0f};
                        const int64_t srcBase = tilingData_->layerWorkspaceOffset[li - 1] + outerIdx * rLen + rStart;
                        DataCopyPad(x, workspaceGM[srcBase], cp, pp);
                        PipeBarrier<PIPE_ALL>();
                        ReduceSum<float>(reduced, x, tmpForAr, static_cast<int32_t>(validRows));
                    }
                    PipeBarrier<PIPE_V>();
                    Add(acc, acc, reduced, 8);
                    PipeBarrier<PIPE_V>();
                    continue;
                }

                const int64_t ubPitchBlocks = tileAlign * sizeof(float) / 32;
                const int64_t copiedBlocks = (validCols * (isFirst ? sizeof(T) : sizeof(float)) + 31) / 32;
                const int64_t gmStride = (inner - validCols) * (isFirst ? sizeof(T) : sizeof(float));
                uint32_t shape[] = {static_cast<uint32_t>(validRows), static_cast<uint32_t>(tileAlign)};

                if (isFirst) {
                    LocalTensor<T> x = multiInBuf.Get<T>();
                    Duplicate(x, static_cast<T>(0), static_cast<int32_t>(validRows * tileAlign));
                    PipeBarrier<PIPE_V>();
                    DataCopyExtParams cp{static_cast<uint16_t>(validRows), static_cast<uint32_t>(validCols * sizeof(T)),
                        static_cast<uint32_t>(gmStride), static_cast<uint32_t>(ubPitchBlocks - copiedBlocks), 0};
                    DataCopyPadExtParams<T> pp{false, 0, 0, static_cast<T>(0)};
                    const int64_t srcBase = outerIdx * rLen * inner + rStart * inner + a0Start;
                    DataCopyPad(x, inputGM[srcBase], cp, pp);
                    PipeBarrier<PIPE_ALL>();
                    if constexpr (isFloatInput) {
                        Mul(x, x, x, static_cast<int32_t>(validRows * tileAlign));
                        PipeBarrier<PIPE_V>();
                        ReduceSum<float, Pattern::Reduce::RA, true>(reduced, x, tmp, shape, true);
                    } else {
                        LocalTensor<float> fp32 = multiComputeBuf.Get<float>();
                        Cast(fp32, x, RoundMode::CAST_NONE, static_cast<int32_t>(validRows * tileAlign));
                        PipeBarrier<PIPE_V>();
                        Mul(fp32, fp32, fp32, static_cast<int32_t>(validRows * tileAlign));
                        PipeBarrier<PIPE_V>();
                        ReduceSum<float, Pattern::Reduce::RA, true>(reduced, fp32, tmp, shape, true);
                    }
                } else {
                    LocalTensor<float> x = multiInBuf.Get<float>();
                    Duplicate(x, 0.0f, static_cast<int32_t>(validRows * tileAlign));
                    PipeBarrier<PIPE_V>();
                    DataCopyExtParams cp{static_cast<uint16_t>(validRows), static_cast<uint32_t>(validCols * sizeof(float)),
                        static_cast<uint32_t>(gmStride), static_cast<uint32_t>(ubPitchBlocks - copiedBlocks), 0};
                    DataCopyPadExtParams<float> pp{false, 0, 0, 0.0f};
                    const int64_t srcBase = tilingData_->layerWorkspaceOffset[li - 1]
                        + outerIdx * rLen * inner + rStart * inner + a0Start;
                    DataCopyPad(x, workspaceGM[srcBase], cp, pp);
                    PipeBarrier<PIPE_ALL>();
                    ReduceSum<float, Pattern::Reduce::RA, true>(reduced, x, tmp, shape, true);
                }
                PipeBarrier<PIPE_V>();
                Add(acc, acc, reduced, static_cast<int32_t>(tileAlign));
                PipeBarrier<PIPE_V>();
            }

            if (isLast) {
                DataCopyExtParams out{1, static_cast<uint32_t>(validCols * sizeof(T)), 0, 0, 0};
                if constexpr (isFloatInput) {
                    DataCopyPad(resultGM[outerIdx * inner + a0Start], acc.template ReinterpretCast<T>(), out);
                } else {
                    LocalTensor<T> y = multiInBuf.Get<T>();
                    Cast(y, acc, RoundMode::CAST_RINT, static_cast<int32_t>(tileAlign));
                    PipeBarrier<PIPE_V>();
                    DataCopyPad(resultGM[outerIdx * inner + a0Start], y, out);
                }
            } else {
                DataCopyExtParams out{1, static_cast<uint32_t>(validCols * sizeof(float)), 0, 0, 0};
                const int64_t dstBase = tilingData_->layerWorkspaceOffset[li] + outerIdx * inner + a0Start;
                DataCopyPad(workspaceGM[dstBase], acc, out);
            }
            PipeBarrier<PIPE_ALL>();
        }
        // Every launched core, including cores with no tile in this layer,
        // reaches the same hard barrier before the next stage swaps buffers.
        SyncAll();
    }
}

} // namespace NsSquareSumV1
#endif // SQUARE_SUM_V1_H
