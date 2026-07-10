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
 *
 * Data flow (fp16/bf16 input):
 *   DataCopyPad -> Cast(half->float) -> Mul(x,x) -> ReduceSum -> Cast(float->half) -> DataCopyPad
 * fp32 input skips all Cast operations.
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
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR result, const SquareSumV1TilingData* tilingData);
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

    // === GM tensors ===
    GlobalTensor<T> inputGM;
    GlobalTensor<T> resultGM;

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

    uint32_t isAlign32B_ = 0;

    static constexpr bool isFloatInput = std::is_same_v<T, float>;
};

// ============================================================
// Init
// ============================================================

template <typename T>
__aicore__ inline void SquareSumV1<T>::Init(GM_ADDR input, GM_ADDR result, const SquareSumV1TilingData* tilingData)
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
        // tmpBuf for ReduceSum (first-n version)
        {
            uint32_t typeSize = sizeof(float);
            uint32_t epr = 256 / typeSize; // 64
            uint32_t epb = 32 / typeSize;  // 8
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

        // tmpBuf for Pattern ReduceSum RA
        {
            uint32_t tmpBufBytes = static_cast<uint32_t>(totalCols * sizeof(float));
            if (tmpBufBytes < 32) tmpBufBytes = 32;
            pipe.InitBuffer(tmpBuf, tmpBufBytes);
        }
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

            // CopyIn chunk via DataCopyPad
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

            // Compute: Cast -> Mul -> ReduceSum -> accumulate
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

            // Read scalar result and accumulate
            float partial = reduceDst.GetValue(0);
            accVal += partial;
        }

        // Write result: store accVal to T and copy out
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

            // CopyIn [R, alignedCols] 2D block
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

            // Compute: Cast -> Mul -> Pattern ReduceSum
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

            // Cast float result back to T and copy out
            LocalTensor<T> yLocal = outQueueYSingle.Get<T>();
            if constexpr (!isFloatInput) {
                Cast(yLocal, reduceDst, RoundMode::CAST_NONE, alignedCols);
                PipeBarrier<PIPE_V>();
            } else {
                // fp32: reuse reduceDst as output
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

            // Initialize accumulator to 0
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

                // CopyIn [rSize, alignedCols] block
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

                // Compute: Cast -> Mul -> Pattern ReduceSum
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

                // Accumulate: accLocal += chunkResult
                PipeBarrier<PIPE_V>();
                Add(accLocal, accLocal, chunkResult, alignedCols);
                PipeBarrier<PIPE_V>();
            }

            // Cast float result back to T and copy out
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

} // namespace NsSquareSumV1
#endif // SQUARESUMV1_H
