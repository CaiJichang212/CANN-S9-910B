/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * \file squaresumv1.h
 * \brief SquareSumV1 kernel class (arch22 / Ascend910B)
 *
 * AR_FULLLOAD mode (TilingKey=0):
 *   - axis=-1 (innermost continuous reduction)
 *   - Entire reduction row fits in UB
 *   - Data flow: DataCopyPad -> Cast(half->float) -> Mul(x,x) -> ReduceSum -> Cast(float->half) -> DataCopyPad
 *   - fp32 input skips Cast
 *   - Double Buffer pipeline (BUFFER_NUM=2)
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
    // Compute in float for precision (fp32 accumulation)
    using ComputeT = float;

public:
    __aicore__ inline SquareSumV1() {};

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR result, const SquareSumV1TilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t rowIdx);
    __aicore__ inline void Compute();
    __aicore__ inline void CopyOut(int64_t rowIdx);

private:
    TPipe pipe;

    // Input queue: holds one row of original dtype data
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    // Output queue: holds one scalar result in original dtype
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;

    // TBuf for compute work areas (not queued, reused across tiles)
    TBuf<TPosition::VECCALC> computeBuf;  // fp32 work buffer for Cast/Mul/ReduceSum source
    TBuf<TPosition::VECCALC> tmpBuf;      // ReduceSum internal work area

    GlobalTensor<T> inputGM;
    GlobalTensor<T> resultGM;

    int64_t totalRows_ = 0;
    int64_t rowsPerCore_ = 0;
    int64_t rLength_ = 0;
    int64_t rLengthAlign_ = 0;
    int64_t myRows_ = 0;
    int64_t myRowOffset_ = 0;

    // Whether input is already float (skip Cast)
    static constexpr bool isFloatInput = std::is_same_v<T, float>;
};

template <typename T>
__aicore__ inline void SquareSumV1<T>::Init(GM_ADDR input, GM_ADDR result, const SquareSumV1TilingData* tilingData)
{
    totalRows_ = tilingData->totalRows;
    rowsPerCore_ = tilingData->rowsPerCore;
    rLength_ = tilingData->rLength;
    rLengthAlign_ = tilingData->rLengthAlign;

    int64_t blockIdx = GetBlockIdx();

    // Calculate this core's row range
    myRowOffset_ = blockIdx * rowsPerCore_;
    if (blockIdx == static_cast<int64_t>(tilingData->usedCoreNum) - 1) {
        // Last core handles remaining rows
        myRows_ = totalRows_ - myRowOffset_;
    } else {
        myRows_ = rowsPerCore_;
    }
    if (myRows_ < 0) {
        myRows_ = 0;
    }

    // Set GM buffers - input is [totalRows, rLength] contiguous
    inputGM.SetGlobalBuffer((__gm__ T*)input);
    resultGM.SetGlobalBuffer((__gm__ T*)result);

    // UB buffer allocation
    // Input queue: BUFFER_NUM * rLengthAlign elements of T
    pipe.InitBuffer(inQueueX, BUFFER_NUM, rLengthAlign_ * sizeof(T));

    // Compute buffer (TBuf): fp32 work area for Cast output / Mul source / ReduceSum source
    // For fp32 input: xLocal is already float, reuse it for Mul. Only need tmpBuf for ReduceSum.
    // For fp16/bf16: need separate fp32 buffer for Cast->Mul->ReduceSum source
    uint32_t computeBufBytes = rLengthAlign_ * sizeof(float);
    if constexpr (!isFloatInput) {
        pipe.InitBuffer(computeBuf, computeBufBytes);
    }

    // tmpBuf for ReduceSum work area
    // Compute minimal tmpBuf size for ReduceSum(count) with float:
    //   elementsPerRepeat = 256/4 = 64
    //   elementsPerBlock = 32/4 = 8
    //   firstMaxRepeat = ceil(count / elementsPerRepeat)
    //   iter1OutputCount = firstMaxRepeat
    //   finalWorkLocalNeedSize = ceilAlign(iter1OutputCount, elementsPerBlock) * elementsPerBlock
    {
        uint32_t typeSize = sizeof(float);
        uint32_t elementsPerRepeat = 256 / typeSize; // 64
        uint32_t elementsPerBlock = 32 / typeSize;   // 8
        uint32_t firstMaxRepeat =
            (static_cast<uint32_t>(rLengthAlign_) + elementsPerRepeat - 1) / elementsPerRepeat;
        if (firstMaxRepeat == 0) firstMaxRepeat = 1;
        uint32_t iter1OutputCount = firstMaxRepeat;
        uint32_t finalWorkLocalNeedSize =
            ((iter1OutputCount + elementsPerBlock - 1) / elementsPerBlock) * elementsPerBlock;
        if (finalWorkLocalNeedSize < elementsPerBlock) {
            finalWorkLocalNeedSize = elementsPerBlock;
        }
        uint32_t tmpBufBytes = finalWorkLocalNeedSize * typeSize;
        pipe.InitBuffer(tmpBuf, tmpBufBytes);
    }

    // Output queue: BUFFER_NUM slots, each 32B (holds 1 scalar result, padded to 32B)
    pipe.InitBuffer(outQueueY, BUFFER_NUM, 32);
}

template <typename T>
__aicore__ inline void SquareSumV1<T>::CopyIn(int64_t rowIdx)
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
__aicore__ inline void SquareSumV1<T>::Compute()
{
    LocalTensor<T> xLocal = inQueueX.template DeQue<T>();
    LocalTensor<T> yLocal = outQueueY.template AllocTensor<T>();

    LocalTensor<float> tmpLocal = tmpBuf.Get<float>();

    if constexpr (isFloatInput) {
        // fp32 path: no Cast needed, compute directly in float
        // Mul: square the input (in-place on xLocal which is float)
        Mul(xLocal, xLocal, xLocal, rLength_);
        // ReduceSum -> result goes to yLocal reinterpreted as float
        ReduceSum<float>(yLocal.template ReinterpretCast<float>(), xLocal, tmpLocal,
                         static_cast<int32_t>(rLength_));
    } else {
        // fp16/bf16 path: Cast -> float -> Mul -> ReduceSum -> Cast back
        LocalTensor<float> xFp32 = computeBuf.Get<float>();

        // Cast half/bf16 -> float (CAST_NONE preserves NaN/inf)
        Cast(xFp32, xLocal, RoundMode::CAST_NONE, rLength_);

        // Mul: square in fp32
        Mul(xFp32, xFp32, xFp32, rLength_);

        // ReduceSum in fp32 -> result goes to yLocal reinterpreted as float
        ReduceSum<float>(yLocal.template ReinterpretCast<float>(), xFp32, tmpLocal,
                         static_cast<int32_t>(rLength_));

        // Cast float result back to T (CAST_NONE preserves NaN/inf)
        PipeBarrier<PIPE_V>();
        Cast(yLocal, yLocal.template ReinterpretCast<float>(), RoundMode::CAST_NONE, 8);
    }

    outQueueY.EnQue(yLocal);
    inQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void SquareSumV1<T>::CopyOut(int64_t rowIdx)
{
    LocalTensor<T> yLocal = outQueueY.template DeQue<T>();

    // Write 1 scalar to GM
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
__aicore__ inline void SquareSumV1<T>::Process()
{
    if (myRows_ == 0) return;

    for (int64_t i = 0; i < myRows_; i++) {
        int64_t globalRowIdx = myRowOffset_ + i;
        CopyIn(globalRowIdx);
        Compute();
        CopyOut(globalRowIdx);
    }
}

} // namespace NsSquareSumV1
#endif // SQUARESUMV1_H
