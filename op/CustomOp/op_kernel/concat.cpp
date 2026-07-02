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
 *   - 按 beforeDim 行数切分给各核（每行内是一个连续的输出段）
 *   - 每行按输入依次拷贝；每个 chunk 内部按固定 TILE_BYTES 分批
 *
 * DataCopyPad（CANN 8.5 Ascend C）：
 *   - DataCopyExtParams(blockCount, blockLen, srcStride, dstStride, rsv)
 *       blockLen 单位为「字节」，uint32_t
 *   - 使用 uint8_t 视角搬运，统一处理所有 dtype
 */
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

using namespace AscendC;

constexpr uint32_t MAX_INPUT_NUM = 64;  // 与 host 端 MAX_CONCAT_INPUT_NUM 保持一致
// 单个 UB tile 字节数：兼顾流水与 UB 上限
constexpr uint32_t TILE_BYTES = 8192;

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

        for (uint32_t i = 0; i < inputNum_; i++) {
            inputCatLen_[i]    = tiling.inputCatLen[i];
            inputCatOffset_[i] = tiling.inputCatOffset[i];
        }

        AscendC::ListTensorDesc listDesc(reinterpret_cast<__gm__ void*>(srcList));
        for (uint32_t i = 0; i < inputNum_; i++) {
            __gm__ uint8_t *ptr = listDesc.GetDataPtr<__gm__ uint8_t>(i);
            inputGm_[i].SetGlobalBuffer(ptr);
        }

        yGm_.SetGlobalBuffer((__gm__ uint8_t *)dst);

        // 使用两个队列：VECIN 用于 GM->UB，VECOUT 用于 UB->GM
        // 中间通过 EnQue/DeQue 完成 MTE2→VECTOR/MTE3 同步
        pipe.InitBuffer(inQueue_,  2, TILE_BYTES);
        pipe.InitBuffer(outQueue_, 2, TILE_BYTES);
    }

    __aicore__ inline void Process()
    {
        uint32_t coreId = GetBlockIdx();
        if (coreId >= usedCoreNum_ || inputNum_ == 0 || beforeDimSize_ == 0) {
            return;
        }

        // 按行切分，每行整段（totalCatLen * afterDimSize 字节）由单核处理
        uint32_t perCoreRows = (beforeDimSize_ + usedCoreNum_ - 1) / usedCoreNum_;
        uint32_t startRow = coreId * perCoreRows;
        uint32_t endRow = startRow + perCoreRows;
        if (endRow > beforeDimSize_) endRow = beforeDimSize_;
        if (startRow >= endRow) return;

        for (uint32_t row = startRow; row < endRow; row++) {
            for (uint32_t i = 0; i < inputNum_; i++) {
                uint32_t catLen = inputCatLen_[i];
                if (catLen == 0) continue;
                uint64_t elemCount = static_cast<uint64_t>(catLen) * afterDimSize_;
                if (elemCount == 0) continue;
                CopyOneInput(row, i, elemCount);
            }
        }
    }

private:
    __aicore__ inline void CopyOneInput(uint32_t row, uint32_t i, uint64_t elemCount)
    {
        uint64_t rowBytes = elemCount * dtypeSize_;

        uint64_t inRowElem = static_cast<uint64_t>(inputCatLen_[i]) * afterDimSize_;
        uint64_t inOffsetBytes = static_cast<uint64_t>(row) * inRowElem * dtypeSize_;

        uint64_t outRowOffset = static_cast<uint64_t>(row) *
                                static_cast<uint64_t>(totalCatLen_) *
                                afterDimSize_ * dtypeSize_;
        uint64_t outOffsetBytes = outRowOffset +
            static_cast<uint64_t>(inputCatOffset_[i]) * afterDimSize_ * dtypeSize_;

        uint64_t remaining = rowBytes;
        uint64_t srcOff = inOffsetBytes;
        uint64_t dstOff = outOffsetBytes;
        while (remaining > 0) {
            uint32_t batchBytes = (remaining > TILE_BYTES) ? TILE_BYTES
                                                          : static_cast<uint32_t>(remaining);
            CopyBatch(i, srcOff, dstOff, batchBytes);
            srcOff += batchBytes;
            dstOff += batchBytes;
            remaining -= batchBytes;
        }
    }

    __aicore__ inline void CopyBatch(uint32_t inputIdx, uint64_t srcOffsetBytes,
                                     uint64_t dstOffsetBytes, uint32_t bytes)
    {
        // GM -> UB (VECIN)
        LocalTensor<uint8_t> xLocal = inQueue_.AllocTensor<uint8_t>();
        DataCopyExtParams copyInParams{1, bytes, 0, 0, 0};
        DataCopyPadExtParams<uint8_t> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, inputGm_[inputIdx][srcOffsetBytes], copyInParams, padParams);
        inQueue_.EnQue(xLocal);

        // 从 VECIN DeQue（保证 GM→UB 完成），拷贝到 VECOUT buffer
        LocalTensor<uint8_t> xLocalDeq = inQueue_.DeQue<uint8_t>();
        LocalTensor<uint8_t> yLocal = outQueue_.AllocTensor<uint8_t>();
        // UB->UB 拷贝：把已就绪的输入数据搬到输出队列 buffer
        // 使用 DataCopy 需要 32B 对齐，元素级拷贝更通用
        uint32_t alignedBytes = (bytes + 31) & ~31u;
        AscendC::DataCopy(yLocal, xLocalDeq, alignedBytes);
        outQueue_.EnQue(yLocal);
        inQueue_.FreeTensor(xLocalDeq);

        // UB -> GM (VECOUT)
        LocalTensor<uint8_t> yLocalDeq = outQueue_.DeQue<uint8_t>();
        DataCopyExtParams copyOutParams{1, bytes, 0, 0, 0};
        DataCopyPad(yGm_[dstOffsetBytes], yLocalDeq, copyOutParams);
        outQueue_.FreeTensor(yLocalDeq);
    }

private:
    uint32_t inputNum_;
    uint32_t afterDimSize_;
    uint32_t dtypeSize_;
    uint32_t totalCatLen_;
    uint32_t usedCoreNum_;
    uint32_t beforeDimSize_;

    uint32_t inputCatLen_[MAX_INPUT_NUM];
    uint32_t inputCatOffset_[MAX_INPUT_NUM];

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN,  2> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 2> outQueue_;
    AscendC::GlobalTensor<uint8_t> inputGm_[MAX_INPUT_NUM];
    AscendC::GlobalTensor<uint8_t> yGm_;
};

extern "C" __global__ __aicore__ void concat(GM_ADDR srcList, GM_ADDR dst,
                                             GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    KernelConcat op;
    op.Init(srcList, dst, tiling_data);
    op.Process();
}
