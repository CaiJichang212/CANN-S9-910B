/**
 * @file transpose.cpp
 * @brief Transpose (torch.permute) kernel -- 通用 permute，输出驱动搬运。
 *
 * host 把任意 permute 归约为「按输出行搬运」：
 *   W = 输出末维 = inShape[dims[ndim-1]]；S = 源步长 = inStride[dims[ndim-1]]（元素）。
 *   外层输出维 (0..ndim-2) 每个组合对应输出的一「行」，源基址由外层维的
 *   inStride[dims[i]] 加权得到（DecodeRow）。
 *
 * mode=0 (COPY)：S==1（输出末维在源端连续）或任意 permute（S>1 兜底）。
 *   - S==1：按 copyTileLen 连续搬运一整行（GM 连续读 -> UB -> GM 连续写，高效）。
 *   - S>1：逐元素从源(间隔 S 元素)读取、连续写出（通用任意 permute，正确优先）。
 * mode=1 (TRANSPOSE)：末两维相邻交换且前缀 identity (dims=[0..,n-1,n-2])，
 *   即 2D 转置 (M,N)->(N,M)（可带前缀 batch）。
 *   按 16×16 块：GM 按行连续读入 UB（紧凑 16×16，因 half 16元素=32B=1block），
 *   half 调硬件 vtranspose(Transpose API，if constexpr 守卫)，其余 dtype 走逐元素转置，
 *   再按输出行连续写出（GM strided 写）。
 *
 * tiling 完全由 shape/dims/dtype 决定，无针对已知用例的定制化。
 *
 * stride 单位约定（参考 DataCopyPad 文档）：
 *   GM 侧 srcStride/dstStride 单位 = 字节；
 *   UB 侧 srcStride/dstStride 单位 = 32B dataBlock（必须 32B 整倍）。
 *   blockCount(uint16_t) ∈ [1,4095]。
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
// MTE3 can still be writing a tile when the next tile is submitted.  Keep two
// buffers so a core never reuses the same UB region before that write retires.
constexpr int32_t TRANSPOSE_BUFFER_NUM = 2;
constexpr uint32_t MAX_DIM = 8;
constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t BLK = 16; // vtranspose 块边长

class KernelTranspose {
public:
    __aicore__ inline KernelTranspose() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TransposeTilingData &t)
    {
        mode_ = t.mode;
        total_ = t.total;
        ndim_ = t.ndim;
        dtypeSize_ = t.dtypeSize;
        blockDim_ = t.blockDim;
        W_ = t.W;
        numRows_ = t.numRows;
        S_ = t.srcStrideInner;
        outerCount_ = t.outerCount;
        copyTileLen_ = t.copyTileLen;
        for (uint32_t i = 0; i < outerCount_; i++) {
            outerOutShape_[i] = t.outerOutShape[i];
            outerSrcStride_[i] = t.outerSrcStride[i];
        }
        transM_ = t.transM;
        transN_ = t.transN;
        transBatch_ = t.transBatch;
        tileM_ = t.tileM;
        tileN_ = t.tileN;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y);
    }

    __aicore__ inline void Process()
    {
        if (mode_ == 1) {
            ProcessTranspose();
        } else {
            ProcessCopy();
        }
    }

private:
    // ================= COPY 路径 (S==1, 源端连续) =================
    __aicore__ inline void ProcessCopy()
    {
        uint32_t numCore = blockDim_;
        if (numCore == 0) {
            return;
        }
        uint32_t coreId = GetBlockIdx();
        uint32_t rowsPerCore = (numRows_ + numCore - 1) / numCore;
        uint32_t rowStart = coreId * rowsPerCore;
        if (rowStart >= numRows_) {
            return;
        }
        uint32_t rowEnd = rowStart + rowsPerCore;
        if (rowEnd > numRows_) {
            rowEnd = numRows_;
        }

        pipe.InitBuffer(inQueue, BUFFER_NUM, copyTileLen_ * dtypeSize_);
        // inQueue 是 TQueBind<VECIN, VECOUT>：EnQue 自动建立 MTE2(GM→UB)→MTE3(UB→GM) 同步，
        // 无需显式事件，也无需 Vector 桥接（参考内置 transpose v35 transpose_tensor_move.h）。
        for (uint32_t row = rowStart; row < rowEnd; row++) {
            uint64_t srcBase = DecodeRow(row);
            uint64_t dstBase = (uint64_t)row * (uint64_t)W_;
            ProcessCopyRow(srcBase, dstBase);
        }
    }

    __aicore__ inline uint64_t DecodeRow(uint32_t row)
    {
        uint32_t rem = row;
        uint64_t srcBase = 0;
        for (int32_t i = (int32_t)outerCount_ - 1; i >= 0; i--) {
            uint32_t dim = outerOutShape_[i];
            uint32_t digit = (dim > 0) ? (rem % dim) : 0;
            rem = (dim > 0) ? (rem / dim) : 0;
            srcBase += (uint64_t)digit * (uint64_t)outerSrcStride_[i];
        }
        return srcBase;
    }

    __aicore__ inline void ProcessCopyRow(uint64_t srcBase, uint64_t dstBase)
    {
        if (S_ == 1) {
            // 源端连续：按 copyTileLen 连续搬运（高效）
            uint32_t offset = 0;
            while (offset < W_) {
                uint32_t curLen = (W_ - offset < copyTileLen_) ? (W_ - offset) : copyTileLen_;
                CopyTileContiguous(srcBase, dstBase, offset, curLen);
                offset += curLen;
            }
        } else {
            // 源端非连续 (S>1，任意 permute)：按元素步长读取。
            // 分块限制 blockCount<=4095，每块搬运 curLen 个元素（各间隔 (S-1)*sizeof 字节）。
            uint32_t offset = 0;
            while (offset < W_) {
                uint32_t curLen = (W_ - offset < copyTileLen_) ? (W_ - offset) : copyTileLen_;
                if (curLen > 4095) {
                    curLen = 4095;
                }
                CopyTileStrided(srcBase, dstBase, offset, curLen);
                offset += curLen;
            }
        }
    }

    // S==1 连续搬运一整段
    __aicore__ inline void CopyTileContiguous(uint64_t srcBase, uint64_t dstBase, uint32_t offset, uint32_t curLen)
    {
        LocalTensor<DTYPE_X> ub = inQueue.AllocTensor<DTYPE_X>();
        uint64_t srcOff = srcBase + (uint64_t)offset;
        DataCopyExtParams inParams;
        inParams.blockCount = 1;
        inParams.blockLen = curLen * dtypeSize_;
        inParams.srcStride = 0;
        inParams.dstStride = 0;
        inParams.rsv = 0;
        DataCopyPadExtParams<DTYPE_X> padExt;
        padExt.isPad = false;
        padExt.leftPadding = 0;
        padExt.rightPadding = 0;
        padExt.paddingValue = (DTYPE_X)0;
        DataCopyPad(ub, xGm[srcOff], inParams, padExt);
        inQueue.EnQue(ub);

        LocalTensor<DTYPE_X> ubOut = inQueue.DeQue<DTYPE_X>();
        DataCopyExtParams outParams;
        outParams.blockCount = 1;
        outParams.blockLen = curLen * dtypeSize_;
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        outParams.rsv = 0;
        DataCopyPad(yGm[dstBase + offset], ubOut, outParams);
        inQueue.FreeTensor(ubOut);
    }

    // S>1 步长搬运：逐元素从源(间隔 S)读、连续写出。通用任意 permute 兜底，正确优先。
    // UB 内顺序存放 curLen 个元素（连续），读用 DataCopyPad 单元素逐个或 GetValue，写连续。
    __aicore__ inline void CopyTileStrided(uint64_t srcBase, uint64_t dstBase, uint32_t offset, uint32_t curLen)
    {
        LocalTensor<DTYPE_X> ub = inQueue.AllocTensor<DTYPE_X>();
        // 逐元素读取（源间隔 S 元素）。GetValue 从 GM 读开销大，改用单元素 DataCopyPad 紧凑写入 UB。
        // 为正确与简洁，这里逐元素 SetValue 到 UB。
        uint64_t srcOff = srcBase + (uint64_t)offset * (uint64_t)S_;
        for (uint32_t k = 0; k < curLen; k++) {
            // 单元素 GM->UB：用 DataCopyPad blockCount=1, blockLen=dtypeSize，目的紧凑(自动 pad 到 32B)。
            // 简单起见用 SetValue(ub[k], xGm.GetValue(srcOff+k*S)) —— 但 GlobalTensor GetValue 在 device 上可用。
            ub.SetValue(k, xGm.GetValue(srcOff + (uint64_t)k * (uint64_t)S_));
        }
        inQueue.EnQue(ub);

        LocalTensor<DTYPE_X> ubOut = inQueue.DeQue<DTYPE_X>();
        // 连续写出 curLen 个元素
        DataCopyExtParams outParams;
        outParams.blockCount = 1;
        outParams.blockLen = curLen * dtypeSize_;
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        outParams.rsv = 0;
        DataCopyPad(yGm[dstBase + offset], ubOut, outParams);
        inQueue.FreeTensor(ubOut);
    }

    // ================= TRANSPOSE 路径 (末两维交换 M,N->N,M) =================
    // 输出 (N,M)。按 [tileN 列 × tileM 行] 源子块分块（tileM 行，每行 tileN 连续元素）。
    //   读：源 [mi..mi+tileM) 行 × [nj..nj+tileN) 列 -> UB（行布局，每行 tileN 元素，
    //       UB 行间距 32B 对齐）。
    //   转：UB 内 src[mh][nw] -> dst[nw][mh]（dst 行布局，每行 mh 元素，对齐到 32B）。
    //   写：dst [nj..nj+nw) 行 × [mi..mi+tileM) 列 -> 输出（每行 mh 连续，GM 行间 stride）。
    __aicore__ inline void ProcessTranspose()
    {
        uint32_t numCore = blockDim_;
        if (numCore == 0) {
            return;
        }
        uint32_t coreId = GetBlockIdx();
        uint32_t nTiles = (transN_ + tileN_ - 1) / tileN_;
        uint32_t mTiles = (transM_ + tileM_ - 1) / tileM_;
        uint32_t tilesPerMat = nTiles * mTiles;
        uint32_t totalTiles = transBatch_ * tilesPerMat; // 含前缀 batch 维
        uint32_t tilesPerCore = (totalTiles + numCore - 1) / numCore;
        uint32_t tStart = coreId * tilesPerCore;
        if (tStart >= totalTiles) {
            return;
        }
        uint32_t tEnd = tStart + tilesPerCore;
        if (tEnd > totalTiles) {
            tEnd = totalTiles;
        }

        // UB：源行布局每行对齐到 32B，存放 tileM 行 tileN 列。
        uint32_t srcRowBytes = Align32(tileN_ * dtypeSize_);
        uint32_t dstRowBytes = Align32(tileM_ * dtypeSize_);
        uint32_t srcBytes = srcRowBytes * tileM_;
        uint32_t dstBytes = dstRowBytes * tileN_;
        uint32_t ubBytes = (srcBytes > dstBytes) ? srcBytes : dstBytes;
        uint32_t ubElems = ubBytes / dtypeSize_;
        pipe.InitBuffer(srcQue, TRANSPOSE_BUFFER_NUM, ubElems * dtypeSize_);
        pipe.InitBuffer(dstQue, TRANSPOSE_BUFFER_NUM, ubElems * dtypeSize_);
        // TQue provides the DMA/vector dependencies used by vtranspose.  The
        // generic path uses scalar LocalTensor GetValue/SetValue (S pipe), so
        // it needs its own explicit DMA-to-scalar and scalar-to-DMA events.
        event_t mte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        event_t sToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));

        uint64_t matElems = (uint64_t)transM_ * (uint64_t)transN_; // 单个矩阵元素数
        for (uint32_t tb = tStart; tb < tEnd; tb++) {
            uint32_t b = tb / tilesPerMat;          // batch 索引
            uint32_t rem = tb - b * tilesPerMat;
            uint32_t ti = rem / nTiles;             // m 方向块索引
            uint32_t tj = rem % nTiles;             // n 方向块索引
            uint32_t mi = ti * tileM_;
            uint32_t nj = tj * tileN_;
            uint32_t mh = (tileM_ < transM_ - mi) ? tileM_ : (transM_ - mi);
            uint32_t nw = (tileN_ < transN_ - nj) ? tileN_ : (transN_ - nj);
            uint64_t matBase = (uint64_t)b * matElems; // 源/目的同 batch 偏移
            TransposeBlk(matBase, mi, nj, mh, nw, srcRowBytes, dstRowBytes, mte2ToS, sToMte3);
        }
    }

    __aicore__ inline void TransposeBlk(uint64_t matBase, uint32_t mi, uint32_t nj, uint32_t mh, uint32_t nw,
                                         uint32_t srcRowBytes, uint32_t dstRowBytes,
                                         event_t mte2ToS, event_t sToMte3)
    {
        // ---- 1) GM->UB：读源 [mi..mi+mh) × [nj..nj+nw)，行布局 ----
        LocalTensor<DTYPE_X> srcUB = srcQue.AllocTensor<DTYPE_X>();
        DataCopyExtParams inParams;
        inParams.blockCount = mh;
        inParams.blockLen = nw * dtypeSize_;
        inParams.srcStride = (transN_ - nw) * dtypeSize_; // GM gap 字节
        uint32_t srcRowBlocks = srcRowBytes / BLOCK_BYTES;
        uint32_t srcCopyBlocks = (nw * dtypeSize_ + BLOCK_BYTES - 1) / BLOCK_BYTES;
        inParams.dstStride = srcRowBlocks - srcCopyBlocks; // UB gap, 单位 block
        inParams.rsv = 0;
        DataCopyPadExtParams<DTYPE_X> padExt;
        padExt.isPad = false;
        padExt.leftPadding = 0;
        padExt.rightPadding = 0;
        padExt.paddingValue = (DTYPE_X)0;
        uint64_t srcOff = matBase + (uint64_t)mi * (uint64_t)transN_ + (uint64_t)nj;
        DataCopyPad(srcUB, xGm[srcOff], inParams, padExt);
        srcQue.EnQue(srcUB);
        LocalTensor<DTYPE_X> src = srcQue.DeQue<DTYPE_X>();

        bool scalarTranspose = NeedsScalarTranspose(mh, nw);
        if (scalarTranspose) {
            SetFlag<HardEvent::MTE2_S>(mte2ToS);
            WaitFlag<HardEvent::MTE2_S>(mte2ToS);
        }

        // ---- 2) UB 内转置 src[mh][nw] -> dst[nw][mh] ----
        LocalTensor<DTYPE_X> dstUB = dstQue.AllocTensor<DTYPE_X>();
        TransposeUB(dstUB, src, mh, nw, srcRowBytes, dstRowBytes);
        dstQue.EnQue(dstUB);
        LocalTensor<DTYPE_X> dst = dstQue.DeQue<DTYPE_X>();

        if (scalarTranspose) {
            SetFlag<HardEvent::S_MTE3>(sToMte3);
            WaitFlag<HardEvent::S_MTE3>(sToMte3);
        }

        // ---- 3) UB->GM：写输出 [nj..nj+nw) × [mi..mi+mh) ----
        DataCopyExtParams outParams;
        outParams.blockCount = nw;
        outParams.blockLen = mh * dtypeSize_;
        uint32_t dstRowBlocks = dstRowBytes / BLOCK_BYTES;
        uint32_t dstCopyBlocks = (mh * dtypeSize_ + BLOCK_BYTES - 1) / BLOCK_BYTES;
        outParams.srcStride = dstRowBlocks - dstCopyBlocks; // UB gap, 单位 block
        outParams.dstStride = (transM_ - mh) * dtypeSize_;  // GM gap 字节
        outParams.rsv = 0;
        uint64_t dstOff = matBase + (uint64_t)nj * (uint64_t)transM_ + (uint64_t)mi;
        DataCopyPad(yGm[dstOff], dst, outParams);
        srcQue.FreeTensor(src);
        dstQue.FreeTensor(dst);
    }

    // UB 内转置。half 走 vtranspose(16×16，紧凑读写已在 TransposeBlk 完成)；其余 dtype 逐元素。
    __aicore__ inline void TransposeUB(LocalTensor<DTYPE_X> &dst, const LocalTensor<DTYPE_X> &src,
                                        uint32_t mh, uint32_t nw, uint32_t srcRowBytes, uint32_t dstRowBytes)
    {
        // AscendC Transpose is a fp16 16x16 instruction.  In particular, it
        // does not define tail-tile behaviour, so every non-full tile must use
        // the scalar layout-aware implementation below.
        if constexpr (sizeof(DTYPE_X) == sizeof(half)) {
            if (mh == BLK && nw == BLK) {
                Transpose(dst, src); // vtranspose：dst[i][j]=src[j][i]
                return;
            }
        }
        TransposeGeneric(dst, src, mh, nw, srcRowBytes, dstRowBytes);
    }

    __aicore__ inline bool NeedsScalarTranspose(uint32_t mh, uint32_t nw)
    {
        if constexpr (sizeof(DTYPE_X) == sizeof(half)) {
            return mh != BLK || nw != BLK;
        }
        return true;
    }

    // 通用 UB 内转置：dst[c][r] = src[r][c]，逐元素（dtype 无关，正确优先）。
    __aicore__ inline void TransposeGeneric(LocalTensor<DTYPE_X> &dst, const LocalTensor<DTYPE_X> &src,
                                            uint32_t mh, uint32_t nw, uint32_t srcRowBytes, uint32_t dstRowBytes)
    {
        uint32_t srcElemStride = srcRowBytes / dtypeSize_;
        uint32_t dstElemStride = dstRowBytes / dtypeSize_;
        for (uint32_t r = 0; r < mh; r++) {
            for (uint32_t c = 0; c < nw; c++) {
                dst.SetValue(c * dstElemStride + r, src.GetValue(r * srcElemStride + c));
            }
        }
    }

    static __aicore__ inline uint32_t Align32(uint32_t bytes)
    {
        return (bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES;
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, TRANSPOSE_BUFFER_NUM> srcQue;
    AscendC::TQue<AscendC::TPosition::VECOUT, TRANSPOSE_BUFFER_NUM> dstQue;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;

    uint32_t mode_;
    uint32_t total_;
    uint32_t ndim_;
    uint32_t dtypeSize_;
    uint32_t blockDim_;
    uint32_t W_;
    uint32_t numRows_;
    uint32_t S_;
    uint32_t outerCount_;
    uint32_t outerOutShape_[MAX_DIM];
    uint32_t outerSrcStride_[MAX_DIM];
    uint32_t copyTileLen_;
    uint32_t transBatch_;
    uint32_t transM_;
    uint32_t transN_;
    uint32_t tileM_;
    uint32_t tileN_;
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
