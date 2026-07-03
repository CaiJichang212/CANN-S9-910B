/**
 * @file greater.cpp
 *
 * Kernel for the Greater (torch.gt) custom operator on Ascend 910B.
 *
 *   out = x > y   (element-wise, NumPy-style broadcast), output dtype = bool.
 *
 * Supported input dtypes: float16, float32, bfloat16, int32, int8.
 *
 * Compute strategy (per dtype, verified against the 910B / dav_c220 impl):
 *   - float16 / float32 : AscendC::Compare(GT) directly.
 *   - bfloat16           : Cast bf16 -> float (exact), then Compare(GT).
 *   - int8               : Cast int8 -> half (exact), then Compare(GT).
 *   - int32              : Compare only supports EQ on 910B, so use an exact
 *                          overflow-safe identity:  gt = (Max(x,y)==x) && (x!=y),
 *                          built from Max + Compare(EQ) + Select.
 *
 * Compare yields a packed uint8 bitmask (8 elems/byte); the bool output is
 * 1 byte/elem, so the mask is expanded with Select(mask, ...) -> half 0/1,
 * then Cast half -> uint8.
 *
 * Broadcast: the output is split into `outerSize` segments of `innerSize`
 * contiguous elements (innerSize = maximal trailing non-broadcast suffix, so
 * identical-shape inputs collapse to one segment = fast flatten path). Outer
 * dims are walked with broadcast strides (0 on broadcast dims); an innermost
 * broadcast dim makes one operand a per-segment scalar (Duplicate-filled).
 */
#include "kernel_operator.h"

using namespace AscendC;

using InputT = DTYPE_X;

constexpr bool kIsHalf = IsSameType<InputT, half>::value;
constexpr bool kIsFloat = IsSameType<InputT, float>::value;
constexpr bool kIsBf16 = IsSameType<InputT, bfloat16_t>::value;
constexpr bool kIsInt32 = IsSameType<InputT, int32_t>::value;
constexpr bool kIsInt8 = IsSameType<InputT, int8_t>::value;

// Compute dtype: half for fp16/int8, float for fp32/bf16, int32 for int32.
using ComputeT = typename std::conditional<
    kIsInt32, int32_t,
    typename std::conditional<(kIsFloat || kIsBf16), float, half>::type>::type;

// Tile length (elements). Multiple of 256 so every op's 256B alignment holds.
constexpr uint32_t TILE = (kIsFloat || kIsBf16 || kIsInt32) ? 4096 : 8192;
constexpr uint32_t COMP_ALIGN = 256;   // elems; 256B for every dtype involved
constexpr uint32_t Z_BLKELEMS = 256;   // bool 256B in elements
constexpr int32_t BUFFER_NUM = 2;

__aicore__ inline uint32_t RoundUpTo(uint32_t n, uint32_t a)
{
    return (n + a - 1) / a * a;
}

class KernelGreater {
public:
    __aicore__ inline KernelGreater() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t totalSize, uint32_t blockDim,
                                uint32_t innerSize, uint32_t outerSize,
                                uint32_t bcastMode, uint32_t outerDim,
                                const uint32_t* outerShape, const uint32_t* xStride,
                                const uint32_t* yStride)
    {
        totalSize_ = totalSize;
        blockDim_ = blockDim;
        innerSize_ = innerSize;
        outerSize_ = outerSize;
        bcastMode_ = bcastMode;
        outerDim_ = outerDim;
        for (int d = 0; d < 8; ++d) {
            outerShape_[d] = outerShape[d];
            xStride_[d] = xStride[d];
            yStride_[d] = yStride[d];
        }

        xGm.SetGlobalBuffer((__gm__ InputT*)x);
        yGm.SetGlobalBuffer((__gm__ InputT*)y);
        zGm.SetGlobalBuffer((__gm__ uint8_t*)z);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE * sizeof(InputT));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE * sizeof(InputT));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE * sizeof(uint8_t));

        pipe.InitBuffer(maskBuf, TILE / 8 * sizeof(uint8_t));
        pipe.InitBuffer(halfOutBuf, TILE * sizeof(half));
        pipe.InitBuffer(halfZeroBuf, TILE * sizeof(half));
        pipe.InitBuffer(halfOneBuf, TILE * sizeof(half));
        pipe.InitBuffer(xCompBuf, TILE * sizeof(ComputeT));
        pipe.InitBuffer(yCompBuf, TILE * sizeof(ComputeT));
        if constexpr (kIsInt32) {
            pipe.InitBuffer(mxBuf, TILE * sizeof(int32_t));
            pipe.InitBuffer(neBuf, TILE * sizeof(half));
            pipe.InitBuffer(maskMxBuf, TILE / 8 * sizeof(uint8_t));
            pipe.InitBuffer(maskEqBuf, TILE / 8 * sizeof(uint8_t));
        }
        if constexpr (kIsBf16) {
            pipe.InitBuffer(bf16TileBuf, TILE * sizeof(bfloat16_t));
        }
        pipe.InitBuffer(scalarBuf, 256);
        pipe.InitBuffer(scalarCTBuf, 512);

        LocalTensor<half> zeroLocal = halfZeroBuf.Get<half>();
        LocalTensor<half> oneLocal = halfOneBuf.Get<half>();
        Duplicate(zeroLocal, (half)0.0f, static_cast<int32_t>(TILE));
        Duplicate(oneLocal, (half)1.0f, static_cast<int32_t>(TILE));
    }

    __aicore__ inline void Process()
    {
        if (totalSize_ == 0 || blockDim_ == 0) {
            return;
        }
        uint32_t coreId = GetBlockIdx();

        uint64_t totalBlks = (totalSize_ + Z_BLKELEMS - 1) / Z_BLKELEMS;
        uint64_t blkStart = totalBlks * coreId / blockDim_;
        uint64_t blkEnd = totalBlks * (coreId + 1) / blockDim_;
        uint64_t coreStart = blkStart * Z_BLKELEMS;
        uint64_t coreEnd = blkEnd * Z_BLKELEMS;
        if (coreEnd > totalSize_) {
            coreEnd = totalSize_;
        }
        if (coreStart >= coreEnd) {
            return;
        }

        uint64_t pos = coreStart;
        uint64_t seg = (innerSize_ > 0) ? (coreStart / innerSize_) : 0;
        uint64_t offInSeg = (innerSize_ > 0) ? (coreStart % innerSize_) : 0;

        while (pos < coreEnd) {
            uint64_t xBase = 0;
            uint64_t yBase = 0;
            ComputeBases(seg, xBase, yBase);

            while (offInSeg < innerSize_ && pos < coreEnd) {
                uint64_t tileN64 = TILE;
                if (innerSize_ - offInSeg < tileN64) {
                    tileN64 = innerSize_ - offInSeg;
                }
                if (coreEnd - pos < tileN64) {
                    tileN64 = coreEnd - pos;
                }
                uint32_t tileN = static_cast<uint32_t>(tileN64);
                ProcessTile(xBase, yBase, offInSeg, pos, tileN);
                pos += tileN;
                offInSeg += tileN;
            }
            offInSeg = 0;
            seg++;
        }
    }

private:
    __aicore__ inline void ComputeBases(uint64_t seg, uint64_t& xBase, uint64_t& yBase)
    {
        xBase = 0;
        yBase = 0;
        if (outerDim_ == 0) {
            return;
        }
        uint64_t rem = seg;
        for (int d = 0; d < outerDim_; ++d) {
            uint64_t stride = 1;
            for (int j = d + 1; j < outerDim_; ++j) {
                stride *= outerShape_[j];
            }
            uint64_t idx = (stride > 0) ? (rem / stride) : 0;
            rem -= idx * stride;
            xBase += idx * xStride_[d];
            yBase += idx * yStride_[d];
        }
    }

    __aicore__ inline void CopyInTensor(LocalTensor<InputT>& local,
                                        GlobalTensor<InputT>& gm, uint64_t offset,
                                        uint32_t n)
    {
        constexpr uint32_t alignElems = 256 / sizeof(InputT);
        uint64_t baseBytes = offset * sizeof(InputT);
        if ((baseBytes % 256 == 0) && (n % alignElems == 0)) {
            DataCopy(local, gm[offset], n);
        } else {
            DataCopyExtParams params;
            params.blockCount = 1;
            params.blockLen = static_cast<uint32_t>(n * sizeof(InputT));
            params.srcStride = 0;
            params.dstStride = 0;
            params.rsv = 0;
            DataCopyPadExtParams<InputT> pad;
            pad.isPad = true;
            pad.leftPadding = 0;
            pad.rightPadding = 0;
            pad.paddingValue = (InputT)0;
            DataCopyPad(local, gm[offset], params, pad);
        }
    }

    __aicore__ inline void ProcessTile(uint64_t xBase, uint64_t yBase,
                                       uint64_t offInSeg, uint64_t zBase, uint32_t n)
    {
        uint32_t compCount = RoundUpTo(n, COMP_ALIGN);

        LocalTensor<InputT> xLocal = inQueueX.AllocTensor<InputT>();
        LocalTensor<InputT> yLocal = inQueueY.AllocTensor<InputT>();
        if (bcastMode_ != 1) {
            CopyInTensor(xLocal, xGm, xBase + offInSeg, n);
        }
        if (bcastMode_ != 2) {
            CopyInTensor(yLocal, yGm, yBase + offInSeg, n);
        }
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);

        LocalTensor<InputT> xIn = inQueueX.DeQue<InputT>();
        LocalTensor<InputT> yIn = inQueueY.DeQue<InputT>();
        LocalTensor<uint8_t> zOut = outQueueZ.AllocTensor<uint8_t>();

        LocalTensor<ComputeT> xc = GetComputeSrcT<ComputeT>(xIn, true, xBase, compCount);
        LocalTensor<ComputeT> yc = GetComputeSrcT<ComputeT>(yIn, false, yBase, compCount);
        ComputeGtT<ComputeT>(zOut, xc, yc, compCount);

        outQueueZ.EnQue<uint8_t>(zOut);
        inQueueX.FreeTensor(xIn);
        inQueueY.FreeTensor(yIn);

        LocalTensor<uint8_t> zLocal = outQueueZ.DeQue<uint8_t>();
        DataCopyExtParams outParams;
        outParams.blockCount = 1;
        outParams.blockLen = n;
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        outParams.rsv = 0;
        DataCopyPad(zGm[zBase], zLocal, outParams);
        outQueueZ.FreeTensor(zLocal);
    }

    template <typename CT>
    __aicore__ inline void ComputeGtT(LocalTensor<uint8_t>& zOut,
                                      LocalTensor<CT>& xc, LocalTensor<CT>& yc,
                                      uint32_t compCount)
    {
        LocalTensor<half> halfOut = halfOutBuf.Get<half>();
        LocalTensor<half> zero = halfZeroBuf.Get<half>();
        LocalTensor<half> one = halfOneBuf.Get<half>();

        if constexpr (IsSameType<CT, int32_t>::value) {
            // gt = (Max(x,y) == x) && (x != y), exact & overflow-safe for int32.
            LocalTensor<int32_t> mx = mxBuf.Get<int32_t>();
            LocalTensor<uint8_t> maskMx = maskMxBuf.Get<uint8_t>();
            LocalTensor<uint8_t> maskEq = maskEqBuf.Get<uint8_t>();
            LocalTensor<half> ne = neBuf.Get<half>();
            Max(mx, xc, yc, static_cast<int32_t>(compCount));
            Compare(maskMx, mx, xc, CMPMODE::EQ, compCount);
            Compare(maskEq, xc, yc, CMPMODE::EQ, compCount);
            // ne = (x != y): bit set (x==y) -> 0, bit clear (x!=y) -> 1.
            // Select semantics: dst = bit ? src0 : src1.
            Select(ne, maskEq, zero, one, SELMODE::VSEL_TENSOR_TENSOR_MODE, compCount);
            // halfOut = (mx==x) ? ne : 0.
            Select(halfOut, maskMx, ne, zero, SELMODE::VSEL_TENSOR_TENSOR_MODE, compCount);
        } else {
            LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
            Compare(mask, xc, yc, CMPMODE::GT, compCount);
            // halfOut = (x>y) ? 1 : 0  (bit set -> src0=one, bit clear -> src1=zero).
            Select(halfOut, mask, one, zero, SELMODE::VSEL_TENSOR_TENSOR_MODE, compCount);
        }

        Cast(zOut, halfOut, RoundMode::CAST_NONE, compCount);
    }

    // Load 1 InputT element from gm[base] into scalarBuf (block-padded).
    __aicore__ inline void LoadScalar(GlobalTensor<InputT>& gm, uint64_t base)
    {
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(sizeof(InputT));
        p.srcStride = 0;
        p.dstStride = 0;
        p.rsv = 0;
        DataCopyPadExtParams<InputT> pad;
        pad.isPad = true;
        pad.leftPadding = 0;
        pad.rightPadding = 0;
        pad.paddingValue = (InputT)0;
        LocalTensor<InputT> sc = scalarBuf.Get<InputT>();
        DataCopyPad(sc, gm[base], p, pad);
    }

    // Return a ComputeT view of an operand: reinterpret (same type), cast
    // (bf16/int8), or Duplicate-fill (innermost-scalar broadcast).
    template <typename CT>
    __aicore__ inline LocalTensor<CT> GetComputeSrcT(LocalTensor<InputT>& in,
                                                     bool isX, uint64_t base,
                                                     uint32_t compCount)
    {
        bool isScalar = (isX && bcastMode_ == 1) || (!isX && bcastMode_ == 2);
        GlobalTensor<InputT>& gm = isX ? xGm : yGm;
        LocalTensor<CT> comp = (isX ? xCompBuf : yCompBuf).Get<CT>();

        if (!isScalar) {
            if constexpr (IsSameType<InputT, CT>::value) {
                return in.ReinterpretCast<CT>();
            } else {
                Cast(comp, in, RoundMode::CAST_NONE, compCount);
                return comp;
            }
        }

        // Scalar broadcast: load 1 element, materialize a CT tile, Duplicate.
        LoadScalar(gm, base);
        LocalTensor<InputT> sc = scalarBuf.Get<InputT>();
        if constexpr (IsSameType<InputT, CT>::value) {
            // half / float / int32 : InputT == CT, no conversion.
            CT s = (CT)sc.GetValue(0);
            Duplicate(comp, s, static_cast<int32_t>(compCount));
        } else if constexpr (IsSameType<CT, half>::value) {
            // int8 -> half : GetValue (syncs MTE2) then convert.
            int8_t v = sc.GetValue(0);
            half s = (half)(float)v;
            Duplicate(comp, s, static_cast<int32_t>(compCount));
        } else {
            // bf16 -> float : backend has no scalar bf16->float, so GetValue
            // (syncs MTE2) the bf16 element, Duplicate a bf16 tile, then Cast.
            InputT v = sc.GetValue(0);
            LocalTensor<InputT> bfTile = bf16TileBuf.Get<InputT>();
            Duplicate(bfTile, v, static_cast<int32_t>(compCount));
            Cast(comp, bfTile, RoundMode::CAST_NONE, compCount);
        }
        return comp;
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<InputT> xGm, yGm;
    GlobalTensor<uint8_t> zGm;

    TBuf<TPosition::VECCALC> maskBuf, halfOutBuf, halfZeroBuf, halfOneBuf;
    TBuf<TPosition::VECCALC> xCompBuf, yCompBuf;
    TBuf<TPosition::VECCALC> mxBuf, neBuf, maskMxBuf, maskEqBuf;
    TBuf<TPosition::VECCALC> bf16TileBuf;
    TBuf<TPosition::VECCALC> scalarBuf, scalarCTBuf;

    uint32_t totalSize_ = 0;
    uint32_t blockDim_ = 1;
    uint32_t innerSize_ = 1;
    uint32_t outerSize_ = 1;
    uint32_t bcastMode_ = 0;
    uint32_t outerDim_ = 0;
    uint32_t outerShape_[8] = {0};
    uint32_t xStride_[8] = {0};
    uint32_t yStride_[8] = {0};
};

extern "C" __global__ __aicore__ void greater(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                              GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelGreater op;
    op.Init(x, y, z, tiling_data.totalSize, tiling_data.blockDim,
            tiling_data.innerSize, tiling_data.outerSize, tiling_data.bcastMode,
            tiling_data.outerDim, tiling_data.outerShape, tiling_data.xStride,
            tiling_data.yStride);
    op.Process();
}
