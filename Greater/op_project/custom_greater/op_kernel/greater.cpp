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
// Sized to use most of the 910B 192KB UB per dtype.
constexpr uint32_t TILE = kIsInt32 ? 4096 :
                          (kIsBf16 ? 6144 :
                           (kIsFloat ? 5120 :
            (kIsInt8 ? 10240 : 9216)));  // fp16 -> 9216
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

        // ---- P1: detect outer-broadcast resident operand ----
        // bcastMode==0 (innermost non-broadcast), outerDim>0, all outer strides
        // of one operand are 0 => that operand's data is a single innerSize-elem
        // block reused across every segment. innerSize must be 256-aligned so
        // per-tile UB slice reads stay 256-aligned, and fit in UB.
        constexpr uint32_t RES_UB_LIMIT = 96 * 1024;  // bytes; leave room for other bufs
        xResident_ = false;
        yResident_ = false;
        if (bcastMode_ == 0 && outerDim_ > 0 && (innerSize_ % COMP_ALIGN) == 0
            && innerSize_ <= TILE) {
            bool xAllZero = true, yAllZero = true;
            for (int d = 0; d < static_cast<int>(outerDim_); ++d) {
                if (xStride_[d] != 0) { xAllZero = false; }
                if (yStride_[d] != 0) { yAllZero = false; }
            }
            uint32_t resBytes = (innerSize_ + COMP_ALIGN) * sizeof(InputT);
            if (xAllZero && resBytes <= RES_UB_LIMIT) {
                xResident_ = true;
                residentElemsX_ = innerSize_ + COMP_ALIGN;
            }
            if (yAllZero && resBytes <= RES_UB_LIMIT) {
                yResident_ = true;
                residentElemsY_ = innerSize_ + COMP_ALIGN;
            }
            // Both resident (both inputs constant) is degenerate; stream x.
            if (xResident_ && yResident_) {
                xResident_ = false;
                residentElemsX_ = 0;
            }
        }
        // An operand uses its input queue iff it is neither resident nor scalar.
        xQueued_ = !xResident_ && (bcastMode_ != 1);
        yQueued_ = !yResident_ && (bcastMode_ != 2);

        // ---- P2: detect innermost-broadcast (scalar per segment) ----
        // The scalar operand's outerSize values are contiguous in GM. Batch-load
        // them once; process big tiles with per-sub-segment materialization.
        // Only enable when the scalar operand is contiguous (stride 1) or a
        // single constant (stride 0); other strides fall back to segment walk.
        innerBcast_ = false;
        if ((bcastMode_ == 1 || bcastMode_ == 2) && outerDim_ > 0
            && (innerSize_ % COMP_ALIGN) == 0 && innerSize_ <= TILE) {
            uint32_t sStr = (bcastMode_ == 1) ? xStride_[0] : yStride_[0];
            if (sStr == 0 || sStr == 1) {
                uint32_t batchCount = (sStr == 0) ? 1u : outerSize_;
                uint32_t batchBytes = (batchCount + COMP_ALIGN) * sizeof(InputT);
                if (batchBytes <= 64 * 1024) {
                    innerBcast_ = true;
                    scalarStride_ = sStr;
                    scalarBatchCount_ = batchCount;
                    scalarBatchElems_ = batchCount + COMP_ALIGN;
                }
            }
        }

        xGm.SetGlobalBuffer((__gm__ InputT*)x);
        yGm.SetGlobalBuffer((__gm__ InputT*)y);
        zGm.SetGlobalBuffer((__gm__ uint8_t*)z);

        // Only allocate the input queue an operand actually uses; the freed UB
        // is used for the resident buffer when applicable.
        if (xQueued_) {
            pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE * sizeof(InputT));
        }
        if (yQueued_) {
            pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE * sizeof(InputT));
        }
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
        if (xResident_) {
            pipe.InitBuffer(residentXBuf, residentElemsX_ * sizeof(InputT));
        }
        if (yResident_) {
            pipe.InitBuffer(residentYBuf, residentElemsY_ * sizeof(InputT));
        }
        if (innerBcast_) {
            pipe.InitBuffer(scalarBatchBuf, scalarBatchElems_ * sizeof(InputT));
        }

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
        // P1+: when an operand is outer-broadcast resident, the other operand
        // and the output are fully contiguous. Flatten to big TILE tiles and
        // reuse the resident operand across innerSize sub-tiles (no per-segment
        // HBM read, no small-tile scalar overhead).
        if (xResident_ || yResident_) {
            ProcessResident();
            return;
        }
        // P2: innermost-broadcast (scalar per segment). Batch-load scalars and
        // process big tiles with per-sub-segment materialization.
        if (innerBcast_) {
            ProcessInnerBcast();
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

    // P1+: flattened resident path. The streamed operand (the one NOT resident)
    // and the output are contiguous in GM; the resident operand is innerSize-
    // periodic. Split work by segments (innerSize-aligned) so each big tile
    // starts on a period boundary, then process big TILE tiles where each tile
    // is a sequence of full innerSize sub-tiles all compared against the same
    // resident slice. This removes both the redundant HBM reads (P1) and the
    // small-tile scalar overhead (P1+).
    __aicore__ inline void ProcessResident()
    {
        LoadResidents();
        uint32_t coreId = GetBlockIdx();
        uint64_t totalSegs = static_cast<uint64_t>(outerSize_);
        uint64_t segStart = totalSegs * coreId / blockDim_;
        uint64_t segEnd = totalSegs * (coreId + 1) / blockDim_;
        uint64_t pos = segStart * innerSize_;
        uint64_t coreEnd = segEnd * innerSize_;
        if (coreEnd > totalSize_) {
            coreEnd = totalSize_;
        }
        if (pos >= coreEnd) {
            return;
        }
        while (pos < coreEnd) {
            // Big tile = largest multiple of innerSize that fits in TILE, so
            // every tile starts on a y-period boundary (pos stays innerSize-aligned).
            uint64_t tileN64 = (static_cast<uint64_t>(TILE) / innerSize_) * innerSize_;
            if (tileN64 == 0) {
                tileN64 = innerSize_;
            }
            if (coreEnd - pos < tileN64) {
                tileN64 = coreEnd - pos;
            }
            uint32_t tileN = static_cast<uint32_t>(tileN64);
            ProcessResidentTile(pos, tileN);
            pos += tileN;
        }
    }

    // Process one big TILE tile at output offset `pos` (innerSize-aligned).
    // The streamed operand is loaded once via its queue; the resident operand
    // is reused from UB for every innerSize sub-tile.
    __aicore__ inline void ProcessResidentTile(uint64_t zBase, uint32_t n)
    {
        uint32_t compCount = RoundUpTo(n, COMP_ALIGN);

        // Stream the non-resident operand. (For broadcast, exactly one operand
        // is resident; the other is streamed. If both were resident the output
        // is constant -- still correct, x is treated as the streamed one.)
        bool streamX = !xResident_;
        LocalTensor<InputT> sIn;
        if (streamX) {
            LocalTensor<InputT> sLocal = inQueueX.AllocTensor<InputT>();
            CopyInTensor(sLocal, xGm, zBase, n);
            inQueueX.EnQue(sLocal);
            sIn = inQueueX.DeQue<InputT>();
        } else {
            LocalTensor<InputT> sLocal = inQueueY.AllocTensor<InputT>();
            CopyInTensor(sLocal, yGm, zBase, n);
            inQueueY.EnQue(sLocal);
            sIn = inQueueY.DeQue<InputT>();
        }

        LocalTensor<uint8_t> zOut = outQueueZ.AllocTensor<uint8_t>();

        // ComputeT view of the streamed tile (cast if dtype requires it).
        LocalTensor<ComputeT> sc;
        if constexpr (IsSameType<InputT, ComputeT>::value) {
            sc = sIn.ReinterpretCast<ComputeT>();
        } else {
            sc = xCompBuf.Get<ComputeT>();
            Cast(sc, sIn, RoundMode::CAST_NONE, compCount);
        }
        // ComputeT view of the resident operand (the full innerSize block).
        LocalTensor<InputT> resRaw = (xResident_ ? residentXBuf : residentYBuf)
            .Get<InputT>();
        LocalTensor<ComputeT> rc;
        if constexpr (IsSameType<InputT, ComputeT>::value) {
            rc = resRaw.ReinterpretCast<ComputeT>();
        } else {
            rc = yCompBuf.Get<ComputeT>();
            Cast(rc, resRaw, RoundMode::CAST_NONE, RoundUpTo(innerSize_, COMP_ALIGN));
        }

        // Sub-tile loop: each sub-tile is one full innerSize period, compared
        // against the resident slice. pos is innerSize-aligned (segment start)
        // and n is a multiple of innerSize, so every sub-tile uses resRaw[0:innerSize].
        // sc = streamed operand's ComputeT, rc = resident operand's ComputeT;
        // map them to x/y correctly (Greater is x > y, order matters).
        uint32_t off = 0;
        while (off < n) {
            uint32_t subN = innerSize_;
            if (n - off < subN) {
                subN = n - off;
            }
            uint32_t subComp = RoundUpTo(subN, COMP_ALIGN);
            LocalTensor<uint8_t> zSub = zOut[off];
            LocalTensor<ComputeT> xSub = streamX ? sc[off] : rc;
            LocalTensor<ComputeT> ySub = streamX ? rc : sc[off];
            ComputeGtT<ComputeT>(zSub, xSub, ySub, subComp);
            off += subN;
        }

        outQueueZ.EnQue<uint8_t>(zOut);
        if (streamX) { inQueueX.FreeTensor(sIn); } else { inQueueY.FreeTensor(sIn); }

        LocalTensor<uint8_t> zLocal = outQueueZ.DeQue<uint8_t>();
        if ((zBase % 256 == 0) && (n % 256 == 0)) {
            DataCopy(zGm[zBase], zLocal, n);
        } else {
            DataCopyExtParams outParams;
            outParams.blockCount = 1;
            outParams.blockLen = n;
            outParams.srcStride = 0;
            outParams.dstStride = 0;
            outParams.rsv = 0;
            DataCopyPad(zGm[zBase], zLocal, outParams);
        }
        outQueueZ.FreeTensor(zLocal);
    }

    // P2: innermost-broadcast flatten. The scalar operand (x for bcastMode==1,
    // y for bcastMode==2) has outerSize contiguous values; the streamed operand
    // and output are contiguous. Batch-load the scalars once, then process big
    // TILE tiles where each innerSize sub-tile materializes its segment's scalar
    // from the UB batch (no per-segment LoadScalar MTE2, ~TILE/innerSize fewer
    // streamed-operand queue ops).
    __aicore__ inline void ProcessInnerBcast()
    {
        LoadScalarBatch();
        uint32_t coreId = GetBlockIdx();
        uint64_t totalSegs = static_cast<uint64_t>(outerSize_);
        uint64_t segStart = totalSegs * coreId / blockDim_;
        uint64_t segEnd = totalSegs * (coreId + 1) / blockDim_;
        uint64_t pos = segStart * innerSize_;
        uint64_t coreEnd = segEnd * innerSize_;
        if (coreEnd > totalSize_) {
            coreEnd = totalSize_;
        }
        if (pos >= coreEnd) {
            return;
        }
        while (pos < coreEnd) {
            uint64_t tileN64 = (static_cast<uint64_t>(TILE) / innerSize_) * innerSize_;
            if (tileN64 == 0) {
                tileN64 = innerSize_;
            }
            if (coreEnd - pos < tileN64) {
                tileN64 = coreEnd - pos;
            }
            uint32_t tileN = static_cast<uint32_t>(tileN64);
            ProcessInnerBcastTile(pos, tileN);
            pos += tileN;
        }
    }

    __aicore__ inline void ProcessInnerBcastTile(uint64_t zBase, uint32_t n)
    {
        ProcessInnerBcastTileT<ComputeT>(zBase, n);
    }

    template <typename CT>
    __aicore__ inline void ProcessInnerBcastTileT(uint64_t zBase, uint32_t n)
    {
        uint32_t compCount = RoundUpTo(n, COMP_ALIGN);
        // Streamed operand = the non-scalar one (x for bcastMode==2, y for ==1).
        bool streamX = (bcastMode_ != 1);
        LocalTensor<InputT> sIn;
        if (streamX) {
            LocalTensor<InputT> sLocal = inQueueX.AllocTensor<InputT>();
            CopyInTensor(sLocal, xGm, zBase, n);
            inQueueX.EnQue(sLocal);
            sIn = inQueueX.DeQue<InputT>();
        } else {
            LocalTensor<InputT> sLocal = inQueueY.AllocTensor<InputT>();
            CopyInTensor(sLocal, yGm, zBase, n);
            inQueueY.EnQue(sLocal);
            sIn = inQueueY.DeQue<InputT>();
        }

        LocalTensor<uint8_t> zOut = outQueueZ.AllocTensor<uint8_t>();

        // ComputeT view of the streamed tile.
        LocalTensor<CT> sc;
        if constexpr (IsSameType<InputT, CT>::value) {
            sc = sIn.ReinterpretCast<CT>();
        } else {
            sc = (streamX ? xCompBuf : yCompBuf).Get<CT>();
            Cast(sc, sIn, RoundMode::CAST_NONE, compCount);
        }
        // Buffer for the scalar operand's materialized ComputeT sub-tile.
        LocalTensor<CT> scSubBuf = (streamX ? yCompBuf : xCompBuf).Get<CT>();
        LocalTensor<InputT> batch = scalarBatchBuf.Get<InputT>();

        // Sub-tile loop: each sub-tile = one segment (innerSize elems). The
        // scalar for segment `seg` is batch[seg] (stride 1) or batch[0] (const).
        uint64_t seg = zBase / innerSize_;
        uint32_t off = 0;
        while (off < n) {
            uint32_t subN = innerSize_;
            if (n - off < subN) {
                subN = n - off;
            }
            uint32_t subComp = RoundUpTo(subN, COMP_ALIGN);
            uint32_t scalarIdx = (scalarStride_ == 0) ? 0u : static_cast<uint32_t>(seg);

            // Materialize the segment's scalar into a ComputeT sub-tile.
            if constexpr (IsSameType<InputT, CT>::value) {
                CT sv = (CT)batch.GetValue(scalarIdx);
                Duplicate(scSubBuf, sv, static_cast<int32_t>(subComp));
            } else if constexpr (IsSameType<CT, half>::value) {
                // int8 -> half
                int8_t v = batch.GetValue(scalarIdx);
                half sv = (half)(float)v;
                Duplicate(scSubBuf, sv, static_cast<int32_t>(subComp));
            } else {
                // bf16 -> float
                InputT v = batch.GetValue(scalarIdx);
                LocalTensor<InputT> bfTile = bf16TileBuf.Get<InputT>();
                Duplicate(bfTile, v, static_cast<int32_t>(subComp));
                Cast(scSubBuf, bfTile, RoundMode::CAST_NONE, subComp);
            }

            LocalTensor<uint8_t> zSub = zOut[off];
            LocalTensor<CT> sSub = sc[off];
            LocalTensor<CT> xc = streamX ? sSub : scSubBuf;   // x = streamed
            LocalTensor<CT> yc = streamX ? scSubBuf : sSub;  // y = scalar
            ComputeGtT<CT>(zSub, xc, yc, subComp);
            off += subN;
            seg++;
        }

        outQueueZ.EnQue<uint8_t>(zOut);
        if (streamX) { inQueueX.FreeTensor(sIn); } else { inQueueY.FreeTensor(sIn); }

        LocalTensor<uint8_t> zLocal = outQueueZ.DeQue<uint8_t>();
        if ((zBase % 256 == 0) && (n % 256 == 0)) {
            DataCopy(zGm[zBase], zLocal, n);
        } else {
            DataCopyExtParams outParams;
            outParams.blockCount = 1;
            outParams.blockLen = n;
            outParams.srcStride = 0;
            outParams.dstStride = 0;
            outParams.rsv = 0;
            DataCopyPad(zGm[zBase], zLocal, outParams);
        }
        outQueueZ.FreeTensor(zLocal);
    }

private:
    // P1: load each resident operand's innerSize elements from GM into its UB
    // buffer once, then sync MTE2->V so all later tiles may read it directly.
    __aicore__ inline void LoadResidents()
    {
        if (!xResident_ && !yResident_) {
            return;
        }
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(innerSize_ * sizeof(InputT));
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;
        DataCopyPadExtParams<InputT> pad;
        pad.isPad = true;
        pad.leftPadding = 0;
        pad.rightPadding = 0;
        pad.paddingValue = (InputT)0;
        if (xResident_) {
            LocalTensor<InputT> rx = residentXBuf.Get<InputT>();
            DataCopyPad(rx, xGm[0], params, pad);
        }
        if (yResident_) {
            LocalTensor<InputT> ry = residentYBuf.Get<InputT>();
            DataCopyPad(ry, yGm[0], params, pad);
        }
        TEventID eid = pipe.AllocEventID<HardEvent::MTE2_V>();
        SetFlag<HardEvent::MTE2_V>(eid);
        WaitFlag<HardEvent::MTE2_V>(eid);
        pipe.ReleaseEventID<HardEvent::MTE2_V>(eid);
    }

    // P2: load all distinct scalars of the innermost-broadcast operand once.
    // scalarStride_==0 -> a single constant (1 elem); ==1 -> outerSize contig.
    __aicore__ inline void LoadScalarBatch()
    {
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(scalarBatchCount_ * sizeof(InputT));
        params.srcStride = 0;
        params.dstStride = 0;
        params.rsv = 0;
        DataCopyPadExtParams<InputT> pad;
        pad.isPad = true;
        pad.leftPadding = 0;
        pad.rightPadding = 0;
        pad.paddingValue = (InputT)0;
        LocalTensor<InputT> batch = scalarBatchBuf.Get<InputT>();
        GlobalTensor<InputT>& gm = (bcastMode_ == 1) ? xGm : yGm;  // scalar operand
        DataCopyPad(batch, gm[0], params, pad);
        TEventID eid = pipe.AllocEventID<HardEvent::MTE2_V>();
        SetFlag<HardEvent::MTE2_V>(eid);
        WaitFlag<HardEvent::MTE2_V>(eid);
        pipe.ReleaseEventID<HardEvent::MTE2_V>(eid);
    }

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

        // Stream the operands that use a queue (MTE2). Resident operands are
        // already in UB (loaded once in LoadResidents); scalar operands are
        // materialized per-tile inside GetComputeSrcT. Both skip the queue.
        LocalTensor<InputT> xIn;   // valid only when xQueued_
        LocalTensor<InputT> yIn;   // valid only when yQueued_
        if (xQueued_) {
            LocalTensor<InputT> xLocal = inQueueX.AllocTensor<InputT>();
            CopyInTensor(xLocal, xGm, xBase + offInSeg, n);
            inQueueX.EnQue(xLocal);
            xIn = inQueueX.DeQue<InputT>();
        }
        if (yQueued_) {
            LocalTensor<InputT> yLocal = inQueueY.AllocTensor<InputT>();
            CopyInTensor(yLocal, yGm, yBase + offInSeg, n);
            inQueueY.EnQue(yLocal);
            yIn = inQueueY.DeQue<InputT>();
        }

        LocalTensor<uint8_t> zOut = outQueueZ.AllocTensor<uint8_t>();

        LocalTensor<ComputeT> xc = GetComputeSrcT<ComputeT>(true, xIn, offInSeg, xBase, compCount);
        LocalTensor<ComputeT> yc = GetComputeSrcT<ComputeT>(false, yIn, offInSeg, yBase, compCount);
        ComputeGtT<ComputeT>(zOut, xc, yc, compCount);

        outQueueZ.EnQue<uint8_t>(zOut);
        if (xQueued_) { inQueueX.FreeTensor(xIn); }
        if (yQueued_) { inQueueY.FreeTensor(yIn); }

        LocalTensor<uint8_t> zLocal = outQueueZ.DeQue<uint8_t>();
        if ((zBase % 256 == 0) && (n % 256 == 0)) {
            DataCopy(zGm[zBase], zLocal, n);
        } else {
            DataCopyExtParams outParams;
            outParams.blockCount = 1;
            outParams.blockLen = n;
            outParams.srcStride = 0;
            outParams.dstStride = 0;
            outParams.rsv = 0;
            DataCopyPad(zGm[zBase], zLocal, outParams);
        }
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

    // Return a ComputeT view of an operand. Source is one of:
    //  - resident UB slice (outer-broadcast operand loaded once): read directly
    //  - streamed queue tile (the common path)
    //  - scalar (innermost-broadcast): LoadScalar + Duplicate per tile
    template <typename CT>
    __aicore__ inline LocalTensor<CT> GetComputeSrcT(bool isX,
                                                     LocalTensor<InputT>& queued,
                                                     uint64_t offInSeg,
                                                     uint64_t base,
                                                     uint32_t compCount)
    {
        bool isScalar = (isX && bcastMode_ == 1) || (!isX && bcastMode_ == 2);
        bool isResident = isX ? xResident_ : yResident_;
        GlobalTensor<InputT>& gm = isX ? xGm : yGm;
        LocalTensor<CT> comp = (isX ? xCompBuf : yCompBuf).Get<CT>();

        if (isScalar) {
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

        // Resident: read the UB slice directly (V-side, no MTE2). offInSeg is
        // 256-aligned (resident requires innerSize % 256 == 0). Streamed: use
        // the queued tile (data at offset 0).
        if (isResident) {
            LocalTensor<InputT> src = (isX ? residentXBuf : residentYBuf)
                .Get<InputT>()[static_cast<uint32_t>(offInSeg)];
            if constexpr (IsSameType<InputT, CT>::value) {
                return src.ReinterpretCast<CT>();
            } else {
                Cast(comp, src, RoundMode::CAST_NONE, compCount);
                return comp;
            }
        }

        if constexpr (IsSameType<InputT, CT>::value) {
            return queued.ReinterpretCast<CT>();
        } else {
            Cast(comp, queued, RoundMode::CAST_NONE, compCount);
            return comp;
        }
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
    TBuf<TPosition::VECCALC> residentXBuf, residentYBuf;  // P1 broadcast-resident
    TBuf<TPosition::VECCALC> scalarBatchBuf;             // P2 innermost-bcast scalars

    uint32_t totalSize_ = 0;
    uint32_t blockDim_ = 1;
    uint32_t innerSize_ = 1;
    uint32_t outerSize_ = 1;
    uint32_t bcastMode_ = 0;
    uint32_t outerDim_ = 0;
    uint32_t outerShape_[8] = {0};
    uint32_t xStride_[8] = {0};
    uint32_t yStride_[8] = {0};
    // ---- P1: outer-broadcast operand resident in UB ----
    // An operand whose outer-dim strides are ALL 0 is constant across segments
    // (its unique data is just innerSize contiguous elements). Load it ONCE into
    // UB and let every tile read it directly (V-side), eliminating the per-tile
    // redundant HBM reads that make broadcast cases ~12x slower than same-shape.
    bool xResident_ = false;          // x loaded once into residentXBuf
    bool yResident_ = false;          // y loaded once into residentYBuf
    bool xQueued_ = true;            // x uses inQueueX (false if resident or scalar)
    bool yQueued_ = true;            // y uses inQueueY
    uint32_t residentElemsX_ = 0;     // capacity of residentXBuf (elems, 256-aligned)
    uint32_t residentElemsY_ = 0;
    // ---- P2: innermost-broadcast scalar batch ----
    // bcastMode 1/2: one operand is a per-segment scalar (outerSize values).
    // Load all scalars once, process big TILE tiles with per-sub-segment scalar
    // materialization (no per-segment LoadScalar MTE2, far fewer queue ops).
    bool innerBcast_ = false;
    uint32_t scalarBatchElems_ = 0;
    uint32_t scalarStride_ = 0;       // P2: scalar operand's outer stride (0=const,1=contig)
    uint32_t scalarBatchCount_ = 0;    // P2: distinct scalars to load
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
