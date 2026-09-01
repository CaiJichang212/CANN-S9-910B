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

// DAV_2201 exposes 192KiB physical UB, but the basic APIs reserve the final
// 8KiB from offset 184KiB as temporary space. P2 allocates one input queue,
// one output queue, comparison scratch, and a dtype-specific scalar batch.
constexpr uint32_t USER_UB_LIMIT_BYTES = 184 * 1024;
constexpr uint32_t P2_BATCH_LIMIT_BYTES = kIsBf16 ? 48 * 1024 :
                                          (kIsInt8 ? 60 * 1024 : 64 * 1024);
constexpr uint32_t P2_COMP_BUFFER_COUNT = kIsBf16 ? 2 :
                                           ((kIsInt8 || kIsInt32) ? 1 : 0);
constexpr uint32_t P2_DTYPE_EXTRA_BYTES = kIsInt32
    ? (TILE * sizeof(int32_t) + TILE * sizeof(half) + 2 * (TILE / 8))
    : (kIsBf16 ? TILE * sizeof(bfloat16_t) : 0);
constexpr uint32_t P2_BRCB_BYTES = 1024;
constexpr uint32_t P2_ROW_EXTRA_BYTES = kIsHalf ? P2_BRCB_BYTES :
    (kIsFloat ? P2_BRCB_BYTES + TILE * sizeof(float) : 0);
constexpr uint32_t P2_FIXED_UB_BYTES =
    2 * TILE * sizeof(InputT) +              // one double-buffered input queue
    2 * TILE * sizeof(uint8_t) +             // double-buffered bool output
    TILE / 8 * sizeof(uint8_t) +             // Compare mask
    3 * TILE * sizeof(half) +                // halfOut, zero and one
    P2_COMP_BUFFER_COUNT * TILE * sizeof(ComputeT) +
    P2_DTYPE_EXTRA_BYTES + P2_ROW_EXTRA_BYTES;
static_assert(P2_FIXED_UB_BYTES + P2_BATCH_LIMIT_BYTES <= USER_UB_LIMIT_BYTES,
              "P2 buffers exceed the DAV_2201 user UB region");
constexpr uint32_t P1_LARGE_FIXED_UB_BYTES =
    2 * TILE * sizeof(InputT) +                    // one double-buffered stream queue
    2 * TILE * sizeof(uint8_t) +                   // double-buffered bool output
    TILE / 8 * sizeof(uint8_t) +                   // Compare mask
    3 * TILE * sizeof(half) +                      // halfOut, zero and one
    2 * TILE * sizeof(ComputeT) +                  // stream and resident compute views
    P2_DTYPE_EXTRA_BYTES +                         // int32/bf16 dtype-specific buffers
    (TILE + COMP_ALIGN) * sizeof(InputT);          // one aligned resident slice
static_assert(P1_LARGE_FIXED_UB_BYTES <= USER_UB_LIMIT_BYTES,
              "large P1 resident buffers exceed the DAV_2201 user UB region");

__aicore__ inline uint32_t RoundUpTo(uint32_t n, uint32_t a)
{
    uint64_t rounded = (static_cast<uint64_t>(n) + a - 1) / a * a;
    return rounded > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(rounded);
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
        if (totalSize_ == 0) {
            return;
        }

        // ---- P1: detect outer-broadcast resident operand ----
        // A zero-stride suffix in an outer broadcast operand means that its
        // innerSize-element block can stay in UB for a whole group of segments.
        // The other operand must be contiguous over that group. This handles
        // both the old full outer broadcast case and partial cases such as
        // [B,M,N] x [B,1,N] without shape-specific special casing.
        constexpr uint32_t RES_UB_LIMIT = 96 * 1024;  // bytes; leave room for other bufs
        xResident_ = false;
        yResident_ = false;
        largeResident_ = false;
        rowElems_ = RoundUpTo(innerSize_, COMP_ALIGN);
        // Vector Compare requires 256-element alignment, whereas DataCopyPad
        // only guarantees a 32-byte row boundary.  For a non-aligned logical
        // row we therefore stage each row in a 256-element slot. The generic
        // path already rounds every individual row to the same vector width;
        // batching short rows removes repeated address and queue overhead
        // without increasing the total vector work.
        rowPadded_ = (innerSize_ % COMP_ALIGN) != 0 && innerSize_ <= TILE &&
                     rowElems_ <= TILE;
        if (bcastMode_ == 0 && outerDim_ > 0 &&
            ((innerSize_ % COMP_ALIGN) == 0 || rowPadded_) && innerSize_ <= TILE) {
            uint32_t residentElems = rowPadded_ ? rowElems_ : (innerSize_ + COMP_ALIGN);
            uint32_t resBytes = residentElems * sizeof(InputT);
            if (resBytes <= RES_UB_LIMIT) {
                uint32_t xGroups = GetResidentGroupSegs(true);
                uint32_t yGroups = GetResidentGroupSegs(false);
                // Prefer the longer reuse run. Keep y on ties for stable
                // selection and to preserve the original full-broadcast path.
                if (xGroups > yGroups) {
                    xResident_ = true;
                    residentGroupSegs_ = xGroups;
                    residentElemsX_ = residentElems;
                } else if (yGroups > 1) {
                    yResident_ = true;
                    residentGroupSegs_ = yGroups;
                    residentElemsY_ = residentElems;
                }
            }
        }
        // Full outer broadcast with an inner row larger than TILE. The Host has
        // selected a product of inner and outer workers; each core only keeps a
        // TILE-sized resident slice in UB.
        if (!xResident_ && !yResident_ && bcastMode_ == 0 && outerDim_ > 0 &&
            outerSize_ > 1 && innerSize_ > TILE) {
            uint32_t xGroups = GetResidentGroupSegs(true);
            uint32_t yGroups = GetResidentGroupSegs(false);
            uint32_t residentElems = TILE + COMP_ALIGN;
            if (xGroups == outerSize_) {
                xResident_ = true;
                largeResident_ = true;
                residentGroupSegs_ = outerSize_;
                residentElemsX_ = residentElems;
            } else if (yGroups == outerSize_) {
                yResident_ = true;
                largeResident_ = true;
                residentGroupSegs_ = outerSize_;
                residentElemsY_ = residentElems;
            }
        }
        // An operand uses its input queue iff it is neither resident nor scalar.
        xQueued_ = !xResident_ && (bcastMode_ != 1);
        yQueued_ = !yResident_ && (bcastMode_ != 2);

        // ---- P2: detect innermost-broadcast (scalar per segment) ----
        // Batch-load the scalar operand's complete contiguous storage range once.
        // The scalar used by each output segment is derived from all outer
        // broadcast strides, so this also handles mixed outer broadcasts such as
        // [B,1,1] and [1,M,1], not only stride[0] == 0/1 layouts.  The streamed
        // operand must still be dense because the P2 loop addresses it by the
        // flattened output offset.
        innerBcast_ = false;
        scalarBatchBlocked_ = false;
        const uint32_t* streamStrides = (bcastMode_ == 1) ? yStride_ : xStride_;
        if ((bcastMode_ == 1 || bcastMode_ == 2) &&
            ((innerSize_ % COMP_ALIGN) == 0 || rowPadded_) && innerSize_ <= TILE &&
            IsStreamIndexContinuous(streamStrides)) {
            uint64_t maxScalarOffset = 0;
            const uint32_t* scalarStrides = (bcastMode_ == 1) ? xStride_ : yStride_;
            for (uint32_t d = 0; d < outerDim_; ++d) {
                maxScalarOffset += static_cast<uint64_t>(outerShape_[d] - 1) * scalarStrides[d];
            }
            uint64_t batchCount = maxScalarOffset + 1;
            // For scalarIndex(seg)==seg each core only consumes its own
            // contiguous segment range.  This removes both the 64KiB cliff
            // (fp32 [16384,1024]x[16384,1]) and redundant all-core reads.
            scalarBatchPerCore_ = IsScalarIndexContinuous(scalarStrides);
            uint64_t allocCount = batchCount;
            if (scalarBatchPerCore_) {
                allocCount = (static_cast<uint64_t>(outerSize_) + blockDim_ - 1) / blockDim_;
            }
            uint64_t batchBytes = (allocCount + COMP_ALIGN) * sizeof(InputT);
            if (allocCount <= UINT32_MAX && batchBytes <= P2_BATCH_LIMIT_BYTES) {
                innerBcast_ = true;
                scalarBatchCount_ = static_cast<uint32_t>(allocCount);
                scalarBatchElems_ = scalarBatchCount_ + COMP_ALIGN;
            } else if constexpr (kIsHalf || kIsFloat) {
                if (scalarBatchPerCore_ && rowPadded_ && rowElems_ == COMP_ALIGN) {
                    // Stream a small aligned scalar block per padded row tile
                    // instead of rejecting P2 when the whole core range is large.
                    innerBcast_ = true;
                    scalarBatchBlocked_ = true;
                    scalarBatchCount_ = kIsHalf ? 32 : 16;
                    scalarBatchElems_ = scalarBatchCount_ + COMP_ALIGN;
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
        if (!innerBcast_ || kIsBf16) {
            pipe.InitBuffer(xCompBuf, TILE * sizeof(ComputeT));
            pipe.InitBuffer(yCompBuf, TILE * sizeof(ComputeT));
        } else if constexpr (kIsInt8) {
            if (bcastMode_ == 1) {
                pipe.InitBuffer(yCompBuf, TILE * sizeof(ComputeT));
            } else {
                pipe.InitBuffer(xCompBuf, TILE * sizeof(ComputeT));
            }
        } else if constexpr (kIsInt32) {
            if (bcastMode_ == 1) {
                pipe.InitBuffer(xCompBuf, TILE * sizeof(ComputeT));
            } else {
                pipe.InitBuffer(yCompBuf, TILE * sizeof(ComputeT));
            }
        }
        if constexpr (kIsInt32) {
            pipe.InitBuffer(mxBuf, TILE * sizeof(int32_t));
            pipe.InitBuffer(neBuf, TILE * sizeof(half));
            pipe.InitBuffer(maskMxBuf, TILE / 8 * sizeof(uint8_t));
            pipe.InitBuffer(maskEqBuf, TILE / 8 * sizeof(uint8_t));
        }
        if constexpr (kIsBf16) {
            pipe.InitBuffer(bf16TileBuf, TILE * sizeof(bfloat16_t));
        }
        if (!innerBcast_ && (bcastMode_ == 1 || bcastMode_ == 2)) {
            pipe.InitBuffer(scalarBuf, 256);
        }
        if (xResident_) {
            pipe.InitBuffer(residentXBuf, residentElemsX_ * sizeof(InputT));
        }
        if (yResident_) {
            pipe.InitBuffer(residentYBuf, residentElemsY_ * sizeof(InputT));
        }
        if (innerBcast_) {
            pipe.InitBuffer(scalarBatchBuf, scalarBatchElems_ * sizeof(InputT));
            if constexpr (kIsHalf || kIsFloat) {
                if (rowPadded_ && rowElems_ == COMP_ALIGN && scalarBatchPerCore_) {
                    pipe.InitBuffer(scalarBrcbBuf, P2_BRCB_BYTES);
                    if constexpr (kIsFloat) {
                        pipe.InitBuffer(scalarRowsBuf, TILE * sizeof(float));
                    }
                }
            }
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
            if (largeResident_) {
                ProcessLargeResident();
                return;
            }
            if (rowPadded_) {
                ProcessResidentPadded();
                return;
            }
            ProcessResident();
            return;
        }
        // P2: innermost-broadcast (scalar per segment). Batch-load scalars and
        // process big tiles with per-sub-segment materialization.
        if (innerBcast_) {
            if (rowPadded_) {
                ProcessInnerBcastPadded();
                return;
            }
            ProcessInnerBcast();
            return;
        }
        uint32_t coreId = GetBlockIdx();

        uint64_t totalBlks = (static_cast<uint64_t>(totalSize_) + Z_BLKELEMS - 1) / Z_BLKELEMS;
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

    // Full-outer P1 with innerSize > TILE. A core owns the Cartesian product of
    // one aligned inner slice and one outer segment range. This keeps output
    // ownership disjoint while reusing the resident slice across assigned rows.
    __aicore__ inline void ProcessLargeResident()
    {
        uint32_t innerTiles = (innerSize_ - 1) / TILE + 1;
        uint32_t innerWorkers = innerTiles < blockDim_ ? innerTiles : blockDim_;
        if (innerWorkers == 0) {
            return;
        }
        uint32_t outerWorkers = blockDim_ / innerWorkers;
        if (outerWorkers == 0) {
            return;
        }

        uint32_t coreId = GetBlockIdx();
        uint32_t innerWorker = coreId % innerWorkers;
        uint32_t outerWorker = coreId / innerWorkers;
        if (outerWorker >= outerWorkers) {
            return;
        }

        uint64_t rawStart = static_cast<uint64_t>(innerSize_) * innerWorker / innerWorkers;
        uint64_t rawEnd = static_cast<uint64_t>(innerSize_) * (innerWorker + 1) / innerWorkers;
        uint32_t sliceStart = innerWorker == 0
            ? 0 : RoundUpTo(static_cast<uint32_t>(rawStart), COMP_ALIGN);
        uint32_t sliceEnd = innerWorker + 1 == innerWorkers
            ? innerSize_ : RoundUpTo(static_cast<uint32_t>(rawEnd), COMP_ALIGN);
        if (sliceEnd > innerSize_) {
            sliceEnd = innerSize_;
        }
        if (sliceStart >= sliceEnd) {
            return;
        }

        uint64_t segStart = static_cast<uint64_t>(outerSize_) * outerWorker / outerWorkers;
        uint64_t segEnd = static_cast<uint64_t>(outerSize_) * (outerWorker + 1) / outerWorkers;
        for (uint32_t off = sliceStart; off < sliceEnd;) {
            uint32_t n = sliceEnd - off;
            if (n > TILE) {
                n = TILE;
            }
            LoadResidentSlice(off, n);
            for (uint64_t seg = segStart; seg < segEnd; ++seg) {
                uint64_t base = seg * innerSize_ + off;
                ProcessLargeResidentTile(base, base, n);
            }
            SyncVToMte2();
            off += n;
        }
    }

    __aicore__ inline void ProcessLargeResidentTile(uint64_t zBase, uint64_t streamBase,
                                                    uint32_t n)
    {
        uint32_t compCount = RoundUpTo(n, COMP_ALIGN);
        bool streamX = !xResident_;
        LocalTensor<InputT> sIn;
        if (streamX) {
            LocalTensor<InputT> sLocal = inQueueX.AllocTensor<InputT>();
            CopyInTensor(sLocal, xGm, streamBase, n);
            inQueueX.EnQue(sLocal);
            sIn = inQueueX.DeQue<InputT>();
        } else {
            LocalTensor<InputT> sLocal = inQueueY.AllocTensor<InputT>();
            CopyInTensor(sLocal, yGm, streamBase, n);
            inQueueY.EnQue(sLocal);
            sIn = inQueueY.DeQue<InputT>();
        }

        LocalTensor<uint8_t> zOut = outQueueZ.AllocTensor<uint8_t>();
        LocalTensor<InputT> resRaw = (xResident_ ? residentXBuf : residentYBuf).Get<InputT>();
        LocalTensor<ComputeT> sc;
        LocalTensor<ComputeT> rc;
        if constexpr (IsSameType<InputT, ComputeT>::value) {
            sc = sIn.ReinterpretCast<ComputeT>();
            rc = resRaw.ReinterpretCast<ComputeT>();
        } else {
            sc = xCompBuf.Get<ComputeT>();
            rc = yCompBuf.Get<ComputeT>();
            Cast(sc, sIn, RoundMode::CAST_NONE, compCount);
            Cast(rc, resRaw, RoundMode::CAST_NONE, compCount);
        }
        LocalTensor<ComputeT> xc = streamX ? sc : rc;
        LocalTensor<ComputeT> yc = streamX ? rc : sc;
        ComputeGtT<ComputeT>(zOut, xc, yc, compCount);

        outQueueZ.EnQue<uint8_t>(zOut);
        if (streamX) {
            inQueueX.FreeTensor(sIn);
        } else {
            inQueueY.FreeTensor(sIn);
        }
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

    // P1+: resident path. Work is split by zero-stride reuse groups. The
    // resident block is loaded once per group, while the other operand is
    // streamed as a contiguous group-sized range.
    __aicore__ inline void ProcessResident()
    {
        // A fully outer-broadcast operand has one resident block for the whole
        // output. Split segments across every core just as the original path
        // did; grouping it as one unit would accidentally serialize the work.
        if (residentGroupSegs_ == outerSize_) {
            ProcessFullResident();
            return;
        }
        uint32_t coreId = GetBlockIdx();
        uint64_t totalGroups = outerSize_ / residentGroupSegs_;
        uint64_t groupStart = totalGroups * coreId / blockDim_;
        uint64_t groupEnd = totalGroups * (coreId + 1) / blockDim_;
        const uint64_t groupElems = static_cast<uint64_t>(residentGroupSegs_) * innerSize_;
        const uint64_t tileElems = (static_cast<uint64_t>(TILE) / innerSize_) * innerSize_;

        for (uint64_t group = groupStart; group < groupEnd; ++group) {
            uint64_t seg = group * residentGroupSegs_;
            uint64_t xBase = 0;
            uint64_t yBase = 0;
            ComputeBases(seg, xBase, yBase);
            LoadResident(xResident_ ? xBase : yBase);

            uint64_t zBase = seg * innerSize_;
            uint64_t streamBase = xResident_ ? yBase : xBase;
            uint64_t off = 0;
            while (off < groupElems) {
                uint64_t n64 = tileElems;
                if (groupElems - off < n64) {
                    n64 = groupElems - off;
                }
                ProcessResidentTile(zBase + off, streamBase + off,
                                    static_cast<uint32_t>(n64));
                off += n64;
            }
            // The next group overwrites the resident UB buffer. V-side reads
            // above are asynchronous with respect to MTE2, so make that reuse
            // explicit before issuing the next DataCopyPad.
            TEventID eid = pipe.AllocEventID<HardEvent::V_MTE2>();
            SetFlag<HardEvent::V_MTE2>(eid);
            WaitFlag<HardEvent::V_MTE2>(eid);
            pipe.ReleaseEventID<HardEvent::V_MTE2>(eid);
        }
    }

    __aicore__ inline void ProcessFullResident()
    {
        LoadResident(0);
        uint32_t coreId = GetBlockIdx();
        uint64_t segStart = static_cast<uint64_t>(outerSize_) * coreId / blockDim_;
        uint64_t segEnd = static_cast<uint64_t>(outerSize_) * (coreId + 1) / blockDim_;
        uint64_t pos = segStart * innerSize_;
        uint64_t coreEnd = segEnd * innerSize_;
        const uint64_t tileElems = (static_cast<uint64_t>(TILE) / innerSize_) * innerSize_;
        while (pos < coreEnd) {
            uint64_t n64 = tileElems;
            if (coreEnd - pos < n64) {
                n64 = coreEnd - pos;
            }
            // The peer is globally contiguous when the resident suffix covers
            // all outer dims, hence output and stream offsets are identical.
            ProcessResidentTile(pos, pos, static_cast<uint32_t>(n64));
            pos += n64;
        }
    }

    // Non-aligned P1.  DataCopyPad lays every GM row into a COMP_ALIGN-sized
    // UB slot, so the start address of every Compare/Select/Cast is aligned.
    // The logical rows remain tightly packed in GM; only the UB staging is
    // padded, and CopyOutRows writes back exactly innerSize bools per row.
    __aicore__ inline void ProcessResidentPadded()
    {
        if (residentGroupSegs_ == outerSize_) {
            LoadResidentPadded(0);
            uint64_t segStart = static_cast<uint64_t>(outerSize_) * GetBlockIdx() / blockDim_;
            uint64_t segEnd = static_cast<uint64_t>(outerSize_) * (GetBlockIdx() + 1) / blockDim_;
            ProcessResidentPaddedRows(segStart * innerSize_, segStart * innerSize_,
                                      static_cast<uint32_t>(segEnd - segStart));
            return;
        }

        uint64_t totalGroups = outerSize_ / residentGroupSegs_;
        uint64_t groupStart = totalGroups * GetBlockIdx() / blockDim_;
        uint64_t groupEnd = totalGroups * (GetBlockIdx() + 1) / blockDim_;
        for (uint64_t group = groupStart; group < groupEnd; ++group) {
            uint64_t seg = group * residentGroupSegs_;
            uint64_t xBase = 0;
            uint64_t yBase = 0;
            ComputeBases(seg, xBase, yBase);
            LoadResidentPadded(xResident_ ? xBase : yBase);
            uint64_t streamBase = xResident_ ? yBase : xBase;
            ProcessResidentPaddedRows(seg * innerSize_, streamBase, residentGroupSegs_);
            SyncVToMte2();  // resident buffer is overwritten by the next group
        }
    }

    __aicore__ inline void ProcessResidentPaddedRows(uint64_t zBase, uint64_t streamBase,
                                                      uint32_t rows)
    {
        const uint32_t maxRows = TILE / rowElems_;
        uint32_t done = 0;
        while (done < rows) {
            uint32_t curRows = rows - done;
            if (curRows > maxRows) {
                curRows = maxRows;
            }
            ProcessResidentPaddedTile(zBase + static_cast<uint64_t>(done) * innerSize_,
                                      streamBase + static_cast<uint64_t>(done) * innerSize_, curRows);
            done += curRows;
        }
    }

    __aicore__ inline void ProcessResidentPaddedTile(uint64_t zBase, uint64_t streamBase,
                                                      uint32_t rows)
    {
        const uint32_t paddedN = rows * rowElems_;
        const bool streamX = !xResident_;
        LocalTensor<InputT> sIn;
        if (streamX) {
            LocalTensor<InputT> sLocal = inQueueX.AllocTensor<InputT>();
            ZeroInput(sLocal, paddedN);
            SyncVToMte2();
            CopyInRows(sLocal, xGm, streamBase, rows);
            inQueueX.EnQue(sLocal);
            sIn = inQueueX.DeQue<InputT>();
        } else {
            LocalTensor<InputT> sLocal = inQueueY.AllocTensor<InputT>();
            ZeroInput(sLocal, paddedN);
            SyncVToMte2();
            CopyInRows(sLocal, yGm, streamBase, rows);
            inQueueY.EnQue(sLocal);
            sIn = inQueueY.DeQue<InputT>();
        }

        LocalTensor<uint8_t> zOut = outQueueZ.AllocTensor<uint8_t>();
        LocalTensor<ComputeT> sc;
        if constexpr (IsSameType<InputT, ComputeT>::value) {
            sc = sIn.ReinterpretCast<ComputeT>();
        } else {
            sc = xCompBuf.Get<ComputeT>();
            Cast(sc, sIn, RoundMode::CAST_NONE, paddedN);
        }
        LocalTensor<InputT> resRaw = (xResident_ ? residentXBuf : residentYBuf).Get<InputT>();
        LocalTensor<ComputeT> rc;
        if constexpr (IsSameType<InputT, ComputeT>::value) {
            rc = resRaw.ReinterpretCast<ComputeT>();
        } else {
            rc = yCompBuf.Get<ComputeT>();
            Cast(rc, resRaw, RoundMode::CAST_NONE, rowElems_);
        }
        if constexpr (kIsHalf || kIsFloat) {
            // Repeat the same padded resident row in VEC without re-reading GM,
            // then compare the whole row batch with one vector call sequence.
            LocalTensor<ComputeT> residentRows = xCompBuf.Get<ComputeT>();
            LocalTensor<ComputeT> residentRow = rc.ReinterpretCast<ComputeT>();
            LocalTensor<ComputeT> streamRows = sc.ReinterpretCast<ComputeT>();
            SetMaskCount();
            SetVectorMask<ComputeT, MaskMode::COUNTER>(rowElems_);
            Copy<ComputeT, false>(residentRows, residentRow, MASK_PLACEHOLDER,
                                  static_cast<uint8_t>(rows),
                                  {1, 1,
                                   static_cast<uint16_t>(rowElems_ * sizeof(ComputeT) / 32), 0});
            PipeBarrier<PIPE_V>();
            SetMaskNorm();
            ResetMask();
            LocalTensor<ComputeT> xRows = streamX ? streamRows : residentRows;
            LocalTensor<ComputeT> yRows = streamX ? residentRows : streamRows;
            ComputeGtT<ComputeT>(zOut, xRows, yRows, paddedN);
        } else {
            for (uint32_t row = 0; row < rows; ++row) {
                uint32_t off = row * rowElems_;
                LocalTensor<ComputeT> xRow = streamX ? sc[off] : rc;
                LocalTensor<ComputeT> yRow = streamX ? rc : sc[off];
                LocalTensor<uint8_t> zRow = zOut[off];
                ComputeGtT<ComputeT>(zRow, xRow, yRow, rowElems_);
            }
        }
        outQueueZ.EnQue<uint8_t>(zOut);
        if (streamX) { inQueueX.FreeTensor(sIn); } else { inQueueY.FreeTensor(sIn); }
        LocalTensor<uint8_t> zLocal = outQueueZ.DeQue<uint8_t>();
        CopyOutRows(zGm, zLocal, zBase, rows);
        outQueueZ.FreeTensor(zLocal);
    }

    // Process one big TILE tile at output offset `pos` (innerSize-aligned).
    // The streamed operand is loaded once via its queue; the resident operand
    // is reused from UB for every innerSize sub-tile.
    __aicore__ inline void ProcessResidentTile(uint64_t zBase, uint64_t streamBase,
                                               uint32_t n)
    {
        uint32_t compCount = RoundUpTo(n, COMP_ALIGN);

        // Stream the non-resident operand. (For broadcast, exactly one operand
        // is resident; the other is streamed. If both were resident the output
        // is constant -- still correct, x is treated as the streamed one.)
        bool streamX = !xResident_;
        LocalTensor<InputT> sIn;
        if (streamX) {
            LocalTensor<InputT> sLocal = inQueueX.AllocTensor<InputT>();
            CopyInTensor(sLocal, xGm, streamBase, n);
            inQueueX.EnQue(sLocal);
            sIn = inQueueX.DeQue<InputT>();
        } else {
            LocalTensor<InputT> sLocal = inQueueY.AllocTensor<InputT>();
            CopyInTensor(sLocal, yGm, streamBase, n);
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
        uint32_t coreId = GetBlockIdx();
        uint64_t totalSegs = static_cast<uint64_t>(outerSize_);
        uint64_t segStart = totalSegs * coreId / blockDim_;
        uint64_t segEnd = totalSegs * (coreId + 1) / blockDim_;
        // A small outerSize may leave trailing cores with no whole segment.
        // DataCopyPad forbids blockLen==0, so they must exit before loading a
        // per-core scalar batch.
        if (segStart >= segEnd) {
            return;
        }
        if (!scalarBatchBlocked_) {
            LoadScalarBatch(segStart, static_cast<uint32_t>(segEnd - segStart));
        }
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

    // Non-aligned P2 counterpart of ProcessInnerBcast.  The streamed input and
    // bool output use the same padded row slots as P1; scalar values still come
    // from a UB batch, so no per-row MTE2 scalar load is reintroduced.
    __aicore__ inline void ProcessInnerBcastPadded()
    {
        uint32_t coreId = GetBlockIdx();
        uint64_t totalSegs = static_cast<uint64_t>(outerSize_);
        uint64_t segStart = totalSegs * coreId / blockDim_;
        uint64_t segEnd = totalSegs * (coreId + 1) / blockDim_;
        if (segStart >= segEnd) {
            return;
        }
        if (!scalarBatchBlocked_) {
            LoadScalarBatch(segStart, static_cast<uint32_t>(segEnd - segStart));
        }
        uint32_t maxRows = TILE / rowElems_;
        if constexpr (kIsHalf || kIsFloat) {
            // A fixed 32-row batch keeps every subsequent half scalar source
            // 32-byte aligned for Brcb; float uses 16 rows for the same reason.
            // Only the final batch may be shorter.
            if (scalarBatchPerCore_ && rowElems_ == COMP_ALIGN) {
                maxRows = kIsHalf ? 32 : 16;
            }
        }
        uint64_t seg = segStart;
        while (seg < segEnd) {
            uint32_t rows = static_cast<uint32_t>(segEnd - seg);
            if (rows > maxRows) {
                rows = maxRows;
            }
            if (scalarBatchBlocked_) {
                LoadScalarBatch(seg, rows);
            }
            ProcessInnerBcastPaddedTile(seg * innerSize_, rows, seg);
            seg += rows;
            if (scalarBatchBlocked_) {
                SyncVToMte2();
            }
        }
    }

    __aicore__ inline void ProcessInnerBcastPaddedTile(uint64_t zBase, uint32_t rows,
                                                        uint64_t firstSeg)
    {
        const uint32_t paddedN = rows * rowElems_;
        const bool streamX = (bcastMode_ != 1);
        LocalTensor<InputT> sIn;
        if (streamX) {
            LocalTensor<InputT> sLocal = inQueueX.AllocTensor<InputT>();
            ZeroInput(sLocal, paddedN);
            SyncVToMte2();
            CopyInRows(sLocal, xGm, zBase, rows);
            inQueueX.EnQue(sLocal);
            sIn = inQueueX.DeQue<InputT>();
        } else {
            LocalTensor<InputT> sLocal = inQueueY.AllocTensor<InputT>();
            ZeroInput(sLocal, paddedN);
            SyncVToMte2();
            CopyInRows(sLocal, yGm, zBase, rows);
            inQueueY.EnQue(sLocal);
            sIn = inQueueY.DeQue<InputT>();
        }

        LocalTensor<uint8_t> zOut = outQueueZ.AllocTensor<uint8_t>();
        LocalTensor<ComputeT> sc;
        if constexpr (IsSameType<InputT, ComputeT>::value) {
            sc = sIn.ReinterpretCast<ComputeT>();
        } else {
            sc = (streamX ? xCompBuf : yCompBuf).Get<ComputeT>();
            Cast(sc, sIn, RoundMode::CAST_NONE, paddedN);
        }
        LocalTensor<InputT> batch = scalarBatchBuf.Get<InputT>();
        if constexpr (kIsHalf || kIsFloat) {
            if (scalarBatchPerCore_ && rowElems_ == COMP_ALIGN) {
                uint32_t localOffset = static_cast<uint32_t>(firstSeg - scalarBatchBase_);
                LocalTensor<ComputeT> scalarBlocks = scalarBrcbBuf.Get<ComputeT>();
                Brcb(scalarBlocks, batch[localOffset].ReinterpretCast<ComputeT>(),
                     static_cast<uint8_t>((rows + 7) / 8), {1, 8});
                PipeBarrier<PIPE_V>();

                LocalTensor<ComputeT> scalarRows;
                if constexpr (kIsHalf) {
                    scalarRows = halfOutBuf.Get<half>().ReinterpretCast<ComputeT>();
                } else {
                    scalarRows = scalarRowsBuf.Get<ComputeT>();
                }
                SetMaskCount();
                SetVectorMask<ComputeT, MaskMode::COUNTER>(rowElems_);
                Copy<ComputeT, false>(scalarRows, scalarBlocks, MASK_PLACEHOLDER,
                                      static_cast<uint8_t>(rows),
                                      {1, 0,
                                       static_cast<uint16_t>(rowElems_ * sizeof(ComputeT) / 32), 1});
                PipeBarrier<PIPE_V>();
                SetMaskNorm();
                ResetMask();

                LocalTensor<ComputeT> streamRows = sc.ReinterpretCast<ComputeT>();
                LocalTensor<ComputeT> xRows = streamX ? streamRows : scalarRows;
                LocalTensor<ComputeT> yRows = streamX ? scalarRows : streamRows;
                ComputeGtT<ComputeT>(zOut, xRows, yRows, paddedN);
            } else {
                ProcessInnerBcastPaddedRows(zOut, sc, batch, streamX, firstSeg, rows);
            }
        } else {
            ProcessInnerBcastPaddedRows(zOut, sc, batch, streamX, firstSeg, rows);
        }
        outQueueZ.EnQue<uint8_t>(zOut);
        if (streamX) { inQueueX.FreeTensor(sIn); } else { inQueueY.FreeTensor(sIn); }
        LocalTensor<uint8_t> zLocal = outQueueZ.DeQue<uint8_t>();
        CopyOutRows(zGm, zLocal, zBase, rows);
        outQueueZ.FreeTensor(zLocal);
    }

    __aicore__ inline void ProcessInnerBcastPaddedRows(LocalTensor<uint8_t>& zOut,
                                                       LocalTensor<ComputeT>& sc,
                                                       LocalTensor<InputT>& batch,
                                                       bool streamX, uint64_t firstSeg,
                                                       uint32_t rows)
    {
        for (uint32_t row = 0; row < rows; ++row) {
            uint32_t off = row * rowElems_;
            uint64_t seg = firstSeg + row;
            uint32_t scalarIdx = scalarBatchPerCore_
                ? static_cast<uint32_t>(seg - scalarBatchBase_)
                : ScalarIndex(seg);
            LocalTensor<ComputeT> sRow = sc[off];
            LocalTensor<uint8_t> zRow = zOut[off];
            if constexpr (kIsHalf || kIsFloat || kIsInt8) {
                ComputeT scalar = GetScalarValue<ComputeT>(batch, scalarIdx);
                ComputeGtScalarT<ComputeT>(zRow, sRow, scalar, streamX, rowElems_);
            } else {
                LocalTensor<ComputeT> scalarRow = (streamX ? yCompBuf : xCompBuf).Get<ComputeT>();
                MaterializeScalar<ComputeT>(scalarRow, batch, scalarIdx, rowElems_);
                LocalTensor<ComputeT> xc = streamX ? sRow : scalarRow;
                LocalTensor<ComputeT> yc = streamX ? scalarRow : sRow;
                ComputeGtT<ComputeT>(zRow, xc, yc, rowElems_);
            }
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
        LocalTensor<InputT> batch = scalarBatchBuf.Get<InputT>();

        // Sub-tile loop: each sub-tile = one segment (innerSize elems). The
        // scalar index is calculated from the complete outer broadcast layout.
        uint64_t seg = zBase / innerSize_;
        uint32_t off = 0;
        while (off < n) {
            uint32_t subN = innerSize_;
            if (n - off < subN) {
                subN = n - off;
            }
            uint32_t subComp = RoundUpTo(subN, COMP_ALIGN);
            LocalTensor<CT> sSub = sc[off];
            LocalTensor<uint8_t> zSub = zOut[off];
            uint32_t scalarIdx = scalarBatchPerCore_
                ? static_cast<uint32_t>(seg - scalarBatchBase_)
                : ScalarIndex(seg);
            if constexpr (kIsHalf || kIsFloat || kIsInt8) {
                // CompareScalar avoids materializing an innerSize-element
                // Duplicate buffer for the common fp16/fp32/int8 paths.
                CT scalar = GetScalarValue<CT>(batch, scalarIdx);
                ComputeGtScalarT<CT>(zSub, sSub, scalar, streamX, subComp);
            } else {
                // bf16 and int32 retain their verified conversion/exact paths.
                LocalTensor<CT> scalarTile = (streamX ? yCompBuf : xCompBuf).Get<CT>();
                MaterializeScalar<CT>(scalarTile, batch, scalarIdx, subComp);
                LocalTensor<CT> xc = streamX ? sSub : scalarTile;
                LocalTensor<CT> yc = streamX ? scalarTile : sSub;
                ComputeGtT<CT>(zSub, xc, yc, subComp);
            }
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
    // P2 copies the non-scalar operand from gm[seg * innerSize].  This is valid
    // only when every non-degenerate outer output dimension advances with the
    // corresponding dense output stride.  Degenerate dimensions never advance,
    // so their operand stride is irrelevant.
    __aicore__ inline bool IsStreamIndexContinuous(const uint32_t* streamStrides)
    {
        uint64_t expected = innerSize_;
        for (int d = static_cast<int>(outerDim_) - 1; d >= 0; --d) {
            uint32_t dim = outerShape_[d];
            if (dim <= 1) {
                continue;
            }
            if (expected > UINT32_MAX || streamStrides[d] != expected) {
                return false;
            }
            expected *= dim;
        }
        return true;
    }

    // True exactly when scalar operand storage advances one element per output
    // segment.  This is the common [outer, 1] form; mixed outer broadcasts
    // deliberately keep the conservative ScalarIndex/whole-range path.
    __aicore__ inline bool IsScalarIndexContinuous(const uint32_t* scalarStrides)
    {
        uint64_t expected = 1;
        for (int d = static_cast<int>(outerDim_) - 1; d >= 0; --d) {
            if (scalarStrides[d] != expected) {
                return false;
            }
            expected *= outerShape_[d];
            if (expected > UINT32_MAX) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline void SyncVToMte2()
    {
        TEventID eid = pipe.AllocEventID<HardEvent::V_MTE2>();
        SetFlag<HardEvent::V_MTE2>(eid);
        WaitFlag<HardEvent::V_MTE2>(eid);
        pipe.ReleaseEventID<HardEvent::V_MTE2>(eid);
    }

    // Duplicate has no int8 overload on dav_c220.  Reinterpret the byte buffer
    // as half for initialization; rowElems is a multiple of 256, so its byte
    // size and the half element count are both naturally aligned.
    __aicore__ inline void ZeroInput(LocalTensor<InputT>& dst, uint32_t count)
    {
        if constexpr (kIsInt8) {
            LocalTensor<half> asHalf = dst.ReinterpretCast<half>();
            Duplicate(asHalf, static_cast<half>(0), static_cast<int32_t>(count / 2));
        } else {
            Duplicate(dst, static_cast<InputT>(0), static_cast<int32_t>(count));
        }
    }

    // Return the largest trailing outer-dimension group whose resident operand
    // has zero strides and whose peer is contiguous. A group of one has no
    // reuse value and is intentionally not selected.
    __aicore__ inline uint32_t GetResidentGroupSegs(bool residentIsX)
    {
        const uint32_t* residentStride = residentIsX ? xStride_ : yStride_;
        const uint32_t* streamStride = residentIsX ? yStride_ : xStride_;
        uint64_t groupSegs = 1;
        uint64_t expectedStride = innerSize_;
        for (int d = static_cast<int>(outerDim_) - 1; d >= 0; --d) {
            if (residentStride[d] != 0 || streamStride[d] != expectedStride) {
                break;
            }
            groupSegs *= outerShape_[d];
            expectedStride *= outerShape_[d];
            if (groupSegs > UINT32_MAX || expectedStride > UINT32_MAX) {
                return 1;
            }
        }
        return static_cast<uint32_t>(groupSegs);
    }

    // P1: load one resident inner block for the current reuse group, then sync
    // MTE2->V before Vector reads it.
    __aicore__ inline void LoadResident(uint64_t base)
    {
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
            DataCopyPad(rx, xGm[base], params, pad);
        }
        if (yResident_) {
            LocalTensor<InputT> ry = residentYBuf.Get<InputT>();
            DataCopyPad(ry, yGm[base], params, pad);
        }
        TEventID eid = pipe.AllocEventID<HardEvent::MTE2_V>();
        SetFlag<HardEvent::MTE2_V>(eid);
        WaitFlag<HardEvent::MTE2_V>(eid);
        pipe.ReleaseEventID<HardEvent::MTE2_V>(eid);
    }

    __aicore__ inline void LoadResidentSlice(uint32_t offset, uint32_t n)
    {
        if (xResident_) {
            LocalTensor<InputT> rx = residentXBuf.Get<InputT>();
            CopyInTensor(rx, xGm, offset, n);
        } else {
            LocalTensor<InputT> ry = residentYBuf.Get<InputT>();
            CopyInTensor(ry, yGm, offset, n);
        }
        TEventID eid = pipe.AllocEventID<HardEvent::MTE2_V>();
        SetFlag<HardEvent::MTE2_V>(eid);
        WaitFlag<HardEvent::MTE2_V>(eid);
        pipe.ReleaseEventID<HardEvent::MTE2_V>(eid);
    }

    __aicore__ inline void LoadResidentPadded(uint64_t base)
    {
        if (xResident_) {
            LocalTensor<InputT> rx = residentXBuf.Get<InputT>();
            ZeroInput(rx, rowElems_);
            SyncVToMte2();
            CopyInRows(rx, xGm, base, 1);
        }
        if (yResident_) {
            LocalTensor<InputT> ry = residentYBuf.Get<InputT>();
            ZeroInput(ry, rowElems_);
            SyncVToMte2();
            CopyInRows(ry, yGm, base, 1);
        }
        TEventID eid = pipe.AllocEventID<HardEvent::MTE2_V>();
        SetFlag<HardEvent::MTE2_V>(eid);
        WaitFlag<HardEvent::MTE2_V>(eid);
        pipe.ReleaseEventID<HardEvent::MTE2_V>(eid);
    }

    // P2: load the complete reachable scalar storage range once. ScalarIndex()
    // maps each output segment back into this batch with broadcast strides.
    __aicore__ inline void LoadScalarBatch(uint64_t segStart, uint32_t coreSegs)
    {
        uint32_t count = scalarBatchPerCore_ ? coreSegs : scalarBatchCount_;
        scalarBatchBase_ = scalarBatchPerCore_ ? segStart : 0;
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(count * sizeof(InputT));
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
        DataCopyPad(batch, gm[scalarBatchBase_], params, pad);
        TEventID vEid = pipe.AllocEventID<HardEvent::MTE2_V>();
        SetFlag<HardEvent::MTE2_V>(vEid);
        WaitFlag<HardEvent::MTE2_V>(vEid);
        pipe.ReleaseEventID<HardEvent::MTE2_V>(vEid);
        // GetValue below runs on the scalar pipe, not vector pipe.
        TEventID sEid = pipe.AllocEventID<HardEvent::MTE2_S>();
        SetFlag<HardEvent::MTE2_S>(sEid);
        WaitFlag<HardEvent::MTE2_S>(sEid);
        pipe.ReleaseEventID<HardEvent::MTE2_S>(sEid);
    }

    // Multi-row logical-GM -> padded-UB transfer.  The slot has been zeroed
    // before this call because DataCopyPad can only explicitly pad <=32 bytes
    // on either side, while COMP_ALIGN padding can be larger.
    __aicore__ inline void CopyInRows(LocalTensor<InputT>& dst, GlobalTensor<InputT>& gm,
                                      uint64_t base, uint32_t rows)
    {
        const uint32_t logicalBytes = innerSize_ * sizeof(InputT);
        const uint32_t roundedBytes = RoundUpTo(logicalBytes, 32);
        DataCopyExtParams params;
        params.blockCount = rows;
        params.blockLen = logicalBytes;
        params.srcStride = 0;
        params.dstStride = (rowElems_ * sizeof(InputT) - roundedBytes) / 32;
        params.rsv = 0;
        DataCopyPadExtParams<InputT> pad;
        pad.isPad = true;
        pad.leftPadding = 0;
        pad.rightPadding = 0;
        pad.paddingValue = static_cast<InputT>(0);
        DataCopyPad(dst, gm[base], params, pad);
    }

    __aicore__ inline void CopyOutRows(GlobalTensor<uint8_t>& gm, LocalTensor<uint8_t>& src,
                                       uint64_t base, uint32_t rows)
    {
        const uint32_t roundedBytes = RoundUpTo(innerSize_, 32);
        DataCopyExtParams params;
        params.blockCount = rows;
        params.blockLen = innerSize_;
        params.srcStride = (rowElems_ - roundedBytes) / 32;
        params.dstStride = 0;
        params.rsv = 0;
        DataCopyPad(gm[base], src, params);
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

    __aicore__ inline uint32_t ScalarIndex(uint64_t seg)
    {
        uint64_t xBase = 0;
        uint64_t yBase = 0;
        ComputeBases(seg, xBase, yBase);
        return static_cast<uint32_t>((bcastMode_ == 1) ? xBase : yBase);
    }

    template <typename CT>
    __aicore__ inline CT GetScalarValue(LocalTensor<InputT>& batch, uint32_t scalarIdx)
    {
        if constexpr (IsSameType<InputT, CT>::value) {
            return static_cast<CT>(batch.GetValue(scalarIdx));
        } else {
            // The only scalar fast-path conversion is int8 -> half.
            int8_t value = batch.GetValue(scalarIdx);
            return static_cast<CT>(static_cast<float>(value));
        }
    }

    template <typename CT>
    __aicore__ inline void MaterializeScalar(LocalTensor<CT>& dst,
                                              LocalTensor<InputT>& batch,
                                              uint32_t scalarIdx, uint32_t count)
    {
        if constexpr (IsSameType<InputT, CT>::value) {
            Duplicate(dst, static_cast<CT>(batch.GetValue(scalarIdx)), static_cast<int32_t>(count));
        } else {
            // bf16 -> float: preserve the established exact vector Cast path.
            InputT value = batch.GetValue(scalarIdx);
            LocalTensor<InputT> bfTile = bf16TileBuf.Get<InputT>();
            Duplicate(bfTile, value, static_cast<int32_t>(count));
            Cast(dst, bfTile, RoundMode::CAST_NONE, count);
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

    // Scalar version for fp16/fp32/int8->fp16. CompareScalar has the same
    // 256B-aligned count contract as Compare, satisfied by compCount.
    template <typename CT>
    __aicore__ inline void ComputeGtScalarT(LocalTensor<uint8_t>& zOut,
                                            LocalTensor<CT>& stream, CT scalar,
                                            bool streamIsX, uint32_t compCount)
    {
        LocalTensor<uint8_t> mask = maskBuf.Get<uint8_t>();
        LocalTensor<half> halfOut = halfOutBuf.Get<half>();
        LocalTensor<half> zero = halfZeroBuf.Get<half>();
        LocalTensor<half> one = halfOneBuf.Get<half>();
        // stream > scalar when stream is x; scalar > stream is stream < scalar.
        CompareScalar(mask, stream, scalar,
                      streamIsX ? CMPMODE::GT : CMPMODE::LT, compCount);
        Select(halfOut, mask, one, zero, SELMODE::VSEL_TENSOR_TENSOR_MODE, compCount);
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
        TEventID eid = pipe.AllocEventID<HardEvent::MTE2_S>();
        SetFlag<HardEvent::MTE2_S>(eid);
        WaitFlag<HardEvent::MTE2_S>(eid);
        pipe.ReleaseEventID<HardEvent::MTE2_S>(eid);
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
    TBuf<TPosition::VECCALC> scalarBuf;
    TBuf<TPosition::VECCALC> residentXBuf, residentYBuf;  // P1 broadcast-resident
    TBuf<TPosition::VECCALC> scalarBatchBuf;             // P2 innermost-bcast scalars
    TBuf<TPosition::VECCALC> scalarBrcbBuf;              // P2 fp16/fp32 scalar blocks (1KiB)
    TBuf<TPosition::VECCALC> scalarRowsBuf;              // P2 fp32 expanded scalar rows

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
    bool largeResident_ = false;      // full-outer P1 sliced across inner workers
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
    uint32_t scalarBatchCount_ = 0;    // P2: contiguous scalar storage range
    bool scalarBatchPerCore_ = false;  // P2: scalarIndex(seg) == seg
    bool scalarBatchBlocked_ = false;  // P2: load one aligned scalar block per row tile
    uint64_t scalarBatchBase_ = 0;     // GM segment represented by batch[0]
    uint32_t residentGroupSegs_ = 1;   // P1: zero-stride reuse run (segments)
    bool rowPadded_ = false;           // non-aligned broadcast UB staging enabled
    uint32_t rowElems_ = 0;            // padded row length (COMP_ALIGN multiple)
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
