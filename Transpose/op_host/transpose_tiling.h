/**
 * @file transpose_tiling.h
 * @brief Tiling contract for the 910B transpose kernel.
 *
 * All geometry that can exceed 32 bits is carried as uint64_t.  The kernel
 * has three paths: contiguous rows, a 2-D rotation of contiguous axis groups,
 * and a generic strided-row DMA fallback.
 */
#ifndef TRANSPOSE_TILING_H
#define TRANSPOSE_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TransposeTilingData)
TILING_DATA_FIELD_DEF(uint32_t, mode);       // 0=COPY_CONTIG, 1=ROTATE_2D, 2=STRIDED_ROWS
TILING_DATA_FIELD_DEF(uint32_t, dtypeSize);
TILING_DATA_FIELD_DEF(uint32_t, blockDim);
TILING_DATA_FIELD_DEF(uint32_t, outerCount);

TILING_DATA_FIELD_DEF(uint64_t, total);
TILING_DATA_FIELD_DEF(uint64_t, rowWidth);
TILING_DATA_FIELD_DEF(uint64_t, numRows);
TILING_DATA_FIELD_DEF(uint64_t, srcInnerStride);
TILING_DATA_FIELD_DEF_ARR(uint64_t, 8, outerOutShape);
TILING_DATA_FIELD_DEF_ARR(uint64_t, 8, outerSrcStride);
TILING_DATA_FIELD_DEF(uint32_t, copyTileElems);
TILING_DATA_FIELD_DEF(uint32_t, stridedTileElems);

TILING_DATA_FIELD_DEF(uint64_t, transBatch);
TILING_DATA_FIELD_DEF(uint64_t, transM);
TILING_DATA_FIELD_DEF(uint64_t, transN);
TILING_DATA_FIELD_DEF(uint32_t, tileM);
TILING_DATA_FIELD_DEF(uint32_t, tileN);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Transpose, TransposeTilingData)
} // namespace optiling
#endif
