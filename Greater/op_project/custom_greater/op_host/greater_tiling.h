/**
 * @file greater_tiling.h
 *
 * Tiling data for the Greater (torch.gt) custom operator.
 * Element-wise x > y with NumPy-style broadcast; output is bool.
 * Supported input dtypes: float16, float32, bfloat16, int32, int8.
 *
 * Note: all fields are uint32_t (uniform 4-byte width) so that the host
 * TilingDef layout and the kernel-side packed struct match exactly.
 */
#ifndef GREATER_TILING_H
#define GREATER_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GreaterTilingData)
// Total number of output elements (after broadcast).
TILING_DATA_FIELD_DEF(uint32_t, totalSize);
// Number of AI cores to use.
TILING_DATA_FIELD_DEF(uint32_t, blockDim);
// ---- broadcast decomposition ----
// The output is `outerSize` segments of `innerSize` contiguous elements.
// innerSize is the maximal trailing non-broadcast suffix (identical-shape
// inputs collapse to one segment = fast flatten path). When the innermost dim
// is broadcast, innerSize is the innermost dim size and one operand is a
// per-segment scalar.
TILING_DATA_FIELD_DEF(uint32_t, innerSize);
TILING_DATA_FIELD_DEF(uint32_t, outerSize);
// 0: both operands full on the inner block; 1: x scalar; 2: y scalar.
TILING_DATA_FIELD_DEF(uint32_t, bcastMode);
// Number of outer dims actually used (0 when outerSize == 1).
TILING_DATA_FIELD_DEF(uint32_t, outerDim);
// Outer dim sizes and per-operand element strides (0 marks a broadcast dim).
TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, outerShape);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, xStride);
TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, yStride);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Greater, GreaterTilingData)
} // namespace optiling
#endif // GREATER_TILING_H
