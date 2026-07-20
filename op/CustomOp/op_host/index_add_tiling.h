#ifndef INDEX_ADD_CUSTOM_TILING_H
#define INDEX_ADD_CUSTOM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

// IndexAdd is viewed as [beforeDimSize, dimLen, afterDimSize].  The kernel
// always uses the same core group for the copy and scatter phases.  Scatter
// uses DMA atomics only when a complete source/output vector is 32B aligned;
// otherwise index ownership (index % scatterCoreNum) removes WAW hazards.
BEGIN_TILING_DATA_DEF(IndexAddCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, dim);
  TILING_DATA_FIELD_DEF(uint32_t, beforeDimSize);
  TILING_DATA_FIELD_DEF(uint32_t, dimLen);
  TILING_DATA_FIELD_DEF(uint32_t, afterDimSize);
  TILING_DATA_FIELD_DEF(uint32_t, indexLen);
  TILING_DATA_FIELD_DEF(uint32_t, dtypeSize);
  TILING_DATA_FIELD_DEF(uint32_t, dtype);
  TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, scatterCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, atomicEnabled);
  TILING_DATA_FIELD_DEF(uint32_t, copyTileBytes);
  TILING_DATA_FIELD_DEF(uint32_t, atomicTileBytes);
  TILING_DATA_FIELD_DEF(uint32_t, atomicThresholdBytes);
  TILING_DATA_FIELD_DEF(uint32_t, rmwTileLen);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexAdd, IndexAddCustomTilingData)

}  // namespace optiling

#endif  // INDEX_ADD_CUSTOM_TILING_H
