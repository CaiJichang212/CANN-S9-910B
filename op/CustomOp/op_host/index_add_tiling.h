#ifndef INDEX_ADD_CUSTOM_TILING_H
#define INDEX_ADD_CUSTOM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

// IndexAdd is viewed as [beforeDimSize, dimLen, afterDimSize].  The kernel
// first copies self to output, then chooses either DMA atomics for aligned
// vectors or stable buckets plus output ownership for arbitrary layouts.
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
  TILING_DATA_FIELD_DEF(uint32_t, path);
  TILING_DATA_FIELD_DEF(uint32_t, atomicEnabled);
  TILING_DATA_FIELD_DEF(uint32_t, copyTileBytes);
  TILING_DATA_FIELD_DEF(uint32_t, atomicTileBytes);
  TILING_DATA_FIELD_DEF(uint32_t, kTile);
  TILING_DATA_FIELD_DEF(uint32_t, targetGroupSize);
  TILING_DATA_FIELD_DEF(uint32_t, indexChunkLen);
  TILING_DATA_FIELD_DEF(uint32_t, positionChunkLen);
  TILING_DATA_FIELD_DEF(uint64_t, workspaceBytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexAdd, IndexAddCustomTilingData)

}  // namespace optiling

#endif  // INDEX_ADD_CUSTOM_TILING_H
