#ifndef INDEX_ADD_CUSTOM_TILING_H
#define INDEX_ADD_CUSTOM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

// IndexAdd 数据视图：把内存按 dim 拆成 [beforeDimSize, dimLen, afterDimSize]，
// afterDimSize 维内存连续。source 为 [beforeDimSize, M, afterDimSize]，index 为 [M]。
// 语义：output = copy(self)；再对每个 i，output[..., index[i], ...] += source[..., i, ...]。
//
// 核切分（两模式，均保证各核输出区域互不重叠 → 无 WAW、无需原子）：
//   ROW   (mode=0)：按 beforeDim 行切分，每核负责若干整行（afterDim 全长）。
//   AFTER (mode=1)：按 afterDim 轴切分，每核负责所有行的一段 afterSlice。
// host 选「能给出更多核」的模式以尽量用满 20 核。
BEGIN_TILING_DATA_DEF(IndexAddCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, dim);            // 拼接维度（已转为非负）
  TILING_DATA_FIELD_DEF(uint32_t, beforeDimSize);  // dim 之前各维乘积
  TILING_DATA_FIELD_DEF(uint32_t, dimLen);         // self 沿 dim 维长度
  TILING_DATA_FIELD_DEF(uint32_t, afterDimSize);   // dim 之后各维乘积（内存连续维）
  TILING_DATA_FIELD_DEF(uint32_t, indexLen);       // M = index 长度 = source 沿 dim 维长度
  TILING_DATA_FIELD_DEF(uint32_t, dtypeSize);      // 每个 element 的字节数
  TILING_DATA_FIELD_DEF(uint32_t, dtype);          // 0=float 1=bf16 2=half 3=int32 4=int8
  TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);    // 实际使用的核数 (<=20)
  TILING_DATA_FIELD_DEF(uint32_t, mode);           // 0=ROW 1=AFTER
  TILING_DATA_FIELD_DEF(uint32_t, scatterTileLen);  // scatter 阶段连续向量内部分块元素数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexAdd, IndexAddCustomTilingData)

}  // namespace optiling

#endif  // INDEX_ADD_CUSTOM_TILING_H
