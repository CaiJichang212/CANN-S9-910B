#ifndef CONCAT_TILING_H
#define CONCAT_TILING_H

#include <cstdint>

#include "register/tilingdata_base.h"

namespace optiling {

// 最多支持 256 路输入拼接（ACLNN 框架 dynamic tensorList 硬上限为 256）
constexpr uint32_t MAX_CONCAT_INPUT_NUM = 256;

BEGIN_TILING_DATA_DEF(ConcatCustomTilingData)
  // 仅保留 kernel 消费的数据。inputNum <= 256，dtypeSize <= 4，splitMode
  // 只有两个取值；紧凑标量让动态 TensorList 的 tiling 从约 2 KiB 降至约 1 KiB。
  TILING_DATA_FIELD_DEF(uint16_t, inputNum);
  TILING_DATA_FIELD_DEF(uint8_t, dtypeSize);
  TILING_DATA_FIELD_DEF(uint8_t, splitMode);
  // beforeDim = dim 之前各维度乘积；afterDim = dim 之后各维度乘积
  TILING_DATA_FIELD_DEF(uint32_t, beforeDimSize);
  TILING_DATA_FIELD_DEF(uint32_t, afterDimSize);
  // 输出沿 dim 维的总长度。长度仍为 uint32_t，避免限制大行输入。
  TILING_DATA_FIELD_DEF(uint32_t, totalCatLen);
  // 每个输入沿 dim 维的长度
  TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_CONCAT_INPUT_NUM, inputCatLen);
  // 多核切分。splitMode=0 为整行切分，1 为行×输出列切分。
  TILING_DATA_FIELD_DEF(uint32_t, rowSliceNum);
  TILING_DATA_FIELD_DEF(uint32_t, colCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, colBlockBytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Concat, ConcatCustomTilingData)

}  // namespace optiling

#endif  // CONCAT_TILING_H
