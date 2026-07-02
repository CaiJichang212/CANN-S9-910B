#ifndef CONCAT_TILING_H
#define CONCAT_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

// 最多支持 64 路输入拼接
constexpr uint32_t MAX_CONCAT_INPUT_NUM = 64;

BEGIN_TILING_DATA_DEF(ConcatCustomTilingData)
  // 基础信息
  TILING_DATA_FIELD_DEF(uint32_t, inputNum);       // 实际输入张量个数
  TILING_DATA_FIELD_DEF(uint32_t, dim);            // 拼接维度（已转为非负）
  TILING_DATA_FIELD_DEF(uint32_t, dimNum);         // 输入张量维数
  TILING_DATA_FIELD_DEF(uint32_t, dtypeSize);      // 每个 element 的字节数
  // beforeDim = dim 之前各维度乘积；afterDim = dim 之后各维度乘积
  TILING_DATA_FIELD_DEF(uint32_t, beforeDimSize);
  TILING_DATA_FIELD_DEF(uint32_t, afterDimSize);
  // 每个输入沿 dim 维的长度
  TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_CONCAT_INPUT_NUM, inputCatLen);
  // 每个输入在输出 dim 维上的起始偏移（前缀和）
  TILING_DATA_FIELD_DEF_ARR(uint32_t, MAX_CONCAT_INPUT_NUM, inputCatOffset);
  // 输出沿 dim 维的总长度
  TILING_DATA_FIELD_DEF(uint32_t, totalCatLen);
  // 多核切分：以 (row, input) 对为最小 work item，共 beforeDimSize*inputNum 个
  TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Concat, ConcatCustomTilingData)

}  // namespace optiling

#endif  // CONCAT_TILING_H
