/**
 * @file transpose_tiling.h
 * @brief Transpose (torch.permute) tiling data definition.
 *
 * 实现分两条路径（host 侧完成全部几何计算，kernel 只搬运/重排）：
 *
 * 1) COPY 路径 (mode=0)：输出末维在源端连续（srcStrideInner==1，即 dims 末维未被移动，
 *    或为前缀 identity 的相邻递减序列）。按输出行连续读 + 连续写。
 *
 * 2) TRANSPOSE 路径 (mode=1)：末两维相邻交换（dims[n-1]==ndim-2 && dims[n-2]==ndim-1，
 *    前缀 identity），即 2D 转置 (M,N)->(N,M)。按 tile 分块：strided 读 tileT_M×tileT_N
 *    入 UB（每行一块，blockLen 对齐 32B），UB 内 strided 重排为转置布局，strided 写出。
 *    tileT_M/tileT_N 取 16 倍数贴满 UB。非对齐尾块用 DataCopyPad 处理。
 *
 * 其余 permute 暂回退到 TRANSPOSE 的逐 2D 交换或 COPY 分解（后续按实测 pattern 扩展）。
 */
#ifndef TRANSPOSE_TILING_H
#define TRANSPOSE_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TransposeTilingData)
// 模式：0=COPY(末维源端连续), 1=TRANSPOSE(末两维 2D 转置)
TILING_DATA_FIELD_DEF(uint32_t, mode);
TILING_DATA_FIELD_DEF(uint32_t, total);          // 元素总数
TILING_DATA_FIELD_DEF(uint32_t, ndim);            // 维度数 (1..5)
TILING_DATA_FIELD_DEF(uint32_t, dtypeSize);      // sizeof(T) 字节
TILING_DATA_FIELD_DEF(uint32_t, blockDim);       // 核数 (<=20)

// --- COPY 路径参数 (mode=0) ---
TILING_DATA_FIELD_DEF(uint32_t, W);              // 末输出维大小 = 每行连续写出长度
TILING_DATA_FIELD_DEF(uint32_t, numRows);        // = total / W
TILING_DATA_FIELD_DEF(uint32_t, srcStrideInner); // S = inStride[dims[ndim-1]] (元素)，COPY 时为 1
TILING_DATA_FIELD_DEF(uint32_t, outerCount);     // = ndim-1（外层维数）
TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, outerOutShape);   // outShape[0..outerCount-1]
TILING_DATA_FIELD_DEF_ARR(uint32_t, 8, outerSrcStride);  // inStride[dims[0..outerCount-1]] (元素)
TILING_DATA_FIELD_DEF(uint32_t, copyTileLen);    // 行内分块（元素）

// --- TRANSPOSE 路径参数 (mode=1) ---
TILING_DATA_FIELD_DEF(uint32_t, transBatch);     // 前缀 identity 维乘积（batch，外层循环数）
TILING_DATA_FIELD_DEF(uint32_t, transM);         // 输入倒数第二维 A (= 输出倒数第一维)
TILING_DATA_FIELD_DEF(uint32_t, transN);        // 输入倒数第一维 B (= 输出倒数第二维)
TILING_DATA_FIELD_DEF(uint32_t, tileM);          // M 方向 tile（16 倍数）
TILING_DATA_FIELD_DEF(uint32_t, tileN);         // N 方向 tile（16 倍数）
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Transpose, TransposeTilingData)
} // namespace optiling
#endif // TRANSPOSE_TILING_H
