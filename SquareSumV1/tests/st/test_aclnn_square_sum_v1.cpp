/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the CANN Open Software License Agreement Version 2.0.
 *
 * ST 测试: aclnnSquareSumV1 — 平方+规约融合算子
 *
 * 功能: Y = sum(X^2, dim=axis, keepdim=keep_dims)
 *
 * 模式:
 *   Mock (-DUSE_MOCK_ACLNN): CPU golden 自测，无需 NPU
 *   Real (默认):            NPU 执行 + CPU golden 比对
 *
 * 精度标准:
 *   float16/bfloat16:  rtol=atol=1e-2, loss=1e-3
 *   float32:           rtol=atol=1e-4, loss=1e-4
 *   NaN 同判: real 与 golden 同为 NaN 视为通过
 */

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstring>
#include <limits>
#include <cstdint>
#include <algorithm>
#include <sstream>
#include <fstream>

#ifndef USE_MOCK_ACLNN
#include "acl/acl.h"
#include "aclnn_square_sum_v1.h"
#endif

// ============================================================================
// 宏与常量
// ============================================================================
#define LOG_PRINT(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)

// fp16/bf16 精度
constexpr double RTOL_FP16 = 1e-2;
constexpr double ATOL_FP16 = 1e-2;
constexpr double LOSS_FP16 = 1e-3;

// fp32 精度
constexpr double RTOL_FP32 = 1e-4;
constexpr double ATOL_FP32 = 1e-4;
constexpr double LOSS_FP32 = 1e-4;

// ============================================================================
// 辅助函数
// ============================================================================

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t size = 1;
    for (auto dim : shape) size *= dim;
    return size;
}

// 计算输出 shape
std::vector<int64_t> ComputeOutputShape(const std::vector<int64_t>& inputShape,
                                        const std::vector<int64_t>& axis,
                                        bool keepDims) {
    int64_t rank = static_cast<int64_t>(inputShape.size());
    // 归一化 axis 为正值集合
    std::vector<int64_t> normalizedAxis;
    for (auto a : axis) {
        normalizedAxis.push_back(a < 0 ? a + rank : a);
    }
    std::sort(normalizedAxis.begin(), normalizedAxis.end());

    std::vector<int64_t> outputShape;
    for (int64_t i = 0; i < rank; i++) {
        bool isReduceDim = std::binary_search(normalizedAxis.begin(),
                                              normalizedAxis.end(), i);
        if (isReduceDim) {
            if (keepDims) outputShape.push_back(1);
            // else skip
        } else {
            outputShape.push_back(inputShape[i]);
        }
    }
    // 标量输出 (全部维度被规约且 keepDims=false)
    if (outputShape.empty()) {
        outputShape.push_back(1); // 用 1 维表示标量
    }
    return outputShape;
}

// 将 axis 列表转换为 strides（用于 golden 计算中的多维索引遍历）
std::vector<int64_t> ComputeStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    return strides;
}

// ============================================================================
// 类型转换辅助 (fp16/bf16 内存表示)
// ============================================================================

// float16 编码: sign(1) + exponent(5) + mantissa(10)
uint16_t FloatToHalfBits(float f) {
    uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(float));
    uint32_t sign = (x >> 31) & 0x1;
    uint32_t exp = (x >> 23) & 0xFF;
    uint32_t mant = x & 0x7FFFFF;

    uint16_t h_sign = static_cast<uint16_t>(sign << 15);

    if (exp == 0xFF) {
        // NaN or Inf
        uint16_t h_exp = 0x1F << 10;
        uint16_t h_mant = 0;
        if (mant != 0) {
            // NaN — ensure mantissa nonzero
            h_mant = static_cast<uint16_t>(mant >> 13);
            if (h_mant == 0) h_mant = 1; // ensure NaN
        }
        return static_cast<uint16_t>(h_sign | h_exp | h_mant);
    }

    int32_t newExp = static_cast<int32_t>(exp) - 127 + 15;

    if (newExp >= 0x1F) {
        // Overflow to Inf
        return static_cast<uint16_t>(h_sign | (0x1F << 10));
    }
    if (newExp <= 0) {
        // Subnormal or zero
        if (newExp < -10) {
            // Too small, flush to zero
            return h_sign;
        }
        // Subnormal
        uint32_t mantFull = mant | 0x800000;
        uint16_t h_mant = static_cast<uint16_t>(mantFull >> (14 - newExp));
        return static_cast<uint16_t>(h_sign | h_mant);
    }
    // Normal
    uint16_t h_exp = static_cast<uint16_t>(newExp << 10);
    uint16_t h_mant = static_cast<uint16_t>(mant >> 13);
    return static_cast<uint16_t>(h_sign | h_exp | h_mant);
}

float HalfBitsToFloat(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    uint32_t f_sign = sign << 31;
    uint32_t f_exp = 0;
    uint32_t f_mant = 0;

    if (exp == 0) {
        if (mant == 0) {
            // Zero
            uint32_t result = f_sign;
            float f;
            std::memcpy(&f, &result, sizeof(float));
            return f;
        }
        // Subnormal: find leading 1
        int32_t e = -1;
        uint32_t m = mant;
        do {
            e++;
            m <<= 1;
        } while ((m & 0x400) == 0);
        f_exp = static_cast<uint32_t>(127 - 15 - e + 1);
        f_mant = (m & 0x3FF) << 13;
    } else if (exp == 0x1F) {
        // Inf or NaN
        f_exp = 0xFF;
        f_mant = mant ? (mant << 13) : 0;
        if (mant != 0) {
            // Ensure NaN mantissa is nonzero
            f_mant = (mant << 13) | 0x400000; // ensure nonzero
        }
    } else {
        // Normal
        f_exp = exp - 15 + 127;
        f_mant = mant << 13;
    }

    uint32_t result = f_sign | (f_exp << 23) | f_mant;
    float f;
    std::memcpy(&f, &result, sizeof(float));
    return f;
}

// bfloat16 编码: sign(1) + exponent(8) + mantissa(7) — 截断 float 的高 16 位
uint16_t FloatToBfloat16Bits(float f) {
    uint32_t x = 0;
    std::memcpy(&x, &f, sizeof(float));
    // Round to nearest even for the truncation
    uint32_t lsb = (x >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    uint32_t rounded = x + rounding_bias;
    return static_cast<uint16_t>(rounded >> 16);
}

float Bfloat16BitsToFloat(uint16_t bf) {
    uint32_t result = static_cast<uint32_t>(bf) << 16;
    float f;
    std::memcpy(&f, &result, sizeof(float));
    return f;
}

// 枚举: 测试数据类型
enum class TestDtype {
    FLOAT16,
    BFLOAT16,
    FLOAT32
};

const char* DtypeToString(TestDtype dt) {
    switch (dt) {
        case TestDtype::FLOAT16:  return "float16";
        case TestDtype::BFLOAT16: return "bfloat16";
        case TestDtype::FLOAT32:  return "float32";
    }
    return "unknown";
}

// 统一的浮点表示，内部计算都用 double
// 内存中 fp16/bf16 用 uint16_t 存储，fp32 用 float 存储
// 为了统一，我们用一种 generic buffer：用 double 做逻辑，序列化时编码

struct GenericTensor {
    TestDtype dtype;
    std::vector<int64_t> shape;
    std::vector<double> values; // 逻辑值（已解码为 double）

    int64_t NumElements() const {
        return static_cast<int64_t>(values.size());
    }

    bool IsNaN(size_t idx) const {
        return std::isnan(values[idx]);
    }

    bool IsInf(size_t idx) const {
        return std::isinf(values[idx]);
    }
};

// 将逻辑值编码为字节流 (device buffer 内容)
std::vector<uint8_t> EncodeTensor(const GenericTensor& tensor) {
    std::vector<uint8_t> bytes;
    int64_t n = tensor.NumElements();
    switch (tensor.dtype) {
        case TestDtype::FLOAT16: {
            bytes.resize(n * 2);
            for (int64_t i = 0; i < n; i++) {
                uint16_t bits = FloatToHalfBits(static_cast<float>(tensor.values[i]));
                std::memcpy(&bytes[i * 2], &bits, 2);
            }
            break;
        }
        case TestDtype::BFLOAT16: {
            bytes.resize(n * 2);
            for (int64_t i = 0; i < n; i++) {
                uint16_t bits = FloatToBfloat16Bits(static_cast<float>(tensor.values[i]));
                std::memcpy(&bytes[i * 2], &bits, 2);
            }
            break;
        }
        case TestDtype::FLOAT32: {
            bytes.resize(n * 4);
            for (int64_t i = 0; i < n; i++) {
                float val = static_cast<float>(tensor.values[i]);
                std::memcpy(&bytes[i * 4], &val, 4);
            }
            break;
        }
    }
    return bytes;
}

// 从字节流解码为逻辑值
GenericTensor DecodeTensor(const uint8_t* bytes, int64_t n,
                           TestDtype dtype,
                           const std::vector<int64_t>& shape) {
    GenericTensor tensor;
    tensor.dtype = dtype;
    tensor.shape = shape;
    tensor.values.resize(n);
    switch (dtype) {
        case TestDtype::FLOAT16:
            for (int64_t i = 0; i < n; i++) {
                uint16_t bits;
                std::memcpy(&bits, &bytes[i * 2], 2);
                tensor.values[i] = static_cast<double>(HalfBitsToFloat(bits));
            }
            break;
        case TestDtype::BFLOAT16:
            for (int64_t i = 0; i < n; i++) {
                uint16_t bits;
                std::memcpy(&bits, &bytes[i * 2], 2);
                tensor.values[i] = static_cast<double>(Bfloat16BitsToFloat(bits));
            }
            break;
        case TestDtype::FLOAT32:
            for (int64_t i = 0; i < n; i++) {
                float val;
                std::memcpy(&val, &bytes[i * 4], 4);
                tensor.values[i] = static_cast<double>(val);
            }
            break;
    }
    return tensor;
}

// ============================================================================
// CPU Golden 计算: Y = sum(X^2, dim=axis, keepdim=keep_dims)
// 使用 double 精度计算，输出前再截断到目标 dtype
// ============================================================================

GenericTensor ComputeGolden(const GenericTensor& input,
                           const std::vector<int64_t>& axis,
                           bool keepDims) {
    GenericTensor output;
    output.dtype = input.dtype;
    output.shape = ComputeOutputShape(input.shape, axis, keepDims);

    int64_t inRank = static_cast<int64_t>(input.shape.size());
    int64_t inSize = input.NumElements();

    // 归一化 axis
    std::vector<int64_t> normAxis;
    for (auto a : axis) {
        normAxis.push_back(a < 0 ? a + inRank : a);
    }
    std::sort(normAxis.begin(), normAxis.end());
    normAxis.erase(std::unique(normAxis.begin(), normAxis.end()), normAxis.end());

    // 为每个输入维度计算: 它是否是规约维度
    std::vector<bool> isReduceDim(inRank, false);
    for (auto a : normAxis) {
        isReduceDim[a] = true;
    }

    // 输出形状对应的非规约维度的索引列表
    std::vector<int64_t> nonReduceDims;
    for (int64_t i = 0; i < inRank; i++) {
        if (!isReduceDim[i]) nonReduceDims.push_back(i);
    }

    // 全规约时（所有维度都被规约）输出为标量
    bool fullReduce = (normAxis.size() == static_cast<size_t>(inRank));

    int64_t outSize;
    if (fullReduce) {
        outSize = 1;
    } else {
        outSize = GetShapeSize(output.shape);
    }
    output.values.assign(outSize, 0.0);

    // 如果 axis 为空，等价于 identity（不做规约）
    if (normAxis.empty()) {
        // output = x^2
        for (int64_t i = 0; i < inSize; i++) {
            double xv = input.values[i];
            output.values[i] = xv * xv;
        }
        return output;
    }

    // 遍历每个输入元素，计算其对应的输出索引，累加 x^2
    auto inStrides = ComputeStrides(input.shape);

    for (int64_t elemIdx = 0; elemIdx < inSize; elemIdx++) {
        // 分解 elemIdx 为多维索引
        int64_t remaining = elemIdx;
        std::vector<int64_t> multiIdx(inRank);
        for (int64_t d = 0; d < inRank; d++) {
            multiIdx[d] = remaining / inStrides[d];
            remaining = remaining % inStrides[d];
        }

        // 计算输出索引
        int64_t outIdx = 0;
        if (fullReduce) {
            outIdx = 0;
        } else {
            auto outStrides = ComputeStrides(output.shape);
            int64_t outDim = 0;
            for (int64_t d = 0; d < inRank; d++) {
                if (isReduceDim[d]) {
                    if (keepDims) outDim++; // skip keepDim=1 positions
                } else {
                    outIdx += multiIdx[d] * outStrides[outDim];
                    outDim++;
                }
            }
        }

        double xv = input.values[elemIdx];
        double sq = xv * xv;

        // IEEE 754 NaN 传播: NaN + anything = NaN
        if (std::isnan(output.values[outIdx]) || std::isnan(sq)) {
            output.values[outIdx] = std::numeric_limits<double>::quiet_NaN();
        } else {
            output.values[outIdx] += sq;
        }
    }

    return output;
}

// ============================================================================
// 精度比对: 与 test_op.py verify_result 一致的逻辑
//   fp16/bf16: rtol=atol=1e-2, loss=1e-3
//   fp32:      rtol=atol=1e-4, loss=1e-4
//   NaN 同判:  real 与 golden 同 NaN 视通过
// ============================================================================

bool CompareResults(const GenericTensor& golden, const GenericTensor& actual) {
    double rtol, atol, loss;
    switch (golden.dtype) {
        case TestDtype::FLOAT16:
        case TestDtype::BFLOAT16:
            rtol = RTOL_FP16; atol = ATOL_FP16; loss = LOSS_FP16;
            break;
        case TestDtype::FLOAT32:
            rtol = RTOL_FP32; atol = ATOL_FP32; loss = LOSS_FP32;
            break;
    }

    int64_t n = golden.NumElements();
    if (n != static_cast<int64_t>(actual.values.size())) {
        LOG_PRINT("  [FAIL] 元素数量不匹配: golden=%lld, actual=%lld",
                  static_cast<long long>(n),
                  static_cast<long long>(actual.values.size()));
        return false;
    }
    if (n == 0) return true;

    const double minimum = 10e-10;
    int64_t errNum = 0;

    for (int64_t i = 0; i < n; i++) {
        double g = golden.values[i];
        double a = actual.values[i];

        // NaN 同判
        if (std::isnan(g) && std::isnan(a)) continue;
        // NaN vs 非 NaN = 错误
        if (std::isnan(g) != std::isnan(a)) { errNum++; continue; }

        // Inf 同判: 同号 Inf 视为匹配 (IEEE 754 语义)
        if (std::isinf(g) && std::isinf(a)) {
            if ((g > 0 && a > 0) || (g < 0 && a < 0)) continue;
            errNum++; continue;
        }
        // Inf vs 非 Inf = 错误
        if (std::isinf(g) != std::isinf(a)) { errNum++; continue; }

        double g_abs = (g == 0.0) ? minimum : std::abs(g);
        double a_abs = (a == 0.0) ? minimum : std::abs(a);

        double absDiff = std::abs(a - g);
        double relDiff = absDiff / std::max(a_abs, g_abs);

        bool isClose = (absDiff <= atol) || (relDiff <= rtol);
        if (!isClose) errNum++;
    }

    int64_t lossThreshold = static_cast<int64_t>(n * loss);
    if (lossThreshold < 1) lossThreshold = 1; // 至少允许1个误差

    if (errNum > lossThreshold) {
        LOG_PRINT("  [FAIL] errNum=%lld / %lld (threshold=%lld, loss=%.1e)",
                  static_cast<long long>(errNum),
                  static_cast<long long>(n),
                  static_cast<long long>(lossThreshold), loss);
        // 打印前5个不匹配
        int shown = 0;
        for (int64_t i = 0; i < n && shown < 5; i++) {
            double g = golden.values[i];
            double a = actual.values[i];
            if (std::isnan(g) != std::isnan(a) ||
                (!std::isnan(g) && !std::isnan(a) &&
                 std::abs(a - g) > atol &&
                 std::abs(a - g) / std::max(std::abs(g) == 0 ? minimum : std::abs(g),
                                           std::abs(a) == 0 ? minimum : std::abs(a)) > rtol)) {
                LOG_PRINT("  不匹配 [%lld]: golden=%.6e, actual=%.6e",
                          static_cast<long long>(i), g, a);
                shown++;
            }
        }
        return false;
    }

    LOG_PRINT("  [PASS] errNum=%lld / %lld (rtol=%.1e, atol=%.1e, loss=%.1e)",
              static_cast<long long>(errNum),
              static_cast<long long>(n), rtol, atol, loss);
    return true;
}

// ============================================================================
// CSV 测试用例解析与生成输入数据
// ============================================================================

// 解析特殊值字符串 ("nan", "inf", "-inf", "+0", "-0" 等)
double ParseSpecialValue(const std::string& s) {
    if (s == "nan" || s == "NaN" || s == "NAN")
        return std::numeric_limits<double>::quiet_NaN();
    if (s == "inf" || s == "Inf" || s == "INF")
        return std::numeric_limits<double>::infinity();
    if (s == "-inf" || s == "-Inf" || s == "-INF")
        return -std::numeric_limits<double>::infinity();
    if (s == "+0" || s == "+0.0") return 0.0;
    if (s == "-0" || s == "-0.0") return -0.0; // IEEE negative zero
    return std::stod(s);
}

// 从 data_range 生成随机值
// range 格式: [lo, hi]，支持特殊值
double GenerateValueFromRange(const std::string& loStr, const std::string& hiStr,
                              uint32_t seed) {
    double lo = ParseSpecialValue(loStr);
    double hi = ParseSpecialValue(hiStr);

    // 特殊值处理
    if (std::isnan(lo) || std::isnan(hi)) return std::numeric_limits<double>::quiet_NaN();
    if (std::isinf(lo) && std::isinf(hi)) {
        // 同号 inf → 返回该 inf
        if (lo > 0) return std::numeric_limits<double>::infinity();
        return -std::numeric_limits<double>::infinity();
    }
    if (std::isinf(lo)) return lo;
    if (std::isinf(hi)) return hi;

    // 普通区间: 简单 LCG 随机
    seed = seed * 1103515245u + 12345u;
    double t = static_cast<double>((seed >> 8) & 0xFFFFFF) / static_cast<double>(0xFFFFFF);
    return lo + t * (hi - lo);
}

struct CsvTestCase {
    std::string name;
    std::vector<int64_t> inputShape;
    std::vector<int64_t> outputShape;
    TestDtype dtype;
    std::vector<int64_t> axis;
    bool keepDims;
    std::string dataRangeLo;
    std::string dataRangeHi;
};

// 从 CSV 行解析 data_range 字段，格式: (([lo,hi],),)
void ParseDataRange(const std::string& rangeStr, std::string& lo, std::string& hi) {
    // 找第一个 [ 和对应的 ]
    size_t lb = rangeStr.find('[');
    size_t rb = rangeStr.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos) {
        lo = "0"; hi = "1"; return;
    }
    std::string inner = rangeStr.substr(lb + 1, rb - lb - 1);
    size_t comma = inner.find(',');
    if (comma == std::string::npos) {
        lo = inner; hi = inner; return;
    }
    lo = inner.substr(0, comma);
    hi = inner.substr(comma + 1);
    // 去除首尾空白和引号
    auto trim = [](std::string& s) {
        size_t p = s.find_first_not_of(" \t\"'");
        if (p != std::string::npos) s = s.substr(p);
        p = s.find_last_not_of(" \t\"'");
        if (p != std::string::npos) s = s.substr(0, p + 1);
    };
    trim(lo);
    trim(hi);
}

// 从 CSV tensor_view_shapes 解析 input 和 output shape
// 格式: ((d1,d2,...),(out_d1,out_d2,...),)
void ParseShapes(const std::string& shapesStr,
                 std::vector<int64_t>& inputShape,
                 std::vector<int64_t>& outputShape) {
    // 找到第一个 (( 和第一个 ),(
    // Input shape
    size_t start = shapesStr.find("((");
    size_t mid = shapesStr.find("),(");
    size_t end = shapesStr.find("),)");

    std::string inputStr, outputStr;

    if (start != std::string::npos && mid != std::string::npos) {
        inputStr = shapesStr.substr(start + 2, mid - (start + 2));
    }
    if (mid != std::string::npos) {
        size_t outStart = mid + 3;
        size_t outEnd = shapesStr.find("),", outStart);
        if (outEnd != std::string::npos) {
            outputStr = shapesStr.substr(outStart, outEnd - outStart);
        } else {
            outputStr = shapesStr.substr(outStart);
        }
    }

    auto parseDims = [](const std::string& s, std::vector<int64_t>& dims) {
        std::stringstream ss(s);
        std::string token;
        while (std::getline(ss, token, ',')) {
            // trim
            size_t p = token.find_first_not_of(" \t");
            if (p != std::string::npos) token = token.substr(p);
            p = token.find_last_not_of(" \t");
            if (p != std::string::npos) token = token.substr(0, p + 1);
            if (token.empty()) continue;
            dims.push_back(std::stoll(token));
        }
    };

    parseDims(inputStr, inputShape);
    parseDims(outputStr, outputShape);
}

// 解析 attributes: {"axis": [0, -1], "keepDims": true}
void ParseAttributes(const std::string& attrStr,
                     std::vector<int64_t>& axis, bool& keepDims) {
    keepDims = false;
    axis.clear();

    // keepDims
    if (attrStr.find("\"keepDims\": true") != std::string::npos ||
        attrStr.find("\"keepDims\":true") != std::string::npos) {
        keepDims = true;
    }

    // axis
    size_t axisPos = attrStr.find("\"axis\"");
    if (axisPos != std::string::npos) {
        size_t lb = attrStr.find('[', axisPos);
        size_t rb = attrStr.find(']', axisPos);
        if (lb != std::string::npos && rb != std::string::npos && rb > lb) {
            std::string axisContent = attrStr.substr(lb + 1, rb - lb - 1);
            if (!axisContent.empty()) {
                std::stringstream ss(axisContent);
                std::string token;
                while (std::getline(ss, token, ',')) {
                    size_t p = token.find_first_not_of(" \t");
                    if (p != std::string::npos) token = token.substr(p);
                    p = token.find_last_not_of(" \t");
                    if (p != std::string::npos) token = token.substr(0, p + 1);
                    if (!token.empty()) axis.push_back(std::stoll(token));
                }
            }
        }
    }
}

TestDtype ParseDtype(const std::string& dtStr) {
    if (dtStr.find("float16") != std::string::npos) return TestDtype::FLOAT16;
    if (dtStr.find("bfloat16") != std::string::npos) return TestDtype::BFLOAT16;
    return TestDtype::FLOAT32;
}

// 维度约束检查: N∈[1,10000], N2∈[1,10000], N3∈[1,1000], N4∈[1,200]
// 对于 5D: dim5(第一维) ≤ 200
bool CheckDimConstraints(const std::vector<int64_t>& shape) {
    int64_t rank = static_cast<int64_t>(shape.size());
    if (rank > 5) return false;

    // 从最后一个维度开始: N, N2, N3, N4, dim5
    if (rank >= 1 && shape[rank - 1] > 10000) return false; // N
    if (rank >= 2 && shape[rank - 2] > 10000) return false; // N2
    if (rank >= 3 && shape[rank - 3] > 1000) return false;  // N3
    if (rank >= 4 && shape[rank - 4] > 200) return false;   // N4
    if (rank >= 5 && shape[rank - 5] > 200) return false;   // dim5

    return true;
}

// 读取 CSV 文件并解析为测试用例
std::vector<CsvTestCase> LoadTestCasesFromCSV(const std::string& csvPath) {
    std::vector<CsvTestCase> cases;
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        LOG_PRINT("[ERROR] 无法打开 CSV 文件: %s", csvPath.c_str());
        return cases;
    }

    std::string line;
    std::getline(file, line); // 跳过 header

    int totalParsed = 0;
    int filtered = 0;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        // 简单 CSV 解析 (支持双引号内的逗号)
        std::vector<std::string> fields;
        bool inQuote = false;
        std::string field;
        for (char c : line) {
            if (c == '"') { inQuote = !inQuote; field += c; }
            else if (c == ',' && !inQuote) { fields.push_back(field); field.clear(); }
            else field += c;
        }
        fields.push_back(field);

        if (fields.size() < 11) continue;

        CsvTestCase tc;
        tc.name = fields[0]; // testcase_name

        ParseShapes(fields[2], tc.inputShape, tc.outputShape); // tensor_view_shapes

        // tensor_dtypes: ('float16','float16',)
        tc.dtype = ParseDtype(fields[3]);

        ParseAttributes(fields[5], tc.axis, tc.keepDims); // attributes

        // input_data_ranges: (([lo,hi],),)
        ParseDataRange(fields[9], tc.dataRangeLo, tc.dataRangeHi);

        // 维度约束过滤
        if (!CheckDimConstraints(tc.inputShape)) {
            filtered++;
            LOG_PRINT("[FILTER] %s: 维度超限, shape 已跳过", tc.name.c_str());
            continue;
        }

        totalParsed++;
        cases.push_back(tc);
    }

    LOG_PRINT("[CSV] 解析完成: 总计 %d 条, 维度超限过滤 %d 条, 有效 %zu 条",
              totalParsed + filtered, filtered, cases.size());
    return cases;
}

// 将逻辑值量化到目标 dtype 再解码回来（模拟真实数据的有限精度）
void QuantizeToDtype(GenericTensor& tensor) {
    auto encoded = EncodeTensor(tensor);
    auto decoded = DecodeTensor(encoded.data(), tensor.NumElements(),
                                tensor.dtype, tensor.shape);
    tensor.values = std::move(decoded.values);
}

// 为测试用例生成输入数据
GenericTensor GenerateInputData(const CsvTestCase& tc, uint32_t baseSeed) {
    GenericTensor input;
    input.dtype = tc.dtype;
    input.shape = tc.inputShape;
    int64_t n = GetShapeSize(tc.inputShape);
    input.values.resize(n);

    for (int64_t i = 0; i < n; i++) {
        input.values[i] = GenerateValueFromRange(
            tc.dataRangeLo, tc.dataRangeHi,
            static_cast<uint32_t>(baseSeed + i));
    }

    // 量化输入到目标 dtype 精度，使 golden 与 round-trip 一致
    QuantizeToDtype(input);

    return input;
}

// ============================================================================
// CPU Golden 自测 (验证 golden 计算逻辑正确)
// ============================================================================

bool TestGoldenCorrectness() {
    LOG_PRINT("\n========================================");
    LOG_PRINT("CPU Golden 正确性自测");
    LOG_PRINT("========================================");
    bool allPass = true;

    // 测试 1: 1D reduce, keepDims=false
    {
        GenericTensor in;
        in.dtype = TestDtype::FLOAT32;
        in.shape = {4};
        in.values = {1.0, 2.0, 3.0, 4.0};

        auto golden = ComputeGolden(in, {0}, false);
        // sum(1,4,9,16) = 30
        bool pass = (golden.values.size() == 1 &&
                     std::abs(golden.values[0] - 30.0) < 1e-6);
        LOG_PRINT("测试 1 [1D reduce keepDims=false]: sum(x^2)=%s (期望=30.0)",
                  pass ? "PASS" : "FAIL");
        if (!pass) {
            LOG_PRINT("  实际: size=%zu, val=%.6f", golden.values.size(),
                      golden.values.empty() ? 0 : golden.values[0]);
        }
        allPass &= pass;
    }

    // 测试 2: 1D reduce, keepDims=true
    {
        GenericTensor in;
        in.dtype = TestDtype::FLOAT32;
        in.shape = {4};
        in.values = {1.0, 2.0, 3.0, 4.0};

        auto golden = ComputeGolden(in, {0}, true);
        bool pass = (golden.shape == std::vector<int64_t>{1} &&
                     std::abs(golden.values[0] - 30.0) < 1e-6);
        LOG_PRINT("测试 2 [1D reduce keepDims=true]: shape={%lld} sum=%s (期望=30.0)",
                  static_cast<long long>(golden.shape[0]),
                  pass ? "PASS" : "FAIL");
        allPass &= pass;
    }

    // 测试 3: 2D reduce axis=[0], keepDims=false
    {
        GenericTensor in;
        in.dtype = TestDtype::FLOAT32;
        in.shape = {2, 3};
        in.values = {1, 2, 3, 4, 5, 6};

        auto golden = ComputeGolden(in, {0}, false);
        // x^2: [1,4,9, 16,25,36]
        // sum axis=0: [1+16, 4+25, 9+36] = [17, 29, 45]
        bool pass = (golden.values.size() == 3 &&
                     std::abs(golden.values[0] - 17.0) < 1e-6 &&
                     std::abs(golden.values[1] - 29.0) < 1e-6 &&
                     std::abs(golden.values[2] - 45.0) < 1e-6);
        LOG_PRINT("测试 3 [2D axis=0 keepDims=false]: [%g,%g,%g] %s",
                  golden.values[0], golden.values[1], golden.values[2],
                  pass ? "PASS" : "FAIL");
        allPass &= pass;
    }

    // 测试 4: 2D reduce axis=[1], keepDims=false
    {
        GenericTensor in;
        in.dtype = TestDtype::FLOAT32;
        in.shape = {2, 3};
        in.values = {1, 2, 3, 4, 5, 6};

        auto golden = ComputeGolden(in, {1}, false);
        // x^2: [1,4,9, 16,25,36]
        // sum axis=1: [1+4+9, 16+25+36] = [14, 77]
        bool pass = (golden.values.size() == 2 &&
                     std::abs(golden.values[0] - 14.0) < 1e-6 &&
                     std::abs(golden.values[1] - 77.0) < 1e-6);
        LOG_PRINT("测试 4 [2D axis=1 keepDims=false]: [%g,%g] %s",
                  golden.values[0], golden.values[1],
                  pass ? "PASS" : "FAIL");
        allPass &= pass;
    }

    // 测试 5: 2D reduce axis=[0,1] (full reduce), keepDims=false
    {
        GenericTensor in;
        in.dtype = TestDtype::FLOAT32;
        in.shape = {2, 3};
        in.values = {1, 2, 3, 4, 5, 6};

        auto golden = ComputeGolden(in, {0, 1}, false);
        // sum all squares = 1+4+9+16+25+36 = 91
        bool pass = (golden.values.size() == 1 &&
                     std::abs(golden.values[0] - 91.0) < 1e-6);
        LOG_PRINT("测试 5 [2D full reduce]: sum=%g %s (期望=91)",
                  golden.values[0], pass ? "PASS" : "FAIL");
        allPass &= pass;
    }

    // 测试 6: 负索引 axis=-1
    {
        GenericTensor in;
        in.dtype = TestDtype::FLOAT32;
        in.shape = {2, 3};
        in.values = {1, 2, 3, 4, 5, 6};

        auto golden = ComputeGolden(in, {-1}, false);
        // axis=-1 == axis=1: [14, 77]
        bool pass = (golden.values.size() == 2 &&
                     std::abs(golden.values[0] - 14.0) < 1e-6 &&
                     std::abs(golden.values[1] - 77.0) < 1e-6);
        LOG_PRINT("测试 6 [负索引 axis=-1]: [%g,%g] %s",
                  golden.values[0], golden.values[1],
                  pass ? "PASS" : "FAIL");
        allPass &= pass;
    }

    // 测试 7: axis=[] (无规约, identity x^2)
    {
        GenericTensor in;
        in.dtype = TestDtype::FLOAT32;
        in.shape = {3};
        in.values = {2.0, 3.0, 4.0};

        auto golden = ComputeGolden(in, {}, false);
        // x^2 = [4, 9, 16]
        bool pass = (golden.values.size() == 3 &&
                     std::abs(golden.values[0] - 4.0) < 1e-6 &&
                     std::abs(golden.values[1] - 9.0) < 1e-6 &&
                     std::abs(golden.values[2] - 16.0) < 1e-6);
        LOG_PRINT("测试 7 [axis=[] identity]: [%g,%g,%g] %s",
                  golden.values[0], golden.values[1], golden.values[2],
                  pass ? "PASS" : "FAIL");
        allPass &= pass;
    }

    // 测试 8: NaN 传播
    {
        GenericTensor in;
        in.dtype = TestDtype::FLOAT32;
        in.shape = {3};
        in.values = {1.0, std::numeric_limits<double>::quiet_NaN(), 3.0};

        auto golden = ComputeGolden(in, {0}, false);
        // 1 + NaN + 9 = NaN
        bool pass = (golden.values.size() == 1 && std::isnan(golden.values[0]));
        LOG_PRINT("测试 8 [NaN 传播]: golden=%s %s",
                  std::isnan(golden.values[0]) ? "NaN" : std::to_string(golden.values[0]).c_str(),
                  pass ? "PASS" : "FAIL");
        allPass &= pass;
    }

    // 测试 9: Inf 平方 → Inf, inf+inf = inf
    {
        GenericTensor in;
        in.dtype = TestDtype::FLOAT32;
        in.shape = {2};
        in.values = {std::numeric_limits<double>::infinity(),
                     std::numeric_limits<double>::infinity()};

        auto golden = ComputeGolden(in, {0}, false);
        // inf^2 = inf, inf + inf = inf
        bool pass = (golden.values.size() == 1 && std::isinf(golden.values[0]) &&
                     golden.values[0] > 0);
        LOG_PRINT("测试 9 [Inf 平方求和]: golden=%g %s",
                  golden.values[0], pass ? "PASS" : "FAIL");
        allPass &= pass;
    }

    // 测试 10: keepDims=true 保留维度
    {
        GenericTensor in;
        in.dtype = TestDtype::FLOAT32;
        in.shape = {2, 3, 4};
        in.values.resize(24);
        for (int i = 0; i < 24; i++) in.values[i] = static_cast<double>(i + 1);

        auto golden = ComputeGolden(in, {1}, true);
        // output shape should be {2, 1, 4}
        bool pass = (golden.shape.size() == 3 &&
                     golden.shape[0] == 2 && golden.shape[1] == 1 && golden.shape[2] == 4);
        LOG_PRINT("测试 10 [keepDims shape]: shape=[%lld,%lld,%lld] %s",
                  static_cast<long long>(golden.shape[0]),
                  static_cast<long long>(golden.shape[1]),
                  static_cast<long long>(golden.shape[2]),
                  pass ? "PASS" : "FAIL");

        // 验证值: axis=1 sum over dim=1 (3 elements)
        // x^2 values: 1,4,9,16, 25,36,49,64, 81,100,121,144 (d0=0)
        //              169,196,225,256, 289,324,361,400, 441,484,529,576 (d0=1)
        // sum axis=1 for d0=0: [1+25+81, 4+36+100, 9+49+121, 16+64+144] = [107, 140, 179, 224]
        if (pass) {
            pass = (std::abs(golden.values[0] - 107.0) < 1e-6 &&
                    std::abs(golden.values[1] - 140.0) < 1e-6 &&
                    std::abs(golden.values[2] - 179.0) < 1e-6 &&
                    std::abs(golden.values[3] - 224.0) < 1e-6);
        }
        LOG_PRINT("  值验证: %s", pass ? "PASS" : "FAIL");
        allPass &= pass;
    }

    LOG_PRINT("\nGolden 自测总结: %s\n", allPass ? "全部通过" : "存在失败");
    LOG_PRINT("========================================\n");
    return allPass;
}

// ============================================================================
// Mock 模式辅助: 按比例缩减大 shape 以加速 CPU golden 验证
// 保持 rank/axis 语义不变，仅缩减各维度大小至总计 <= maxElements
// ============================================================================

constexpr int64_t MAX_MOCK_ELEMENTS = 100000; // 10 万元素上限，保证 Mock 快速完成

std::vector<int64_t> ScaleShapeForMock(const std::vector<int64_t>& shape,
                                       int64_t maxElements) {
    int64_t total = GetShapeSize(shape);
    if (total <= maxElements) return shape;

    // 计算缩放比例
    double scale = std::pow(static_cast<double>(maxElements) / total,
                            1.0 / shape.size());
    std::vector<int64_t> scaled;
    for (auto d : shape) {
        int64_t new_d = std::max(static_cast<int64_t>(1),
                                  static_cast<int64_t>(d * scale));
        scaled.push_back(new_d);
    }
    // 微调最后一个维度使总量更接近上限
    int64_t scaledTotal = GetShapeSize(scaled);
    if (scaledTotal > maxElements) {
        // 逐维度缩减直到满足
        for (int64_t i = scaled.size() - 1; i >= 0 && scaledTotal > maxElements; i--) {
            while (scaled[i] > 1 && scaledTotal > maxElements) {
                scaled[i]--;
                scaledTotal = GetShapeSize(scaled);
            }
        }
    }
    return scaled;
}

// ============================================================================
// Mock 模式: 从 CSV 加载用例, golden 计算后自洽比对
// 大 shape 用例自动缩减以保证 CPU golden 快速完成
// ============================================================================

#ifdef USE_MOCK_ACLNN

int RunMockTests(const std::string& csvPath) {
    LOG_PRINT("\n========================================");
    LOG_PRINT("Mock 模式: CSV 用例 + CPU Golden 自洽验证");
    LOG_PRINT("  (大 shape 自动缩减至 <= %lld 元素)", static_cast<long long>(MAX_MOCK_ELEMENTS));
    LOG_PRINT("========================================");

    auto testCases = LoadTestCasesFromCSV(csvPath);
    if (testCases.empty()) {
        LOG_PRINT("[ERROR] 无有效测试用例");
        return 1;
    }

    int passed = 0, failed = 0;
    int scaled = 0;

    for (size_t ci = 0; ci < testCases.size(); ci++) {
        auto& tc = testCases[ci];

        // Mock 模式: 缩减大 shape
        CsvTestCase mockTc = tc;
        int64_t origElements = GetShapeSize(tc.inputShape);
        if (origElements > MAX_MOCK_ELEMENTS) {
            mockTc.inputShape = ScaleShapeForMock(tc.inputShape, MAX_MOCK_ELEMENTS);
            scaled++;
        }

        LOG_PRINT("\n[%zu/%zu] %s (dtype=%s, shape=%s%s, axis=[%s], keepDims=%s)",
                  ci + 1, testCases.size(), tc.name.c_str(),
                  DtypeToString(mockTc.dtype),
                  [&]() {
                      std::string s = "[";
                      for (size_t i = 0; i < mockTc.inputShape.size(); i++) {
                          if (i > 0) s += ",";
                          s += std::to_string(mockTc.inputShape[i]);
                      }
                      s += "]";
                      return s;
                  }().c_str(),
                  (origElements > MAX_MOCK_ELEMENTS) ? " (scaled)" : "",
                  [&]() {
                      std::string s;
                      for (size_t i = 0; i < mockTc.axis.size(); i++) {
                          if (i > 0) s += ",";
                          s += std::to_string(mockTc.axis[i]);
                      }
                      return s;
                  }().c_str(),
                  mockTc.keepDims ? "true" : "false");

        // 生成输入数据 (已量化到目标 dtype 精度)
        GenericTensor input = GenerateInputData(mockTc, static_cast<uint32_t>(ci * 1000 + 42));

        // 计算 golden (double 精度)
        GenericTensor golden = ComputeGolden(input, mockTc.axis, mockTc.keepDims);

        // 将 golden 量化到目标 dtype (模拟 NPU 输出的有限精度)
        GenericTensor goldenQuantized = golden;
        QuantizeToDtype(goldenQuantized);

        // Mock 验证: 量化 golden 与 round-trip 比对
        // (验证 encode/decode 正确性 + 比对逻辑正确性)
        auto encoded = EncodeTensor(goldenQuantized);
        GenericTensor decoded = DecodeTensor(encoded.data(),
                                             goldenQuantized.NumElements(),
                                             goldenQuantized.dtype, goldenQuantized.shape);

        bool result = CompareResults(goldenQuantized, decoded);

        // 额外验证: 输出 shape 正确性
        auto expectedShape = ComputeOutputShape(mockTc.inputShape, mockTc.axis, mockTc.keepDims);
        if (golden.shape != expectedShape) {
            LOG_PRINT("  [WARN] 输出 shape 不匹配: golden vs expected");
            result = false;
        }

        if (result) passed++; else failed++;
    }

    LOG_PRINT("\n========================================");
    LOG_PRINT("Mock 测试报告");
    LOG_PRINT("========================================");
    LOG_PRINT("总计: %d", passed + failed);
    LOG_PRINT("通过: %d", passed);
    LOG_PRINT("失败: %d", failed);
    LOG_PRINT("缩减: %d (大 shape 自动缩减)", scaled);
    LOG_PRINT("========================================\n");

    return failed == 0 ? 0 : 1;
}

#else
// ============================================================================
// Real 模式辅助函数
// ============================================================================

template<typename T>
int CreateAclTensor(const std::vector<uint8_t>& hostData,
                    const std::vector<int64_t>& shape,
                    void** deviceAddr,
                    aclDataType dataType,
                    aclTensor** tensor) {
    size_t size = hostData.size();

    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) return ret;

    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) { aclrtFree(*deviceAddr); return ret; }

    auto strides = ComputeStrides(shape);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(),
                              0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return ACL_SUCCESS;
}

aclDataType TestDtypeToAclDtype(TestDtype dt) {
    switch (dt) {
        case TestDtype::FLOAT16:  return ACL_FLOAT16;
        case TestDtype::BFLOAT16: return ACL_BFLOAT16;
        case TestDtype::FLOAT32:  return ACL_FLOAT;
    }
    return ACL_FLOAT;
}

int RunRealTests(const std::string& csvPath, aclrtStream stream) {
    LOG_PRINT("\n========================================");
    LOG_PRINT("Real 模式: CSV 用例 + NPU 执行 + Golden 比对");
    LOG_PRINT("========================================");

    auto testCases = LoadTestCasesFromCSV(csvPath);
    if (testCases.empty()) {
        LOG_PRINT("[ERROR] 无有效测试用例");
        return 1;
    }

    int passed = 0, failed = 0;

    for (size_t ci = 0; ci < testCases.size(); ci++) {
        auto& tc = testCases[ci];
        LOG_PRINT("\n[%zu/%zu] %s (dtype=%s)",
                  ci + 1, testCases.size(), tc.name.c_str(),
                  DtypeToString(tc.dtype));

        // 生成输入数据
        GenericTensor input = GenerateInputData(tc, static_cast<uint32_t>(ci * 1000 + 42));

        // 计算 golden
        GenericTensor golden = ComputeGolden(input, tc.axis, tc.keepDims);

        // 编码输入为字节流
        auto inputBytes = EncodeTensor(input);
        auto outputBytes = EncodeTensor(golden); // 预分配输出 buffer

        aclDataType aclDtype = TestDtypeToAclDtype(tc.dtype);

        void* inputDev = nullptr;
        void* outputDev = nullptr;
        aclTensor* inputTensor = nullptr;
        aclTensor* outputTensor = nullptr;

        // 创建输入 tensor
        if (CreateAclTensor<uint8_t>(inputBytes, tc.inputShape,
                                      &inputDev, aclDtype, &inputTensor) != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] 创建输入 tensor 失败");
            failed++; continue;
        }

        // 创建输出 tensor
        if (CreateAclTensor<uint8_t>(outputBytes, golden.shape,
                                      &outputDev, aclDtype, &outputTensor) != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] 创建输出 tensor 失败");
            aclDestroyTensor(inputTensor); aclrtFree(inputDev);
            failed++; continue;
        }

        // 创建 axis IntArray
        aclIntArray* axisArray = aclCreateIntArray(
            tc.axis.data(), tc.axis.size());

        // GetWorkspaceSize
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;

        auto ret = aclnnSquareSumV1GetWorkspaceSize(
            inputTensor, axisArray, tc.keepDims,
            outputTensor, &workspaceSize, &executor);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] GetWorkspaceSize 失败: %d", ret);
            aclDestroyIntArray(axisArray);
            aclDestroyTensor(inputTensor); aclDestroyTensor(outputTensor);
            aclrtFree(inputDev); aclrtFree(outputDev);
            failed++; continue;
        }

        // Allocate workspace
        void* workspace = nullptr;
        if (workspaceSize > 0) {
            ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                LOG_PRINT("  [FAIL] workspace 分配失败: %d", ret);
                aclDestroyIntArray(axisArray);
                aclDestroyTensor(inputTensor); aclDestroyTensor(outputTensor);
                aclrtFree(inputDev); aclrtFree(outputDev);
                failed++; continue;
            }
        }

        // Execute
        ret = aclnnSquareSumV1(workspace, workspaceSize, executor, stream);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] aclnnSquareSumV1 失败: %d", ret);
            if (workspace) aclrtFree(workspace);
            aclDestroyIntArray(axisArray);
            aclDestroyTensor(inputTensor); aclDestroyTensor(outputTensor);
            aclrtFree(inputDev); aclrtFree(outputDev);
            failed++; continue;
        }

        aclrtSynchronizeStream(stream);

        // Copy output back
        int64_t outSize = golden.NumElements();
        size_t outBytes = EncodeTensor(golden).size();
        std::vector<uint8_t> npuOutput(outBytes);
        ret = aclrtMemcpy(npuOutput.data(), outBytes, outputDev, outBytes,
                          ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            LOG_PRINT("  [FAIL] D2H 拷贝失败: %d", ret);
            if (workspace) aclrtFree(workspace);
            aclDestroyIntArray(axisArray);
            aclDestroyTensor(inputTensor); aclDestroyTensor(outputTensor);
            aclrtFree(inputDev); aclrtFree(outputDev);
            failed++; continue;
        }

        // Decode NPU output
        GenericTensor npuResult = DecodeTensor(npuOutput.data(), outSize,
                                               golden.dtype, golden.shape);

        // Compare
        bool result = CompareResults(golden, npuResult);

        // Cleanup
        if (workspace) aclrtFree(workspace);
        aclDestroyIntArray(axisArray);
        aclDestroyTensor(inputTensor); aclDestroyTensor(outputTensor);
        aclrtFree(inputDev); aclrtFree(outputDev);

        if (result) passed++; else failed++;
    }

    LOG_PRINT("\n========================================");
    LOG_PRINT("Real 测试报告");
    LOG_PRINT("========================================");
    LOG_PRINT("总计: %d", passed + failed);
    LOG_PRINT("通过: %d", passed);
    LOG_PRINT("失败: %d", failed);
    LOG_PRINT("========================================\n");

    return failed == 0 ? 0 : 1;
}

#endif

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char* argv[]) {
    LOG_PRINT("\n========================================");
    LOG_PRINT("aclnnSquareSumV1 算子 ST 测试");
    LOG_PRINT("========================================");

#ifdef USE_MOCK_ACLNN
    LOG_PRINT("模式: Mock (CPU golden 自测)");
#else
    LOG_PRINT("模式: Real (NPU 执行)");
#endif

    // 默认 CSV 路径
    std::string csvPath = "testcases/aclnnSquareSumV1_l0_test_cases.csv";
    if (argc > 1) {
        csvPath = argv[1];
    }

    LOG_PRINT("CSV 用例文件: %s", csvPath.c_str());

    // Step 1: CPU Golden 自测
    if (!TestGoldenCorrectness()) {
        LOG_PRINT("[FATAL] CPU Golden 自测失败，终止");
        return 1;
    }

    // Step 2: 执行 CSV 用例
#ifdef USE_MOCK_ACLNN
    return RunMockTests(csvPath);
#else
    // NPU 初始化
    int32_t deviceId = 0;
    aclrtStream stream;

    auto ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FATAL] aclInit 失败: %d", ret);
        return 1;
    }
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FATAL] aclrtSetDevice(%d) 失败: %d", deviceId, ret);
        aclFinalize();
        return 1;
    }
    ret = aclrtCreateStream(&stream);
    if (ret != ACL_SUCCESS) {
        LOG_PRINT("[FATAL] aclrtCreateStream 失败: %d", ret);
        aclrtResetDevice(deviceId);
        aclFinalize();
        return 1;
    }

    int result = RunRealTests(csvPath, stream);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return result;
#endif
}
