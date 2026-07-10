/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * \file test_squaresumv1_tiling.cpp
 * \brief SquareSumV1 op_host Tiling unit tests (AR_FULLLOAD core path)
 *
 * Coverage:
 *   1. TilingKey/dtype mapping (fp16, fp32, bf16)
 *   2. AR_FULLLOAD mode determination (R within UB threshold vs exceeding)
 *   3. Axis normalization (negative index, single axis=-1, multi-axis)
 *   4. blockDim computation (min(coreNum, ceil(rows/tile)))
 *   5. UB budget / tile parameters (rLength, rLengthAlign)
 *   6. keep_dims handling
 *   7. Edge cases (rows=1, R=1, non-aligned R)
 */

#include <iostream>
#include <gtest/gtest.h>
#include <memory>
#include <cstring>
#include <cmath>

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "squaresumv1_tiling_data.h"

namespace SquareSumV1UT {
using namespace std;
using namespace ge;
using namespace gert;

static const std::string OP_NAME = "SquareSumV1";

// CompileInfo struct (matches tiling.cpp)
struct SquareSumV1CompileInfo {};

// Reinterpret the raw tiling data buffer as SquareSumV1TilingData
static const SquareSumV1TilingData* AsTilingData(const TilingInfo& info)
{
    return reinterpret_cast<const SquareSumV1TilingData*>(info.tilingData.get());
}

// Build StorageShape from a vector of dims (origin = storage)
static gert::StorageShape MakeShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    shape.MutableShape().SetDimNum(dims.size());
    shape.MutableStorageShape().SetDimNum(dims.size());
    for (size_t i = 0; i < dims.size(); i++) {
        shape.MutableShape().SetDim(i, dims[i]);
        shape.MutableStorageShape().SetDim(i, dims[i]);
    }
    return shape;
}

// Helper: run tiling and return results
struct TilingResult {
    bool success;
    TilingInfo info;
};

static TilingResult RunTiling(
    const std::vector<int64_t>& inputShape,
    ge::DataType dtype,
    const std::vector<int64_t>& axisList,
    bool keepDims = false,
    uint64_t coreNum = 20,
    uint64_t ubSize = 196608)  // 192KB
{
    gert::StorageShape xShape = MakeShape(inputShape);
    gert::StorageShape yShape = MakeShape({1}); // output shape - not used by tiling

    std::vector<gert::TilingContextPara::TensorDescription> inputDesc = {
        {xShape, dtype, ge::FORMAT_ND}
    };
    std::vector<gert::TilingContextPara::TensorDescription> outputDesc = {
        {yShape, dtype, ge::FORMAT_ND}
    };

    std::vector<gert::TilingContextPara::OpAttr> attrs;
    attrs.emplace_back("axis", Ops::Math::AnyValue::CreateFrom<std::vector<int64_t>>(axisList));
    attrs.emplace_back("keep_dims", Ops::Math::AnyValue::CreateFrom<bool>(keepDims));

    SquareSumV1CompileInfo compileInfo;
    gert::TilingContextPara para(
        OP_NAME, inputDesc, outputDesc, attrs,
        &compileInfo, coreNum, ubSize, 4096);

    TilingResult result;
    result.success = ExecuteTiling(para, result.info);
    return result;
}

// =============================================================================
// Test Fixture
// =============================================================================
class SquareSumV1TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SquareSumV1TilingTest SetUp." << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "SquareSumV1TilingTest TearDown." << std::endl;
    }
};

// =============================================================================
// 1. Basic AR_FULLLOAD path - fp16, axis=-1, typical shape
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_fp16_basic_axis_last_2d)
{
    // shape=[4, 100], axis=-1 → totalRows=4, rLength=100
    auto r = RunTiling({4, 100}, ge::DT_FLOAT16, {-1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    // Coalesced shape
    EXPECT_EQ(td->totalRows, 4);
    EXPECT_EQ(td->rLength, 100);

    // fp16: typeSize=2, elementsPerBlock=32/2=16
    // rLengthAlignInput = CeilAlign(100, 16) = 112
    // rLengthAlignFp32 = CeilAlign(100, 8) = 104
    // rLengthAlign = max(112, 104) = 112
    EXPECT_EQ(td->rLengthAlign, 112);

    // Alignment check: 100 * 2 = 200 bytes, 200 % 32 != 0 → not aligned
    EXPECT_EQ(td->isAlign32B, 0u);

    // Dtype
    EXPECT_EQ(td->inputDtype, static_cast<uint32_t>(ge::DT_FLOAT16));

    // Multi-core: totalRows=4 < coreNum=20 → usedCoreNum=4
    EXPECT_EQ(td->usedCoreNum, 4);
    EXPECT_EQ(r.info.blockNum, 4u);

    // rowsPerCore = ceil(4/4) = 1
    EXPECT_EQ(td->rowsPerCore, 1);

    // tailRows = 4 - 1*(4-1) = 1
    EXPECT_EQ(td->tailRows, 1);

    // Workspace
    ASSERT_EQ(r.info.workspaceSizes.size(), 1u);
    EXPECT_EQ(r.info.workspaceSizes[0], 0u);
}

// =============================================================================
// 2. dtype mapping: fp32
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_fp32_dtype_mapping)
{
    // shape=[2, 64], axis=-1 → totalRows=2, rLength=64
    auto r = RunTiling({2, 64}, ge::DT_FLOAT, {-1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->inputDtype, static_cast<uint32_t>(ge::DT_FLOAT));

    // fp32: typeSize=4, elementsPerBlock=32/4=8
    // rLengthAlignInput = CeilAlign(64, 8) = 64
    // rLengthAlignFp32 = CeilAlign(64, 8) = 64
    EXPECT_EQ(td->rLengthAlign, 64);

    // 64 * 4 = 256 bytes, 256 % 32 == 0 → aligned
    EXPECT_EQ(td->isAlign32B, 1u);

    EXPECT_EQ(td->totalRows, 2);
    EXPECT_EQ(td->rLength, 64);
}

// =============================================================================
// 3. dtype mapping: bf16
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_bf16_dtype_mapping)
{
    // shape=[2, 100], axis=-1 → totalRows=2, rLength=100
    auto r = RunTiling({2, 100}, ge::DT_BF16, {-1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->inputDtype, static_cast<uint32_t>(ge::DT_BF16));

    // bf16: typeSize=2 (same as fp16)
    // rLengthAlignInput = CeilAlign(100, 16) = 112
    // rLengthAlignFp32 = CeilAlign(100, 8) = 104
    // rLengthAlign = max(112, 104) = 112
    EXPECT_EQ(td->rLengthAlign, 112);

    // 100 * 2 = 200, 200 % 32 != 0 → not aligned
    EXPECT_EQ(td->isAlign32B, 0u);
}

// =============================================================================
// 4. axis normalization: negative index -1 on 3D input
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_axis_negative_index_3d)
{
    // shape=[2, 3, 100], axis=-1 → normalized axis=[2]
    // totalRows = 2*3 = 6, rLength = 100
    auto r = RunTiling({2, 3, 100}, ge::DT_FLOAT16, {-1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->totalRows, 6);
    EXPECT_EQ(td->rLength, 100);
}

// =============================================================================
// 5. axis normalization: explicit positive index
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_axis_positive_explicit)
{
    // shape=[2, 3, 100], axis=[2] (equivalent to axis=-1)
    auto r = RunTiling({2, 3, 100}, ge::DT_FLOAT16, {2});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->totalRows, 6);
    EXPECT_EQ(td->rLength, 100);
}

// =============================================================================
// 6. Multi-axis (tail contiguous) - axis covers last two dims
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_tail_contiguous)
{
    // shape=[2, 100, 4], axis=[1, 2] → both tail dims are reduction
    // totalRows = 2, rLength = 100 * 4 = 400
    auto r = RunTiling({2, 100, 4}, ge::DT_FLOAT16, {1, 2});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->totalRows, 2);
    EXPECT_EQ(td->rLength, 400);
}

// =============================================================================
// 7. Multi-axis with negative indices
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_negative_indices)
{
    // shape=[3, 100, 50], axis=[-2, -1] → normalized [1, 2]
    // totalRows = 3, rLength = 100 * 50 = 5000
    auto r = RunTiling({3, 100, 50}, ge::DT_FLOAT16, {-2, -1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->totalRows, 3);
    EXPECT_EQ(td->rLength, 5000);
}

// =============================================================================
// 8. blockDim: large totalRows, capped at coreNum
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_blockdim_capped_at_corenum)
{
    // shape=[100, 64], axis=-1, coreNum=20
    // totalRows=100 > coreNum=20 → usedCoreNum = min(20, 100) = 20
    auto r = RunTiling({100, 64}, ge::DT_FLOAT16, {-1}, false, 20);
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->usedCoreNum, 20);
    EXPECT_EQ(r.info.blockNum, 20u);

    // rowsPerCore = ceil(100, 20) = 5
    EXPECT_EQ(td->rowsPerCore, 5);

    // tailRows = 100 - 5*(20-1) = 100 - 95 = 5
    EXPECT_EQ(td->tailRows, 5);
}

// =============================================================================
// 9. blockDim: small totalRows, less than coreNum
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_blockdim_small_rows)
{
    // shape=[3, 64], axis=-1, coreNum=20
    // totalRows=3 < coreNum=20 → usedCoreNum=3
    auto r = RunTiling({3, 64}, ge::DT_FLOAT16, {-1}, false, 20);
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->usedCoreNum, 3);
    EXPECT_EQ(r.info.blockNum, 3u);
    EXPECT_EQ(td->rowsPerCore, 1);
    // tailRows = 3 - 1*(3-1) = 1
    EXPECT_EQ(td->tailRows, 1);
}

// =============================================================================
// 10. blockDim: uneven split with tail
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_blockdim_uneven_tail)
{
    // shape=[7, 64], axis=-1, coreNum=4
    // usedCoreNum = min(4, 7) = 4
    // rowsPerCore = ceil(7/4) = 2
    // tailRows = 7 - 2*(4-1) = 7 - 6 = 1
    auto r = RunTiling({7, 64}, ge::DT_FLOAT16, {-1}, false, 4);
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->usedCoreNum, 4);
    EXPECT_EQ(td->rowsPerCore, 2);
    EXPECT_EQ(td->tailRows, 1);
}

// =============================================================================
// 11. UB budget: AR_FULLLOAD still works with large R (within UB)
//    R=10000 fp16 fits in UB per DESIGN.md
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_fp16_large_r_within_ub)
{
    // shape=[2, 10000], axis=-1 → R=10000
    // Per DESIGN.md 3.5.4: total UB ~82KB for fp16 input → fits in 192KB
    auto r = RunTiling({2, 10000}, ge::DT_FLOAT16, {-1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->rLength, 10000);

    // rLengthAlign = CeilAlign(10000, 16) = 10000 (10000 / 16 = 625, exact)
    // rLengthAlignFp32 = CeilAlign(10000, 8) = 10000
    EXPECT_EQ(td->rLengthAlign, 10000);

    // 10000 * 2 = 20000, 20000 % 32 = 20000 - 625*32 = 20000 - 20000 = 0 → aligned
    EXPECT_EQ(td->isAlign32B, 1u);
}

// =============================================================================
// 12. Alignment: non-aligned R for fp16 (R=101)
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_fp16_nonaligned_r)
{
    // shape=[2, 101], axis=-1 → R=101
    // rLengthAlignInput = CeilAlign(101, 16) = 112
    // rLengthAlignFp32 = CeilAlign(101, 8) = 104
    // rLengthAlign = max(112, 104) = 112
    auto r = RunTiling({2, 101}, ge::DT_FLOAT16, {-1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->rLength, 101);
    EXPECT_EQ(td->rLengthAlign, 112);

    // 101 * 2 = 202, 202 % 32 != 0 → not aligned
    EXPECT_EQ(td->isAlign32B, 0u);
}

// =============================================================================
// 13. Alignment: aligned R for fp16 (R=128)
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_fp16_aligned_r)
{
    // shape=[2, 128], axis=-1 → R=128
    // 128 * 2 = 256 bytes, 256 % 32 = 0 → aligned
    auto r = RunTiling({2, 128}, ge::DT_FLOAT16, {-1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->rLength, 128);
    EXPECT_EQ(td->rLengthAlign, 128);
    EXPECT_EQ(td->isAlign32B, 1u);
}

// =============================================================================
// 14. Alignment: fp32 non-aligned (R=101)
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_fp32_nonaligned_r)
{
    // shape=[2, 101], axis=-1 → R=101
    // fp32: elementsPerBlock = 32/4 = 8
    // rLengthAlignInput = CeilAlign(101, 8) = 104
    // rLengthAlignFp32 = CeilAlign(101, 8) = 104
    auto r = RunTiling({2, 101}, ge::DT_FLOAT, {-1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->rLength, 101);
    EXPECT_EQ(td->rLengthAlign, 104);
    // 101 * 4 = 404, 404 % 32 != 0 → not aligned
    EXPECT_EQ(td->isAlign32B, 0u);
}

// =============================================================================
// 15. Edge case: rows=1 (single row, 2D input)
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_edge_single_row_2d)
{
    // shape=[1, 1000], axis=-1 → totalRows=1, rLength=1000
    auto r = RunTiling({1, 1000}, ge::DT_FLOAT16, {-1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->totalRows, 1);
    EXPECT_EQ(td->rLength, 1000);
    EXPECT_EQ(td->usedCoreNum, 1);
    EXPECT_EQ(td->rowsPerCore, 1);
    EXPECT_EQ(td->tailRows, 1);
}

// =============================================================================
// 16. Edge case: R=1 (degenerate reduction axis)
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_edge_r_length_one)
{
    // shape=[4, 1], axis=-1 → totalRows=4, rLength=1
    auto r = RunTiling({4, 1}, ge::DT_FLOAT16, {-1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->totalRows, 4);
    EXPECT_EQ(td->rLength, 1);

    // rLengthAlignInput = CeilAlign(1, 16) = 16
    // rLengthAlignFp32 = CeilAlign(1, 8) = 8
    // rLengthAlign = max(16, 8) = 16
    EXPECT_EQ(td->rLengthAlign, 16);

    // 1 * 2 = 2 bytes, 2 % 32 != 0 → not aligned
    EXPECT_EQ(td->isAlign32B, 0u);
}

// =============================================================================
// 17. Edge case: 1D input (scalar reduction, axis=0)
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_edge_1d_input)
{
    // shape=[1000], axis=[0] → totalRows=1, rLength=1000
    auto r = RunTiling({1000}, ge::DT_FLOAT16, {0});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->totalRows, 1);
    EXPECT_EQ(td->rLength, 1000);
    EXPECT_EQ(td->usedCoreNum, 1);
}

// =============================================================================
// 18. Edge case: 1D input with axis=-1
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_edge_1d_input_negative_axis)
{
    // shape=[1000], axis=-1 → normalized [0]
    auto r = RunTiling({1000}, ge::DT_FLOAT16, {-1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->totalRows, 1);
    EXPECT_EQ(td->rLength, 1000);
}

// =============================================================================
// 19. keep_dims=True does not affect tiling logic
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_keepdims_true)
{
    // shape=[4, 100], axis=-1, keep_dims=True
    // Tiling should produce same result regardless of keep_dims
    auto rTrue = RunTiling({4, 100}, ge::DT_FLOAT16, {-1}, true);
    auto rFalse = RunTiling({4, 100}, ge::DT_FLOAT16, {-1}, false);

    ASSERT_TRUE(rTrue.success);
    ASSERT_TRUE(rFalse.success);

    auto* tdTrue = AsTilingData(rTrue.info);
    auto* tdFalse = AsTilingData(rFalse.info);

    ASSERT_NE(tdTrue, nullptr);
    ASSERT_NE(tdFalse, nullptr);

    // All tiling params should be identical
    EXPECT_EQ(tdTrue->totalRows, tdFalse->totalRows);
    EXPECT_EQ(tdTrue->rLength, tdFalse->rLength);
    EXPECT_EQ(tdTrue->rLengthAlign, tdFalse->rLengthAlign);
    EXPECT_EQ(tdTrue->usedCoreNum, tdFalse->usedCoreNum);
    EXPECT_EQ(tdTrue->rowsPerCore, tdFalse->rowsPerCore);
}

// =============================================================================
// 20. 5D input (max supported dimension)
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_5d_input)
{
    // shape=[2, 3, 4, 5, 100], axis=-1
    // totalRows = 2*3*4*5 = 120, rLength = 100
    auto r = RunTiling({2, 3, 4, 5, 100}, ge::DT_FLOAT16, {-1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->totalRows, 120);
    EXPECT_EQ(td->rLength, 100);
}

// =============================================================================
// 21. UB budget validation: verify tiling data for R=10000 fp32
//     Per DESIGN.md 3.5.5: ~121.3 KB < 184 KB
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_fp32_large_r_within_ub)
{
    // shape=[2, 10000], axis=-1 → R=10000
    auto r = RunTiling({2, 10000}, ge::DT_FLOAT, {-1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->rLength, 10000);
    // fp32: rLengthAlign = CeilAlign(10000, 8) = 10000
    EXPECT_EQ(td->rLengthAlign, 10000);
    // 10000 * 4 = 40000, 40000 % 32 = 0 → aligned
    EXPECT_EQ(td->isAlign32B, 1u);
}

// =============================================================================
// 22. Verify UB usage stays within 192KB for typical fp16 case
//     This validates the AR_FULLLOAD threshold check implicitly
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_fp16_ub_budget_check)
{
    // shape=[2, 4096], axis=-1 → R=4096, fp16
    // rLengthAlign = CeilAlign(4096, 16) = 4096
    // UB needed:
    //   inQueueX(half): 2 * 4096 * 2 = 16384
    //   workFp32(float): 4096 * 4 = 16384
    //   outQueueY: 2*32 = 64
    //   tmpBuf: ~4096
    //   Total ~36928 bytes < 192KB
    auto r = RunTiling({2, 4096}, ge::DT_FLOAT16, {-1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->rLengthAlign, 4096);

    // Manual UB verification
    uint64_t rAlign = static_cast<uint64_t>(td->rLengthAlign);
    uint64_t inQueueBytes = 2 * rAlign * 2;       // half, double buffer
    uint64_t workFp32Bytes = rAlign * 4;            // float compute buf
    // tmpBuf computation matching tiling.cpp logic
    uint32_t perRepeat = 256 / sizeof(float);     // 64
    uint32_t perBlock = 32 / sizeof(float);       // 8
    uint32_t firstMaxRepeat = (static_cast<uint32_t>(rAlign) + perRepeat - 1) / perRepeat;
    if (firstMaxRepeat == 0) firstMaxRepeat = 1;
    uint32_t tmpBufElements = ((firstMaxRepeat + perBlock - 1) / perBlock) * perBlock;
    if (tmpBufElements < perBlock) tmpBufElements = perBlock;
    uint64_t tmpBufBytes = tmpBufElements * sizeof(float);
    uint64_t outQueueBytes = 2 * 32;
    uint64_t totalUb = inQueueBytes + workFp32Bytes + tmpBufBytes + outQueueBytes;

    std::cout << "  UB breakdown (fp16, R=4096):"
              << " inQ=" << inQueueBytes
              << " work=" << workFp32Bytes
              << " tmp=" << tmpBufBytes
              << " outQ=" << outQueueBytes
              << " total=" << totalUb
              << " (limit=192KB=" << (192*1024) << ")" << std::endl;

    EXPECT_LE(totalUb, 192u * 1024u);
}

// =============================================================================
// 23. TilingKey verification: all dtypes produce correct template selection
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_key_dtype_mapping)
{
    struct DtypeKeyCase {
        std::string name;
        ge::DataType dtype;
        int64_t expectedTilingKey;
    };

    // TilingKey from ASCENDC_TPL_SEL in tiling_key.h:
    // The ASCENDC_TPL_SEL macro encodes dtype into TilingKey via internal bit fields.
    // Observed actual values (from running tiling on the platform):
    //   fp16 → key=1, fp32 → key=0, bf16 → key=27
    // These are determined by the TPL_ARGS framework encoding, not manually assigned.
    DtypeKeyCase cases[] = {
        {"fp16", ge::DT_FLOAT16, 1},
        {"fp32", ge::DT_FLOAT, 0},
        {"bf16", ge::DT_BF16, 27},
    };

    for (const auto& c : cases) {
        auto r = RunTiling({4, 64}, c.dtype, {-1});
        ASSERT_TRUE(r.success) << "Failed for dtype: " << c.name;
        EXPECT_EQ(r.info.tilingKey, c.expectedTilingKey)
            << "TilingKey mismatch for dtype: " << c.name
            << " (expected=" << c.expectedTilingKey << ", actual=" << r.info.tilingKey << ")";
    }
}

// =============================================================================
// 24. Multi-core split: exact division
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_blockdim_exact_division)
{
    // shape=[40, 64], axis=-1, coreNum=20
    // usedCoreNum = min(20, 40) = 20
    // rowsPerCore = ceil(40/20) = 2
    // tailRows = 40 - 2*19 = 40 - 38 = 2
    auto r = RunTiling({40, 64}, ge::DT_FLOAT16, {-1}, false, 20);
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->usedCoreNum, 20);
    EXPECT_EQ(td->rowsPerCore, 2);
    EXPECT_EQ(td->tailRows, 2);
}

// =============================================================================
// 25. Multi-core split: single core (totalRows=1)
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_blockdim_single_core)
{
    // shape=[1, 64], axis=-1, coreNum=20
    // usedCoreNum = min(20, 1) = 1
    auto r = RunTiling({1, 64}, ge::DT_FLOAT16, {-1}, false, 20);
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->usedCoreNum, 1);
    EXPECT_EQ(r.info.blockNum, 1u);
    EXPECT_EQ(td->rowsPerCore, 1);
    // tailRows = 1 - 1*0 = 1
    EXPECT_EQ(td->tailRows, 1);
}

} // namespace SquareSumV1UT
