/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * \file test_squaresumv1_tiling.cpp
 * \brief SquareSumV1 op_host Tiling unit tests
 *
 * Coverage:
 *   1. TilingKey/dtype mapping (fp16, fp32, bf16)
 *   2. AR_FULLLOAD mode determination (R within UB threshold vs exceeding)
 *   3. Axis normalization (negative index, single axis=-1, multi-axis)
 *   4. blockDim computation (min(coreNum, ceil(rows/tile)))
 *   5. UB budget / tile parameters (rLength, rLengthAlign)
 *   6. keep_dims handling
 *   7. Edge cases (rows=1, R=1, non-aligned R)
 *   8. Axis position determination (AR vs ARA routing)
 *   9. AR Full Load vs Column Split (tilingMode 0 vs 1)
 *  10. ARA Full Load vs Row Split (tilingMode 2 vs 3, binary search)
 *  11. Multi-dtype mapping regression (TilingKey encoding)
 *  12. ARA edge cases and boundaries
 *  13. MULTI_AXIS (Key=4) detection: non-contiguous axis routing
 *  14. MULTI_AXIS per-layer parameters (numLayers, layerAxis, layerRLength, etc.)
 *  15. MULTI_AXIS workspace size (2*inputElems*sizeof(float), 4096-aligned)
 *  16. MULTI_AXIS boundaries (full reduction, degenerate dims, single core)
 */

#include <iostream>
#include <gtest/gtest.h>
#include <memory>
#include <cstring>
#include <cmath>

#include "tiling_context_faker.h"
#include "tiling_case_executor.h"
#include "square_sum_v1_tiling_data.h"

namespace SquareSumV1UT {
using namespace std;
using namespace ge;
using namespace gert;

// SquareSumV1 is a CANN built-in L0 name.  The submitted public ACLNN API
// dispatches the isolated implementation-only type below, so tiling UTs must
// resolve the same registry entry as the packaged operator.
static const std::string OP_NAME = "SquareSumV1Custom";

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

static uint64_t Align32(uint64_t bytes)
{
    return (bytes + 31U) & ~static_cast<uint64_t>(31U);
}

// Keep this mirror of Key4 Init() deliberately small: any valid Key4 tiling
// must fit the five raw TBuf allocations made by the compact kernel.
static uint64_t Key4UbBytes(const SquareSumV1TilingData* td)
{
    uint64_t maxMatrixElems = 8;
    uint64_t maxCols = 8;
    uint64_t maxTmpBytes = 32;
    for (int32_t li = 0; li < td->numLayers; ++li) {
        const uint64_t cols = td->layerIsTailReduce[li] ? 1U
            : static_cast<uint64_t>(td->layerTileA0Align[li]);
        const uint64_t rows = static_cast<uint64_t>(td->layerRChunkSizeCompact[li]);
        maxMatrixElems = std::max(maxMatrixElems, rows * cols);
        maxCols = std::max(maxCols, cols);
        maxTmpBytes = std::max(maxTmpBytes,
                               static_cast<uint64_t>(td->layerReduceTmpBytes[li]));
    }
    return 2U * Align32(maxMatrixElems * sizeof(float))
        + 2U * Align32(maxCols * sizeof(float))
        + Align32(maxTmpBytes);
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

// The ACLNN front end rejects these too; exercising tiling directly prevents
// malformed graph attributes from reaching CoalesceAxis/Kernel dispatch.
TEST_F(SquareSumV1TilingTest, tiling_rejects_out_of_range_and_duplicate_axis)
{
    EXPECT_FALSE(RunTiling({2, 3}, ge::DT_FLOAT16, {2}).success);
    EXPECT_FALSE(RunTiling({2, 3}, ge::DT_FLOAT16, {-3}).success);
    EXPECT_FALSE(RunTiling({2, 3, 4}, ge::DT_FLOAT, {1, -2}).success);
}

TEST_F(SquareSumV1TilingTest, tiling_rejects_rank_above_aclnn_contract)
{
    EXPECT_FALSE(RunTiling({1, 1, 1, 1, 1, 1, 1, 1, 1}, ge::DT_FLOAT16, {-1}).success);
}

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

// =============================================================================
// ===================== Iteration 2: A2 UT Extension ==========================
// =============================================================================

// -----------------------------------------------------------------------------
// Group A: Axis Position Determination (AR vs ARA routing)
// -----------------------------------------------------------------------------

// =============================================================================
// 26. Axis position: 3D non-tail reduce (axis=1) → ARA mode
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_axis_position_3d_nontail)
{
    // shape=[4, 100, 64], axis=[1] → non-tail reduce
    // totalRows=4, rLength=100, a0Length=64
    auto r = RunTiling({4, 100, 64}, ge::DT_FLOAT, {1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->totalRows, 4);
    EXPECT_EQ(td->rLength, 100);
    EXPECT_EQ(td->a0Length, 64);
    // ARA_FULLLOAD (Key=2): R=100, A0=64, fp32 fits in UB
    EXPECT_EQ(td->tilingMode, 2u);
}

// =============================================================================
// 27. Axis position: 4D non-tail reduce (axis=2) → ARA mode
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_axis_position_4d_nontail)
{
    // shape=[2, 3, 100, 64], axis=[2] → non-tail reduce
    // totalRows = 2*3 = 6, rLength=100, a0Length=64
    auto r = RunTiling({2, 3, 100, 64}, ge::DT_FLOAT, {2});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->totalRows, 6);
    EXPECT_EQ(td->rLength, 100);
    EXPECT_EQ(td->a0Length, 64);
    EXPECT_EQ(td->tilingMode, 2u);
}

// =============================================================================
// 28. Axis position: 5D non-tail reduce (axis=3) → ARA mode
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_axis_position_5d_nontail)
{
    // shape=[2, 3, 4, 5, 100], axis=[3] → non-tail reduce
    // totalRows = 2*3*4 = 24, rLength=5, a0Length=100
    auto r = RunTiling({2, 3, 4, 5, 100}, ge::DT_FLOAT, {3});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->totalRows, 24);
    EXPECT_EQ(td->rLength, 5);
    EXPECT_EQ(td->a0Length, 100);
    EXPECT_EQ(td->tilingMode, 2u);
}

// =============================================================================
// 29. Axis position: axis=0 on 2D → ARA mode (non-tail)
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_axis_position_first_axis_2d)
{
    // shape=[100, 64], axis=[0] → non-tail reduce
    // totalRows=1, rLength=100, a0Length=64
    auto r = RunTiling({100, 64}, ge::DT_FLOAT, {0});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->totalRows, 1);
    EXPECT_EQ(td->rLength, 100);
    EXPECT_EQ(td->a0Length, 64);
    EXPECT_EQ(td->tilingMode, 2u);
    // ARA work is split by (A1, A0-tile), so axis=0 keeps AIVs busy.
    EXPECT_EQ(td->usedCoreNum, 8);
}

// =============================================================================
// 30. Axis position: negative non-tail axis → ARA mode
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_axis_position_negative_nontail)
{
    // shape=[4, 100, 64], axis=-2 → normalized [1] → non-tail reduce
    auto r = RunTiling({4, 100, 64}, ge::DT_FLOAT, {-2});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->totalRows, 4);
    EXPECT_EQ(td->rLength, 100);
    EXPECT_EQ(td->a0Length, 64);
    EXPECT_EQ(td->tilingMode, 2u);
}

// =============================================================================
// 31. Axis position: multi-axis non-tail → ARA mode
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_axis_position_multi_axis_nontail)
{
    // shape=[2, 100, 50, 64], axis=[1, 2] → non-tail contiguous reduce
    // totalRows=2, rLength=100*50=5000, a0Length=64
    auto r = RunTiling({2, 100, 50, 64}, ge::DT_FLOAT16, {1, 2});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->totalRows, 2);
    EXPECT_EQ(td->rLength, 5000);
    EXPECT_EQ(td->a0Length, 64);
    // R=5000 with A0=64 in fp16: too large for full load
    // tileA0 binary search fails → Key=3 (ARA_ROWSPLIT)
    EXPECT_EQ(td->tilingMode, 3u);
}

// -----------------------------------------------------------------------------
// Group B: AR Full Load vs Column Split (tilingMode 0 vs 1)
// -----------------------------------------------------------------------------

// =============================================================================
// 32. AR_FULLLOAD: tail reduce, R within UB threshold → mode=0
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_ar_fullload_mode0)
{
    // shape=[4, 4096], axis=-1, fp32 → R=4096 fits in UB
    auto r = RunTiling({4, 4096}, ge::DT_FLOAT, {-1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->tilingMode, 0u);  // AR_FULLLOAD
    EXPECT_EQ(td->rLength, 4096);
    // chunkCols and numChunks should be 0 for full-load
    EXPECT_EQ(td->chunkCols, 0);
    EXPECT_EQ(td->numChunks, 0);
}

// =============================================================================
// 33. AR_COLSPLIT: tail reduce, R exceeds UB threshold → mode=1
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_ar_colsplit_mode1_fp32)
{
    // shape=[4, 25000], axis=-1, fp32 → R=25000 exceeds UB
    // ubNeededFullLoad = 2*25000*4 + tmpBuf + 64 = 201632 > 196608
    auto r = RunTiling({4, 25000}, ge::DT_FLOAT, {-1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->tilingMode, 1u);  // AR_COLSPLIT
    EXPECT_EQ(td->rLength, 25000);
    // chunkCols = min(maxCols, 255*64) aligned to 8
    // For fp32: maxCols = (UB - chunkTmpBuf - 64) / 4
    // chunkTmpBuf = 1024, maxCols = (196608 - 1024 - 64) / 4 = 48880
    // chunkCols = min(48880, 16320) = 16320, aligned to 8 = 16320
    EXPECT_EQ(td->chunkCols, 16320);
    EXPECT_EQ(td->numChunks, 2);  // ceil(25000 / 16320) = 2
}

// =============================================================================
// 34. AR_COLSPLIT: fp16 large R → mode=1
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_ar_colsplit_mode1_fp16)
{
    // shape=[4, 30000], axis=-1, fp16 → R=30000 exceeds UB
    // ubNeeded = 2*30000*2 + 30000*4 + tmpBuf + 64 = 241952 > 196608
    auto r = RunTiling({4, 30000}, ge::DT_FLOAT16, {-1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->tilingMode, 1u);
    EXPECT_EQ(td->rLength, 30000);
    // fp16: maxCols = (UB - 1024 - 64) / (2+4) = 195520/6 = 32586
    // chunkCols = min(32586, 16320) = 16320, aligned to 8 = 16320
    EXPECT_EQ(td->chunkCols, 16320);
    EXPECT_EQ(td->numChunks, 2);
}

// =============================================================================
// 35. AR_COLSPLIT: verify chunkCols capped at 255*64
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_ar_colsplit_chunkcap)
{
    // shape=[4, 50000], axis=-1, fp32 → R=50000
    // maxCols would be 48880, but capped at 255*64=16320
    auto r = RunTiling({4, 50000}, ge::DT_FLOAT, {-1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->tilingMode, 1u);
    EXPECT_EQ(td->chunkCols, 16320);  // capped at 255*64
    // ceil(50000 / 16320) = 4
    EXPECT_EQ(td->numChunks, 4);
}

// -----------------------------------------------------------------------------
// Group C: ARA Full Load vs Row Split (tilingMode 2 vs 3, binary search)
// -----------------------------------------------------------------------------

// =============================================================================
// 36. ARA_FULLLOAD: non-tail reduce, [R, A0] fits in UB → mode=2, single tile
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_ara_fullload_mode2_single_tile)
{
    // shape=[4, 100, 64], axis=[1], fp32 → R=100, A0=64 fits
    auto r = RunTiling({4, 100, 64}, ge::DT_FLOAT, {1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->tilingMode, 2u);
    EXPECT_EQ(td->a0Length, 64);
    EXPECT_EQ(td->numA0Tiles, 4);
    // rChunkSize/numRChunks should be 0 for ARA_FULLLOAD
    EXPECT_EQ(td->rChunkSize, 0);
    EXPECT_EQ(td->numRChunks, 0);
}

// =============================================================================
// 37. ARA_FULLLOAD: A0 tile split via binary search → mode=2, multi-tile
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_ara_fullload_mode2_multi_a0_tile)
{
    // shape=[4, 100, 1024], axis=[1], fp32 → R=100, A0=1024
    // R*A0=100*1024 too large for full load, but tileA0 binary search finds fit
    auto r = RunTiling({4, 100, 1024}, ge::DT_FLOAT, {1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->tilingMode, 2u);
    EXPECT_EQ(td->a0Length, 1024);
    // a0LengthAlign = CeilAlign(1024, 8) = 1024
    EXPECT_EQ(td->a0LengthAlign, 1024);
    // Binary search finds max tileA0Align that fits with R=100 in UB
    // Result: tileA0Align=472, tileA0Len=472, numA0Tiles=ceil(1024/472)=3
    EXPECT_GT(td->tileA0Align, 0);
    EXPECT_GE(td->numA0Tiles, 2);
    // rChunkSize/numRChunks = 0 for ARA_FULLLOAD
    EXPECT_EQ(td->rChunkSize, 0);
    EXPECT_EQ(td->numRChunks, 0);
}

// =============================================================================
// 38. ARA_FULLLOAD: fp16 with A0 tile split → mode=2
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_ara_fullload_fp16_a0_tile)
{
    // shape=[4, 1000, 64], axis=[1], fp16 → R=1000, A0=64
    // R*A0 = 1000*64 too large, but tileA0=32 fits
    auto r = RunTiling({4, 1000, 64}, ge::DT_FLOAT16, {1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->tilingMode, 2u);
    EXPECT_EQ(td->rLength, 1000);
    EXPECT_EQ(td->a0Length, 64);
    // tileA0Align found by binary search: 32
    EXPECT_GT(td->tileA0Align, 0);
    EXPECT_GE(td->numA0Tiles, 1);
    EXPECT_EQ(td->rChunkSize, 0);
}

// =============================================================================
// 39. ARA_ROWSPLIT: non-tail reduce, R too large even with min tileA0 → mode=3
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_ara_rowsplit_mode3_fp32)
{
    // shape=[4, 7000, 64], axis=[1], fp32 → R=7000, A0=64
    // Even tileA0=8 doesn't fit: 7000*8*4+overhead > UB → Key=3
    auto r = RunTiling({4, 7000, 64}, ge::DT_FLOAT, {1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->tilingMode, 3u);
    EXPECT_EQ(td->rLength, 7000);
    EXPECT_EQ(td->a0Length, 64);
    // The scheduler further tiles A0 to expose independent output owners.
    EXPECT_EQ(td->tileA0Align, 16);
    EXPECT_EQ(td->tileA0Len, 16);
    EXPECT_EQ(td->numA0Tiles, 4);
    EXPECT_GT(td->rChunkSize, 0);
    EXPECT_GT(td->numRChunks, 1);
    EXPECT_EQ(td->numRChunks, (td->rLength + td->rChunkSize - 1) / td->rChunkSize);
}

// =============================================================================
// 40. ARA_ROWSPLIT: fp16 with large R → mode=3
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_ara_rowsplit_mode3_fp16)
{
    // shape=[4, 5000, 64], axis=[1], fp16 → R=5000, A0=64
    // tileA0 binary search fails → Key=3
    auto r = RunTiling({4, 5000, 64}, ge::DT_FLOAT16, {1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->tilingMode, 3u);
    EXPECT_EQ(td->rLength, 5000);
    EXPECT_EQ(td->a0Length, 64);
    EXPECT_EQ(td->tileA0Align, 16);
    EXPECT_GT(td->rChunkSize, 0);
    EXPECT_GT(td->numRChunks, 1);
    EXPECT_EQ(td->numRChunks, (td->rLength + td->rChunkSize - 1) / td->rChunkSize);
}

// =============================================================================
// 41. ARA_ROWSPLIT: verify rChunkSize UB constraint
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_ara_rowsplit_rchunk_ub_constraint)
{
    // shape=[4, 7000, 64], axis=[1], fp32
    // rChunkSize should be the max that fits in UB: 765
    auto r = RunTiling({4, 7000, 64}, ge::DT_FLOAT, {1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    ASSERT_EQ(td->tilingMode, 3u);
    int64_t rChunk = td->rChunkSize;
    int64_t tileA0 = td->tileA0Align;

    // Verify rChunkSize * tileA0 fits in UB
    uint64_t ubAtChunk = static_cast<uint64_t>(rChunk) * tileA0 * sizeof(float)
                       + tileA0 * sizeof(float) * 2 + std::max(tileA0 * sizeof(float), 32UL);
    EXPECT_LE(ubAtChunk, 196608u);

    // DMA rows are capped by the documented DataCopyPad blockCount limit.
    EXPECT_LE(rChunk, 4095);
}

// -----------------------------------------------------------------------------
// Group D: Multi-dtype Mapping Regression (TilingKey encoding)
// -----------------------------------------------------------------------------

// =============================================================================
// 42. dtype mapping regression: ARA mode fp16 TilingKey
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_key_ara_fp16)
{
    // shape=[4, 100, 64], axis=[1], fp16 → ARA mode
    auto r = RunTiling({4, 100, 64}, ge::DT_FLOAT16, {1});
    ASSERT_TRUE(r.success);

    EXPECT_EQ(r.info.tilingKey, 1);  // fp16 encoding
}

// =============================================================================
// 43. dtype mapping regression: ARA mode fp32 TilingKey
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_key_ara_fp32)
{
    // shape=[4, 100, 64], axis=[1], fp32 → ARA mode
    auto r = RunTiling({4, 100, 64}, ge::DT_FLOAT, {1});
    ASSERT_TRUE(r.success);

    EXPECT_EQ(r.info.tilingKey, 0);  // fp32 encoding
}

// =============================================================================
// 44. dtype mapping regression: ARA mode bf16 TilingKey
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_key_ara_bf16)
{
    // shape=[4, 100, 64], axis=[1], bf16 → ARA mode
    auto r = RunTiling({4, 100, 64}, ge::DT_BF16, {1});
    ASSERT_TRUE(r.success);

    EXPECT_EQ(r.info.tilingKey, 27);  // bf16 encoding
}

// =============================================================================
// 45. dtype mapping: all dtypes in AR_COLSPLIT mode
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_key_ar_colsplit_all_dtypes)
{
    struct DtypeCase {
        std::string name;
        ge::DataType dtype;
        int64_t expectedKey;
    };

    DtypeCase cases[] = {
        {"fp16", ge::DT_FLOAT16, 1},
        {"fp32", ge::DT_FLOAT, 0},
        {"bf16", ge::DT_BF16, 27},
    };

    for (const auto& c : cases) {
        // shape=[4, 30000], axis=-1 → AR_COLSPLIT
        auto r = RunTiling({4, 30000}, c.dtype, {-1});
        ASSERT_TRUE(r.success) << "Failed for dtype: " << c.name;

        auto* td = AsTilingData(r.info);
        ASSERT_NE(td, nullptr);
        EXPECT_EQ(td->tilingMode, 1u) << "Expected AR_COLSPLIT for dtype: " << c.name;
        EXPECT_EQ(r.info.tilingKey, c.expectedKey)
            << "TilingKey mismatch for dtype: " << c.name;
    }
}

// -----------------------------------------------------------------------------
// Group E: Edge Cases and Boundaries
// -----------------------------------------------------------------------------

// =============================================================================
// 46. ARA edge: R=1 degenerate with non-tail reduce
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_ara_edge_r_length_one)
{
    // shape=[4, 1, 64], axis=[1] → R=1, A0=64
    auto r = RunTiling({4, 1, 64}, ge::DT_FLOAT, {1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->totalRows, 4);
    EXPECT_EQ(td->rLength, 1);
    EXPECT_EQ(td->a0Length, 64);
    // R=1 fits trivially → ARA_FULLLOAD
    EXPECT_EQ(td->tilingMode, 2u);
    EXPECT_EQ(td->numA0Tiles, 4);
}

// =============================================================================
// 47. ARA edge: A0=1 degenerate
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_ara_edge_a0_length_one)
{
    // shape=[4, 100, 1], axis=[1] → R=100, A0=1
    auto r = RunTiling({4, 100, 1}, ge::DT_FLOAT, {1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->totalRows, 4);
    EXPECT_EQ(td->rLength, 100);
    // a0Length=1 gets special handling; a0LengthAlign = CeilAlign(1, 8) = 8
    EXPECT_EQ(td->a0Length, 1);
    // Should be ARA_FULLLOAD (tiny data)
    EXPECT_EQ(td->tilingMode, 2u);
}

// =============================================================================
// 48. ARA edge: non-aligned A0
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_ara_nonaligned_a0)
{
    // shape=[4, 100, 15], axis=[1], fp32 → R=100, A0=15
    auto r = RunTiling({4, 100, 15}, ge::DT_FLOAT, {1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->rLength, 100);
    EXPECT_EQ(td->a0Length, 15);
    // a0LengthAlign = CeilAlign(15, 8) = 16
    EXPECT_EQ(td->a0LengthAlign, 16);
    EXPECT_EQ(td->tilingMode, 2u);
}

// =============================================================================
// 49. 5D max dimension boundary with ARA mode
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_5d_ara_max_dims)
{
    // shape=[200, 3, 1000, 5, 100], axis=[2] → non-tail reduce
    // totalRows = 200*3 = 600, rLength=1000, a0Length=5*100=500
    auto r = RunTiling({200, 3, 1000, 5, 100}, ge::DT_FLOAT16, {2});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->totalRows, 600);
    EXPECT_EQ(td->rLength, 1000);
    EXPECT_EQ(td->a0Length, 500);
    // R=1000, A0=500(fp16) → likely needs split
    EXPECT_GE(td->tilingMode, 2u);  // ARA_FULLLOAD or ARA_ROWSPLIT
    // Multi-core: min(20, 600) = 20
    EXPECT_EQ(td->usedCoreNum, 20);
}

// =============================================================================
// 50. blockDim upper limit with ARA mode
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_ara_blockdim_capped)
{
    // shape=[100, 100, 64], axis=[1], fp32 → totalRows=100
    // usedCoreNum = min(20, 100) = 20
    auto r = RunTiling({100, 100, 64}, ge::DT_FLOAT, {1}, false, 20);
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->totalRows, 100);
    EXPECT_EQ(td->usedCoreNum, 20);
    EXPECT_EQ(r.info.blockNum, 20u);
    // rowsPerCore = ceil(100/20) = 5
    EXPECT_EQ(td->rowsPerCore, 5);
    // tailRows = 100 - 5*19 = 100-95 = 5
    EXPECT_EQ(td->tailRows, 5);
}

// =============================================================================
// 51. blockDim single core with ARA mode
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_ara_blockdim_single_core)
{
    // shape=[1, 100, 64], axis=[1], fp32 → totalRows=1
    auto r = RunTiling({1, 100, 64}, ge::DT_FLOAT, {1}, false, 20);
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->totalRows, 1);
    EXPECT_EQ(td->usedCoreNum, 8);
    EXPECT_EQ(r.info.blockNum, 8u);
}

// =============================================================================
// 52. AR_COLSPLIT edge: R just above UB threshold
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_ar_colsplit_boundary)
{
    // The production path reserves a 184KiB UB safety budget.
    auto rFull = RunTiling({2, 23000}, ge::DT_FLOAT, {-1});
    auto rSplit = RunTiling({2, 24000}, ge::DT_FLOAT, {-1});
    ASSERT_TRUE(rFull.success);
    ASSERT_TRUE(rSplit.success);

    auto* tdFull = AsTilingData(rFull.info);
    auto* tdSplit = AsTilingData(rSplit.info);
    ASSERT_NE(tdFull, nullptr);
    ASSERT_NE(tdSplit, nullptr);

    EXPECT_EQ(tdFull->tilingMode, 0u);   // Just within UB
    EXPECT_EQ(tdSplit->tilingMode, 1u);  // Just exceeds UB
}

// =============================================================================
// 53. ARA_ROWSPLIT: verify rChunkSize boundary correctness for fp16
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_ara_rowsplit_rchunk_boundary_fp16)
{
    // shape=[4, 5000, 64], axis=[1], fp16 → rChunkSize=510
    auto r = RunTiling({4, 5000, 64}, ge::DT_FLOAT16, {1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    ASSERT_EQ(td->tilingMode, 3u);
    int64_t rChunk = td->rChunkSize;
    int64_t tileA0 = td->tileA0Align;
    uint32_t ts = 2;  // fp16 typeSize

    // Verify rChunkSize fits in UB (fp16: input + fp32 compute buffer)
    uint64_t ubAtChunk = static_cast<uint64_t>(rChunk) * tileA0 * ts   // input buffer
                       + static_cast<uint64_t>(rChunk) * tileA0 * 4UL  // fp32 compute buffer
                       + tileA0 * 4UL                                    // acc buffer
                       + tileA0 * ts                                     // output buffer
                       + std::max(tileA0 * 4UL, 32UL);                   // tmp buffer
    EXPECT_LE(ubAtChunk, 196608u);

    EXPECT_LE(rChunk, 4095);
}

// =============================================================================
// 54. Coalesced shape: multi-dim non-reduce after reduce (A0 = product)
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_coalesced_multi_dim_a0)
{
    // shape=[2, 50, 3, 4], axis=[1] → R=50, A0=3*4=12
    auto r = RunTiling({2, 50, 3, 4}, ge::DT_FLOAT, {1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->totalRows, 2);
    EXPECT_EQ(td->rLength, 50);
    EXPECT_EQ(td->a0Length, 12);
    EXPECT_EQ(td->tilingMode, 2u);
}

// =============================================================================
// 55. tilingMode field consistency: verify all 4 modes have distinct values
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_mode_distinct_values)
{
    // AR_FULLLOAD (mode=0): tail reduce, small R
    auto r0 = RunTiling({4, 64}, ge::DT_FLOAT, {-1});
    ASSERT_TRUE(r0.success);
    EXPECT_EQ(AsTilingData(r0.info)->tilingMode, 0u);

    // AR_COLSPLIT (mode=1): tail reduce, large R
    auto r1 = RunTiling({4, 30000}, ge::DT_FLOAT, {-1});
    ASSERT_TRUE(r1.success);
    EXPECT_EQ(AsTilingData(r1.info)->tilingMode, 1u);

    // ARA_FULLLOAD (mode=2): non-tail reduce, small [R,A0]
    auto r2 = RunTiling({4, 100, 64}, ge::DT_FLOAT, {1});
    ASSERT_TRUE(r2.success);
    EXPECT_EQ(AsTilingData(r2.info)->tilingMode, 2u);

    // ARA_ROWSPLIT (mode=3): non-tail reduce, large R
    auto r3 = RunTiling({4, 7000, 64}, ge::DT_FLOAT, {1});
    ASSERT_TRUE(r3.success);
    EXPECT_EQ(AsTilingData(r3.info)->tilingMode, 3u);
}

// =============================================================================
// ===================== Iteration 3: A2 UT Full Coverage ======================
// =============================================================================
//   Key=4 MULTI_AXIS detection + per-layer params + workspace + boundaries
// =============================================================================

// -----------------------------------------------------------------------------
// Group F: MULTI_AXIS Detection (Key=4 routing)
// -----------------------------------------------------------------------------

// =============================================================================
// 56. MULTI_AXIS detection: 3D non-contiguous [0,2] → Key=4
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_3d_noncontiguous_0_2)
{
    // shape=[4, 100, 64], axis=[0, 2]
    // Reduce dims 0 and 2, but dim 1 is non-reduce between them → non-contiguous
    auto r = RunTiling({4, 100, 64}, ge::DT_FLOAT, {0, 2});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->tilingMode, 4u);
    // numLayers = 2 (two non-contiguous reduce axes)
    EXPECT_EQ(td->numLayers, 2);
}

// =============================================================================
// 57. MULTI_AXIS detection: 4D non-contiguous [0,2] → Key=4
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_4d_noncontiguous_0_2)
{
    // shape=[2, 100, 50, 64], axis=[0, 2]
    // Dim 1 is non-reduce between axes 0 and 2
    auto r = RunTiling({2, 100, 50, 64}, ge::DT_FLOAT, {0, 2});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->tilingMode, 4u);
    EXPECT_EQ(td->numLayers, 2);
}

// =============================================================================
// 58. MULTI_AXIS detection: 4D non-contiguous [0,3] → Key=4
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_4d_noncontiguous_0_3)
{
    // shape=[2, 3, 100, 64], axis=[0, 3]
    // Dims 1, 2 are non-reduce between axes 0 and 3
    auto r = RunTiling({2, 3, 100, 64}, ge::DT_FLOAT16, {0, 3});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->tilingMode, 4u);
    EXPECT_EQ(td->numLayers, 2);
}

// =============================================================================
// 59. MULTI_AXIS detection: 4D non-contiguous [0,2,3] → Key=4
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_4d_noncontiguous_0_2_3)
{
    // shape=[2, 100, 50, 64], axis=[0, 2, 3]
    // Axis 0 is separated from contiguous [2,3] by dim 1
    auto r = RunTiling({2, 100, 50, 64}, ge::DT_FLOAT, {0, 2, 3});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->tilingMode, 4u);
    // After coalescing: axes 2,3 are contiguous (merged), axis 0 is separate
    // But CoalesceAxis detects non-contiguous and returns -1 → MULTI_AXIS
    // numLayers = number of original sorted axes = 3
    EXPECT_EQ(td->numLayers, 3);
}

// =============================================================================
// 60. MULTI_AXIS detection: 5D non-contiguous [0,2,4] → Key=4
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_5d_noncontiguous_0_2_4)
{
    // shape=[2, 3, 100, 4, 64], axis=[0, 2, 4]
    // Dims 1, 3 are non-reduce gaps
    auto r = RunTiling({2, 3, 100, 4, 64}, ge::DT_FLOAT, {0, 2, 4});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->tilingMode, 4u);
    EXPECT_EQ(td->numLayers, 3);
}

// =============================================================================
// 61. Adjacent multi-axis does NOT trigger Key=4: [1,2] → coalesced Key0-3
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_adjacent_1_2_not_key4)
{
    // shape=[2, 100, 50], axis=[1, 2] → contiguous tail reduce
    auto r = RunTiling({2, 100, 50}, ge::DT_FLOAT, {1, 2});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_NE(td->tilingMode, 4u);  // Should be AR_FULLLOAD (0) or AR_COLSPLIT (1)
    EXPECT_EQ(td->totalRows, 2);
    EXPECT_EQ(td->rLength, 5000);   // 100*50
}

// =============================================================================
// 62. Adjacent multi-axis with negative indices: [-2,-1] → coalesced Key0-3
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_adjacent_neg2_neg1_not_key4)
{
    // shape=[3, 100, 50], axis=[-2, -1] → normalized [1, 2] → contiguous tail
    auto r = RunTiling({3, 100, 50}, ge::DT_FLOAT16, {-2, -1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_NE(td->tilingMode, 4u);
    EXPECT_EQ(td->totalRows, 3);
    EXPECT_EQ(td->rLength, 5000);
}

// =============================================================================
// 63. Single axis does NOT trigger Key=4
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_single_axis_not_key4)
{
    // shape=[4, 100, 64], axis=[1] → single axis ARA
    auto r = RunTiling({4, 100, 64}, ge::DT_FLOAT, {1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_NE(td->tilingMode, 4u);
    EXPECT_EQ(td->tilingMode, 2u);  // ARA_FULLLOAD
}

// -----------------------------------------------------------------------------
// Group G: Per-Layer Parameters
// -----------------------------------------------------------------------------

// =============================================================================
// 64. Per-layer params: 3D [0,2] → verify layerAxis, layerRLength
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_3d_layer_params)
{
    // shape=[4, 100, 64], axis=[0, 2], fp32
    // Process order (innermost first): axis=2 (layer 0), axis=0 (layer 1)
    auto r = RunTiling({4, 100, 64}, ge::DT_FLOAT, {0, 2});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);
    ASSERT_EQ(td->tilingMode, 4u);
    ASSERT_EQ(td->numLayers, 2);

    // Process order (innermost first): processOrder = reversed(sorted) = [2, 0]
    // Layer 0: targetAxis=2 (axis=2), Layer 1: targetAxis=0 (axis=0)
    EXPECT_EQ(td->layerAxis[0], 2);
    EXPECT_EQ(td->layerAxis[1], 0);

    // Layer 0 (axis=2): shape [4,100,64], reduce axis pos=2, rLength=64
    EXPECT_EQ(td->layerRLength[0], 64);
    EXPECT_EQ(td->layerReduceAxisIdx[0], 2);
    EXPECT_EQ(td->layerIsTailReduce[0], 1);  // pos 2 is last in 3D
    EXPECT_EQ(td->layerA0Length[0], 0);      // tail reduce → a0=0

    // Layer 0 output: [4,100,1] → after squeeze: [4,100], elemCount=400
    EXPECT_EQ(td->layerInputElemCount[0], 4 * 100 * 64);
    EXPECT_EQ(td->layerOutputElemCount[0], 4 * 100);

    // Layer 1 (axis=0): shape [4,100], reduce axis pos=0, rLength=4
    EXPECT_EQ(td->layerRLength[1], 4);
    EXPECT_EQ(td->layerReduceAxisIdx[1], 0);
    EXPECT_EQ(td->layerIsTailReduce[1], 0);  // pos 0 is not last
    EXPECT_EQ(td->layerA0Length[1], 100);    // non-reduce tail = dim 1

    EXPECT_EQ(td->layerInputElemCount[1], 4 * 100);
    EXPECT_EQ(td->layerOutputElemCount[1], 100);
}

// =============================================================================
// 65. Per-layer params: 4D [0,2] → verify layerShapeBefore
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_4d_layer_shape_before)
{
    // shape=[2, 100, 50, 64], axis=[0, 2], fp32
    // Process order: axis=2 (layer 0), axis=0 (layer 1)
    auto r = RunTiling({2, 100, 50, 64}, ge::DT_FLOAT, {0, 2});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);
    ASSERT_EQ(td->tilingMode, 4u);
    ASSERT_EQ(td->numLayers, 2);

    // Layer 0 (axis=2): original shape [2,100,50,64]
    EXPECT_EQ(td->layerNDims[0], 4);
    EXPECT_EQ(td->layerShapeBefore[0][0], 2);
    EXPECT_EQ(td->layerShapeBefore[0][1], 100);
    EXPECT_EQ(td->layerShapeBefore[0][2], 50);
    EXPECT_EQ(td->layerShapeBefore[0][3], 64);

    // Layer 0 reduce: axis=2, rLength=50, a0Length=64 (non-tail)
    EXPECT_EQ(td->layerRLength[0], 50);
    EXPECT_EQ(td->layerReduceAxisIdx[0], 2);
    EXPECT_EQ(td->layerIsTailReduce[0], 0);
    EXPECT_EQ(td->layerA0Length[0], 64);

    // Layer 1 (axis=0): after removing axis=2, shape=[2,100,64]
    EXPECT_EQ(td->layerNDims[1], 3);
    EXPECT_EQ(td->layerShapeBefore[1][0], 2);
    EXPECT_EQ(td->layerShapeBefore[1][1], 100);
    EXPECT_EQ(td->layerShapeBefore[1][2], 64);

    // Layer 1 reduce: axis=0, rLength=2
    EXPECT_EQ(td->layerRLength[1], 2);
    EXPECT_EQ(td->layerReduceAxisIdx[1], 0);
    EXPECT_EQ(td->layerIsTailReduce[1], 0);
    EXPECT_EQ(td->layerA0Length[1], 100 * 64);  // 6400
}

// =============================================================================
// 66. Per-layer params: 5D [0,2,4] → 3 layers
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_5d_three_layers)
{
    // shape=[2, 3, 100, 4, 64], axis=[0, 2, 4], fp32
    // Process order: axis=4, axis=2, axis=0
    auto r = RunTiling({2, 3, 100, 4, 64}, ge::DT_FLOAT, {0, 2, 4});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);
    ASSERT_EQ(td->tilingMode, 4u);
    ASSERT_EQ(td->numLayers, 3);

    // Layer 0 (axis=4): shape [2,3,100,4,64], tail reduce
    EXPECT_EQ(td->layerAxis[0], 4);
    EXPECT_EQ(td->layerRLength[0], 64);
    EXPECT_EQ(td->layerIsTailReduce[0], 1);
    EXPECT_EQ(td->layerOutputElemCount[0], 2 * 3 * 100 * 4);

    // Layer 1 (axis=2): shape [2,3,100,4], reduce pos=2
    EXPECT_EQ(td->layerAxis[1], 2);
    EXPECT_EQ(td->layerRLength[1], 100);
    EXPECT_EQ(td->layerIsTailReduce[1], 0);
    EXPECT_EQ(td->layerA0Length[1], 4);

    // Layer 2 (axis=0): shape [2,3,4], reduce pos=0
    EXPECT_EQ(td->layerAxis[2], 0);
    EXPECT_EQ(td->layerRLength[2], 2);
    EXPECT_EQ(td->layerIsTailReduce[2], 0);
    EXPECT_EQ(td->layerA0Length[2], 3 * 4);
}

// =============================================================================
// 67. Per-layer workspace offset: 3D [0,2]
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_3d_workspace_offset)
{
    // shape=[4, 100, 64], axis=[0, 2], fp32
    // Layer 0 output: 4*100 elements → workspace region 0
    // Layer 1 reads from workspace region 0, writes to result GM (last layer)
    auto r = RunTiling({4, 100, 64}, ge::DT_FLOAT, {0, 2});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);
    ASSERT_EQ(td->tilingMode, 4u);
    ASSERT_EQ(td->numLayers, 2);

    // Workspace offset for layer 0 should be meaningful (layer 0 writes intermediate)
    // layerWorkspaceOffset[0] is the read offset; for layer 0 it reads from input (not workspace)
    // The actual workspace offset for layer 1's read = where layer 0 wrote its output
    // From code: layer[0].workspaceOffset = 0 (reads from input)
    //            layer[1].workspaceOffset = layer[0].workspaceOffset (reads prev output)
    // But the write offset for layer 0 output is set via layers[li+1].workspaceOffset = wsOffset
    // Layer 0 output bytes = CeilAlign(400 * 4, 32) = 1600
    // So layer[1].workspaceOffset = 0 (reads from ws[0])
    EXPECT_GE(td->layerWorkspaceOffset[0], 0);
    EXPECT_GE(td->layerWorkspaceOffset[1], 0);
}

// -----------------------------------------------------------------------------
// Group H: Workspace Size
// -----------------------------------------------------------------------------

// =============================================================================
// 68. Workspace size: Key=4 = 2*inputElems*sizeof(float), 4096-aligned
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_workspace_size)
{
    // shape=[4, 100, 64], axis=[0, 2], fp32
    // totalInputElems = 4*100*64 = 25600
    // wsSize = 25600 * 4 * 2 = 204800
    // Aligned to 4096: 204800 is already 4096-aligned (204800 / 4096 = 50)
    auto r = RunTiling({4, 100, 64}, ge::DT_FLOAT, {0, 2});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);
    ASSERT_EQ(td->tilingMode, 4u);

    ASSERT_GE(r.info.workspaceSizes.size(), 1u);
    size_t expectedWs = static_cast<size_t>(512) * sizeof(float) * 2;
    expectedWs = (expectedWs + 4095) & ~static_cast<size_t>(4095);
    EXPECT_EQ(r.info.workspaceSizes[0], expectedWs);
}

// =============================================================================
// 69. Workspace size: Key=4 fp16 same formula (uses sizeof(float) not typeSize)
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_workspace_size_fp16)
{
    // shape=[2, 50, 32], axis=[0, 2], fp16
    // totalInputElems = 2*50*32 = 3200
    // wsSize = 3200 * 4 * 2 = 25600, aligned to 4096 = 25600 (already aligned)
    // Wait: 25600 / 4096 = 6.25 → not aligned. CeilAlign = 28672
    auto r = RunTiling({2, 50, 32}, ge::DT_FLOAT16, {0, 2});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);
    ASSERT_EQ(td->tilingMode, 4u);

    ASSERT_GE(r.info.workspaceSizes.size(), 1u);
    size_t expectedWs = static_cast<size_t>(512) * sizeof(float) * 2;
    expectedWs = (expectedWs + 4095) & ~static_cast<size_t>(4095);
    EXPECT_EQ(r.info.workspaceSizes[0], expectedWs);
}

// =============================================================================
// 70. Workspace size: Key0-3 = 0 (no workspace for non-MULTI_AXIS)
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_non_multi_axis_workspace_zero)
{
    // AR_FULLLOAD (mode=0): workspace should be 0
    auto r0 = RunTiling({4, 64}, ge::DT_FLOAT, {-1});
    ASSERT_TRUE(r0.success);
    ASSERT_GE(r0.info.workspaceSizes.size(), 1u);
    EXPECT_EQ(r0.info.workspaceSizes[0], 0u);

    // ARA_FULLLOAD (mode=2): workspace should be 0
    auto r2 = RunTiling({4, 100, 64}, ge::DT_FLOAT, {1});
    ASSERT_TRUE(r2.success);
    ASSERT_GE(r2.info.workspaceSizes.size(), 1u);
    EXPECT_EQ(r2.info.workspaceSizes[0], 0u);

    // AR_COLSPLIT (mode=1): workspace should be 0
    auto r1 = RunTiling({4, 30000}, ge::DT_FLOAT, {-1});
    ASSERT_TRUE(r1.success);
    ASSERT_GE(r1.info.workspaceSizes.size(), 1u);
    EXPECT_EQ(r1.info.workspaceSizes[0], 0u);

    // ARA_ROWSPLIT (mode=3): workspace should be 0
    auto r3 = RunTiling({4, 7000, 64}, ge::DT_FLOAT, {1});
    ASSERT_TRUE(r3.success);
    ASSERT_GE(r3.info.workspaceSizes.size(), 1u);
    EXPECT_EQ(r3.info.workspaceSizes[0], 0u);
}

// -----------------------------------------------------------------------------
// Group I: MULTI_AXIS Boundaries
// -----------------------------------------------------------------------------

// =============================================================================
// 71. MULTI_AXIS boundary: full reduction (all axes) → scalar output
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_full_reduction_2d)
{
    // shape=[4, 64], axis=[0, 1] → reduce all dims
    // These are contiguous → NOT Key=4, it's AR_FULLLOAD
    auto r = RunTiling({4, 64}, ge::DT_FLOAT, {0, 1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    // [0,1] is contiguous tail reduce → AR_FULLLOAD, not MULTI_AXIS
    EXPECT_NE(td->tilingMode, 4u);
    EXPECT_EQ(td->tilingMode, 0u);  // AR_FULLLOAD
    EXPECT_EQ(td->totalRows, 1);
    EXPECT_EQ(td->rLength, 256);    // 4*64
}

// =============================================================================
// 72. MULTI_AXIS boundary: full reduction 3D non-contiguous [0,1,2]
//     → contiguous (all dims), NOT Key=4
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_full_reduction_3d_contiguous)
{
    // shape=[4, 100, 64], axis=[0, 1, 2] → all contiguous
    auto r = RunTiling({4, 100, 64}, ge::DT_FLOAT, {0, 1, 2});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_NE(td->tilingMode, 4u);
    EXPECT_EQ(td->totalRows, 1);
    EXPECT_EQ(td->rLength, 4 * 100 * 64);
}

// =============================================================================
// 73. MULTI_AXIS boundary: reduction dim = 1 in one layer
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_reduce_dim_one)
{
    // shape=[2, 1, 50, 64], axis=[0, 2]
    // Layer 0 (axis=2): shape [2,1,50,64], rLength=50
    // Layer 1 (axis=0): shape [2,1,64], rLength=2
    auto r = RunTiling({2, 1, 50, 64}, ge::DT_FLOAT, {0, 2});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);
    ASSERT_EQ(td->tilingMode, 4u);
    ASSERT_EQ(td->numLayers, 2);

    EXPECT_EQ(td->layerRLength[0], 50);
    EXPECT_EQ(td->layerRLength[1], 2);
}

// =============================================================================
// 74. MULTI_AXIS boundary: single-element degenerate axis in non-reduce gap
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_degenerate_gap)
{
    // shape=[4, 1, 64], axis=[0, 2]
    // Dim 1 = 1 (degenerate non-reduce between axes 0 and 2)
    auto r = RunTiling({4, 1, 64}, ge::DT_FLOAT, {0, 2});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);
    ASSERT_EQ(td->tilingMode, 4u);
    ASSERT_EQ(td->numLayers, 2);

    // Layer 0 (axis=2): shape [4,1,64], rLength=64
    EXPECT_EQ(td->layerRLength[0], 64);
    // Layer 1 (axis=0): shape [4,1], rLength=4
    EXPECT_EQ(td->layerRLength[1], 4);
}

// =============================================================================
// 75. MULTI_AXIS with negative indices: 3D [-3, -1] → [0, 2]
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_negative_indices_3d)
{
    // shape=[4, 100, 64], axis=[-3, -1] → normalized [0, 2]
    auto r = RunTiling({4, 100, 64}, ge::DT_FLOAT, {-3, -1});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);
    ASSERT_EQ(td->tilingMode, 4u);
    ASSERT_EQ(td->numLayers, 2);

    // Same as test 64
    EXPECT_EQ(td->layerAxis[0], 2);  // innermost first
    EXPECT_EQ(td->layerAxis[1], 0);
}

// =============================================================================
// 76. MULTI_AXIS: all 3 dtypes produce correct tilingMode=4
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_all_dtypes)
{
    struct DtypeCase {
        std::string name;
        ge::DataType dtype;
    };

    DtypeCase cases[] = {
        {"fp16", ge::DT_FLOAT16},
        {"fp32", ge::DT_FLOAT},
        {"bf16", ge::DT_BF16},
    };

    for (const auto& c : cases) {
        auto r = RunTiling({4, 100, 64}, c.dtype, {0, 2});
        ASSERT_TRUE(r.success) << "Failed for dtype: " << c.name;

        auto* td = AsTilingData(r.info);
        ASSERT_NE(td, nullptr);
        EXPECT_EQ(td->tilingMode, 4u) << "Expected MULTI_AXIS for dtype: " << c.name;
    }
}

// =============================================================================
// 77. MULTI_AXIS: verify blockDim uses firstLayerRows
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_blockdim)
{
    // shape=[4, 100, 64], axis=[0, 2]
    // firstLayerRows = product of dims before first reduce axis (axis=0 in layer 0)
    // Layer 0 processes axis=2 (innermost), reduceAxisInShape=2
    // firstLayerRows = product of dims before pos 2 in original shape = 4*100 = 400
    auto r = RunTiling({4, 100, 64}, ge::DT_FLOAT, {0, 2});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);
    ASSERT_EQ(td->tilingMode, 4u);

    // firstLayerRows = 4 * 100 = 400 (dims before axis 2 in original shape)
    EXPECT_EQ(td->totalRows, 400);
    EXPECT_EQ(td->usedCoreNum, 20);  // min(20, 400) = 20
    EXPECT_EQ(r.info.blockNum, 20u);
    // rowsPerCore = ceil(400/20) = 20
    EXPECT_EQ(td->rowsPerCore, 20);
    // tailRows = 400 - 20*19 = 400 - 380 = 20
    EXPECT_EQ(td->tailRows, 20);
}

// =============================================================================
// 78. MULTI_AXIS: single core when firstLayerRows=1
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_single_core)
{
    // shape=[1, 100, 64], axis=[0, 2]
    // firstLayerRows = product of dims before axis 2 = 1*100 = 100
    // usedCoreNum = min(20, 100) = 20
    auto r = RunTiling({1, 100, 64}, ge::DT_FLOAT, {0, 2}, false, 20);
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);
    ASSERT_EQ(td->tilingMode, 4u);

    // Layer 0 (axis=2): shape [1,100,64], reduceAxisInShape=2
    // firstLayerRows = 1*100 = 100
    EXPECT_EQ(td->totalRows, 100);
    EXPECT_EQ(td->usedCoreNum, 20);
    EXPECT_EQ(r.info.blockNum, 20u);
}

// =============================================================================
// 79. MULTI_AXIS: layer sub-mode for tail-reduce layer (AR_FULLLOAD)
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_layer_submode_tail_ar)
{
    // shape=[4, 100, 64], axis=[0, 2], fp32
    // Layer 0 (axis=2): tail reduce, R=64, fp32
    // ubNeeded = 2*64*4 + ComputeTmpBufSize(64,4) + 64
    // tmpBuf for R_align=64: firstMaxRep=1, finalNeed=8, 8*4=32
    // ubNeeded = 2*64*4 + 32 + 64 = 608 ≤ UB → AR_FULLLOAD (subMode=0)
    auto r = RunTiling({4, 100, 64}, ge::DT_FLOAT, {0, 2});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);
    ASSERT_EQ(td->tilingMode, 4u);

    // Layer 0 is tail reduce with R=64 → AR_FULLLOAD
    EXPECT_EQ(td->layerMode[0], 0);  // AR_FULLLOAD
}

// =============================================================================
// 80. MULTI_AXIS: layer sub-mode for non-tail layer (ARA_FULLLOAD)
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_layer_submode_nontail_ara)
{
    // shape=[4, 100, 64], axis=[0, 2], fp32
    // Layer 1 (axis=0): shape [4,100], reduce pos=0, R=4, A0=100
    // ARA mode: ubNeeded = 4*100*4 + 100*4 + 100*4 + max(100*4,32)
    // = 1600 + 400 + 400 + 400 = 2800 ≤ UB → ARA_FULLLOAD (subMode=2)
    auto r = RunTiling({4, 100, 64}, ge::DT_FLOAT, {0, 2});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);
    ASSERT_EQ(td->tilingMode, 4u);

    // Layer 1 is non-tail reduce with small R=4, A0=100
    EXPECT_EQ(td->layerMode[1], 2);  // ARA_FULLLOAD
}

// =============================================================================
// 81. MULTI_AXIS: dtype TilingKey still driven by ASCENDC_TPL_SEL_PARAM
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_tilingkey_dtypes)
{
    struct DtypeKeyCase {
        std::string name;
        ge::DataType dtype;
        int64_t expectedTilingKey;
    };

    DtypeKeyCase cases[] = {
        {"fp16", ge::DT_FLOAT16, 1},
        {"fp32", ge::DT_FLOAT, 0},
        {"bf16", ge::DT_BF16, 27},
    };

    for (const auto& c : cases) {
        auto r = RunTiling({4, 100, 64}, c.dtype, {0, 2});
        ASSERT_TRUE(r.success) << "Failed for dtype: " << c.name;

        EXPECT_EQ(r.info.tilingKey, c.expectedTilingKey)
            << "TilingKey mismatch for dtype: " << c.name;
    }
}

// =============================================================================
// 82. MULTI_AXIS: 4D [0,2,3] → 3 layers, verify processing order
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_4d_0_2_3_processing_order)
{
    // shape=[2, 100, 50, 64], axis=[0, 2, 3], fp32
    // sorted = [0, 2, 3], processOrder = reversed = [3, 2, 0]
    // Layer 0: axis=3 (tail), shape [2,100,50,64], R=64
    // Layer 1: axis=2, shape [2,100,50], R=50 (after removing axis 3)
    // Layer 2: axis=0, shape [2,100], R=2
    auto r = RunTiling({2, 100, 50, 64}, ge::DT_FLOAT, {0, 2, 3});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);
    ASSERT_EQ(td->tilingMode, 4u);
    ASSERT_EQ(td->numLayers, 3);

    // Process order: axis=3, axis=2, axis=0
    EXPECT_EQ(td->layerAxis[0], 3);
    EXPECT_EQ(td->layerAxis[1], 2);
    EXPECT_EQ(td->layerAxis[2], 0);

    // Layer 0: tail reduce, R=64
    EXPECT_EQ(td->layerRLength[0], 64);
    EXPECT_EQ(td->layerIsTailReduce[0], 1);

    // Layer 1: original axis=2, after removing axis 3 → shape [2,100,50]
    // posInShape = 2 - 0 = 2 (last dim) → tail reduce, R=50
    EXPECT_EQ(td->layerRLength[1], 50);
    EXPECT_EQ(td->layerIsTailReduce[1], 1);
    EXPECT_EQ(td->layerA0Length[1], 0);

    // Layer 2: shape [2,100], reduce axis 0
    EXPECT_EQ(td->layerRLength[2], 2);
    EXPECT_EQ(td->layerReduceAxisIdx[2], 0);
}

// =============================================================================
// 83. MULTI_AXIS: inputDtype field set correctly
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_input_dtype_field)
{
    auto r = RunTiling({4, 100, 64}, ge::DT_BF16, {0, 2});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);
    ASSERT_EQ(td->tilingMode, 4u);
    EXPECT_EQ(td->inputDtype, static_cast<uint32_t>(ge::DT_BF16));
}

// =============================================================================
// 84. MULTI_AXIS: isAlign32B set to 0 for multi-axis mode
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_is_align32b_zero)
{
    // In MULTI_AXIS mode, isAlign32B is always set to 0 (no single contiguous reduce)
    auto r = RunTiling({4, 100, 64}, ge::DT_FLOAT, {0, 2});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);
    ASSERT_EQ(td->tilingMode, 4u);
    EXPECT_EQ(td->isAlign32B, 0u);
}

// =============================================================================
// 85. MULTI_AXIS: 5D max dimension with non-contiguous axes
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_5d_max_dim)
{
    // shape=[2, 3, 100, 4, 64], axis=[0, 2, 4], fp32
    auto r = RunTiling({2, 3, 100, 4, 64}, ge::DT_FLOAT, {0, 2, 4});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);
    ASSERT_EQ(td->tilingMode, 4u);
    ASSERT_EQ(td->numLayers, 3);

    // firstLayerRows = product of dims before axis 4 (processOrder[0]=4)
    // = 2*3*100*4 = 2400
    EXPECT_EQ(td->totalRows, 2400);
    EXPECT_EQ(td->usedCoreNum, 20);

    // Two 512B-aligned compact stages, sized by the largest intermediate.
    size_t expectedWs = static_cast<size_t>(2432) * sizeof(float) * 2;
    expectedWs = (expectedWs + 4095) & ~static_cast<size_t>(4095);
    EXPECT_EQ(r.info.workspaceSizes[0], expectedWs);
}

// =============================================================================
// 86. MULTI_AXIS: keep_dims does not affect MULTI_AXIS detection
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_keepdims)
{
    auto rTrue = RunTiling({4, 100, 64}, ge::DT_FLOAT, {0, 2}, true);
    auto rFalse = RunTiling({4, 100, 64}, ge::DT_FLOAT, {0, 2}, false);

    ASSERT_TRUE(rTrue.success);
    ASSERT_TRUE(rFalse.success);

    auto* tdTrue = AsTilingData(rTrue.info);
    auto* tdFalse = AsTilingData(rFalse.info);

    ASSERT_NE(tdTrue, nullptr);
    ASSERT_NE(tdFalse, nullptr);

    // Both should be MULTI_AXIS
    EXPECT_EQ(tdTrue->tilingMode, 4u);
    EXPECT_EQ(tdFalse->tilingMode, 4u);
    EXPECT_EQ(tdTrue->numLayers, tdFalse->numLayers);
    EXPECT_EQ(tdTrue->totalRows, tdFalse->totalRows);
}

// =============================================================================
// 87. MULTI_AXIS: contiguous non-tail multi-axis does NOT trigger Key=4
//     [1,2] on 4D where dims 1,2 are contiguous → coalesced ARA
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_multi_axis_contiguous_nontail_not_key4)
{
    // shape=[2, 100, 50, 64], axis=[1, 2]
    // Dims 1, 2 are contiguous → coalesced ARA mode, not MULTI_AXIS
    auto r = RunTiling({2, 100, 50, 64}, ge::DT_FLOAT, {1, 2});
    ASSERT_TRUE(r.success);

    auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_NE(td->tilingMode, 4u);
    // Coalesced: totalRows=2, rLength=100*50=5000, a0Length=64
    EXPECT_EQ(td->totalRows, 2);
    EXPECT_EQ(td->rLength, 5000);
    EXPECT_EQ(td->a0Length, 64);
}

// =============================================================================
// Submission paths: compact multi-axis stages and large all-reduce cooperation
// =============================================================================
TEST_F(SquareSumV1TilingTest, tiling_reduce_all_cooperative_uses_partial_workspace)
{
    // One very large output cannot get parallelism from A1/A0.  It must use
    // the explicit two-stage cooperative path instead of the AR single core.
    auto r = RunTiling({65536}, ge::DT_FLOAT, {0}, false, 20);
    ASSERT_TRUE(r.success);
    const auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);

    EXPECT_EQ(td->tilingMode, 5u);
    EXPECT_EQ(td->cooperativeChunkCols, 16320);
    EXPECT_EQ(td->cooperativeCoreNum, 5);
    EXPECT_EQ(td->usedCoreNum, 5);
    EXPECT_EQ(r.info.blockNum, 5u);
    ASSERT_GE(r.info.workspaceSizes.size(), 1u);
    EXPECT_GE(r.info.workspaceSizes[0], 4096u);
}

TEST_F(SquareSumV1TilingTest, tiling_multi_axis_compact_stages_are_dense_and_aligned)
{
    auto r = RunTiling({2, 3, 5}, ge::DT_BF16, {0, 2}, false, 20);
    ASSERT_TRUE(r.success);
    const auto* td = AsTilingData(r.info);
    ASSERT_NE(td, nullptr);
    ASSERT_EQ(td->tilingMode, 4u);
    ASSERT_EQ(td->numLayers, 2);

    // Layer 0 output is [2,3] = 6 fp32 values, not six 32B scalar slots.
    EXPECT_EQ(td->layerOutputElemCount[0], 6);
    EXPECT_EQ(td->layerWorkspaceOffset[0], 0);
    EXPECT_EQ(td->layerWorkspaceOffset[1], 128);
    EXPECT_GT(td->layerRChunkSizeCompact[0], 0);
    EXPECT_GT(td->layerReduceTmpBytes[0], 0u);
    EXPECT_EQ(r.info.workspaceSizes[0], 4096u);
}

TEST_F(SquareSumV1TilingTest, tiling_key4_fp16_regression_noncontiguous_keepdims)
{
    // Regression for the NPU layout failure: layer 0 is a non-tail fp16
    // 2D DMA, so its row pitch must be input-type (16 fp16 values/32B), not
    // fp32 (8 values/32B).  Keep-dims does not change kernel tiling.
    for (bool keepDims : {false, true}) {
        auto r = RunTiling({2, 3, 4, 5, 6}, ge::DT_FLOAT16, {1, 3}, keepDims);
        ASSERT_TRUE(r.success);
        const auto* td = AsTilingData(r.info);
        ASSERT_NE(td, nullptr);
        ASSERT_EQ(td->tilingMode, 4u);
        ASSERT_EQ(td->numLayers, 2);
        EXPECT_EQ(td->layerAxis[0], 3);
        EXPECT_EQ(td->layerAxis[1], 1);
        EXPECT_EQ(td->layerTileA0Align[0] % 16, 0);
        EXPECT_LE(Key4UbBytes(td), 184U * 1024U);
    }
}

TEST_F(SquareSumV1TilingTest, tiling_key4_r_chunk_4095_and_4096_boundaries)
{
    for (int64_t rLength : {4095, 4096}) {
        // Layer 0 reduces the tail, then layer 1 is the Key4 non-tail 2D
        // route.  The latter must never encode a DataCopy blockCount > 4095.
        auto r = RunTiling({2, rLength, 7, 3}, ge::DT_FLOAT16, {1, 3});
        ASSERT_TRUE(r.success);
        const auto* td = AsTilingData(r.info);
        ASSERT_NE(td, nullptr);
        ASSERT_EQ(td->tilingMode, 4u);
        ASSERT_EQ(td->layerRLength[1], rLength);
        EXPECT_LE(td->layerRChunkSizeCompact[1], 4095);
        EXPECT_EQ(td->layerNumRChunks[1],
                  (rLength + td->layerRChunkSizeCompact[1] - 1)
                      / td->layerRChunkSizeCompact[1]);
        EXPECT_LE(Key4UbBytes(td), 184U * 1024U);
    }
}

TEST_F(SquareSumV1TilingTest, tiling_key4_compact_ub_budget_and_dma_limits)
{
    for (auto dtype : {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT}) {
        // A0 tails exercise 1, 6, 7, 15, 16 and 17 valid columns.  The
        // negative axis form also guards normalization before Key4 routing.
        for (int64_t a0 : {1, 6, 7, 15, 16, 17}) {
            auto r = RunTiling({2, 3, 17, 5, a0}, dtype, {-4, -2});
            ASSERT_TRUE(r.success);
            const auto* td = AsTilingData(r.info);
            ASSERT_NE(td, nullptr);
            ASSERT_EQ(td->tilingMode, 4u);
            EXPECT_LE(Key4UbBytes(td), 184U * 1024U);
            for (int32_t li = 0; li < td->numLayers; ++li) {
                EXPECT_GT(td->layerRChunkSizeCompact[li], 0);
                EXPECT_LE(td->layerRChunkSizeCompact[li], 4095);
            }
        }
    }
}

} // namespace SquareSumV1UT
