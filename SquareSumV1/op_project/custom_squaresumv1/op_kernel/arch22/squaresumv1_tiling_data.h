/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * \file squaresumv1_tiling_data.h
 * \brief SquareSumV1 TilingData struct (arch22 / Ascend910B)
 */

#ifndef _SQUARESUMV1_TILING_DATA_H_
#define _SQUARESUMV1_TILING_DATA_H_

#include <cstdint>

struct SquareSumV1TilingData {
    // Multi-core splitting
    int64_t totalRows;        // Total rows to process (A1 dimension)
    int64_t rowsPerCore;      // Rows assigned to each core
    int64_t tailRows;         // Rows for the last core (if different)
    int64_t usedCoreNum;      // Actual number of cores used

    // Reduction axis parameters
    int64_t rLength;          // Reduction axis length (valid element count)
    int64_t rLengthAlign;     // Reduction axis length aligned to 32B (in element count)

    // General parameters
    uint32_t inputDtype;      // Input dtype (ge::DataType value)
    uint32_t isAlign32B;      // Whether data is 32B aligned (0=no, 1=yes)
};

#endif // _SQUARESUMV1_TILING_DATA_H_
