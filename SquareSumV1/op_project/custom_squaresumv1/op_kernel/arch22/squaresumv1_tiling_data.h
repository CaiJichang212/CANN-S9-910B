/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * \file squaresumv1_tiling_data.h
 * \brief SquareSumV1 TilingData struct (arch22 / Ascend910B)
 *
 * TilingKey=0: AR_FULLLOAD  - tail axis (innermost) reduce, full load
 * TilingKey=1: AR_COLSPLIT  - tail axis reduce, column chunk
 * TilingKey=2: ARA_FULLLOAD - non-tail axis reduce, full load (Pattern::Reduce::RA)
 * TilingKey=3: ARA_ROWSPLIT - non-tail axis reduce, row chunk
 */

#ifndef _SQUARESUMV1_TILING_DATA_H_
#define _SQUARESUMV1_TILING_DATA_H_

#include <cstdint>

struct SquareSumV1TilingData {
    // === Multi-core splitting (all modes) ===
    int64_t totalRows;        // Total rows to process (A1 dimension: product of non-reduce outer dims)
    int64_t rowsPerCore;      // Rows assigned to each core
    int64_t tailRows;         // Rows for the last core (if different)
    int64_t usedCoreNum;      // Actual number of cores used

    // === Reduction axis parameters (all modes) ===
    int64_t rLength;          // Reduction axis length (valid element count)
    int64_t rLengthAlign;     // Reduction axis length aligned to 32B (in element count, using larger of input/fp32 alignment)

    // === AR_COLSPLIT (Key=1) parameters ===
    int64_t chunkCols;        // Column chunk size for AR_COLSPLIT mode
    int64_t numChunks;        // Number of column chunks for AR_COLSPLIT mode

    // === ARA mode (Key=2,3) parameters ===
    int64_t a0Length;         // Non-reduce tail axis length (product of dims after R)
    int64_t a0LengthAlign;    // Non-reduce tail axis length aligned to 32B (in fp32 elements)
    int64_t tileA0Len;        // A0 tile length per iteration (for multi-tile A0 splitting)
    int64_t tileA0Align;      // A0 tile length aligned to 32B (in fp32 elements)
    int64_t numA0Tiles;       // Number of A0 tiles per row
    int64_t rChunkSize;       // R chunk size for ARA_ROWSPLIT mode
    int64_t numRChunks;       // Number of R chunks for ARA_ROWSPLIT mode

    // === General parameters ===
    uint32_t tilingMode;      // 0=AR_FULLLOAD, 1=AR_COLSPLIT, 2=ARA_FULLLOAD, 3=ARA_ROWSPLIT
    uint32_t inputDtype;      // Input dtype (ge::DataType value)
    uint32_t isAlign32B;      // Whether rLength data is 32B aligned (0=no, 1=yes)
};

#endif // _SQUARESUMV1_TILING_DATA_H_
