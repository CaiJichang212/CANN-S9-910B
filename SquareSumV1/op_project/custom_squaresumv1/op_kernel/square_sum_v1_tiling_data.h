/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * \file square_sum_v1_tiling_data.h
 * \brief SquareSumV1 TilingData struct (arch22 / Ascend910B)
 *
 * TilingKey=0: AR_FULLLOAD  - tail axis (innermost) reduce, full load
 * TilingKey=1: AR_COLSPLIT  - tail axis reduce, column chunk
 * TilingKey=2: ARA_FULLLOAD - non-tail axis reduce, full load (Pattern::Reduce::RA)
 * TilingKey=3: ARA_ROWSPLIT - non-tail axis reduce, row chunk
 * TilingMode=4: MULTI_AXIS_COMPACT - non-contiguous multi-axis, compact fp32 ping-pong stages
 * TilingMode=5: REDUCE_ALL_COOPERATIVE - large all-reduce, per-core fp32 partials + merge
 * TilingMode=6: NO_REDUCE - axis=[] elementwise square
 * TilingMode=7: EMPTY_REDUCE - reduction over an empty axis, write zeroes
 */

#ifndef _SQUARE_SUM_V1_TILING_DATA_H_
#define _SQUARE_SUM_V1_TILING_DATA_H_

#include <cstdint>

// ACLNN accepts rank <= 8, so an arbitrary axis list can require eight
// sequential layers in the non-contiguous multi-axis fallback.
constexpr int32_t SS_MAX_LAYERS = 8;

struct SquareSumV1TilingData {
    // === Multi-core splitting (all modes) ===
    int64_t totalRows;        // Total rows to process (A1 dimension: product of non-reduce outer dims)
    int64_t totalWorkItems;   // AR: totalRows; ARA: totalRows * numA0Tiles
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
    uint32_t reduceTmpBytes;  // Pattern::Reduce::RA temporary space (fp32)
    uint32_t reserved0;

    // === REDUCE_ALL_COOPERATIVE (mode=5) ===
    // Every participating AIV owns one contiguous input range and writes one
    // fp32 partial.  The final merge is deterministic and has no atomics.
    int64_t cooperativeChunkCols;
    int64_t cooperativeCoreNum;

    // === NO_REDUCE / EMPTY_REDUCE (modes=6,7) ===
    // Work is assigned in 32B blocks.  The final valid core owns the only
    // potentially partial block, preventing cross-core short-DMA writes.
    uint64_t noReduceTotalElements;
    int64_t noReduceBlocksPerCore;
    uint32_t noReduceTileElements;
    uint32_t noReduceTailElements;

    // === MULTI_AXIS (Key=4) parameters ===
    int32_t  numLayers;                                          // Number of reduce layers (sorted innermost first)
    int32_t  layerAxis[SS_MAX_LAYERS];                           // Original axis index for each layer (sorted ascending)
    int64_t  layerShapeBefore[SS_MAX_LAYERS][SS_MAX_LAYERS + 1]; // Shape before each layer reduce (up to rank 8)
    int32_t  layerNDims[SS_MAX_LAYERS];                          // Number of dims in shape before each layer
    int64_t  layerReduceAxisIdx[SS_MAX_LAYERS];                  // Position of the reduce axis within the current shape (0-indexed)
    int64_t  layerRLength[SS_MAX_LAYERS];                        // Reduce axis length for each layer
    int64_t  layerA0Length[SS_MAX_LAYERS];                       // Non-reduce tail length for each layer (0 = tail reduce)
    int64_t  layerInputElemCount[SS_MAX_LAYERS];                 // Total element count at each layer input
    int64_t  layerOutputElemCount[SS_MAX_LAYERS];                // Total element count at each layer output
    int64_t  layerIsTailReduce[SS_MAX_LAYERS];                   // 1 = tail reduce (AR), 0 = non-tail (ARA)
    int64_t  layerWorkspaceOffset[SS_MAX_LAYERS];                // Workspace offset in fp32 elements for each layer's output
    int64_t  layerChunkCols[SS_MAX_LAYERS];                      // Chunk columns for tail-reduce layers (AR_COLSPLIT fallback)
    int64_t  layerNumChunks[SS_MAX_LAYERS];                      // Number of chunks for tail-reduce layers
    int64_t  layerTileA0Align[SS_MAX_LAYERS];                    // Tile A0 aligned for ARA layers
    int64_t  layerNumA0Tiles[SS_MAX_LAYERS];                     // Number of A0 tiles for ARA layers
    int64_t  layerTileA0Len[SS_MAX_LAYERS];                      // Tile A0 length for ARA layers
    int64_t  layerRChunkSize[SS_MAX_LAYERS];                     // R chunk size for ARA_ROWSPLIT layers
    int64_t  layerNumRChunks[SS_MAX_LAYERS];                     // Number of R chunks for ARA_ROWSPLIT layers
    int64_t  layerMode[SS_MAX_LAYERS];                           // Sub-mode per layer: 0=AR_FULLLOAD, 1=AR_COLSPLIT, 2=ARA_FULLLOAD, 3=ARA_ROWSPLIT
    int64_t  layerOuterLength[SS_MAX_LAYERS];                    // Product of dimensions before R in this layer
    int64_t  layerRChunkSizeCompact[SS_MAX_LAYERS];              // Valid R rows/elements per compact-stage iteration
    uint32_t layerReduceTmpBytes[SS_MAX_LAYERS];                 // Pattern::Reduce::RA scratch for this layer
    uint32_t reserved1[SS_MAX_LAYERS];

    // === General parameters ===
    uint32_t tilingMode;      // 0=AR_FULLLOAD, 1=AR_COLSPLIT, 2=ARA_FULLLOAD, 3=ARA_ROWSPLIT, 4=MULTI_AXIS_COMPACT, 5=REDUCE_ALL_COOPERATIVE, 6=NO_REDUCE, 7=EMPTY_REDUCE
    uint32_t inputDtype;      // Input dtype (ge::DataType value)
    uint32_t isAlign32B;      // Whether rLength data is 32B aligned (0=no, 1=yes)
};

#endif // _SQUARE_SUM_V1_TILING_DATA_H_
