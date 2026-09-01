#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "graph/tensor.h"
#include "tiling/reduce/reduce_tiling.h"

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "usage: reduce_tmp_probe <rows> <aligned_cols>\n";
        return 2;
    }

    try {
        const int64_t rows = std::stoll(argv[1]);
        const int64_t cols = std::stoll(argv[2]);
        if (rows <= 0 || cols <= 0) {
            throw std::invalid_argument("dimensions must be positive");
        }

        ge::Shape shape(std::vector<int64_t>{rows, cols});
        uint32_t maxBytes = 0;
        uint32_t minBytes = 0;
        AscendC::GetReduceSumMaxMinTmpSize(
            shape, ge::DT_FLOAT, AscendC::ReducePattern::RA,
            true, true, maxBytes, minBytes);
        std::cout << "rows=" << rows << " cols=" << cols
                  << " max_tmp_bytes=" << maxBytes
                  << " min_tmp_bytes=" << minBytes << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
    return 0;
}
