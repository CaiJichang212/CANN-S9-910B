# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# op_kernel CMakeLists for SquareSumV1
# ----------------------------------------------------------------------------------------------------------

if(ASCEND_COMPUTE_UNIT STREQUAL "ascend950")
    npu_op_kernel_sources(ascendc_kernels
        OP_TYPE SquareSumV1
        KERNEL_DIR .
        KERNEL_FILE squaresumv1_arch35.cpp
    )
else()
    npu_op_kernel_sources(ascendc_kernels
        OP_TYPE SquareSumV1
        KERNEL_DIR .
        KERNEL_FILE squaresumv1_arch22.cpp
    )
endif()

npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR}
    TILING_LIBRARY cust_optiling
)

npu_op_package_add(${package_name}
    LIBRARY
        ascendc_kernels
)
