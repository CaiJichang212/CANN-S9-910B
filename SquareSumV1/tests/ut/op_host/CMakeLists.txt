# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

# =============================================================================
# Configuration
# =============================================================================
set(OP_NAME squaresumv1)
set(UT_EXE ${OP_NAME}_op_host_ut)

# Default SOC version for UT
if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION "Ascend910B")
endif()

message(STATUS "Building UT for SOC_VERSION: ${SOC_VERSION}")

# =============================================================================
# Op Host Source Files
# =============================================================================
set(OP_HOST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_project/custom_squaresumv1/op_host)

# The packaged source uses underscore-separated filenames directly under
# op_host/op_kernel; the old arch22 layout was removed with the 910B port.
set(TILING_SRC ${OP_HOST_DIR}/square_sum_v1_tiling.cpp)
set(TILING_DATA_H ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_project/custom_squaresumv1/op_kernel/square_sum_v1_tiling_data.h)

set(OP_HOST_SRCS
    ${OP_HOST_DIR}/square_sum_v1.cpp
    ${OP_HOST_DIR}/square_sum_v1_infershape.cpp
    ${TILING_SRC}
)

# =============================================================================
# UT Test Sources
# =============================================================================
set(UT_TEST_SRCS
    test_${OP_NAME}_tiling.cpp
)

# =============================================================================
# UT Test Main
# =============================================================================
set(UT_MAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_op_host_main.cpp)

# =============================================================================
# Build op_host as SHARED library
# =============================================================================
set(OP_HOST_SO ${OP_NAME}_op_host_ut_lib)

add_library(${OP_HOST_SO} SHARED
    ${OP_HOST_SRCS}
)

# Include Directories for .so
target_include_directories(${OP_HOST_SO} PRIVATE
    ${UT_COMMON_INCLUDE_DIRS}
    ${OP_HOST_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_project/custom_squaresumv1/op_kernel
)

# Compile Definitions for .so
target_compile_definitions(${OP_HOST_SO} PRIVATE
    BUILD_SOC_VERSION=${SOC_VERSION}
    _GLIBCXX_USE_CXX11_ABI=0
)

# Link Libraries for .so
target_link_libraries(${OP_HOST_SO} PRIVATE
    -Wl,--no-as-needed

    # whole-archive for static libraries with registration code
    -Wl,--whole-archive
        $ENV{ASCEND_HOME_PATH}/lib64/librt2_registry.a
        $ENV{ASCEND_HOME_PATH}/lib64/libtiling_api.a
    -Wl,--no-whole-archive

    $ENV{ASCEND_HOME_PATH}/lib64/libopp_registry.so
    $ENV{ASCEND_HOME_PATH}/lib64/libregister.so
    $ENV{ASCEND_HOME_PATH}/lib64/libmmpa.so
)

# =============================================================================
# Build UT Executable
# =============================================================================
add_executable(${UT_EXE}
    ${UT_MAIN_SRC}
    ${UT_TEST_SRCS}
    ${UT_COMMON_SRCS}
)

# Include Directories for UT
target_include_directories(${UT_EXE} PRIVATE
    ${UT_COMMON_INCLUDE_DIRS}
    ${OP_HOST_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_project/custom_squaresumv1/op_kernel
)

# Compile Definitions for UT
target_compile_definitions(${UT_EXE} PRIVATE
    BUILD_SOC_VERSION=${SOC_VERSION}
    _GLIBCXX_USE_CXX11_ABI=0
)

# Compile Options
target_compile_options(${UT_EXE} PUBLIC
    -fPIE
    -fno-access-control
)

# Link Directories for UT
target_link_directories(${UT_EXE} PRIVATE
    ${UT_COMMON_LIB_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
)

# =============================================================================
# Link Libraries for UT
# =============================================================================
target_link_libraries(${UT_EXE} PRIVATE
    gtest
    gtest_main

    -Wl,--no-as-needed

    $ENV{ASCEND_HOME_PATH}/lib64/libmetadef.so
    $ENV{ASCEND_HOME_PATH}/lib64/libunified_dlog.so
    $ENV{ASCEND_HOME_PATH}/lib64/libopp_registry.so
    $ENV{ASCEND_HOME_PATH}/lib64/libregister.so
    $ENV{ASCEND_HOME_PATH}/lib64/libgraph.so
    $ENV{ASCEND_HOME_PATH}/lib64/libgraph_base.so
    $ENV{ASCEND_HOME_PATH}/lib64/libplatform.so
    $ENV{ASCEND_HOME_PATH}/lib64/libc_sec.so
    $ENV{ASCEND_HOME_PATH}/lib64/libmmpa.so

    dl
    pthread
)

# =============================================================================
# Set RPATH for UT
# =============================================================================
set_target_properties(${UT_EXE} PROPERTIES
    INSTALL_RPATH "$ORIGIN;$ENV{ASCEND_HOME_PATH}/lib64"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "=== UT Build Configuration ===")
message(STATUS "  UT Executable: ${UT_EXE}")
message(STATUS "  OP Host Library: lib${OP_HOST_SO}.so")
message(STATUS "  SOC Version: ${SOC_VERSION}")
message(STATUS "  Tiling Source: ${TILING_SRC}")
message(STATUS "================================")
message(STATUS "")
