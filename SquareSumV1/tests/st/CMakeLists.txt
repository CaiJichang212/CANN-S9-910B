# ----------------------------------------------------------------------------------------------------------
# CMakeLists.txt for aclnnSquareSumV1 ST test
# 支持两种模式:
#   Mock (-DUSE_MOCK=ON): CPU golden 自测，无需 NPU
#   Real (默认):          NPU 执行 + CPU golden 比对
# ----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.10)
project(aclnn_square_sum_v1_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_MOCK "Use mock aclnn for testing without real operator" OFF)

if(USE_MOCK)
    add_definitions(-DUSE_MOCK_ACLNN)
    message(STATUS "Building with MOCK aclnn + CPU golden - full test validation without NPU")
else()
    message(STATUS "Building with REAL aclnn - NPU output vs CPU golden comparison")
endif()

# Set CANN paths (use official ASCEND_HOME_PATH)
if(NOT USE_MOCK)
    if(NOT DEFINED ENV{ASCEND_HOME_PATH})
        message(FATAL_ERROR "ERROR: ASCEND_HOME_PATH environment variable is not set! source set_env.sh in cann install directory first.")
    endif()
    if(NOT EXISTS "$ENV{ASCEND_HOME_PATH}")
        message(FATAL_ERROR "ERROR: ASCEND_HOME_PATH directory does not exist!")
    endif()

    set(CANN_ROOT $ENV{ASCEND_HOME_PATH})
    set(ASCENDCL_INCLUDE_DIRS ${CANN_ROOT}/include)
    set(ASCENDCL_LIBRARIES ${CANN_ROOT}/lib64/libascendcl.so)
    set(CANN_INCLUDE_DIRS ${CANN_ROOT}/include)
    set(CANN_LIBRARIES ${CANN_ROOT}/lib64/libopapi.so)

    message(STATUS "CANN_ROOT: ${CANN_ROOT}")
endif()

add_executable(test_aclnn_square_sum_v1 test_aclnn_square_sum_v1.cpp)

# Link Ascend libraries for Real mode
if(NOT USE_MOCK)
    # Find custom operator include directory (squaresumv1_custom vendor)
    find_path(CUSTOM_OP_INCLUDE_DIR
        NAMES aclnn_square_sum_v1.h
        PATHS
            ${CANN_ROOT}/opp/vendors/squaresumv1_custom/op_api/include
            /usr/local/Ascend/cann-8.5.0/opp/vendors/squaresumv1_custom/op_api/include
            /usr/local/Ascend/opp/vendors/squaresumv1_custom/op_api/include
            $ENV{HOME}/Ascend/opp/vendors/squaresumv1_custom/op_api/include
    )

    if(CUSTOM_OP_INCLUDE_DIR)
        message(STATUS "Found custom operator include dir: ${CUSTOM_OP_INCLUDE_DIR}")
    else()
        message(FATAL_ERROR "Custom operator include directory not found. Please install the operator first (bash build_out/custom_opp_*.run).")
    endif()

    target_include_directories(test_aclnn_square_sum_v1 PRIVATE
        ${ASCENDCL_INCLUDE_DIRS}
        ${CANN_INCLUDE_DIRS}
        ${CUSTOM_OP_INCLUDE_DIR}
    )

    # Find custom operator library (libcust_opapi.so)
    find_library(CUSTOM_OP_LIBRARY cust_opapi
        PATHS
            ${CANN_ROOT}/opp/vendors/squaresumv1_custom/op_api/lib
            /usr/local/Ascend/cann-8.5.0/opp/vendors/squaresumv1_custom/op_api/lib
            /usr/local/Ascend/opp/vendors/squaresumv1_custom/op_api/lib
            $ENV{HOME}/Ascend/opp/vendors/squaresumv1_custom/op_api/lib
    )

    if(NOT CUSTOM_OP_LIBRARY)
        message(WARNING "cust_opapi library not found, using default linkage")
    endif()

    target_link_libraries(test_aclnn_square_sum_v1
        ${ASCENDCL_LIBRARIES}
        ${CANN_LIBRARIES}
        ${CANN_ROOT}/lib64/libnnopbase.so
        ${CUSTOM_OP_LIBRARY}
    )
endif()

# Install target
install(TARGETS test_aclnn_square_sum_v1 DESTINATION bin)
