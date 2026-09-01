# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)
project(squaresumv1_op_prj)
find_package(ASC REQUIRED)

# 强制启用 C++17：CANN 8.5 的 ACLNN 头文件 (op_dfx.h / op_arg_def.h / op_cache_container.h)
# 依赖 std::string_view、std::apply、std::is_same_v、if constexpr、fold-expressions 等 C++17 特性。
# GCC 12+ 默认 C++17，但 GCC 7.3 (Ascend 官方镜像) 默认 C++14，必须显式开启。
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Architecture configuration
set(ARCH32_COMPUTE_UNITS ascend910b ascend910_93)

if(NOT DEFINED ASCEND_COMPUTE_UNIT OR ASCEND_COMPUTE_UNIT STREQUAL "")
    set(ASCEND_COMPUTE_UNIT ${ARCH32_COMPUTE_UNITS})
endif()
# The evaluator discovers the submitted dynamic implementation under the
# standard `customize` vendor and its SquareSumV1 registration.  Keep this
# package identity aligned with that published operator name.
set(package_name customize)

npu_op_package(${package_name}
    TYPE RUN
    CONFIG
        # The dynamic kernel python entry references square_sum_v1.cpp at
        # install/compile time.  Make source packaging explicit here instead
        # of relying on a toolchain-default CMake cache value.
        ENABLE_SOURCE_PACKAGE TRUE
        INSTALL_PATH ${CMAKE_BINARY_DIR}
)

# Some CANN 8.5 layouts omit toolkit/version.info even though the package
# helper still reads that legacy path. Rewrite the generated vendor identity
# from the compiler version file before CPack creates the .run package.
set(CANN_COMPILER_VERSION_FILE "${ASCEND_CANN_PACKAGE_PATH}/compiler/version.info")
if(NOT EXISTS "${CANN_COMPILER_VERSION_FILE}")
    set(CANN_COMPILER_VERSION_FILE "${ASCEND_CANN_PACKAGE_PATH}/toolkit/version.info")
endif()
if(NOT EXISTS "${CANN_COMPILER_VERSION_FILE}")
    message(FATAL_ERROR "CANN compiler version file not found under ${ASCEND_CANN_PACKAGE_PATH}")
endif()
file(STRINGS "${CANN_COMPILER_VERSION_FILE}" CANN_COMPILER_VERSION_LINE
    REGEX "^Version=" LIMIT_COUNT 1)
string(REGEX REPLACE "^Version=" "" CANN_COMPILER_VERSION "${CANN_COMPILER_VERSION_LINE}")
if(CANN_COMPILER_VERSION STREQUAL "")
    message(FATAL_ERROR "CANN compiler version is empty in ${CANN_COMPILER_VERSION_FILE}")
endif()
set(WRITE_VENDOR_VERSION_SCRIPT "${CMAKE_BINARY_DIR}/write_vendor_version.cmake")
file(GENERATE OUTPUT "${WRITE_VENDOR_VERSION_SCRIPT}"
    CONTENT "file(WRITE \"${CMAKE_BINARY_DIR}/version.info\" \"custom_opp_compiler_version=${CANN_COMPILER_VERSION}\\n\")\n")
add_custom_command(TARGET gen_version_info POST_BUILD
    COMMAND ${CMAKE_COMMAND} -P "${WRITE_VENDOR_VERSION_SCRIPT}"
    VERBATIM)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_host")
    add_subdirectory(op_host)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel")
    add_subdirectory(op_kernel)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/op_graph")
    add_subdirectory(op_graph)
endif()

message(WARNING "cmake 'make' does NOT build kernel binary by default. Use: bash build.sh --soc=<soc>")
