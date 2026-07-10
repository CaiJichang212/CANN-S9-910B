/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * \file tiling_case_executor.cpp
 * \brief Simplified tiling case executor (no nlohmann/json dependency)
 *
 * Replaces the json-based platform info parsing with direct map construction.
 */

#include "tiling_case_executor.h"
#include <algorithm>
#include <sstream>
#include <gtest/gtest.h>
#include "platform/platform_infos_def.h"
#include "base/registry/op_impl_space_registry_v2.h"

#define STR_IMPL(x) #x
#define STR(x) STR_IMPL(x)

// =============================================================================
// DO_TILING macro: builds a TilingContext, runs the tiling function
// =============================================================================
#define DO_TILING(tilingContextPara)                                                                                   \
    auto contextFaker = gert::TilingContextFaker();                                                                    \
    size_t inputNum = tilingContextPara.inputTensorDesc_.size();                                                       \
    size_t outputNum = tilingContextPara.outputTensorDesc_.size();                                                     \
    if (tilingContextPara.inputInstanceNum_.size() != 0 || tilingContextPara.outputInstanceNum_.size() != 0) {         \
        contextFaker.IrInstanceNum(tilingContextPara.inputInstanceNum_, tilingContextPara.outputInstanceNum_);         \
    } else {                                                                                                           \
        contextFaker.NodeIoNum(inputNum, outputNum);                                                                   \
    }                                                                                                                  \
    std::vector<gert::Tensor *> inputTensors = {};                                                                     \
    std::vector<gert::Tensor *> outputTensors = {};                                                                    \
    std::vector<std::unique_ptr<gert::Tensor>> inputTensorsKeepAlive = {};                                             \
    std::vector<std::unique_ptr<gert::Tensor>> outputTensorsKeepAlive = {};                                            \
    for (size_t index = 0; index < inputNum; index++) {                                                                \
        std::unique_ptr<gert::Tensor> curTensor = std::make_unique<gert::Tensor>(                                      \
            tilingContextPara.inputTensorDesc_[index].shape_,                                                          \
            gert::StorageFormat(tilingContextPara.inputTensorDesc_[index].format_,                                     \
             tilingContextPara.inputTensorDesc_[index].format_,                                                        \
             gert::ExpandDimsType()),                                                                                  \
            gert::TensorPlacement::kOnHost,                                                                            \
            tilingContextPara.inputTensorDesc_[index].dtype_,                                                          \
            tilingContextPara.inputTensorDesc_[index].isConst_ ?                                                       \
            tilingContextPara.inputTensorDesc_[index].constValue_:                                                     \
            nullptr);                                                                                                  \
        inputTensors.push_back(curTensor.get());                                                                       \
        inputTensorsKeepAlive.push_back(std::move(curTensor));                                                         \
    }                                                                                                                  \
    for (size_t index = 0; index < outputNum; index++) {                                                               \
        std::unique_ptr<gert::Tensor> curTensor = std::make_unique<gert::Tensor>(                                      \
            tilingContextPara.outputTensorDesc_[index].shape_,                                                         \
            gert::StorageFormat(tilingContextPara.outputTensorDesc_[index].format_,                                    \
             tilingContextPara.outputTensorDesc_[index].format_,                                                       \
             gert::ExpandDimsType()),                                                                                  \
            gert::TensorPlacement::kOnHost,                                                                            \
            tilingContextPara.outputTensorDesc_[index].dtype_,                                                         \
            tilingContextPara.outputTensorDesc_[index].isConst_ ?                                                      \
            tilingContextPara.outputTensorDesc_[index].constValue_:                                                    \
            nullptr);                                                                                                  \
        outputTensors.push_back(curTensor.get());                                                                      \
        outputTensorsKeepAlive.push_back(std::move(curTensor));                                                        \
    }                                                                                                                  \
    contextFaker.InputTensors(inputTensors).OutputTensors(outputTensors);                                              \
    for (auto& attrInfo : tilingContextPara.attrs_) {                                                                  \
        switch (attrInfo.attr_.type_) {                                                                                \
            case Ops::Math::AnyValue::ValueType::VT_BOOL: {                                                            \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<bool*>(attrInfo.attr_.valuePtr_.get()));       \
                break;}                                                                                                \
            case Ops::Math::AnyValue::ValueType::VT_INT: {                                                             \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<int64_t*>(attrInfo.attr_.valuePtr_.get()));    \
                break;}                                                                                                \
            case Ops::Math::AnyValue::ValueType::VT_FLOAT: {                                                           \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<float*>(attrInfo.attr_.valuePtr_.get()));      \
                break;}                                                                                                \
            case Ops::Math::AnyValue::ValueType::VT_STRING: {                                                          \
                contextFaker.Attr(attrInfo.attrName_, ge::AscendString(reinterpret_cast<std::string*>(attrInfo.attr_.valuePtr_.get())->c_str()));\
                break;}                                                                                                \
            case Ops::Math::AnyValue::ValueType::VT_LIST_BOOL: {                                                       \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<std::vector<bool>*>(attrInfo.attr_.valuePtr_.get()));\
                break;}                                                                                                \
            case Ops::Math::AnyValue::ValueType::VT_LIST_INT: {                                                        \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<std::vector<int64_t>*>(attrInfo.attr_.valuePtr_.get()));\
                break;}                                                                                                \
            case Ops::Math::AnyValue::ValueType::VT_LIST_LIST_INT: {                                                   \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<std::vector<std::vector<int64_t>>*>(attrInfo.attr_.valuePtr_.get()));\
                break;}                                                                                                \
            case Ops::Math::AnyValue::ValueType::VT_LIST_FLOAT: {                                                      \
                contextFaker.Attr(attrInfo.attrName_, *reinterpret_cast<std::vector<float>*>(attrInfo.attr_.valuePtr_.get()));\
                break;}                                                                                                \
            default:                                                                                                   \
                std::cout << "[ERROR] Unsupported attr type: " << attrInfo.attr_.type_ << std::endl;                   \
        }                                                                                                              \
    }                                                                                                                  \
    /* 2. base information */                                                                                          \
    fe::PlatFormInfos platformInfo;                                                                                    \
    platformInfo.Init();                                                                                               \
    auto tilingData = gert::TilingData::CreateCap(tilingContextPara.tilingDataSize_);                                  \
    auto workspace = gert::ContinuousVector::Create<size_t>(4096);                                                     \
    auto contextHolder = contextFaker.SetOpType(tilingContextPara.opName_.c_str())                                     \
                                     .CompileInfo(tilingContextPara.compileInfo_)                                      \
                                     .PlatformInfo(reinterpret_cast<char*>(&platformInfo))                             \
                                     .TilingData(tilingData.get())                                                     \
                                     .Workspace(reinterpret_cast<gert::ContinuousVector *>(workspace.get()))           \
                                     .Build();                                                                         \
    /* Build platform info maps directly (no json) */                                                                  \
    std::string buildSocVersion = STR(BUILD_SOC_VERSION);                                                              \
    map<string, string> socToUpper = {                                                                                  \
        {"ascend910b", "Ascend910B"},                                                                                   \
        {"ascend910_93", "Ascend910_93"},                                                                               \
        {"ascend950", "Ascend950"},                                                                                     \
        {"ascend310p", "Ascend310P"},                                                                                   \
        {"ascend910", "Ascend910"},                                                                                     \
        {"ascend310b", "Ascend310B"}                                                                                    \
    };                                                                                                                 \
    if (!buildSocVersion.empty()) {                                                                                    \
        buildSocVersion = socToUpper[buildSocVersion];                                                                 \
    }                                                                                                                  \
    map<string, string> socToArch = {                                                                                  \
        {"Ascend310P", "2002"},                                                                                        \
        {"Ascend910B", "2201"},                                                                                        \
        {"Ascend910_93", "2201"},                                                                                      \
        {"Ascend950", "3510"},                                                                                         \
        {"Ascend910", "1001"}                                                                                          \
    };                                                                                                                 \
    map<string, string> socInfos;                                                                                      \
    map<string, string> aicoreSpec;                                                                                    \
    map<string, string> intrinsics;                                                                                    \
    socInfos["core_type_list"] = "AICore";                                                                             \
    socInfos["ai_core_cnt"] = std::to_string(tilingContextPara.coreNum_);                                              \
    socInfos["l2_size"] = "33554432";                                                                                  \
    socInfos["vector_core_cnt"] = std::to_string(tilingContextPara.coreNum_);                                          \
    aicoreSpec["ub_size"] = std::to_string(tilingContextPara.ubSize_);                                                 \
    aicoreSpec["l0_a_size"] = "65536";                                                                                 \
    aicoreSpec["l0_b_size"] = "65536";                                                                                 \
    aicoreSpec["l0_c_size"] = "262144";                                                                                \
    aicoreSpec["l1_size"] = "1048576";                                                                                 \
    aicoreSpec["bt_size"] = "0";                                                                                       \
    aicoreSpec["load3d_constraints"] = "1";                                                                            \
    aicoreSpec["cube_freq"] = "cube_freq";                                                                             \
    intrinsics["Intrinsic_data_move_l12ub"] = "float16";                                                               \
    intrinsics["Intrinsic_data_move_l0c2ub"] = "float16";                                                              \
    map<string, string> socversions = {                                                                                \
        {"NpuArch", socToArch[buildSocVersion]}, {"Short_SoC_version", buildSocVersion}};                              \
    auto tilingContext = contextHolder.GetContext();                                                                   \
    tilingContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);                                             \
    tilingContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");                                                  \
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);                                        \
    tilingContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);                           \
    tilingContext->GetPlatformInfo()->SetPlatformRes("version", socversions);                                          \
    /* 3. get tiling func */                                                                                           \
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();                         \
    if (spaceRegistry == nullptr) {                                                                                   \
        throw std::invalid_argument("not found spaceRegistry");                                                        \
    }                                                                                                                  \
    auto functionStruct = spaceRegistry->GetOpImpl(tilingContextPara.opName_.c_str());                                 \
    if (functionStruct == nullptr) {                                                                                   \
        throw std::invalid_argument("not found " + tilingContextPara.opName_);                                         \
    }                                                                                                                  \
    auto tilingFunc = functionStruct->tiling;                                                                          \
    /* 4. run tiling func */                                                                                           \
    auto tilingRet = tilingFunc(tilingContext);

// =============================================================================
// Helper: convert raw tiling data to string
// =============================================================================
template <typename T>
static string to_string(void* buf, size_t size) {
    string result;
    const T* data = reinterpret_cast<const T*>(buf);
    size_t len = size / sizeof(T);
    for (size_t i = 0; i < len; i++) {
        result += std::to_string(data[i]);
        result += " ";
    }
    return result;
}

// =============================================================================
// ExecuteTestCase: runs tiling and validates expected results
// =============================================================================
void ExecuteTestCase(const gert::TilingContextPara& tilingContextPara,
                     ge::graphStatus                expectResult,
                     uint64_t                       expectTilingKey,
                     const string&                  expectTilingData,
                     const std::vector<size_t>&     expectWorkspaces)
{
    DO_TILING(tilingContextPara);

    EXPECT_EQ(tilingRet, expectResult);
    if (expectResult == ge::GRAPH_FAILED) {
        return;
    }

    // check workspace
    size_t workspaceCount = tilingContext->GetWorkspaceNum();
    if (workspaceCount > 0) {
        ASSERT_EQ(workspaceCount, expectWorkspaces.size());
        auto workspaceSizes = tilingContext->GetWorkspaceSizes(workspaceCount);
        for (size_t i = 0; i < workspaceCount; i++) {
            ASSERT_EQ(workspaceSizes[i], expectWorkspaces[i]);
        }
    }

    // check tiling key
    auto tilingKeyResult = tilingContext->GetTilingKey();
    ASSERT_EQ(tilingKeyResult, expectTilingKey);

    // check tiling data
    if (expectTilingData == EMPTY_EXPECT_TILING_DATA) {
        return;
    }
    auto rawTilingData = tilingContext->GetRawTilingData();
    auto tilingDataResult = to_string<int64_t>(rawTilingData->GetData(), rawTilingData->GetDataSize());
    EXPECT_EQ(tilingDataResult, expectTilingData);
}

void ExecuteTestCase(const gert::TilingContextPara& tilingContextPara,
                     ge::graphStatus                expectResult,
                     uint64_t                       expectTilingKey,
                     const std::vector<size_t>&     expectWorkspaces)
{
    ExecuteTestCase(tilingContextPara, expectResult, expectTilingKey, EMPTY_EXPECT_TILING_DATA, expectWorkspaces);
}

// =============================================================================
// ExecuteTiling: runs tiling and returns TilingInfo (no validation)
// =============================================================================
bool ExecuteTiling(const gert::TilingContextPara& tilingContextPara, TilingInfo& tilingInfo)
{
    DO_TILING(tilingContextPara);

    if (tilingRet != ge::GRAPH_SUCCESS) {
        return false;
    }

    tilingInfo.tilingKey = tilingContext->GetTilingKey();
    tilingInfo.blockNum = tilingContext->GetBlockDim();
    size_t workspaceCount = tilingContext->GetWorkspaceNum();
    if (workspaceCount > 0) {
        auto workSpaceSizes = tilingContext->GetWorkspaceSizes(workspaceCount);
        for (size_t i = 0; i < workspaceCount; i++) {
            tilingInfo.workspaceSizes.push_back(workSpaceSizes[i]);
        }
    }
    auto rawTilingData = tilingContext->GetRawTilingData();
    tilingInfo.tilingData = std::make_unique<uint8_t[]>(rawTilingData->GetDataSize());
    tilingInfo.tilingDataSize = rawTilingData->GetDataSize();
    std::copy_n(static_cast<const uint8_t*>(rawTilingData->GetData()), rawTilingData->GetDataSize(), tilingInfo.tilingData.get());

    return true;
}
