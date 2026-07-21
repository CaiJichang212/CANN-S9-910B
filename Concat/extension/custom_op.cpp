/**
*
* Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
*
 * Concat 算子 pybind 调用入口：调用自定义算子 aclnnConcatCustom。
*/
#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include "../common/pytorch_npu_helper.hpp"
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;
using namespace at;


at::Tensor my_op_impl_npu(const tensor_list inputs, int64_t dim,
                    const at::IntArrayRef& output_shape) {

    auto round = 30;
    at::Tensor result;

    // 预热：使用 aclnnMul 占位，性能统计会过滤掉 aclnnMul
    auto a = at::empty(
        {4096, 4096},
        at::TensorOptions()
            .device(at::kPrivateUse1)
            .dtype(at::kFloat)
    );
    auto b = at::empty(
        {4096, 4096},
        at::TensorOptions()
            .device(at::kPrivateUse1)
            .dtype(at::kFloat)
    );
    auto c = at::empty(
        {4096, 4096},
        at::TensorOptions()
            .device(at::kPrivateUse1)
            .dtype(at::kFloat)
    );

    for (size_t i = 0; i < round; i++)
    {
        // 关键：确保每个输入是内存连续的
        // torch.split 返回的是原张量的 view（strided/带 offset），
        // 传给 aclnn 后 kernel 侧按连续内存读取会读到错误位置，导致精度错误。
        tensor_list inputs_contig;
        inputs_contig.reserve(inputs.size());
        for (const auto &t : inputs) {
            inputs_contig.emplace_back(t.contiguous());
        }
        at::TensorList inputs_x = at::TensorList(inputs_contig);
        result = at::empty(
            output_shape,
            inputs[0].options()  // 复用input的dtype/device（NPU）
        );
        EXEC_NPU_CMD(aclnnMul, a, b, c);
        // Python 接口保持不变；内部使用不与内置 Concat 冲突的注册名。
        EXEC_NPU_CMD(aclnnConcatCustom, inputs_x, dim, result);
    }
    return result;
}


// 修改my_op的输入输出
TORCH_LIBRARY(myops, m) {
		m.def("my_op(Tensor[] inputs, int dim, int[] output_shape) -> Tensor");
}

// 不修改
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
		m.impl("my_op", &my_op_impl_npu);
}

// 不修改
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
		m.def("custom_op", &my_op_impl_npu, "torch.cat");
}
