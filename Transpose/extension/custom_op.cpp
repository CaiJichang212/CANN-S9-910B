/**
*
* Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <torch/extension.h>
#include <torch/csrc/autograd/custom_function.h>
#include "../common/pytorch_npu_helper.hpp"
using torch::autograd::Function;
using torch::autograd::AutogradContext;
using tensor_list = std::vector<at::Tensor>;
using namespace at;


at::Tensor my_op_impl_npu(const at::Tensor& input, const at::IntArrayRef& dims, 
            const at::IntArrayRef& result_shape) {
    // at::Tensor result = at::empty_like(input);
    auto round = 30;
    at::Tensor result;

    auto a = at::empty(
        {4096, 4096},
        at::TensorOptions()
            .device(at::kPrivateUse1)  // 昇腾NPU固定设备标识 kPrivateUse1
            .dtype(at::kFloat)         // float32
    );
    auto b = at::empty(
        {4096, 4096},
        at::TensorOptions()
            .device(at::kPrivateUse1)  // 昇腾NPU固定设备标识 kPrivateUse1
            .dtype(at::kFloat)         // float32
    );
    auto c = at::empty(
        {4096, 4096},
        at::TensorOptions()
            .device(at::kPrivateUse1)  // 昇腾NPU固定设备标识 kPrivateUse1
            .dtype(at::kFloat)         // float32
    );
    for (size_t i = 0; i < round; i++)
    {
        result = at::empty(result_shape, input.options());
        EXEC_NPU_CMD(aclnnMul, a, b, c);
        EXEC_NPU_CMD(aclnnTranspose, input, dims, result);
    }
    return result;
}



// 修改my_op的输入输出
TORCH_LIBRARY(myops, m) {
		m.def("my_op(Tensor input, int[] dims, int[] result_shape) -> Tensor");
}

// 不修改
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
		m.impl("my_op", &my_op_impl_npu);
}

// 不修改
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
		m.def("custom_op", &my_op_impl_npu, "torch.sum(torch.permute)");
}
