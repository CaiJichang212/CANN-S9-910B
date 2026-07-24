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


at::Tensor my_op_impl_npu(const at::Tensor& input, const at::Tensor& index, const at::Tensor& source,
                int64_t dim) {
    // 30 轮 aclnnIndexAdd（前若干轮自然预热，取稳定区间中位数）。
    // 不再创建占位 a/b/c 与 aclnnMul：在 set_device(k>0) + msprof 环境下，
    // 额外的 aclnnMul 会污染 stream 上下文，使后续 aclnnIndexAdd 的 source
    // 参数在 GetWorkspaceSize 阶段被判为 null。去掉后与 builtin 路径对齐。
    constexpr int round = 30;
    at::Tensor result;
    for (int i = 0; i < round; i++)
    {
        result = at::empty_like(input);
        EXEC_NPU_CMD(aclnnIndexAdd, input, index, source, dim, result);
    }
    return result;
}



// 修改my_op的输入输出
TORCH_LIBRARY(myops, m) {
		m.def("my_op(Tensor input, Tensor index, Tensor source, int dim) -> Tensor");
}

// 不修改
TORCH_LIBRARY_IMPL(myops, PrivateUse1, m) {
		m.impl("my_op", &my_op_impl_npu);
}

// 不修改
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
		m.def("custom_op", &my_op_impl_npu, "torch.index_add");
}
