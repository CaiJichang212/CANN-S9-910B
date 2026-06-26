import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
import sys  

case_data = {
    'case1': {
        'input':np.random.uniform(-10, 10, [123, 31]).astype(np.float16),
        'axis': -1,
        'keep_dims': True
    }
}

def ensure_tuple(variable):
    # 判断是否为元组
    if isinstance(variable, tuple):
        # 是元组则直接返回
        return variable
    else:
        # 不是元组则转换为单元素元组
        return (variable,)

def verify_result(real_result, golden):
    # 根据数据类型设置误差阈值（与原逻辑一致）
    if golden.dtype == torch.float32:
        rtol = 1e-4  # fp32相对误差阈值
        atol = 1e-4  # fp32绝对误差阈值
        loss = 1e-4
    else:
        rtol = 1e-2  # fp16相对误差阈值
        atol = 1e-2  # fp16绝对误差阈值
        loss = 1e-3

    minimum = 10e-10
    golden = torch.where(golden == 0, minimum, golden)
    real_result = torch.where(real_result == 0, minimum, real_result)

    abs_diff = torch.abs(real_result - golden)
    rel_diff = abs_diff / torch.max(torch.abs(real_result), torch.abs(golden))
    is_close = (abs_diff <= atol) | (rel_diff <= rtol)
    # 补充NaN判断（与keep_dims=True保持一致）
    both_nan = torch.isnan(real_result) & torch.isnan(golden)
    is_close = is_close | both_nan
    err_num = torch.sum(~is_close).item()  # 取反统计不满足的数量

    # 与原逻辑相同的误差数量判断
    if real_result.numel() * loss < err_num:
        print(f"[ERROR] result error")
        return False
    print("test pass")
    return True


class TestCustomOP(TestCase):
    def test_custom_op_case(self,num):
        print(num)
        caseName='case'+str(num) 
        input_x = None
        if int(num) == 2 or int(num) == 3:
            input_x = case_data[caseName]["input"]
        else:
            input_x = torch.from_numpy(case_data[caseName]["input"])
        axis = ensure_tuple(case_data[caseName]['axis'])
        keep_dims = case_data[caseName]["keep_dims"]
        
        output = torch.sum(torch.square(input_x), axis, keepdim=keep_dims)
        output_shape = list(output.shape)
        # 修改输入
        output_npu = custom_ops_lib.custom_op(input_x.npu(), axis, keep_dims, output_shape)
        if output_npu is None:
            print(f"{caseName} execution timed out!")
        else:
            if verify_result(output_npu.cpu(), output):
                print(f"{caseName} verify result pass!")
            else:
                print(f"{caseName} verify result failed!")

if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
    
