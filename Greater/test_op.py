import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
import copy
import sys  

case_data = {
    'case1': {
        'other':np.random.uniform(-1000, 1000, [32,64]).astype(np.float16),
        'x':np.random.uniform(-1000, 1000, [32,64]).astype(np.float16)
    }
}

def insert_special_values(
    x: torch.Tensor,
    prob_inf: float = 0.05,
    prob_ninf: float = 0.05,
    prob_nan: float = 0.05
) -> None:
    """
    原地修改tensor，随机插入 inf / -inf / nan
    自动识别dtype：浮点型正常填充；整型直接跳过不做任何修改
    :param x: 待原地修改的tensor
    :param prob_inf: 单个元素赋值inf概率
    :param prob_ninf: 单个元素赋值-inf概率
    :param prob_nan: 单个元素赋值nan概率
    """
    # 整型无法存储inf/nan，直接返回
    if not torch.is_floating_point(x):
        return 

    dtype = x.dtype
    r = torch.rand_like(x)

    # 构造各特殊值掩码
    mask_inf = r < prob_inf
    mask_ninf = (r >= prob_inf) & (r < prob_inf + prob_ninf)
    mask_nan = (r >= prob_inf + prob_ninf) & (r < prob_inf + prob_ninf + prob_nan)

    # 原地填充，自动适配当前tensor dtype
    x.masked_fill_(mask_inf, torch.tensor(float("inf"), dtype=dtype))
    x.masked_fill_(mask_ninf, torch.tensor(float("-inf"), dtype=dtype))
    x.masked_fill_(mask_nan, torch.tensor(float("nan"), dtype=dtype))

def verify_result(real_result, golden):
    # 根据数据类型设置误差阈值（与原逻辑一致）
    if golden.dtype == torch.float32:
        rtol = 1e-4  # fp32相对误差阈值
        atol = 1e-4  # fp32绝对误差阈值
    else:
        rtol = 1e-3  # fp16相对误差阈值
        atol = 1e-3  # fp16绝对误差阈值

    minimum = 10e-10
    golden = torch.where(golden == 0, minimum, golden)
    real_result = torch.where(real_result == 0, minimum, real_result)

    abs_diff = torch.abs(real_result - golden)
    rel_diff = abs_diff / torch.max(torch.abs(real_result), torch.abs(golden))
    is_close = (abs_diff <= atol) | (rel_diff <= rtol)
    # 补充NaN判断（与equal_nan=True保持一致）
    both_nan = torch.isnan(real_result) & torch.isnan(golden)
    is_close = is_close | both_nan
    err_num = torch.sum(~is_close).item()  # 取反统计不满足的数量

    # 与原逻辑相同的误差数量判断
    if real_result.numel() * rtol < err_num:
        print(f"[ERROR] result error")
        return False
    print("test pass")
    return True

    
class TestCustomOP(TestCase):
    def test_custom_op_case(self,num):
        print(num)
        caseName='case'+str(num) 

        if int(num) == 3:
            input_x = case_data[caseName]["other"]
            input_other = case_data[caseName]["x"]
        else:
            input_x = torch.from_numpy(case_data[caseName]["other"])
            input_other = torch.from_numpy(case_data[caseName]["x"])
        insert_special_values(input_x)
        insert_special_values(input_other)
        output = torch.gt(input_x, input_other)
        # 修改输入
        output_npu = custom_ops_lib.custom_op(input_x.npu(), input_other.npu())
        if output_npu is None:
            print(f"{caseName} execution timed out!")
        else:
            if verify_result(output_npu.cpu(), output):
                print(f"{caseName} verify result pass!")
            else:
                print(f"{caseName} verify result failed!")

if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
    
