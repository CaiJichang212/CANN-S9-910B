import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
import sys  

case_data = {
    'case1': {
        'input':np.random.uniform(-50, 50, [32, 128]).astype(np.int8),
        'index':np.random.randint(low=0, high=32, size=120).astype(np.int32),
        'source':np.random.uniform(-10, 10, [120,128]).astype(np.int8),
        'dim': 0
    }
}

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
    #print(err_num)
    #print(real_result[~is_close])
    #print(golden[~is_close])

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

        index = torch.from_numpy(case_data[caseName]["index"])
        if int(num) == 3:
            input_x = case_data[caseName]["input"]
            source = case_data[caseName]["source"]
        else:
            input_x = torch.from_numpy(case_data[caseName]["input"])
            source = torch.from_numpy(case_data[caseName]["source"])

        dim = case_data[caseName]['dim']
        output = torch.index_add(input_x, dim, index, source)
        # print(output)

        # 修改输入
        output_npu = custom_ops_lib.custom_op(input_x.npu(), index.npu(), source.npu(), dim)
        # print(output_npu.cpu())
        if output_npu is None:
            print(f"{caseName} execution timed out!")
        else:
            if verify_result(output_npu.cpu(), output):
                print(f"{caseName} verify result pass!")
            else:
                print(f"{caseName} verify result failed!")

if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
    
