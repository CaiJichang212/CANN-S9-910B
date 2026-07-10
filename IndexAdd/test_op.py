import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
import sys

# numpy 无 bfloat16：bf16 case 以 float32 存放 + 'dtype' 字段标记，测试时转 torch.bfloat16。
case_data = {
    'case1': {
        'input':np.random.uniform(-50, 50, [32, 128]).astype(np.int8),
        'index':np.random.randint(low=0, high=32, size=120).astype(np.int32),
        'source':np.random.uniform(-10, 10, [120,128]).astype(np.int8),
        'dim': 0
    },
    # 本地自测用例（覆盖 5 dtype / dim / 非对齐 / index 重复）。评测系统会注入真实 case2-5。
    'case2': {
        'input':np.random.uniform(-1, 1, [256, 512]).astype(np.float32),
        'index':np.random.randint(low=0, high=256, size=500).astype(np.int32),
        'source':np.random.uniform(-1, 1, [500, 512]).astype(np.float32),
        'dim': 0
    },
    'case3': {
        'input':np.random.uniform(-1, 1, [128, 256, 64]).astype(np.float16),
        'index':np.random.randint(low=0, high=128, size=200).astype(np.int32),
        'source':np.random.uniform(-1, 1, [128, 200, 64]).astype(np.float16),
        'dim': 1
    },
    'case4': {
        'input':np.random.uniform(-1, 1, [200, 400]).astype(np.float32),
        'index':np.random.randint(low=0, high=200, size=300).astype(np.int32),
        'source':np.random.uniform(-1, 1, [300, 400]).astype(np.float32),
        'dim': 0,
        'dtype': torch.bfloat16
    },
    'case5': {
        'input':np.random.randint(-50, 50, [64, 1000]).astype(np.int32),
        'index':np.random.randint(low=0, high=64, size=400).astype(np.int32),
        'source':np.random.randint(-50, 50, [400, 1000]).astype(np.int32),
        'dim': 0
    },
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
        case = case_data[caseName]
        target_dt = case.get("dtype")
        input_x = torch.from_numpy(case["input"])
        source = torch.from_numpy(case["source"])
        if target_dt is not None:
            input_x = input_x.to(target_dt)
            source = source.to(target_dt)

        dim = case['dim']
        output = torch.index_add(input_x, dim, index, source)

        # 修改输入
        output_npu = custom_ops_lib.custom_op(input_x.npu(), index.npu(), source.npu(), dim)
        if output_npu is None:
            print(f"{caseName} execution timed out!")
        else:
            if verify_result(output_npu.cpu(), output):
                print(f"{caseName} verify result pass!")
            else:
                print(f"{caseName} verify result failed!")

if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
