import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import custom_ops_lib
torch.npu.config.allow_internal_format = False
import numpy as np
import sys  

case_data = {
    # --- case1: 基线 fp16 2D 转置 (128,256) ---
    'case1': {
        'input':np.random.uniform(-10, 10, [128, 256]).astype(np.float16),
        'dims': (1,0)
    },
    # --- case2: fp32 2D 转置，非对齐列 (37x53) ---
    'case2': {
        'input':np.random.uniform(-10, 10, [37, 53]).astype(np.float32),
        'dims': (1,0)
    },
    # --- case3: int8 2D 转置，非对齐 (100, 33) ---
    'case3': {
        'input':np.random.randint(-127, 127, [100, 33]).astype(np.int8),
        'dims': (1,0)
    },
    # --- case4: int32 2D 转置，大对齐 (256, 512) ---
    'case4': {
        'input':np.random.randint(-10000, 10000, [256, 512]).astype(np.int32),
        'dims': (1,0)
    },
    # --- case5: fp16 3D 末两维交换 (12, 64, 97) 非对齐 ---
    'case5': {
        'input':np.random.uniform(-10, 10, [12, 64, 97]).astype(np.float16),
        'dims': (0, 2, 1)
    },
    # --- case6: fp32 4D 末两维交换 (3, 5, 37, 53) 非对齐 ---
    'case6': {
        'input':np.random.uniform(-10, 10, [3, 5, 37, 53]).astype(np.float32),
        'dims': (0, 1, 3, 2)
    },
    # --- case7: int8 5D 末两维交换 (2, 3, 4, 17, 9) 非对齐 ---
    'case7': {
        'input':np.random.randint(-127, 127, [2, 3, 4, 17, 9]).astype(np.int8),
        'dims': (0, 1, 2, 4, 3)
    },
    # --- case8: fp16 恒等 permute (无实际重排, 走 COPY) ---
    'case8': {
        'input':np.random.uniform(-10, 10, [64, 128]).astype(np.float16),
        'dims': (0, 1)
    },
    # --- case9: fp16 2D 转置 非对齐行 (33, 100) ---
    'case9': {
        'input':np.random.uniform(-10, 10, [33, 100]).astype(np.float16),
        'dims': (1,0)
    },
    # --- case10: 边界 1 元素 (1,1) ---
    'case10': {
        'input':np.random.uniform(-10, 10, [1, 1]).astype(np.float16),
        'dims': (1,0)
    },
    # --- case11: fp16 3D 任意 permute (非末两维交换: 维 0 移到末尾) (7, 11, 13) ---
    'case11': {
        'input':np.random.uniform(-10, 10, [7, 11, 13]).astype(np.float16),
        'dims': (1, 2, 0)
    },
    # --- case12: fp32 3D 任意 permute (2, 19, 23) ---
    'case12': {
        'input':np.random.uniform(-10, 10, [2, 19, 23]).astype(np.float32),
        'dims': (2, 0, 1)
    },
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
    # 根据数据类型设置误差阈值
    # int8/int32: 无误差（精确搬运）；fp32: 万分之一；fp16: 千分之一
    if golden.dtype in (torch.int8, torch.int32):
        # 整型要求完全一致
        is_close = (real_result == golden)
        err_num = torch.sum(~is_close).item()
        if err_num > 0:
            print(f"[ERROR] int result error, err_num={err_num}")
            return False
        print("test pass")
        return True
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

        input_x = torch.from_numpy(case_data[caseName]["input"])
        dims = ensure_tuple(case_data[caseName]['dims'])
        
        output = torch.permute(input_x, dims)
        output_shape = list(output.shape)
        # 修改输入
        output_npu = custom_ops_lib.custom_op(input_x.npu(), dims, output_shape)
        if output_npu is None:
            print(f"{caseName} execution timed out!")
        else:
            if verify_result(output_npu.cpu(), output):
                print(f"{caseName} verify result pass!")
            else:
                print(f"{caseName} verify result failed!")

if __name__ == "__main__":
    TestCustomOP().test_custom_op_case(sys.argv[1])
    
