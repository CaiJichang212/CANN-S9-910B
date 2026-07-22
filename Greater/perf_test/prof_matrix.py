"""Greater 算子系统化性能测试矩阵 (torch.gt / aclnnGreater 自定义 kernel)。

覆盖 5 dtype (fp16/fp32/bf16/int32/int8) × 多种 shape 模式
(同形 / 外维广播 / 内维广播 / 双维 / 3D / 5D / 标量 / 非对齐 / 1D 向量)。

每个 spec: 注入 inf/-inf/nan (整型跳过), warmup+30 轮 custom_op (实际跑的
是已安装的自定义 aclnnGreater, 通过 libcust_opapi.so 覆盖内置符号),
最后一次 torch.equal 精度校验。配合 msprof 采集 AICore 时间与流水占比。

Usage:
    python3 prof_matrix.py <spec>      # 运行单个用例
    python3 prof_matrix.py __list__    # 打印所有 spec 名 (空格分隔)
"""
import os
import sys
import torch
import torch_npu
import numpy as np
import custom_ops_lib

torch.npu.config.allow_internal_format = False

# 独占指定 NPU (用户要求 4-7); set_device 比 ASCEND_VISIBLE_DEVICES 更可靠
DEV = int(os.environ.get('GREATER_DEV', '4'))
torch.npu.set_device(DEV)

# spec -> (xshape, yshape, dtype, note)
# dtype: np.float16/np.float32/np.int32/np.int8 或 torch.bfloat16
MATRIX = {
    # ===== fp16 参考组 (最全模式覆盖, fp16 是最优路径) =====
    'f16_same_big':    ([8192, 8192], [8192, 8192], np.float16, 'fp16 同形大 67M elem (带宽天花板基线)'),
    'f16_same_med':    ([2024, 3000], [2024, 3000], np.float16, 'fp16 同形中 6M (评分参考 case)'),
    'f16_same_sml':    ([32, 64], [32, 64], np.float16, 'fp16 同形小 (launch 开销探测)'),
    'f16_bouter':      ([16384, 1024], [1, 1024], np.float16, 'fp16 外维广播 (P1 驻留路径)'),
    'f16_bouter_big':  ([65536, 1024], [1, 1024], np.float16, 'fp16 外维广播大 (P1)'),
    'f16_binner':      ([16384, 1024], [16384, 1], np.float16, 'fp16 内维广播 (P2 标量批量路径)'),
    'f16_bboth':       ([16384, 1024], [1, 1], np.float16, 'fp16 双维广播'),
    'f16_3d_bouter':   ([8192, 4, 1024], [1, 4, 1024], np.float16, 'fp16 3D 外维广播'),
    'f16_5d_bcast':    ([2, 4, 8, 128, 32], [1, 1, 8, 128, 32], np.float16, 'fp16 5D 广播'),
    'f16_tail_same':   ([8192, 1000], [8192, 1000], np.float16, 'fp16 非对齐同形 (退化通用路径)'),
    'f16_tail_bouter': ([8192, 1000], [1, 1000], np.float16, 'fp16 非对齐外维广播 (退化)'),
    'f16_vec':         ([67108864,], [67108864,], np.float16, 'fp16 1D 向量 64M (纯 flatten)'),
    # ===== fp32 (ComputeT=float, Select/Cast 仍走 half 中转) =====
    'f32_same_big':    ([8192, 8192], [8192, 8192], np.float32, 'fp32 同形大'),
    'f32_bouter':      ([16384, 1024], [1, 1024], np.float32, 'fp32 外维广播'),
    'f32_binner':      ([16384, 1024], [16384, 1], np.float32, 'fp32 内维广播'),
    'f32_tail_same':   ([4096, 1000], [4096, 1000], np.float32, 'fp32 非对齐同形'),
    # ===== bf16 (Cast bf16->float -> Compare) =====
    'bf16_same_big':   ([8192, 4096], [8192, 4096], torch.bfloat16, 'bf16 同形大 (Cast 路径)'),
    'bf16_bouter':     ([4096, 1024], [1, 1024], torch.bfloat16, 'bf16 外维广播'),
    'bf16_binner':     ([4096, 1024], [4096, 1], torch.bfloat16, 'bf16 内维广播'),
    # ===== int32 (Max+Compare(EQ)x2+Selectx2, 6-op 精确路径) =====
    'i32_same_big':    ([4096, 4096], [4096, 4096], np.int32, 'int32 同形大 (6-op 路径)'),
    'i32_bouter':      ([4096, 1024], [1, 1024], np.int32, 'int32 外维广播'),
    'i32_binner':      ([4096, 1024], [4096, 1], np.int32, 'int32 内维广播'),
    # ===== int8 (Cast int8->half -> Compare) =====
    'i8_same_big':     ([8192, 8192], [8192, 8192], np.int8, 'int8 同形大 (Cast->half)'),
    'i8_bouter':       ([16384, 1024], [1, 1024], np.int8, 'int8 外维广播'),
    'i8_binner':       ([16384, 1024], [16384, 1], np.int8, 'int8 内维广播'),
    # ===== 边界 / 退化 =====
    'scalar':          ([1], [1], np.float16, '标量 vs 标量 (极小, launch 主导)'),
    'f16_bscalar_x':   ([1, 1024], [8192, 1024], np.float16, 'fp16 x 标量广播 (x 为广播方)'),
    'f16_tail_1d':     ([1000,], [1000,], np.float16, 'fp16 1D 非对齐小'),
}


def make(shape, dtype):
    """按 dtype 生成确定性随机数据 (int8 限制在 [-128,127])."""
    if dtype is torch.bfloat16:
        return torch.from_numpy(np.random.uniform(-1000, 1000, shape).astype(np.float32)).to(torch.bfloat16)
    if isinstance(dtype, torch.dtype):
        npdt = torch.empty(0, dtype=dtype).numpy().dtype
        return torch.from_numpy(np.random.uniform(-1000, 1000, shape).astype(npdt))
    if np.issubdtype(dtype, np.integer):
        lo, hi = (-128, 128) if dtype == np.int8 else (-1000, 1000)
        return torch.from_numpy(np.random.randint(lo, hi, shape).astype(dtype))
    return torch.from_numpy(np.random.uniform(-1000, 1000, shape).astype(dtype))


def inject(x):
    """浮点型随机注入 inf/-inf/nan (15%); 整型跳过."""
    if not torch.is_floating_point(x):
        return
    r = torch.rand_like(x)
    x.masked_fill_(r < 0.05, float('inf'))
    x.masked_fill_((r >= 0.05) & (r < 0.10), float('-inf'))
    x.masked_fill_((r >= 0.10) & (r < 0.15), float('nan'))


def dtype_name(dt):
    if isinstance(dt, torch.dtype):
        return str(dt).replace('torch.', '')
    return np.dtype(dt).name


def run(spec):
    xs, ys, dt, note = MATRIX[spec]
    x = make(xs, dt)
    y = make(ys, dt)
    inject(x)
    inject(y)
    golden = torch.gt(x, y)          # CPU golden (含 inf/-inf/nan 语义)
    xn = x.npu()
    yn = y.npu()
    out = None
    for _ in range(35):              # warmup + 30 measured (mirrors harness)
        out = custom_ops_lib.custom_op(xn, yn)
    out_cpu = out.cpu()
    ok = bool(torch.equal(out_cpu, golden))
    print(f"[{spec}] {note} | nelem={out_cpu.numel()} dtype={dtype_name(dt)} "
          f"xshape={list(xs)} yshape={list(ys)} acc={'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        diff = (out_cpu != golden)
        idx = torch.nonzero(diff)[0]
        print(f"  first mismatch {idx.tolist()}: got={out_cpu[idx].item()} golden={golden[idx].item()}", flush=True)


if __name__ == "__main__":
    spec = sys.argv[1]
    if spec == '__list__':
        print(' '.join(MATRIX.keys()))
    else:
        run(spec)
