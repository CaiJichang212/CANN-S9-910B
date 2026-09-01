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

# spec -> (xshape, yshape, dtype, note)
# dtype: np.float16/np.float32/np.int32/np.int8 或 torch.bfloat16
MATRIX = {
    # ===== fp16 参考组 (最全模式覆盖, fp16 是最优路径) =====
    'f16_same_big':    ([8192, 8192], [8192, 8192], np.float16, 'fp16 同形大 67M elem (带宽天花板基线)'),
    'f16_same_med':    ([2024, 3000], [2024, 3000], np.float16, 'fp16 同形中 6M (评分参考 case)'),
    'f16_same_sml':    ([32, 64], [32, 64], np.float16, 'fp16 同形小 (launch 开销探测)'),
    'f16_same_12m':    ([4096, 3072], [4096, 3072], np.float16,
                        'fp16 同形 12.6M (64MiB 估算 IO 阈值下侧)'),
    'f16_same_14m':    ([4096, 3584], [4096, 3584], np.float16,
                        'fp16 同形 14.7M (64MiB 估算 IO 阈值上侧)'),
    'f16_bouter':      ([16384, 1024], [1, 1024], np.float16, 'fp16 外维广播 (P1 驻留路径)'),
    'f16_bouter_big':  ([65536, 1024], [1, 1024], np.float16, 'fp16 外维广播大 (P1)'),
    'f16_binner':      ([16384, 1024], [16384, 1], np.float16, 'fp16 内维广播 (P2 标量批量路径)'),
    'f16_bboth':       ([16384, 1024], [1, 1], np.float16, 'fp16 双维广播'),
    'f16_3d_bouter':   ([8192, 4, 1024], [1, 4, 1024], np.float16, 'fp16 3D 外维广播'),
    'f16_5d_bcast':    ([2, 4, 8, 128, 32], [1, 1, 8, 128, 32], np.float16, 'fp16 5D 广播'),
    'f16_tail_same':   ([8192, 1000], [8192, 1000], np.float16, 'fp16 非对齐同形 (退化通用路径)'),
    'f16_tail_bouter': ([8192, 1000], [1, 1000], np.float16, 'fp16 非对齐外维广播 (退化)'),
    'f16_tail_bouter_rev': ([1, 1000], [8192, 1000], np.float16, 'fp16 非对齐外维广播镜像 (P1)'),
    'f16_tail_binner': ([8192, 1000], [8192, 1], np.float16, 'fp16 非对齐内维广播 (P2)'),
    'f16_tail_binner_rev': ([8192, 1], [8192, 1000], np.float16, 'fp16 非对齐内维广播镜像 (P2)'),
    'f16_p1_rows2': ([2, 1000], [1, 1000], np.float16, 'P1 两行 DataCopyPad 读写 smoke'),
    'f16_p1_rows2_rev': ([1, 1000], [2, 1000], np.float16, 'P1 两行 DataCopyPad 镜像 smoke'),
    'f16_p1_partial_tail': ([16, 512, 1000], [16, 1, 1000], np.float16, 'P1 部分外维复用 + 非对齐'),
    'f16_p2_5d_tail': ([2, 2, 2, 2, 1000], [2, 2, 2, 2, 1], np.float16, 'P2 5D 非对齐'),
    'f16_p1_large_inner': ([8192, 10000], [1, 10000], np.float16,
                           'P1 大 innerSize=N=10000，探测 resident 分块缺口'),
    'f16_p1_large_inner_rev': ([1, 10000], [8192, 10000], np.float16,
                               'P1 大 innerSize=N=10000，x resident 镜像'),
    'f16_p2_large_inner': ([8192, 10000], [8192, 1], np.float16,
                           'P2 大 innerSize=N=10000，探测 scalar batch 分块缺口'),
    'f16_p1_large_inner_partial': ([8, 64, 10000], [8, 1, 10000], np.float16,
                                   'P1 大 inner partial-group fallback'),
    'f16_p1_tiles21_outer2': ([2, 32, 6048], [1, 32, 6048], np.float16,
                              'P1 innerTiles=21 outerSize=2 二维 worker 边界'),
    'f16_p1_tile_minus_1': ([4096, 9215], [1, 9215], np.float16,
                            'P1 fp16 TILE-1 边界'),
    'f16_p1_tile_exact': ([4096, 9216], [1, 9216], np.float16,
                          'P1 fp16 TILE 精确边界'),
    'f16_p1_tile_plus_1': ([4096, 9217], [1, 9217], np.float16,
                           'P1 fp16 TILE+1 边界'),
    'f16_p1_tiny_inner_big': ([1000, 1000, 31], [1, 1, 31], np.float16,
                              'P1 小 innerSize=N=31、大 outer，探测逐行开销'),
    'f16_p2_tiny_inner_big': ([1000, 1000, 31], [1000, 1000, 1], np.float16,
                              'P2 小 innerSize=N=31、大 outer，探测逐行开销'),
    'f16_vec':         ([67108864,], [67108864,], np.float16, 'fp16 1D 向量 64M (纯 flatten)'),
    # ===== fp32 (ComputeT=float, Select/Cast 仍走 half 中转) =====
    'f32_same_big':    ([8192, 8192], [8192, 8192], np.float32, 'fp32 同形大'),
    'f32_same_4m':     ([2048, 2048], [2048, 2048], np.float32,
                        'fp32 同形 4.2M (64MiB 估算 IO 阈值下侧)'),
    'f32_same_8m':     ([2048, 4096], [2048, 4096], np.float32,
                        'fp32 同形 8.4M (64MiB 估算 IO 阈值上侧)'),
    'f32_bouter':      ([16384, 1024], [1, 1024], np.float32, 'fp32 外维广播'),
    'f32_binner':      ([16384, 1024], [16384, 1], np.float32, 'fp32 内维广播'),
    'f32_binner_16128': ([16128, 1024], [16128, 1], np.float32, 'fp32 P2 连续 scalar 门限前'),
    'f32_binner_16129': ([16129, 1024], [16129, 1], np.float32, 'fp32 P2 连续 scalar 原门限点'),
    'f32_tail_same':   ([4096, 1000], [4096, 1000], np.float32, 'fp32 非对齐同形'),
    'f32_p1_large_inner': ([4096, 10000], [1, 10000], np.float32,
                           'fp32 P1 大 innerSize=N=10000'),
    'f32_p1_large_inner_rev': ([1, 10000], [4096, 10000], np.float32,
                               'fp32 P1 大 innerSize=N=10000，x resident 镜像'),
    'f32_p2_large_inner': ([4096, 10000], [4096, 1], np.float32,
                           'fp32 P2 大 innerSize=N=10000'),
    'f32_p1_tile_minus_1': ([4096, 5119], [1, 5119], np.float32,
                            'P1 fp32 TILE-1 边界'),
    'f32_p1_tile_exact': ([4096, 5120], [1, 5120], np.float32,
                          'P1 fp32 TILE 精确边界'),
    'f32_p1_tile_plus_1': ([4096, 5121], [1, 5121], np.float32,
                           'P1 fp32 TILE+1 边界'),
    'f32_p1_tiny_inner_big': ([1000, 1000, 31], [1, 1, 31], np.float32,
                              'fp32 P1 小 innerSize=N=31、大 outer'),
    'f32_p2_tiny_inner_big': ([1000, 1000, 31], [1000, 1000, 1], np.float32,
                              'fp32 P2 小 innerSize=N=31、大 outer'),
    # ===== bf16 (Cast bf16->float -> Compare) =====
    'bf16_same_big':   ([8192, 4096], [8192, 4096], torch.bfloat16, 'bf16 同形大 (Cast 路径)'),
    'bf16_bouter':     ([4096, 1024], [1, 1024], torch.bfloat16, 'bf16 外维广播'),
    'bf16_binner':     ([4096, 1024], [4096, 1], torch.bfloat16, 'bf16 内维广播'),
    'bf16_p1_large_inner': ([2048, 10000], [1, 10000], torch.bfloat16,
                            'bf16 P1 大 innerSize resident 切片'),
    # ===== int32 (Max+Compare(EQ)x2+Selectx2, 6-op 精确路径) =====
    'i32_same_big':    ([4096, 4096], [4096, 4096], np.int32, 'int32 同形大 (6-op 路径)'),
    'i32_bouter':      ([4096, 1024], [1, 1024], np.int32, 'int32 外维广播'),
    'i32_binner':      ([4096, 1024], [4096, 1], np.int32, 'int32 内维广播'),
    'i32_p1_large_inner': ([2048, 10000], [1, 10000], np.int32,
                           'int32 P1 大 innerSize resident 切片'),
    # ===== int8 (Cast int8->half -> Compare) =====
    'i8_same_big':     ([8192, 8192], [8192, 8192], np.int8, 'int8 同形大 (Cast->half)'),
    'i8_bouter':       ([16384, 1024], [1, 1024], np.int8, 'int8 外维广播'),
    'i8_binner':       ([16384, 1024], [16384, 1], np.int8, 'int8 内维广播'),
    'i8_p1_large_inner_3d': ([2048, 2, 6000], [1, 2, 6000], np.int8,
                             'int8 合法多维 innerSize=12000 resident 切片'),
    # ===== 边界 / 退化 =====
    'scalar':          ([1], [1], np.float16, '标量 vs 标量 (极小, launch 主导)'),
    'f16_bscalar_x':   ([1, 1024], [8192, 1024], np.float16, 'fp16 x 标量广播 (x 为广播方)'),
    'f16_tail_1d':     ([1000,], [1000,], np.float16, 'fp16 1D 非对齐小'),
}

# 非对齐广播覆盖：所有 dtype 的 P1/P2 正反向、多个行宽与 core 分割边界。
# 单个 spec 仍可独立执行，避免默认 profiling 意外扩大到这组压力矩阵。
for _tag, _dtype in [('f16', np.float16), ('f32', np.float32), ('bf16', torch.bfloat16),
                     ('i32', np.int32), ('i8', np.int8)]:
    MATRIX.update({
        f'{_tag}_tail_p1': ([21, 1000], [1, 1000], _dtype, f'{_tag} 非对齐 P1'),
        f'{_tag}_tail_p1_rev': ([1, 1000], [21, 1000], _dtype, f'{_tag} 非对齐 P1 镜像'),
        f'{_tag}_tail_p2': ([21, 1000], [21, 1], _dtype, f'{_tag} 非对齐 P2'),
        f'{_tag}_tail_p2_rev': ([21, 1], [21, 1000], _dtype, f'{_tag} 非对齐 P2 镜像'),
    })
for _n in (1, 7, 31, 33, 255, 257, 777, 1000, 10000):
    MATRIX[f'f16_p1_n{_n}'] = ([21, _n], [1, _n], np.float16, f'P1 行宽 N={_n}')
    MATRIX[f'f16_p2_n{_n}'] = ([21, _n], [21, 1], np.float16, f'P2 行宽 N={_n}')
for _outer in (1, 19, 20, 21):
    MATRIX[f'f16_p2_outer{_outer}'] = ([_outer, 1000], [_outer, 1], np.float16,
                                        f'P2 core 分割 outer={_outer}')
for _n in (7, 31, 255):
    MATRIX[f'f32_p2_n{_n}'] = ([21, _n], [21, 1], np.float32,
                                f'fp32 P2 Brcb 行宽 N={_n}')


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
    """注入随机特殊值，并在每个逻辑行尾固定覆盖边界特殊值/整数极值。"""
    if not torch.is_floating_point(x):
        if x.dtype == torch.int32 and x.numel():
            flat = x.reshape(-1, x.shape[-1] if x.dim() else 1)
            flat[:, -1] = torch.iinfo(torch.int32).min
            if flat.shape[1] > 1:
                flat[:, -2] = torch.iinfo(torch.int32).max
        return
    r = torch.rand_like(x)
    x.masked_fill_(r < 0.05, float('inf'))
    x.masked_fill_((r >= 0.05) & (r < 0.10), float('-inf'))
    x.masked_fill_((r >= 0.10) & (r < 0.15), float('nan'))
    flat = x.reshape(-1, x.shape[-1] if x.dim() else 1)
    flat[:, -1] = float('nan')
    if flat.shape[1] > 1:
        flat[:, -2] = float('inf')
    if flat.shape[1] > 2:
        flat[:, -3] = float('-inf')


def dtype_name(dt):
    if isinstance(dt, torch.dtype):
        return str(dt).replace('torch.', '')
    return np.dtype(dt).name


def run(spec):
    torch.npu.set_device(DEV)
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
        raise SystemExit(1)


if __name__ == "__main__":
    spec = sys.argv[1]
    if spec == '__list__':
        print(' '.join(MATRIX.keys()))
    else:
        run(spec)
