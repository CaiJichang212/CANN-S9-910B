"""Accuracy sweep for Greater: dtypes, special values and generic broadcasts."""
import torch, torch_npu, numpy as np, custom_ops_lib, sys
torch.npu.config.allow_internal_format = False

def mk(shape, dtype):
    if dtype == torch.bfloat16:
        return torch.from_numpy(np.random.uniform(-1000,1000,shape).astype(np.float32)).to(torch.bfloat16)
    if isinstance(dtype, torch.dtype):
        return torch.empty(shape, dtype=dtype).copy_(torch.from_numpy(np.random.uniform(-1000,1000,shape)))
    if np.issubdtype(dtype, np.integer):
        return torch.from_numpy(np.random.randint(-1000,1000,shape).astype(dtype))
    return torch.from_numpy(np.random.uniform(-1000,1000,shape).astype(dtype))

DTYPES = [(np.float16,'fp16'),(np.float32,'fp32'),(torch.bfloat16,'bf16'),(np.int32,'int32'),(np.int8,'int8')]
# (xshape, yshape, tag) covering same-shape, outer-bcast, inner-bcast, mid-bcast,
# all-scalar, non-align and the resident/scalar-batch fast paths.
CASES = [
    ([2048,512],[2048,512],'same'),
    ([2048,512],[1,512],'y-outer-bcast'),
    ([1,512],[2048,512],'x-outer-bcast'),
    ([2048,512],[2048,1],'y-inner-bcast'),
    ([2048,512],[1,1],'both-scalar'),
    ([64,128,256],[1,128,256],'3d-outer'),
    ([64,128,256],[64,1,256],'3d-mid'),
    ([64,128,256],[64,128,1],'3d-inner'),
    ([1000,777],[1,777],'nonalign-outer'),   # non-256 innerSize -> resident disabled
    ([37],[1],'1d-bcast'),
    ([1,2,3,4,5],[2,3,4,5],'5d'),
    ([5,7,256],[5,7,1],'3d-inner-index'),
    ([5,7,256],[5,1,1],'3d-inner-mixed-index'),
    ([5,7,256],[5,1,256],'3d-partial-resident-y'),
    ([5,1,256],[5,7,256],'3d-partial-resident-x'),
    ([1],[256],'1d-scalar-x'),
    ([256],[1],'1d-scalar-y'),
]

def inject_specials(t):
    if not torch.is_floating_point(t) or t.numel() < 4:
        return
    flat = t.reshape(-1)
    flat[0] = float('nan')
    flat[1] = float('inf')
    flat[2] = float('-inf')
    flat[3] = 0.0

fails = 0; total = 0
for dt, dname in DTYPES:
    for xs, ys, tag in CASES:
        total += 1
        try:
            x = mk(xs, dt); y = mk(ys, dt)
        except Exception as e:
            continue
        inject_specials(x); inject_specials(y)
        golden = torch.gt(x, y)
        out = custom_ops_lib.custom_op(x.npu(), y.npu()).cpu()
        ok = bool(torch.equal(out, golden))
        if not ok:
            fails += 1
            # show first mismatch
            diff = torch.nonzero(out != golden)
            n_mm = diff.shape[0]
            idx = diff[0].tolist() if n_mm else []
            print(f"FAIL {dname:5} {tag:16} x={xs} y={ys} mismatches={n_mm} first={idx} got={out.flatten()[0].item() if out.numel() else 0} gold={golden.flatten()[0].item() if golden.numel() else 0}")
print(f"\n{total-fails}/{total} passed, {fails} failed")
