"""Measure the real evaluation prof_sum for each case:
   custom_ops_lib.custom_op (our optimized aclnnGreater) + inf injection,
   30 rounds like the harness. Reports median of the Greater op."""
import torch, torch_npu, numpy as np, sys, custom_ops_lib
torch.npu.config.allow_internal_format = False
CASES = {
    'c1_small':       ([32,64],[32,64], np.float16),
    'c2_outer_bcast': ([16384,1024],[1,1024], np.float16),
    'c3_inner_bcast': ([16384,1024],[16384,1], np.float16),
    'c4_int32':       ([4096,4096],[4096,4096], np.int32),
    'c5_bf16':        ([8192,4096],[8192,4096], torch.bfloat16),
}
def mk(shape, dt):
    if dt is torch.bfloat16:
        return torch.from_numpy(np.random.uniform(-1000,1000,shape).astype(np.float32)).to(torch.bfloat16)
    if isinstance(dt, torch.dtype): return torch.empty(shape,dtype=dt)
    if np.issubdtype(dt, np.integer): return torch.from_numpy(np.random.randint(-1000,1000,shape).astype(dt))
    return torch.from_numpy(np.random.uniform(-1000,1000,shape).astype(dt))
def inject(x):
    if not torch.is_floating_point(x): return
    r=torch.rand_like(x); x.masked_fill_(r<0.05,float('inf'))
    x.masked_fill_((r>=0.05)&(r<0.10),float('-inf')); x.masked_fill_((r>=0.10)&(r<0.15),float('nan'))
spec=sys.argv[1]; xs,ys,dt=CASES[spec]
x=mk(xs,dt); y=mk(ys,dt); inject(x); inject(y); xn=x.npu(); yn=y.npu()
g=torch.gt(x,y)  # golden (not timed, for sanity)
for _ in range(35):
    out=custom_ops_lib.custom_op(xn, yn)
ok = bool(torch.equal(out.cpu(), g))
print(f"[{spec}] accuracy={'PASS' if ok else 'FAIL'} out_elems={out.numel()}", flush=True)
