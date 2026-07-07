"""Mirror the evaluation harness: torch.gt with inf/-inf/nan injection, 30 rounds,
report per-op median so we see what torch.gt actually dispatches to and the real
prof_sum that the leaderboard measures."""
import torch, torch_npu, numpy as np, sys
torch.npu.config.allow_internal_format = False

CASES = {
    'c1_small': ([32,64],[32,64], np.float16),
    'c2_outer_bcast': ([16384,1024],[1,1024], np.float16),  # suspected Case2 (biggest)
    'c3_inner_bcast': ([16384,1024],[16384,1], np.float16),
    'c4_int32': ([4096,4096],[4096,4096], np.int32),
    'c5_bf16': ([8192,4096],[8192,4096], torch.bfloat16),
}
def mk(shape, dt):
    if dt is torch.bfloat16:
        return torch.from_numpy(np.random.uniform(-1000,1000,shape).astype(np.float32)).to(torch.bfloat16)
    if isinstance(dt, torch.dtype):
        return torch.empty(shape,dtype=dt)
    if np.issubdtype(dt, np.integer):
        return torch.from_numpy(np.random.randint(-1000,1000,shape).astype(dt))
    return torch.from_numpy(np.random.uniform(-1000,1000,shape).astype(dt))

def inject(x):
    if not torch.is_floating_point(x): return
    r=torch.rand_like(x); m1=r<0.05; m2=(r>=0.05)&(r<0.10); m3=(r>=0.10)&(r<0.15)
    x.masked_fill_(m1, float('inf')); x.masked_fill_(m2, float('-inf')); x.masked_fill_(m3, float('nan'))

spec=sys.argv[1]
xs,ys,dt=CASES[spec]
x=mk(xs,dt); y=mk(ys,dt); inject(x); inject(y)
xn=x.npu(); yn=y.npu()
for _ in range(35):
    out=torch.gt(xn, yn)
print(f"[{spec}] torch.gt done, out={out.shape}", flush=True)
