import os, sys, json
os.environ.setdefault("LD_LIBRARY_PATH",
    "/usr/local/Ascend/cann-8.5.0/opp/vendors/customize/op_api/lib:/usr/local/python3.11.14/lib/python3.11/site-packages/torch/lib:" + os.environ.get("LD_LIBRARY_PATH",""))
import numpy as np, torch, torch_npu
torch.npu.config.allow_internal_format = False
import custom_ops_lib
from stress_test import gen_splits, verify_result

# argv: shape_csv dtype dim max_step  (e.g. "2024,3000" fp16 -1 128)
shape = [int(x) for x in sys.argv[1].split(",")]
dtstr = sys.argv[2]; dim = int(sys.argv[3]); ms = int(sys.argv[4])
dtmap = {"fp16": np.float16, "fp32": np.float32, "int32": np.int32, "int8": np.int8}
dt = dtmap[dtstr]
if dt == np.int8:    x = np.random.randint(-50, 50, shape).astype(np.int8)
elif dt == np.int32: x = np.random.randint(-1000, 1000, shape).astype(np.int32)
else:                x = np.random.uniform(-500, 500, shape).astype(dt)
input_x = torch.from_numpy(x)
sizes = gen_splits(input_x.shape[dim], ms)
inputs = list(torch.split(input_x, sizes, dim=dim))
inputs_npu = [t.npu() for t in inputs]
out = custom_ops_lib.custom_op(inputs_npu, dim, input_x.shape)
ok, err = verify_result(out.cpu(), input_x)
nz = sum(1 for s in sizes if s>0)
print(f"CONFIG shape={shape} dtype={dtstr} dim={dim} total_inputs={len(sizes)} nz={nz} -> {'PASS' if ok else 'FAIL'} err={err}", flush=True)
