import ctypes
import os

out = []
for name in ["libcust_opapi.so", "libopapi.so"]:
    try:
        h = ctypes.CDLL(name)
        has_indexadd = hasattr(h, "aclnnIndexAdd")
        has_getws = hasattr(h, "aclnnIndexAddGetWorkspaceSize")
        out.append(f"dlopen({name}) OK  aclnnIndexAdd={has_indexadd} GetWorkspaceSize={has_getws}")
    except Exception as e:
        out.append(f"dlopen({name}) FAILED: {e}")
msg = "\n".join(out)
print(msg, flush=True)
with open("/tmp/diag_lib.txt", "w") as f:
    f.write(msg + "\nLD=" + os.environ.get("LD_LIBRARY_PATH", "NONE"))
