# NPU Driver 不可用（容器内 device_count=0）

**发现时间**: 2026-07-10（阶段二开发前环境核查）
**状态**: ⚠️ 阻塞 NPU 实跑验证；**不阻塞** kernel 开发 / 编译 / simulator 功能验证

## 现象
- `npu-smi info` 所有设备报 `device is used (ret=-8020)` + `DrvMngGetConsoleLogLevel failed (ret=4)`
- python torch_npu：`rtGetDeviceCount execution failed, reason=driver error:internal error`（runtime 507899）
- `torch.npu.device_count() = 0`（**0 设备可见**）
- `import torch` 报 `get platform info failed, drvErr=87`

## 排查结论
- `/dev/davinci0-7` 存在，权限 `crw-rw-rw-`（root 可访问，权限非问题）
- **无任何进程占用** davinci fd（lsof + /proc 遍历均空）
- driver `version.info` 存在（Version=25.5.1）
- `/usr/local/Ascend/driver/lib64`（在 LD_LIBRARY_PATH 中）**无 libdrv/libdcmi/libruntime**——driver 用户态库缺失或路径不符
- `/var/davinci`、`/etc/davinci` 不存在
- CANN runtime 库（libruntime.so/libascendcl.so）在 `cann-8.5.0/lib64` 存在且可加载
- ccec 编译器、msopgen、simulator（msopgen sim）均可用

## 判定
driver 用户态与设备的通信通道在容器内不可用（宿主机侧 driver daemon/服务异常，或容器 capability/挂载缺失）。**容器内无法修复**，需宿主机层面处理：
1. 检查宿主机 driver 状态（宿主 `npu-smi info`）
2. 重启 driver / 容器
3. 确认容器启动参数（`--device /dev/davinci*`、driver 目录挂载、capability）

## 影响与降级策略
- **阻塞**：A1-P NPU 穿刺、汇合/迭代验收 NPU 精度、最终精度验收 NPU、性能验收 msprof
- **不阻塞**：算子工程创建（msopgen）、kernel 开发、编译（build.sh）、**simulator 功能精度验证**（msopgen sim / ascendc-ops-simulator skill）
- **降级**：开发阶段用 CANN simulator 做功能精度验证（保证 kernel 计算正确）；NPU 实跑验证（精度复验 + msprof 性能采集）延后至 driver 修复后补做，届时可能需针对 NPU 实际行为做小幅修正（对齐/时序）