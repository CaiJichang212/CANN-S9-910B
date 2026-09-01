# SquareSumV1 算子 mode3-fulltile-nozero 阶段执行报告

## 决策

`mode3-fulltile-nozero-20260901_020621` 在第一次 NPU 正确性矩阵中失败，状态为 `rejected_correctness`。按硬门禁未进行第二次重复或任何性能采集。

## 对象与差异

| 项 | 值 |
| --- | --- |
| parent `.run` | `199592a838ebe3c8adbbd6558e3b70a67685d4be85bb8ffecfafda02a16fb416` |
| candidate `.run` | `d2060f6791a2da06df98b236f64b840347d4d08ab291ecdd296e03309868702c` |
| parent Kernel 源 | `5d3593d0173141b21a01c5615946f841c02fe752b40118076c52b96e181dcdcf` |
| candidate Kernel 源 | `9a5215385903a5193494fba14ebddc3c702de13d0f0b30a9a961200c2db8ce46` |
| patch | `4e43c18ecab5bbe062b2db2a9e0a5fda3ac1f2f4748d2dd1eca09b1dbe9a481e` |

Host Tiling、Proto、OpAPI 和 config 与 parent 哈希一致，三 dtype Kernel 均从候选源重编并改变。唯一逻辑差异是：完整 A0 tile 跳过 pre-copy `Duplicate(xLocal, 0, ...)` 和紧随的 `PipeBarrier<PIPE_ALL>()`；尾 tile 保留两者。

## 失败证据

| 用例 | 结果 |
| --- | --- |
| fp16/fp32 score path | 42/44 PASS |
| BF16 | 3/4 PASS |
| invalid | 4/4 PASS |
| `ara_fp32_r4096` | 8/8 错，max_rel 0.99928 |
| `ara_fp32_r5000` | 96/100 错，出现 `2.98e24` 量级异常 |
| `bf16_ara_rowsplit` | 46/400 错，max_rel 0.00490 |

fp16 热点通过不能说明同步安全；多 dtype/不同 R chunk 失败表现为随机或未就绪数据，而不是累加顺序导致的小数值误差。

## 根因

DataCopyPad 对完整 tile 的覆盖判断本身成立，DataCopy 之后的 `PIPE_ALL` 也保留。错误是把上一 chunk Add 后的 `PIPE_V` 当成了 raw `TBuf` 从 Vector 访问切换到下一次 MTE2 覆盖的跨流水依赖。原 pre-copy `PIPE_ALL` 不只等待 Duplicate，也是 TBuf 重用边界。

`ascendc-api-best-practices` 和 `ascendc-precision-debug` 都要求 DataCopy/Vector 间用队列或显式跨流水同步。本地 CANN 8.5 头文件索引未提供能把 `PIPE_V` 外推为 V->MTE2 依赖的保证，实测已直接证伪该假设。

## 五门禁

| 门禁 | 结果 |
| --- | --- |
| 正确性 | **FAIL** |
| 目标 | NOT RUN |
| 全局 | NOT RUN |
| 回归 | NOT RUN |
| 结构 | INCOMPLETE |

回滚后的下一独立假设只可跳过完整 tile 的 Duplicate，pre-copy `PipeBarrier<PIPE_ALL>()` 必须无条件保留。

