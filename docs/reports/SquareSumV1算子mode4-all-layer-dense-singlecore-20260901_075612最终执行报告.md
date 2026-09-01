# SquareSumV1 mode4 全层 dense 单核最终执行报告

## 结论

候选 `mode4-all-layer-dense-singlecore-20260901_075612` 已通过本地正确性、内存安全、六轮成对性能和 s8 包身份门禁，状态为 `package-verified-local`。最终交付包是 `SquareSumV1-20260901_090646`。当前没有官方隐藏 Case 回执，不能据本地结果宣称已达到官方 `<=1500 us` 目标。

## 实现

- mode 4 保持单核且不调用 `SyncAll()`。
- 每个非末层使用独立、连续的 FP32 workspace stage；下一层只读前一 stage，不原地覆盖。
- 删除逐标量 32B-slot fallback，三层以上也统一使用 dense `DataCopyPad + ReduceSum`。
- 补齐 modes 1/4/5/6/7 raw TBuf 的 MTE2、Vector、Scalar、MTE3 和 `SyncAll` 前后依赖。
- mode 4/5 workspace 的乘法、4 KiB 对齐和 16 MiB reserve 均做溢出检查。
- CMake 从 `compiler/version.info` 写入 vendor 版本，`build.sh` 在打包后强校验。

## 正确性与安全

| 门禁 | 最终结果 |
| --- | --- |
| Host UT | 107/107 PASS |
| 最终 release Real ST | L0 106/106、L1 361/361 PASS；11/101 条超资源上限用例显式 skip |
| 评分/BF16/非法输入 | 44/44、4/4、4/4 PASS；int32 按契约拒绝 |
| rank-8 mode 4 | 三 dtype、三路径、每条 10 次 PASS |
| Case4 | >4 GiB、空规约、fp16/fp32 各 100 轮 wrapper 序列 PASS |
| memcheck | 三 dtype 均捕获真实 `SquareSumV1_*_mix_aiv`，无错误 |
| initcheck/synccheck | BF16 mode 4 / mode 5 均捕获真实 Kernel，无错误 |

早期带 `--kernel-name=square_sum_v1` 的 sanitizer 运行没有匹配动态 Kernel，不能作为证据；最终结论只引用 `raw/sanitizer/unfiltered_*`。

## 性能

42 workload 在物理卡 7 上执行六轮交替顺序 A/B。每个 workload 发射 30 个目标 task，丢弃前 10 个，统计后 20 个 P50。

| 指标 | Parent | Candidate | 变化 |
| --- | ---: | ---: | ---: |
| 42 workload 合计 | 六轮中位基线 | 六轮中位候选 | `-134.801 us` / `+17.4918%` |
| 标准 mode4 fp16 | 67.2495 us | 18.4205 us | `+72.1972%` |
| 标准 mode4 fp32 | 67.154 us | 17.62625 us | `+73.3784%` |
| 三层 mode4 fp16 | 248.7875 us | 47.426 us | `+80.9372%` |
| 三层 mode4 fp32 | 244.1495 us | 46.832 us | `+80.8183%` |
| 三层 mode4 BF16 | 253.858 us | 50.766 us | `+80.0022%` |

六轮均改善，结构检查通过，无 material regression。最终 s8 release 的 42 workload smoke 合计为 `635.1235 us`；三种 dtype Kernel `.o` 与六轮候选逐字节一致。

## 包身份

- Release：`releases/SquareSumV1-20260901_090646/`
- `SquareSumV1-20260901_090646.zip` SHA256：`dfc50b075bd867604559c9ee2e78c632dc9b8938fb4b3ed8ba0f8078ef1e5442`
- `.run` SHA256：`a1caa4208c9fca6911ea464a6651c94401683c35a0aef369c99fe99a83e24f2e`
- 源码集合 SHA256：`b8d705aa30e69aea163e5139687aacf82ed092e269a1c6fb9ff39755dd1a776a`
- 反装版本：`custom_opp_compiler_version=8.5.0`
- zip 只有一个根目录，`.run --list` 包含动态 `square_sum_v1.{cpp,h,py}` 和三 dtype Kernel。

## 边界

历史四个外部包的 Case4 仍为 `Run failed`；本轮包尚未提交。正式条例检视的概要/API 预研和分波计划已完成，但含源码片段的 localhost collector payload 未获用户授权，因此未生成最终条例检视报告。
