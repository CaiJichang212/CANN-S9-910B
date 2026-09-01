# SquareSumV1 mode4-dense-output-v-mte3-sync 阶段执行报告

## 结论

候选通过动态正确性、内存安全和性能门禁，但独立源码审计发现非规约零维可达 `CeilDiv(0,0)`，因此该包拒绝；其 Kernel 性能证据可由最终 Host-only 修复包按二进制哈希继承。

## 正确性与安全

- `.run` SHA256：`0e06264911dcbe4944df0d18bcd036db25ccce45e8b1b9f666cfc544e1f9f86d`。
- Host UT 105/105；Mock ST 全通过；受控 Real ST L0 106/106、L1 361/361。
- 44+4+4、rank-8 三 dtype、Case4 100 轮通过。
- fp16/fp32/bf16 mssanitizer 均捕获对应 Kernel，未报告内存错误。

## 性能

- 6 轮 42 workload 全部整体变快。
- paired median 合计：`-131.97175 us`，提升 `17.66334%`。
- 标准 mode 4：fp16 提升 `74.88190%`，fp32 提升 `75.21362%`。
- 负轴 mode 4：fp16/fp32 分别提升约 `19.48%/20.10%`。
- 无 material regression；mode 6 专项合计 `51.330 -> 50.514 us`。

## 拒绝与继承边界

非连续多轴且 0 维仅位于非规约轴时，Host 会构造 `a0Length=tileA0Len=0` 的 layer。最终候选新增零 work item 路由；三 dtype Kernel `.o` 与本候选逐字节一致，因此仅性能与 memcheck Kernel 证据可继承，Host 正确性必须在最终包重新验证。
