# Greater 算子开发与优化经验（Ascend 910B）

## 修订记录

| 日期 | 版本 | 内容 |
| --- | --- | --- |
| 2026-07-20 | 1.0 | 记录 Greater 的广播优化、精度保障、性能验证和制品闭环经验。 |

## 1. 目标与约束

Greater 实现 `x > y` 的逐元素比较，遵循 NumPy/PyTorch 广播规则，输出为
`bool`（每元素 1 字节）。目标平台为 Ascend 910B / CANN 8.5.0，输入 dtype
覆盖 fp16、fp32、bf16、int32、int8，最多支持 5 维广播。

实现和优化时必须同时满足以下约束：

- 浮点输入需正确处理 `NaN`、`+inf`、`-inf`；`NaN > x` 和 `x > NaN` 均为 false。
- int32 必须精确比较，不能通过可能溢出的减法转换为符号判断。
- `Compare` 和 `CompareScalar` 的有效比较长度必须满足 256B 对齐；尾块需要向上补齐后计算，只写回有效元素。
- GM 与 UB 的普通 `DataCopy` 仅用于 32B 对齐地址和长度；其他情况使用 `DataCopyPad`。
- 输出是字节 bool，而 `Compare` 的结果是 packed bit mask，需要经 `Select` 和 `Cast` 展开。

## 2. 基础计算路径

不同 dtype 采用不同的计算路径，避免为了统一实现而牺牲正确性：

| 输入 dtype | Compute dtype | 比较策略 | 关键点 |
| --- | --- | --- |
| fp16 / fp32 | 原 dtype | `Compare(GT)` | 直接产生 bit mask。 |
| int8 | fp16 | `Cast(int8→fp16)` 后比较 | int8 值可被 fp16 精确表示。 |
| bf16 | fp32 | `Cast(bf16→fp32)` 后比较 | 保留已验证的向量转换路径。 |
| int32 | int32 | `Max + EQ + Select` | `gt = (max(x,y)==x) && (x!=y)`，无溢出。 |

公共输出路径是：`Compare`/`CompareScalar` 得到 mask，再以 `Select(mask, one, zero)`
生成 half 的 0/1，最后 `Cast(half→uint8)` 写出 bool。注意 `Select` 的 mask
为真时选择第一个源操作数，源操作数顺序写反会使整个比较结果反向。

实现位置：[op_kernel/greater.cpp](../op_project/custom_greater/op_kernel/greater.cpp)。

## 3. 广播布局：从 shape 推导，而不是硬编码 case

Host tiling 将输出拆成 `outerSize × innerSize`：

- `innerSize` 是最长的末尾非广播连续后缀；普通同形场景可以展平为一个大段。
- 当最内维广播时，广播一侧在每个 outer segment 中是一个标量。
- `outerShape`、`xStride`、`yStride` 记录每个 outer 维的输出尺寸和输入物理步长；广播维的 stride 为 0。

Kernel 的 `ComputeBases(seg)` 用这些 stride 将一个逻辑 segment 映射回 x/y 的真实 GM 偏移。所有广播快路径都应以这套映射为唯一依据，避免只检查某一个维度的 stride。

## 4. 内维标量广播：批量读取 + `CompareScalar`

### 问题

原始标量广播路径会对每一个 segment 执行一次 `LoadScalar`、同步和
`Duplicate`。这会造成大量小 MTE2 搬运和同步开销。只根据 `stride[0]` 是否为
0 或 1 来启用批量路径也不正确：例如 `[B,M,N] × [B,1,1]` 的标量索引依赖
多个 outer 维，`stride[0]` 可能大于 1。

### 通用解法

1. 根据广播一侧完整的 `outerShape` 与 stride 计算可访问最大物理偏移：
   `maxOffset = Σ((shape[d]-1) * stride[d])`。
2. 一次 `DataCopyPad` 将 `[0, maxOffset]` 的连续标量存储范围读入 UB；若加上
   对齐空间后超过 64KB，则安全回退到常规路径。
3. 对每个 output segment 用 `ComputeBases(seg)` 得到实际 scalar index，而不是用
   segment 编号直接索引。
4. fp16、fp32、int8 使用 `CompareScalar`，直接和 UB 中的 scalar 比较，省去
   `Duplicate` 整个子块；当标量是 x 时，将比较改写为 `stream < scalar`，保持
   `x > y` 的语义。
5. bf16 保留 `Duplicate(bf16) + Cast(fp32)`，int32 保留精确比较恒等式，避免引入
   未验证的标量转换或算术路径。

这条路径覆盖全标量、连续标量、`[B,M,N] × [B,M,1]`、
`[B,M,N] × [B,1,1]` 及镜像方向；不满足 UB 容量或对齐条件时仍走安全回退。

## 5. 外维广播：最长零 stride 后缀驻留

### 判定规则

对于 `bcastMode == 0`，某操作数 outer 维的连续零 stride 后缀表示它的
`innerSize` 块可在多个 segment 间复用。快路径选择规则：

1. 从最内 outer 维向外找最长后缀，要求 resident operand 的 stride 全为 0。
2. 要求另一操作数在该后缀内连续，其 stride 必须依次等于
   `innerSize`、`innerSize * shape[last]` 等。
3. 两侧均可驻留时选择复用 segment 更多的一侧；长度相同固定选择 y。
4. resident block 加对齐空间后不超过 96KB，且 `innerSize` 与比较对齐要求兼容。

这将全外维广播和部分外维广播统一处理。例如 `[B,M,N] × [B,1,N]` 可在
每个 B 组内把 y 的 N 元素块驻留一次，流式处理 M 个连续 x segment。

### 切核与同步

- 部分驻留按复用组切核：每组先加载一次 resident block，再按 `TILE` 的
  `innerSize` 整数倍流式比较。
- 若一个核连续处理多个组，下一组的 `DataCopyPad` 会覆盖 resident UB。因此在
  组尾必须插入 `V_MTE2` 的 `SetFlag/WaitFlag`，确保 Vector 已读完旧 resident
  数据，再启动下一次 MTE2 载入。
- 全外维广播只有一个 resident block，不能按“一个组一个核”处理，否则会退化为
  单核。应让每个核各自载入该 block，再按 segment 范围切分输出。

上述同步和全广播特判是本次最重要的稳定性经验：缺少 V→MTE2 同步时，小组可能
正确，但一个核处理多个组时会出现末尾 segment 随机错误。

## 6. UB、对齐与回退策略

- TILE 依据 dtype 的 UB 占用分别设置：int32 4096、bf16 6144、fp32 5120、
  int8 10240、fp16 9216。
- resident buffer 预留 `innerSize + COMP_ALIGN` 元素，避免 `DataCopyPad` 和向上
  补齐的向量访问越界。
- 快路径只在 `innerSize` 满足比较对齐且不超过 TILE 时启用；尾块、非对齐 inner
  size、UB 不足等场景保持 `DataCopyPad` 回退。
- 不要为了“统一”路径移除 bf16/int32 的现有实现；保守回退的性能成本通常远低于
  错误结果或 AIV 异常的风险。

## 7. 性能与制品验证门禁

性能数据只有在正确的自定义制品被加载时才可信。建议固定如下门禁：

1. 从当前源码构建 `.run`，安装至 `ASCEND_OPP_PATH`，并重新构建/安装 pybind。
2. 确认 `libcust_opapi.so` 导出 `aclnnGreater`，并确认 OPP 内存在 Greater 注册和内核。
3. 使用 `msprof --aic-metrics=PipeUtilization` 采样；`op_summary` 中必须包含
   `Greater`，否则停止记录性能结论。
4. 以预热后的 10–30 个 Greater 样本取中位数；不要把注入稳定性的 `aclnnMul`
   计入结果。
5. 最终从交付 zip 内的 `.run` 重装并再次执行精度/制品门禁。

本次参考采样（2026-07-20，五个评测形状）如下：

| Case | 中位时延（µs） |
| --- | ---: |
| c1_small | 2.800 |
| c2_outer_bcast | 75.152 |
| c3_inner_bcast | 73.401 |
| c4_int32 | 173.263 |
| c5_bf16 | 183.973 |
| **prof_sum** | **508.589** |

全部 profiler 输出都含自定义 `Greater`，小于验收门槛 675.254 µs。精度 sweep
覆盖 5 dtype、NaN/±inf、双向广播、5D、尾块和新增定向场景，共 85/85 通过。

## 8. 可复用的检查清单

- [ ] 广播索引是否由全部 outer stride 推导，而非某一维的特判？
- [ ] 广播复用是否按连续零 stride 后缀判定，且另一侧在组内连续？
- [ ] 复用 UB 被下一次 MTE 写入前，是否完成 V→MTE2 同步？
- [ ] 全广播是否仍利用全部可用核？
- [ ] Compare/CompareScalar 的长度、UB 偏移、GM 搬运是否满足对齐要求？
- [ ] bf16/int32 是否保持精确且已验证的路径？
- [ ] profiler 中是否确实出现了自定义 Greater，而不是其他 vendor 制品？
- [ ] 交付 zip 中的 `.run` 是否与源码一致，并可独立重装复验？

## 9. 复现入口

- 精度 sweep：[acc_sweep.py](../acc_sweep.py)
- 制品安装、API/OPP 门禁与可选 profiling：
  [verification/verify_artifact.sh](../verification/verify_artifact.sh)
- 构建和打包：[build_and_pack.sh](../../build_and_pack.sh)
- 交付验证结果摘要：[verification/README.md](../verification/README.md)
