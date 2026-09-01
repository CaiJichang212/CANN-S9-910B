# AscendC算子性能优化工作流与Greater实战经验

> 整理日期：2026-08-31
>
> 适用平台：以Ascend 910B4-1 / DAV_2201、CANN 8.5.0为实证环境
>
> 资料范围：Greater工作树、2026-08-30至2026-08-31 Codex会话历史、原始profiling结果、候选台账、正式包和官方反馈
>
> 最新官方状态：`Greater_20260831_104337.zip`为5/5 Pass，`prof_sum=784.33us`；较990.62us基线改善20.8243%，但尚未达到`<=500us`

## 1. 结论先行

Greater这轮优化最值得复用的不是某个固定`TILE`或64MiB阈值，而是一套可以审计、回滚和继续迭代的证据闭环：

1. 先冻结官方基线、源码、二进制、环境和测量口径，再建立正确性安全父版本。
2. 把执行路径按内存访问和复用模式分类，优化真实成本，而不是只盯算术指令。
3. 每个候选只验证一个主要假设，依次经过编译、定向正确性、相邻A/B、全矩阵和结构门禁。
4. 加速某些目标但造成其他合法输入明显回退的候选必须拒绝，不能用总和改善掩盖局部退化。
5. Host与Kernel必须对快速路径的谓词、核数、对齐和UB预算保持镜像一致。
6. 最终提交物必须从最后源码在指定环境重编，包内源码、`.run`、安装后二进制和上板证据通过哈希闭环。
7. 本地代理矩阵只用于筛选方向，是否达到赛题目标只认同一正式包的官方隐藏评测。

最新官方结果如下。正数“改善”表示时延下降，负数表示回退。

| 官方case | 历史基线(us) | 最终包(us) | 改善(us) | 改善率 | 最终占比 |
|---|---:|---:|---:|---:|---:|
| Case1 | 2.8600 | 2.9400 | -0.0800 | -2.80% | 0.37% |
| Case2 | 657.7830 | 472.5300 | 185.2530 | 28.16% | 60.25% |
| Case3 | 62.1215 | 46.2400 | 15.8815 | 25.57% | 5.90% |
| Case4 | 73.5920 | 71.0200 | 2.5720 | 3.49% | 9.05% |
| Case5 | 194.2635 | 191.6000 | 2.6635 | 1.37% | 24.43% |
| **总计** | **990.6200** | **784.3300** | **206.2900** | **20.8243%** | **100%** |

最终成绩距离500us仍有284.33us。Case2和Case5合计占最终时延84.67%，只能用于确定官方反馈下的优化优先级，不能据此反推隐藏shape或写shape特判。

## 2. 会话历史检索与证据边界

### 2.1 检索方法

会话历史位于：

- `/home/liyc/.codex/sessions/2026/08/30`
- `/home/liyc/.codex/sessions/2026/08/31`

JSONL中大量子会话会继承父会话上下文，因此仅按“Greater”文本搜索会把权限审查、上下文副本和当前总结任务误判为独立技术结论。本次采用以下识别顺序：

1. 解析`session_meta.cwd`，只保留工作目录精确等于Greater工作树的记录。
2. 结合`session_id`、`forked_from_id`和`thread_source`区分用户主会话、子任务和guardian权限审查。
3. 从结构化的用户消息、助手进度、工具调用和工具输出还原时间线，不从文件名猜测内容。
4. 主优化会话只取原优化任务结束前的记录，避免后续“总结会话历史”请求反向污染历史结论。
5. 所有会话结论再与当前源码、原始CSV、manifest、候选台账、zip哈希和官方反馈交叉校验。

两天目录中筛得`cwd`精确匹配的60份JSONL，其中37份metadata标记为subagent（含项目整理和优化子会话），18份标记为guardian审查。guardian记录只证明权限动作经过审查，不作为算子技术结论来源。

### 2.2 关键会话

| 用途 | 会话文件 | 作用 |
|---|---|---|
| 项目规则整理 | `2026/08/30/rollout-2026-08-30T23-21-23-01a05342-d45f-7943-a081-f3b2b92c9f3e.jsonl` | 更新工程指南、基线和证据边界 |
| 项目现状梳理 | `2026/08/30/rollout-2026-08-30T23-26-33-01a05347-8c46-7bf0-b208-05dd70cc5f5a.jsonl` | 建立源码、测试和历史材料索引 |
| 性能优化主线 | `2026/08/30/rollout-2026-08-30T23-50-08-01a0535d-26c9-7383-bca5-a76c5b6f23ff.jsonl` | 从基线冻结到最终候选、代码检视和首版正式包的完整过程 |
| 正式打包自动化 | `2026/08/31/rollout-2026-08-31T10-33-17-01a055a9-f684-7be2-a0a0-a0bfab7e91e4.jsonl` | 改造`s8`一键重编、官方打包、时间戳命名和回验 |

主优化会话派发了34个有边界的技术子任务，覆盖源码模型、证据审计、环境审计、Host契约、数值安全、DMA对齐、UB预算、流水同步、Tiling、接口注册、性能循环和最终问题分诊。这些子任务的结论由主控制者结合实测统一收敛，而不是直接拼接进最终实现。

### 2.3 证据优先级

本项目出现过“报告写着官方待提交，但原始反馈已经更新”的情况。正确的证据优先级应是：

```text
同一正式包的官方原始反馈
  > 当前源码/二进制/安装树哈希与原始测试数据
  > 候选台账、stage plan和最终检视记录
  > 会话中的过程性总结
  > 静态方案、历史报告和文件名暗示
```

因此，本文件以`docs/result-20260720-2.txt`中指向`Greater_20260831_104337.zip`的784.33us为最新官方事实；早先报告中的“官方待提交”只代表报告生成时状态。

## 3. 先建立可解释的性能模型

### 3.1 算子语义与路径分类

Greater实现`x > y`，支持5种输入dtype、最多8维的NumPy风格广播和bool输出。性能模型不能只写成“读两个输入、做一次Compare、写一个输出”，因为广播会改变地址生成、GM访问次数和数据复用关系。

当前实现可抽象为三类执行路径：

| 路径 | 数据特征 | 主要成本 | 主要优化手段 |
|---|---|---|---|
| Generic | 任意合法stride，逐segment计算两个输入base | 地址分解、碎片DMA、逐tile API | 稳健兜底、合理切核、对齐搬运 |
| P1 resident | 一个操作数的连续行或复用组可驻留UB | 流式输入搬运、resident row重复展开 | 驻留复用、整批Vector Copy |
| P2 scalar batch | 最内维为scalar广播，另一侧线性连续 | scalar重复读取、逐row启动 | scalar批量搬运、`Brcb+Copy`、分块加载 |

一个更实用的近似成本模型是：

```text
T_total ~= T_launch(blockDim)
         + segment_count * T_address_decode
         + dma_count * T_dma_startup
         + GM_bytes / GM_bandwidth
         + vector_work / vector_throughput
         + T_sync
```

对于百万outer、极短inner的广播，`segment_count`、`T_address_decode`和小DMA/API启动远大于一次Compare。此时继续微调Compare指令几乎没有意义，必须先合并行、复用GM数据和减少调用次数。

### 3.2 dtype路径必须保持语义

| dtype | 计算路径 | 需要守住的语义 |
|---|---|---|
| fp16/fp32 | 直接`Compare(GT)` | `NaN`比较为false，Inf遵循IEEE语义 |
| bf16 | bf16精确Cast到fp32后GT | Cast和标量物化同步正确 |
| int8 | int8精确Cast到fp16后GT | int8全范围在fp16中可精确表示 |
| int32 | `Max + EQ + Select` | 避免用减法判断大小导致溢出 |

int32使用的恒等式为：

```text
x > y  <=>  (max(x, y) == x) && (x != y)
```

该表达式不依赖`x-y`，因此覆盖`INT32_MIN`和`INT32_MAX`时仍然精确。性能优化不能破坏这种dtype特定实现，也不能“按直觉”改动已由实测确认的`Select` source/bit语义。

## 4. 可复用的端到端优化工作流

### 4.1 总体状态机

```text
冻结官方基线和交付身份
        |
        v
建立正确性安全父版本 ----失败----> 修复语义/内存问题
        |
        v
构建分层测试矩阵和严格采集器
        |
        v
提出单一性能假设 -> stage plan -> 最小补丁
        |
        v
编译 + 定向正确性 + 安全边界
        |
        v
screening -> 相邻A/B或ABBA -> 固定全矩阵
        |                         |
      回退-----------------------+
        |
        v
正式代码检视 -> 指定环境重编 -> 包内安装回验
        |
        v
官方反馈 -> 保留、拒绝或进入下一轮通用优化
```

### 4.2 各阶段的输入、门禁和产物

| 阶段 | 必做动作 | 通过门槛 | 固化产物 |
|---|---|---|---|
| 0. 身份冻结 | 记录branch、HEAD、dirty diff、Host/Tiling/Kernel/`.run`哈希 | 能唯一回答“测的是哪份代码” | baseline manifest |
| 1. 环境冻结 | 核验SoC、CANN、容器、逻辑设备、卡空闲状态、AIV/AIC/UB | 环境与目标平台一致，无共享状态污染 | environment manifest |
| 2. 语义建模 | 列出dtype、广播、rank、tail、空Tensor和异常值约束 | Host和Kernel契约完整 | spec/路径表 |
| 3. 安全父版本 | 先跑穿刺和全dtype正确性，修复已有错误 | 正确性门禁全过，父版本可复现 | frozen parent `.run` |
| 4. 严格测量 | 固定Op Name、样本数、warmup、统计量和输出目录 | 采集失败即停止，不接受脏数据 | raw profile + summary |
| 5. 假设设计 | 用成本模型解释瓶颈，限定一个主要变量 | 有目标集、控制集、回滚条件 | stage plan |
| 6. 最小实现 | Host/Kernel同步改动，控制补丁范围 | 编译成功，静态资源约束通过 | candidate artifact |
| 7. 局部门禁 | 定向精度、安全边界、screening | 目标改善且控制集无明显回退 | screening report |
| 8. 成对A/B | 同设备相邻采集，必要时ABBA控制漂移 | 重复方向一致，达到预设阈值 | paired A/B report |
| 9. 全矩阵 | 固定spec集合逐项比较 | 全部精确PASS，0 material回退 | full matrix report |
| 10. 代码检视 | 检查契约、越界、数值、同步、资源和ABI | 真阻断项闭环，误报有证据豁免 | review resolution |
| 11. 正式交付 | 指定镜像重编、官方脚本打包、包内安装上板 | 源码/二进制/安装树/测试同源 | timestamp zip + hashes |
| 12. 官方迭代 | 比较逐case反馈，不猜隐藏shape | 正确性全过，性能变化可解释 | official feedback ledger |

### 4.3 六类硬门禁

1. **身份门禁**：源码、Tiling、Kernel、`.run`、pybind、安装后的OPP对象必须能追溯。
2. **正确性门禁**：任何dtype、广播或边界错误都优先于性能；错误候选不能成为性能父版本。
3. **安全门禁**：UB容量、GM边界、整数溢出、DataCopy合法性和流水同步必须通过静态与动态验证。
4. **性能门禁**：目标集要有收益，控制集不能出现material回退，重复A/B方向要稳定。
5. **泛化门禁**：路径选择只能依赖dtype、shape结构、stride、对齐、资源和平台能力，不能依赖已知case编号或猜测shape。
6. **交付门禁**：最后源码必须在要求的工具链重编；历史`.run`或历史zip不能替代正式产物。

## 5. Greater候选如何收敛

### 5.1 先修正确性再优化

性能工作开始时，合法混合广播`x=[2,1,256]`、`y=[1,3,1]`触发旧P2错误：非scalar的stream operand在outer维仍需广播，但Kernel把它当作全局线性连续输入，fp32实测出现852个bool mismatch。

修复原则不是为这个shape特判，而是定义通用谓词：只有stream operand的outer stride与输出稠密stride一致时才进入P2，否则回退Generic的`ComputeBases`。修复后mixed 40/40和sweep 85/85通过，形成`safe_b20`安全父版本。

这说明快速路径的资格必须同时满足：

```text
fast_path_eligible = semantic_compatible
                   && address_mapping_valid
                   && vector_api_legal
                   && ub_budget_safe
                   && enough_useful_work
```

### 5.2 候选决策账

| 候选 | 假设 | 结果 | 决策与经验 |
|---|---|---|---|
| `safe_b20` | P2仅接收outer线性连续的stream operand | mixed 40/40、sweep 85/85 | 接受为所有性能实验的安全父版本 |
| `p_aiv_host` | 全路径固定20核改为全部AIV | 目标快13.1%，低工作量广播回退32%-48% | 拒绝；核多不等于快 |
| `p_aiv_tile_grain` | 每核至少分到一个dtype TILE | 修复空核，但`f16_same_med`回退8.3%和2.74us | 拒绝；粗粒度仍未区分执行路径 |
| `p_bcast_aiv_tile` | 广播路由使用AIV/TILE核数 | 总和改善11.65%，full79有5个fallback material回退 | 拒绝；总收益不能掩盖局部退化 |
| `p_route_aware_cores` | Host镜像P1/P2资格，只给真实快路径扩核 | full79改善12.50%，0 material回退 | 接受；核数必须按有效工作单元分配 |
| `p_flat_large_aiv` | 大连续同形IO达到阈值才用全部AIV | 阈值上侧两轮改善5.50%和5.23%，控制稳定 | 接受；64MiB是本平台实测阈值，不是通用常数 |
| `p_all_row_padded` | 移除短行padding不超过2倍的旧门限 | P1有效，P2触发MPU非法访问 | 拒绝并保留崩溃证据；性能路径先过资源门禁 |
| `p_all_row_padded_ub_safe` | 按184KiB用户UB重做P2预算 | fp16短行P1达14.47x/16.43x，P2达8.76x/8.77x | 接受；解除伪门限必须同时修正资源模型 |
| `p_p1_row_vector_copy` | resident row用`Copy(srcRepeatStride=0)`整批展开 | fp16 P1两轮1.74x | 接受；减少逐row向量API启动 |
| `p_p2_row_brcb` | scalar用`Brcb+Copy`整批展开 | fp16 P2两轮1.67x | 接受；32B对齐和scratch计入UB |
| `p_fp32_p1_row_vector_copy` | 将已验证P1方法推广到fp32 | fp32 P1 1.59x-1.60x | 接受；先在一种dtype验证再受控推广 |
| `p_fp32_p2_row_brcb` | fp32安全batch使用Brcb | N=7/31/255改善1.19x-1.35x | 接受；只覆盖整批能放入UB的情况 |
| `p_p2_blocked_scalar` | 整核scalar超预算时按tile分块加载 | fp32大短行P2改善10.9x-11.5x | 接受；分块复用优于退回逐rowGeneric |
| `final_candidate` | 补齐Host契约和u32整数安全 | Host 5/5、UB 5/5、mixed 40/40、sweep 85/85、full94 94/94 | 接受并进入正式打包 |

### 5.3 本地与官方结果必须分开解释

最终候选相对安全父版本的共同79项本地P50总和为：

```text
3864.307us -> 3200.351us
改善663.956us / 17.18%
79/79精确PASS，0个material回退
```

这组数据证明固定公开矩阵上的泛化收益，不是官方5个隐藏case的总分。正式包官方结果为784.33us，说明优化方向确实覆盖了部分隐藏瓶颈，但没有达到500us目标。两种数据都重要，作用不同，不能互相替代。

## 6. 核心技术经验

### 6.1 广播优化的首要目标是减少重复工作

广播算子的主要浪费通常来自：

- 同一resident row或scalar被反复从GM读取；
- 每个短行重复做多维stride分解；
- 大量极小`DataCopyPad`和Vector API启动；
- 每行单独同步和写回。

通用优化顺序应为：

1. 用stride识别连续段、复用组和scalar序列。
2. 将可复用操作数驻留UB或一次批量加载。
3. 把多个逻辑短行映射到固定对齐的UB槽。
4. 用Vector Copy或Broadcast类API成批展开。
5. 合并Compare/Select/Cast和写回。
6. 最后再调核数和TILE。

如果先扩核而不减少重复工作，只会让更多核同时支付初始化、地址计算和小DMA开销。

### 6.2 非对齐应在UB内补齐，GM保持逻辑长度

Greater的Vector Compare要求256元素粒度，而输入维度可能不对齐。可靠做法是：

```text
GM读取：只读真实innerSize，使用DataCopyPad处理尾部
UB布局：rowElems = RoundUp(innerSize, 256)
向量计算：按rowElems执行，保证每行起点合法对齐
GM写回：只写真实innerSize个bool
```

旧实现用`padding <= 2 * logical`限制是否批行，但Generic路径本来就会把每个短行计算上取整到256元素。对`innerSize=1/7/31`而言，拒绝批处理并不会省掉向量padding，反而保留百万次地址分解和小搬运。因此门限应来自“能否放入TILE、API是否合法、UB是否安全”，而不是padding比例的直觉。

### 6.3 核数按有效任务数分配

Greater在910B实测形成的路径感知策略为：

```text
Generic:
  min(平台实测较优上限, ceil(totalSize / 256))

P1 full resident:
  min(AIV数量, outerSize, ceil(totalSize / dtypeTile))

P1 partial resident:
  min(AIV数量, outerSize / residentGroupSegs,
      ceil(totalSize / dtypeTile))

P2 scalar batch:
  min(AIV数量, outerSize, ceil(totalSize / dtypeTile))
```

可复用原则是：

- 核数上限不是唯一变量，真实可并行的group、segment或tile数同样重要。
- 每个启动核都应拿到足够工作，能摊薄常量Buffer初始化和流水启动。
- 不同路径的单位工作量不同，不能共用一个“元素数除固定粒度”的公式。
- 连续大张量可用更多核，但阈值必须在目标SoC、工具链和算子上重新A/B标定。

### 6.4 UB预算按用户可用容量做完整生命周期建模

DAV_2201物理UB为192KiB，但当前API运行时保留末端8KiB临时区，算子应按184KiB用户区规划。P2第一次崩溃的根因是只看约50.5KiB scalar batch，没有把149,376B固定Buffer一起计入。

通用预算可以采用下面两个等价口径，二选一，不能重复扣减保留区：

```text
UB_operator <= UB_user = UB_physical - UB_api_reserved

等价于

UB_operator + UB_api_reserved <= UB_physical
```

其中算子自身预算应覆盖：

```text
UB_operator = input_queues * queue_depth * aligned_input_tile_bytes
            + output_queues * queue_depth * aligned_output_tile_bytes
            + dtype_compute_buffers
            + resident_or_scalar_batch_buffer
            + mask/select/cast_buffers
            + broadcast_scratch
```

可靠做法包括：

- 删除不可达或未使用的Buffer；
- 按dtype、广播方向和实际路径条件初始化Buffer；
- batch设置显式字节上限，而不是只限制元素数；
- 用`static_assert`守住编译期最坏预算；
- 增加“刚好进入、刚好回退、历史崩溃shape”的5 dtype可达回归；
- 每次新增scratch、双缓冲或dtype路径后重新计算总量。

### 6.5 Vector展开既要看吞吐，也要看对齐、repeat和同步

P1使用`Copy(srcRepeatStride=0)`把一条resident row复制为多行；P2使用`Brcb+Copy`把连续scalar扩展为多行。实现时必须同时核对：

- 源和目标32B对齐；
- mask、repeatTimes、block stride和repeat stride的单位；
- repeat上限及分块策略；
- 临时Buffer是否计入UB；
- 修改mask后是否恢复；
- V计算结束后，MTE2覆盖同一Buffer前是否有`V_MTE2`同步；
- MTE2搬运结束后，Vector读取前是否有`MTE2_V`同步；
- 分块边界和最后不足一块的逻辑写回长度。

API调用能编译不代表依赖关系正确。同步应按“谁生产、谁消费、是否复用同一片UB”建立数据依赖图，再选择事件或Barrier。

### 6.6 Host与Kernel谓词必须镜像

Host决定`blockDim`和Tiling，Kernel决定实际是否走P1/P2。若两侧资格不一致，会出现两类问题：

- Host按快路径扩到40核，Kernel却回退Generic，低工作量输入显著退化；
- Host认为batch安全，Kernel按另一种allocCount实际分配，导致UB越界。

每个快速路径应维护一份可审计的条件清单，至少覆盖：

```text
dtype能力
innerSize与TILE
向量对齐或row-padded资格
resident/scalar/stream stride连续性
复用组大小
UB总预算
有效工作单元数
平台AIV上限
```

新增或修改条件时应同时检查Host和Kernel，并用边界shape验证路径两侧得到相同结论。

### 6.7 Host契约是性能实现的安全边界

最终代码检视发现的真实阻断项包括空指针、rank、负维、非法广播、dtype不一致、shape乘法溢出、TilingData的`uint32_t`表示范围和raw tiling容量。修复这些问题不是“性能无关清理”，因为错误的shape或溢出值会直接污染核数、GM offset和UB大小。

通用Host检查顺序应为：

1. context、输入shape/desc、输出、platform和raw tiling判空；
2. rank上下限与维度非负；
3. dtype支持且输入dtype一致；
4. 广播兼容；
5. shape乘法和stride乘法使用checked arithmetic；
6. Tiling字段的可表示性；
7. `SetBlockDim`、输出dtype等API返回值；
8. 空Tensor保持短路，不进入非法搬运。

Kernel中的ceil-div和RoundUp也应在第一次加法前提升到`uint64_t`，避免`uint32_t`先回绕再转换。

## 7. 严格profiling与A/B方法

### 7.1 为什么要自建严格采集器

共享custom OPP、历史`.run`、错误pybind、设备映射和多算子CSV都可能让“看似成功”的profiling测到错误对象。Greater的严格采集器执行以下检查：

- `--out`必须是新目录，禁止覆盖历史证据；
- 安装指定`.run`失败立即停止；
- 只接受精确`Op Name == Greater`；
- 每个spec必须恰好1050条任务；
- 丢弃前150条真实warmup，对后900条取P50；
- 同时记录P95、均值、CV、最小/最大值和pipe ratio；
- 拒绝多份CSV、多个BlockDim、非PASS精度和缺字段；
- 采集前后校验源码、run、已安装OpAPI/Tiling/opmaster和Kernel object哈希；
- 保存设备前后快照，确认没有争用或shared OPP被覆盖。

这些规则比“多跑几次取最小值”更可靠。最小值会系统性偏向偶然噪声，不能代表稳定收益。

### 7.2 相邻A/B与ABBA

候选比较应尽量使用同一设备上的相邻采集：

```text
A(parent) -> B(candidate)
B(candidate) -> A(parent)
```

若两轮方向不一致或存在明显时间漂移，升级为ABBA：

```text
A -> B -> B -> A
```

Greater采用的material回退判据是同时满足：

```text
relative_regression >= 5%
absolute_regression >= 0.5us
```

该双阈值避免把亚微秒抖动放大为严重问题，也避免大case的显著绝对回退被百分比稀释。阈值应在新项目开始时冻结，不能看到结果后再修改。

### 7.3 目标集、控制集和全矩阵缺一不可

- 目标集验证假设是否真的命中预期路径。
- 控制集验证未命中的dtype、对齐、方向和路径没有被误伤。
- 全矩阵防止定向测试漏掉长尾回退。

`p_bcast_aiv_tile`就是典型反例：定向和总和都好看，但full79发现5个fallback material回退，因此被拒绝。优化验收应看逐case分布，而不只看总和。

### 7.4 每个采集必须带身份清单

建议manifest至少包含：

| 类别 | 字段 |
|---|---|
| Git | branch、HEAD、dirty状态、dirty diff哈希 |
| 源码 | Host、TilingData、Kernel、caller、helper哈希 |
| 二进制 | `.run`、pybind、安装后OpAPI和Kernel tree哈希 |
| 环境 | SoC、CANN、容器镜像、Python、逻辑/物理设备映射 |
| 测试 | spec顺序及哈希、样本数、warmup数、统计口径 |
| 运行 | 开始/结束时间、退出码、最后spec、设备前后快照 |

如果无法用这些字段重建“哪份代码在什么环境跑了哪些用例”，该数据只能作为线索，不能成为接受候选的证据。

## 8. 测试矩阵如何覆盖泛化能力

94个spec不是目标本身，关键是覆盖执行路径和边界组合。新算子可按下表设计矩阵：

| 维度 | 必测因素 |
|---|---|
| dtype | 所有声明dtype及其独立计算路径 |
| rank/shape | 标量、1D到最大rank、退化维、多个外维 |
| 广播 | 同形、x广播、y广播、最内维广播、外维广播、混合广播 |
| 对齐 | 32B对齐、256元素对齐、N=1/7/31/32/33/255/256/257等边界 |
| 路径 | Generic、P1 full、P1 partial、P2 safe batch、P2 blocked、回退 |
| 资源 | UB刚好可进入、刚好超限、核数阈值两侧、TILE边界两侧 |
| 数值 | 随机范围、NaN、+Inf、-Inf、int8全范围、int32极值 |
| 规模 | 极小启动开销、中等、超大连续、大outer短inner、大inner |
| Host契约 | 空Tensor、非法广播、负维、超rank、溢出、dtype不一致 |

推荐分层执行：

| 层级 | 内容 | 何时运行 |
|---|---|---|
| L0 | 编译、Host契约、历史崩溃和目标shape | 每次补丁后 |
| L1 | 目标dtype/路径定向精度和screening | 候选初筛 |
| L2 | mixed广播、全dtype sweep、UB边界 | 进入A/B前 |
| L3 | 固定完整profiling矩阵 | 接受候选前 |
| L4 | 指定提交环境重编后的代表集和全精度 | 正式打包后 |

不同spec集合的P50总和不可直接比较。Greater最终94项比旧79项多了15个压力项，因此只能比较共同79项，不能把94项总和与79项总和相减。

## 9. 多智能体协作的有效组织方式

### 9.1 角色分工

Greater主会话的并行任务可归纳为四组：

| 角色 | 典型任务 | 交付物 |
|---|---|---|
| 事实审计 | 源码模型、历史证据、环境与设备 | 带路径和原始数据的事实清单 |
| 假设分析 | 大同形扩核、短行向量化、广播缺口 | 可证伪假设、目标/控制spec、风险 |
| 专项检视 | Host契约、整数、DMA、UB、同步、API、ABI | 按严重度排序的问题和条例证据 |
| 收敛控制 | 候选台账、A/B、回滚、最终打包 | 单一父版本、决策记录、正式产物 |

早期并行审计能缩短“读代码、查证据、核环境”的串行等待；后期专项检视能把大文件按风险面拆开。主控制者仍应独占以下职责：

- 决定当前唯一父版本；
- 修改共享Host/Kernel源码；
- 安装shared custom OPP；
- 运行正式A/B并接受或回滚候选；
- 合并互相矛盾的审计结论；
- 生成最终包和结论。

### 9.2 子任务要可验证、可合并

推荐给子任务使用固定输出格式：

```text
scope: 本任务读取的文件和数据
facts: 可直接从源码/数据确认的事实
hypothesis: 尚需实验验证的解释
risks: 可能破坏的路径或资源
proposed_gate: 最小目标集、控制集和拒绝条件
evidence: 文件路径、行号、CSV/manifest字段
```

一个子任务只处理一个风险面，默认只读。多人同时修改共享源码、安装OPP或运行同一张卡，会让候选身份和数据失去可信度。

### 9.3 会话历史本身也需要去重

子会话继承完整父上下文，简单全文检索会把同一个用户任务重复几十次。复盘多智能体任务时应先按metadata构建父子图，再提取各子任务新增的消息和工具证据；guardian权限会话应单独分类，不能计作技术审计代理。

## 10. 失败模式与对应修正

| 失败模式 | 表面现象 | 根因 | 可复用修正 |
|---|---|---|---|
| 全路径直接扩到40核 | 大case变快，小广播慢32%-48% | 每核工作不足，初始化和启动开销放大 | 按路径和有效任务数限核 |
| 只看总和 | 总和改善11.65%，仍有5个明显回退 | 目标收益掩盖长尾合法输入 | 固定逐case material门槛 |
| 用padding比例拒绝批处理 | 短行仍走逐row Generic | 忽略Generic本来就向量补齐 | 以API合法性、TILE和UB作为门槛 |
| 按192KiB规划UB | 某P2合法shape发生MPU异常 | 末端8KiB为API临时区 | 按184KiB用户区做全量预算 |
| 只计算新增Buffer | 单独batch看似能放下，组合后越界 | 忽略队列、compute和scratch生命周期叠加 | 建立路径/dtype总UB公式和static_assert |
| 广播只判断最内维 | 混合outer广播出现错结果 | stream operand并非全局线性连续 | 校验所有outer stride，不满足即回退 |
| Host/Kernel资格不一致 | Host扩核但Kernel回退，或资源模型不同 | 两侧条件漂移 | 维护镜像谓词和边界路径测试 |
| 只取单次最小值 | 候选收益不可重复 | 预热、漂移、设备争用 | 固定warmup、P50、相邻A/B/ABBA |
| 测到错误kernel | 数据正常但改动无影响或异常 | shared OPP被覆盖、pybind解析错误库 | 每轮重装并核对Op Name和安装树哈希 |
| 混用物理/逻辑设备号 | set_device失败或采错卡 | 容器映射与宿主编号不同 | 同时记录`npu-smi`映射和框架逻辑编号 |
| 丢失原PYTHONPATH | 缺少CANN `tbe`模块 | 隔离扩展时覆盖环境路径 | 前置工作目录但保留原变量 |
| 从文件名判断新旧 | 使用了过期报告或历史zip | 文件名与内容状态不同步 | 读原始内容、时间和哈希 |
| 比较不同矩阵总和 | 得出虚假整体加速/回退 | spec集合不同 | 只比较交集或逐spec对齐 |
| 从隐藏case时延猜shape | 局部上榜但失去泛化资格 | 用不可验证假设硬编码 | 只做由公开规格和结构决定的通用优化 |

## 11. 建议固化的工程产物

### 11.1 目录结构

```text
perf_opt/YYYYMMDD/
├── README.md                 # 环境、入口、证据边界
├── candidate_ledger.csv      # 所有接受/拒绝候选
├── stage_plan_<id>.yaml      # 单候选假设和门禁
├── collect_strict.sh         # 不覆盖历史的采集入口
├── parse_strict.py           # 严格解析和统计
├── compare_ab.py             # 逐spec成对比较
├── artifacts/
│   └── <candidate>/
│       ├── custom_opp_*.run
│       └── source_manifest.txt
└── results/
    └── <unique_run_id>/
        ├── run_manifest.txt
        ├── spec_order.txt
        ├── summary.csv
        ├── metadata/
        └── specs/
```

### 11.2 stage plan模板

```yaml
candidate_id: p_xxx
parent_id: frozen_parent
hypothesis: 一个可证伪的性能解释
patch_scope:
  - op_host/foo.cpp
target_specs:
  - target_a
control_specs:
  - control_a
correctness_gates:
  - all dtype exact/within tolerance
performance_thresholds:
  target_min_gain: 0.05
  material_relative: 0.05
  material_absolute_us: 0.5
reject_when:
  - any correctness failure
  - any safety failure
  - repeated material regression
rollback: restore frozen parent artifact
requires_official_feedback: true
```

### 11.3 candidate ledger最小字段

```text
candidate_id,parent_id,kind,hypothesis,patch_scope,
correctness,screening,paired_ab,structure,decision,reason
```

拒绝项也必须记录。没有失败台账，后续人员很容易重新尝试“全开40核”“只看padding比例”等已经证伪的方向。

## 12. 新算子的快速诊断框架

拿到一个已有AscendC算子时，可以先按现象选择调查方向：

| 现象 | 优先怀疑 | 首个实验 |
|---|---|---|
| 极小shape占时固定 | launch/API/初始化开销 | 减核、合并API、检查空核 |
| 大连续shape随字节数增长 | GM/MTE带宽或切核不足 | 查询pipe ratio，受控增加有效核数 |
| 短inner、大outer异常慢 | 地址分解和小DMA启动 | 行批处理、resident/scalar复用 |
| 广播比同形慢很多 | 重复GM读取、stride计算 | 按stride识别复用组和连续段 |
| 非对齐远慢于对齐 | 每行tail处理碎片化 | GM逻辑搬运、UB固定槽补齐 |
| 某dtype独有瓶颈 | Cast/替代算法/额外Buffer | 分dtype拆解流水和UB |
| 扩核后反而慢 | 每核工作不足或路径回退 | 记录BlockDim和实际路径，按有效单元限核 |
| 偶发崩溃或随机错 | UB越界、GM越界、同步 | 先停止性能实验，做边界回归和资源审计 |
| 本地快但官方无变化 | 代理矩阵未覆盖隐藏瓶颈或包身份错误 | 先核包同源，再扩充公开规格的结构压力项 |

推荐的最小迭代单位是“一份冻结父`.run`、一个假设、一个补丁、一组目标/控制用例、两轮相邻A/B、一次接受或拒绝决定”。

## 13. Greater当前边界与下一轮原则

最终实现已解决混合广播错误、非对齐短行退化、fp32超预算P2、UB规划、路径感知核数和Host契约；正式包官方成绩从990.62us降到784.33us。仍需明确以下边界：

1. `innerSize > dtype TILE`的大inner resident广播仍回退Generic，是公开实现结构中可继续验证的通用方向。
2. rank上限8、TilingData主要字段为`uint32_t`、184KiB用户UB和当前dtype TILE都是显式实现边界。
3. 64MiB同形扩核阈值和20/40核策略来自910B实测，换SoC或工具链必须重做A/B。
4. 官方Case2仍占60.25%，Case5占24.43%；只能据此排序反馈优先级，不能断言它们对应某个本地spec或某种shape。
5. 下一轮候选仍应从公开规格、源码路径和profiling结构出发，经过全矩阵后再提交。官方逐case变化用于判断方向是否覆盖隐藏工作负载，而不是用于构造隐藏case特判。

一个合理的下一轮闭环是：

```text
冻结784.33us正式包为官方父版本
  -> 重新确认当前工作树和父包同源
  -> 对大inner、dtype专有路径和剩余逐segment开销做公开规格probe
  -> 每次只实施一个通用候选
  -> 94项或扩展固定矩阵0 material回退
  -> s8重编和包内回验
  -> 获取新的官方5-case向量并更新台账
```

## 14. 提交前检查清单

- [ ] 官方父基线、逐case数据和对应zip已冻结。
- [ ] branch、HEAD、dirty diff和源码哈希已记录。
- [ ] SoC、CANN、容器、逻辑设备和卡空闲状态已核验。
- [ ] Host/Kernel广播谓词、核数和资源模型一致。
- [ ] 所有dtype、正反向广播、tail、NaN/Inf和整数极值通过。
- [ ] UB预算包含队列、双缓冲、compute、mask和scratch，并通过用户容量或等价物理容量口径计入API保留区。
- [ ] 目标集有稳定收益，控制集和全矩阵无material回退。
- [ ] A/B数据来自同设备相邻采集，样本和统计口径一致。
- [ ] 采集的Op Name、BlockDim、`.run`和安装树身份正确。
- [ ] 正式代码检视的真阻断项已闭环，豁免项有实验证据。
- [ ] 最终`.run`由最后源码在指定CANN环境重建。
- [ ] zip仅含规定清单，包内源码和`.run`哈希一致。
- [ ] 包内`.run`重新安装并完成精度及必要profiling。
- [ ] 本地结果与官方结果分栏记录，没有把代理总和称为官方成绩。
- [ ] 没有case编号、已知shape或猜测隐藏shape的硬编码。

## 15. 证据索引

| 证据 | 路径 |
|---|---|
| 最新官方反馈 | `docs/result-20260720-2.txt` |
| 最终实现与本地证据报告 | `docs/Greater算子性能优化最终报告-20260831.md` |
| 中间route-aware报告 | `docs/Greater算子性能优化阶段报告-20260831.md` |
| 候选接受/拒绝台账 | `Greater/perf_test/opt_20260831/candidate_ledger.csv` |
| 严格采集说明 | `Greater/perf_test/opt_20260831/README.md` |
| 严格采集入口 | `Greater/perf_test/opt_20260831/collect_strict.sh` |
| 最终94项原始结果 | `Greater/perf_test/opt_20260831/results/full94_final/` |
| 正式包烟测 | `Greater/perf_test/opt_20260831/results/timestamp_zip_smoke_104337/` |
| 最终代码检视闭环 | `Greater/review_work/20260831/greater_review_resolution.md` |
| Host源码 | `Greater/op_project/custom_greater/op_host/greater.cpp` |
| Kernel源码 | `Greater/op_project/custom_greater/op_kernel/greater.cpp` |
| 正式打包脚本 | `build_and_pack.sh` |
| 正式包 | `Greater_20260831_104337.zip` |

正式包SHA256为`9ac0049ed05a72e12102645dadb87026bfd45e740fe027cceaba5d585bbc537e`，其中s8 `.run` SHA256为`0c1afa6a1d352757b8156fc2c3b7cf76bfb67cee23389144959d421d75b5a043`。这两个身份与784.33us官方反馈相对应，后续迭代应将它们作为新的官方父版本标识。
