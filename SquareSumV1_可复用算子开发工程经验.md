# Ascend C 算子开发工程经验（可复用）

## 修订记录

| 版本 | 内容 | 日期 |
| --- | --- | --- |
| v1.0 | 基于 SquareSumV1（Ascend 910B / CANN 8.5.0）的问题定位、修复、上板回归与性能采集沉淀 | 2026-07-21 |
| v1.1 | 补充私有 L0 注册、提交包闭环、设备映射、证据分级及 910B 多级规约经验 | 2026-07-24 |
| v1.2 | 补充提交评分器的发布身份契约、源码包验证方法及 SquareSumV1 兼容性回归 | 2026-07-24 |
| v1.3 | 补充 Case4 复盘：语义分支漏建、短 DMA 伪规约、32B 所有权、全核同步、测试身份与动态源码包漂移的防呆门禁 | 2026-07-25 |
| v1.4 | 补充已验证的 mode 4/workspace/BF16 修复、mssanitizer 目录 ACL、42 workload 性能口径和可靠性优先的性能决策 | 2026-07-25 |

## 0. 先建立可复现的证据闭环

算子问题常常不在 Kernel 本身，而在“编译产物—安装包—运行时加载—测试程序”之间的某一环。开发开始就应把下表的证据作为交付物，而不是最后补查。

| 层次 | 必须确认的事实 | 推荐证据 |
| --- | --- | --- |
| 源码 | Host、Kernel、API 的版本一致 | 提交号或 SHA-256、`git diff --check` |
| 构建 | 目标 SoC、CANN 与三种 dtype 二进制均正确 | 完整 build log、`.o` 清单 |
| 安装 | 被测 `.run` 是本轮构建产物 | 隔离 OPP 根目录安装、安装后文件和源码哈希 |
| 运行时 | 调用实际命中私有 `libcust_opapi.so` 和 kernel binary | 环境变量、加载路径、plog/运行日志 |
| 功能 | 每条 tiling 路径在真机通过且可重复 | 用例矩阵、golden 比对、连续多轮结果 |
| 性能 | 时延来自包含真实 AICore task 的同口径采样 | 原始 CSV、预热/样本范围、统计脚本 |

不要以“编译通过”替代安装验证，也不要以 mock/simulator 通过替代真机验证。性能 CSV 若只有 profiling 开关事件、没有实际 Kernel task 行，只能证明采集链路失效，不能用于计算 P50 或得出性能结论。

## 1. 推荐开发闭环

1. **先固化规格**：明确 dtype、shape、axis、空 tensor、NaN/Inf、精度阈值与输出 shape 责任边界。
2. **按数据布局设计 TilingKey**：连续尾轴规约、非尾轴规约、不连续多轴规约分别实现，不用单一路径覆盖全部场景。
3. **先做最小可运行路径**：完成 Host Tiling、Kernel、ACLNN 接口、安装包和真实调用链，再扩展性能分支；接口兼容性与内部实现名解耦。
4. **每个复杂路径有针对性 NPU 用例**：Simulator/Mock 只能辅助，二维 DMA、Pattern Reduce、非对齐尾块和跨流水同步必须上板验证。
5. **正确性稳定后再优化**：性能修改只动已证明的热点，并保留每次 profile 和回归证据。
6. **最后以提交物复验**：从最终 zip 解出 `.run`，安装到全新目录后重新跑 smoke；工作区成功不等于提交包成功。

## 2. API、注册与动态加载：先锁定发布身份契约

公开 API、L0 注册名、vendor 目录、动态 Python 名和 kernel 源码名并不是可独立替换的字段；它们共同决定安装后运行时和评分器如何发现算子。提交前应先从一份已通过的包提取这份契约，再在后续优化中保持一致。

以本项目通过评分器的 SquareSumV1 包为例，契约为：

| 层次 | 必须一致的值 | 验证位置 |
| --- | --- | --- |
| 对外 API | `aclnnSquareSumV1*` | `libcust_opapi.so`、调用样例 |
| L0 / GE / tiling / infer-shape 注册 | `SquareSumV1` | `OP_ADD`、`REG_OP`、`IMPL_OP_*`、`OP_TYPE` |
| package / vendor | `customize` | `set(package_name customize)`、安装后 `vendors/customize/` |
| 动态实现目录 | `customize_impl/dynamic/` | `.run --list` |
| 动态入口和源码 | `square_sum_v1.py`、`square_sum_v1.cpp` | `.run --list`、`npu_supported_ops.json` |

私有 L0 名称（例如 `SquareSumV1Custom`）有时可规避内置算子冲突，但只有调用框架、安装路径、动态实现生成规则和评分器发现规则都同步适配时才可采用。不能只改 `OP_TYPE` 或 `package_name`：这会连带改变注册 JSON、动态 Python 文件名和源码目录。若评分器仍按公开名查找，就可能出现“cannot find square_sum_v1.cpp after pkg install”这类表面是源码缺失、实际是**安装后路径不匹配**的错误。

因此，名称调整后的最小验证集必须包含：

1. `CMakePresets.json`、`CMakeLists.txt` 与 `CMakeCache.txt` 均确认 `ENABLE_SOURCE_PACKAGE=True`；
2. 对最终 `.run` 执行 `--list`，检查预期 vendor、动态目录、`.py` 和 `.cpp` 的完整路径；
3. 解包 `.run` 后读取 `framework/plugin/npu_supported_ops.json`，确认注册名为调用方和评分器期望的名称；
4. 与上一份成功包逐项比较上述字段，而不是只比较 `.run` 内是否存在任意一个同名 `.cpp`。

公开接口契约稳定后，tiling key、workspace 和 kernel 算法仍可在不改变这组发现字段的前提下独立演进。

## 3. DataCopyPad 的核心约束

以 CANN 8.5.0 官方文档 `DataCopyPad(ISASI).md` 为准：

| 参数/场景 | 规则 | 常见后果 |
| --- | --- | --- |
| `DataCopyExtParams.blockLen` | 单位为**字节**，可非 32B 对齐 | 错把元素数或 DataBlock 数传入，导致少搬/多搬 |
| `blockCount` | `1..4095` | 行数直接写入且超过上限，可能运行失败 |
| GM 侧 stride | 单位为**字节** | 误除以 32 会让二维搬运行地址错位 |
| VECIN/VECOUT 侧 stride | 单位为 **32B DataBlock** | 忽略 UB 行距会使尾 tile 覆盖或行布局错误 |
| 非对齐最后块 | 用 `DataCopyPad`，GM 只传真实有效 `blockLen` | 传入对齐后的长度会读越过 GM 末行 |

实践要点：

- 2D GM→UB 搬运时，先计算 `srcStride = (source_row_width - valid_width) * sizeof(T)`。
- UB 的目标行距按 32B 向上对齐；对于尾 tile，显式设置 `dstStride`，让每一行落在预期的 UB pitch。
- `DataCopyPad` 的非对齐填充不等于可以扩大 GM 读取范围；GM 传输长度始终是有效元素长度。
- Host 侧根据 `blockCount <= 4095` 决定全载或按 R 分块，Kernel 不应依赖窄类型截断来“自然分块”。

## 4. UB 容量、对齐和类型转换

- UB 预算要同时包含输入、fp32 Cast/Mul 工作区、累加器、输出、Reduce 临时区和队列的 buffer 倍数。
- `DataCopyPad` 可能在 UB 侧占用完整的 32B 尾块；输入缓冲和 fp32 工作缓冲都要为这部分容量留余量。
- fp16 的平方与累加应提升至 fp32。BF16 不能只写成“转 fp32 后平方”：若框架语义是 `sum(square(bf16_x))`，必须先得到可观察的 BF16 square 再转 fp32 累加。无原生 BF16 Mul 时用“FP32 Mul → Cast BF16 → Cast FP32”显式模拟这一舍入点。
- `Cast` 的 count 不能为满足“看似对齐”而超过目标 buffer 实际容量。优先按已分配容量传入精确 count；若 API 确有对齐要求，则在 Host 侧同步扩容。
- 任何 `int64_t → uint32_t/uint16_t` 的 DMA 参数转换都应有显式上界来源，例如 shape 约束、UB 预算或 `4095` 上限。

## 5. 流水同步：队列与原始 TBuf 不可混淆

`TQue` 的 `EnQue/DeQue` 自带阶段同步；直接通过 `TBuf.Get()` 获得的原始 LocalTensor 则没有这层保护。

对复用同一 TBuf 的路径，至少检查三类依赖：

1. **MTE2 → Vector**：`DataCopyPad` 完成后，Vector 读取前需要对应同步。
2. **Vector → MTE3**：计算/类型转换结果写回 GM 前，需要等待 Vector 完成。
3. **MTE3 → 下一轮复用**：CopyOut 后，下一 tile 的 `Duplicate`、Cast 或 CopyIn 复用该 UB 前，需要等待 MTE3 读完源 buffer。

当使用原始 TBuf 且没有更精确的事件同步方案时，`PipeBarrier<PIPE_ALL>` 是安全的诊断与保守实现方式。确认正确性后，应改为队列同步或更细粒度事件；不要把 `PIPE_ALL` 留在热循环中，也不要为了减少同步牺牲数据依赖正确性。

## 6. 规约算子的分支策略

| 布局 | 推荐路径 | 原因 |
| --- | --- | --- |
| 归约轴位于最内层（AR） | 整行 CopyIn → fp32 square → ReduceSum | 内存连续，HBM 流量最低 |
| AR 超出 UB | 按列/R chunk 分载，fp32 标量累加 | 保持连续访问，避免超 UB |
| 中间轴（ARA） | 2D DataCopyPad 重排后沿 R 累加 | 需明确 GM/UB stride 和行 pitch |
| 大 R 或 blockCount 超限 | R 分块，fp32 向量累加器跨块累加 | 同时满足 DMA 编码上限与 UB 容量 |
| 不连续多轴 | 从内向外逐层规约，workspace 保存 fp32 中间结果 | 每层降低后续数据量，逻辑更可验证 |
| 全规约且输出极少、大 R | 每核 FP32 partial，跨核同步后确定性归并 | 消除单核大规约瓶颈，避免跨核原子写 |

对于布局敏感的 Pattern Reduce API，不要只以 Simulator 结果为依据。若真实 NPU 出现不稳定或错误，使用清晰的 `Add` 累加回退路径先保证正确性，再单独评估替代方案的性能收益。

### 多轴与协作规约的工作区原则

- 多轴规约应把每一层定义为明确的 `(outer, reduce, inner)` 映射，保证不同核拥有不重叠的输出 tile。
- 第一层可融合 Cast + square；后续 workspace 层仅做 FP32 规约，最后一层才 Cast 并写用户输出。这样同时避免重复转换和中间精度损失。
- 优先使用经验证的 dense FP32 stage/ping-pong；但在多核 workspace 读写或同步协议未证明安全时，先降级到单核、完整 32B slot 的保守 staging。该降级是正确性基线，不应伪装成性能优化；恢复 dense/multi-core 前必须通过 sanitizer 和压力回归。
- 协作全规约的 partial 数量、每核 tile 与触发阈值应由输出数、规约长度、UB 预算和硬件核数推导；不能对公开 shape 写死条件。跨核同步仅能用于目标架构明确支持的专用路径，普通路径保持 AIV-only。

## 7. 测试与定位方法

### 用例矩阵

每个 TilingKey 至少应覆盖：

- fp16、fp32、bf16（若支持）；
- 32B 对齐与非对齐的最后维；
- 最小规约长度、临界 UB 大小、超过 DMA blockCount 上限；
- `keep_dims` 的两种取值；
- 不连续多轴、负轴；
- 全零、NaN、正负 Inf；
- 每种 CopyIn/CopyOut 的最后一行或最后一 tile。

建议额外固定以下“硬件编码边界”样例：`blockCount=4094/4095/4096`、rank 1–5、`N/N2=10000`、`N3=1000`、`N4=200`，以及一个全规约协作样例。多轴 axis 同时覆盖负数、乱序和不连续组合。

### 定位顺序

1. 先确认调用的确是新安装的自定义 `libcust_opapi.so` 与新构建的 Python 扩展，避免 ABI/旧包误判。
2. 用最小 shape 复现后，逐步只增加一个变量：非对齐、R 长度、A0 长度、dtype、axis。
3. 错误只在 NPU 出现时，优先审查 DataCopyPad 参数、UB 对齐、Pattern API 布局约束与 pipe 同步，而不是先怀疑数学公式。
4. 对 Run failed，先核对 DMA 参数范围、GM 有效长度、UB 总预算和所有 raw TBuf 的复用时序。

### 设备与环境映射

容器内逻辑设备号不一定等于宿主物理卡号。测试记录必须同时写明“容器逻辑 ID、宿主物理 ID、可见设备环境变量”。例如宿主 NPU 6 映射到容器 NPU 0 时，ACL 测试应显式选择逻辑 `0`，并检查是否有残留的可见设备变量改变映射。出现“无进程/无 profile 数据”时，先核对这层映射，再判断算子是否没有执行。

## 8. 性能采集与决策

- 使用与比赛一致的调用方式、预热次数、采样区间和统计量；记录 shape、容器逻辑/宿主物理卡号、包版本与 CSV 路径。
- 每个候选配置独立采集至少 3 轮；每轮固定发射次数，剔除预热后按约定样本窗口计算 P50。只有确认 CSV 含真实 AICore Kernel 行后才可统计。
- 小算子常受固定 launch、标量和 DMA 开销限制。若 profile 没有任一流水线长期接近饱和，激进重写主路径的风险通常高于收益。
- 正确性修复也可能改善性能，例如消除非法 DMA、错误同步造成的异常等待；但必须以同口径的中位数复测确认。
- 不要为可见样例硬编码 Tiling。性能分支的条件必须由 dtype、shape、UB 预算和 API 硬约束推导，确保隐藏形状可泛化。

性能报告应分为三类，避免混用：功能 smoke（只证明可运行）、性能趋势（采样口径不等同官方）和官方口径结果。前两类都不能替代第三类；没有有效 task 数据时，应明确标注“未验收”，而非填入推测的时延。

## 9. 交付前检查清单

- [ ] Host Tiling 与 Kernel 对齐、tile 大小、buffer 容量的推导一致。
- [ ] 所有 `DataCopyPad` 的 `blockLen`、stride 单位和 `blockCount` 都按官方文档核对。
- [ ] raw TBuf 的 MTE2→V、V→MTE3、MTE3→复用依赖均有同步。
- [ ] 调用 API、L0/GE/tiling/infer-shape 注册名、`package_name`、vendor、动态 `.py` 和 `.cpp` 路径与已验证提交契约一致；如有私有化改名，已重新完成安装后发现验证。
- [ ] 多轴 workspace 的 stage 数、FP32 字节数、对齐和每层输出所有权均由 Host 与 Kernel 共享的 tiling 数据描述。
- [ ] 回归覆盖每个 TilingKey 的典型、非对齐和边界用例，并在真实 NPU 通过。
- [ ] `git diff --check`、编译、安装、实际加载路径均确认无误；记录容器/宿主设备映射。
- [ ] 提交包由项目指定脚本生成，且 `.run` 产物时间晚于最后一次源码修改；zip 不含绝对路径或无关构建目录。
- [ ] `.run --list` 同时证明 `ENABLE_SOURCE_PACKAGE` 的实际结果：目标 `dynamic/` 目录中存在评分器将查找的 `<kernel>.cpp`，而非只检查 staging 目录中的源码。
- [ ] 解包最终 zip 后，把其中 `.run` 安装到独立目录并重新跑真实 NPU smoke；比较安装包与工作区源码哈希。
- [ ] 性能结果注明采集口径和 task 数据有效性，不将同类回归或无效 profiler 输出误写成隐藏测评已通过。

## 10. Case4 复盘：把“反复出现”变成可阻断的工程规则

### 10.1 先区分三件事：错误、重复行为与证据边界

本次 Case4 的经验不是“再加一个 mode”这么简单，而是同一类问题在不同层反复出现：**把语义不同的场景压进通用路径、把逻辑元素当作 DMA/并发所有权单位、用工作区成功替代提交物成功**。它们分别表现为正确性、性能和发布问题。

下表只记录已由源码、构建或包内容证实的事实；评分器隐藏 Case4 的精确 shape、超时阈值和 plog 仍未取得，因此不能把本地已修复路径表述为“已证明覆盖官方 Case4”。

| 类别 | 已证实的重复模式 | 直接后果 | 应固化的规则 |
| --- | --- | --- | --- |
| 语义建模 | `axis=[]` 被折叠为 `A1=numel, R=1` 的规约 | 无规约语义退化为逐元素 `ReduceSum(1)` | 属性规范化后先按**数学语义**选模式；`axis=[]`、空规约和普通规约必须是独立分支 |
| 粒度选择 | 每个元素执行一次 2/4B CopyIn、平方、`ReduceSum(1)`、CopyOut | 循环与 DMA 均为 `O(numel)`，并放大大偏移/MTE 和超时风险 | 先确定最小高效工作单元（这里为 32B block + UB tile），再做多核切分 |
| 并发所有权 | 多核分别写相邻 4B partial 或共享尾部 DataBlock | 多个 MTE3 短写可能竞争/覆盖同一 32B 块 | GM 写所有权的最小单位是 **32B DataBlock**，不是一个逻辑标量 |
| 同步协议 | 通用 `myRows_==0` 早退发生在 `SyncAll()` 模式 dispatch 之前 | 部分核提前返回，其余核可能永久等待 | 含全核 barrier 的模式必须先 dispatch；文档中写明参与核数与每核到达次数 |
| 空值语义 | 将“输入或规约轴为空”统一视为 no-op | 输出元素仍存在时会保持未初始化 | 分别计算 input、reduce 和 output 的元素数；仅 `outputElements==0` 才允许 no-op |
| 发布链路 | 本地二进制已更新，动态源码 staging/`.run` 却保留旧 header | 安装后动态编译回到旧实现，工作区修复未交付 | 每次打包强制刷新源码 staging，并逐文件比较源码、staging 和最终包 |
| 测试基础设施 | UT 使用已不再注册的 `SquareSumV1Custom`，而发布身份为 `SquareSumV1` | 所有用例在进入 tiling 前失败，掩盖真实回归结论 | 测试注册名必须由同一发布身份清单生成或校验，不能手写漂移 |

### 10.2 重复执行反模式：不要让“单元素正确”变成“整体可运行”

最危险的重复行为是把通用路径的最小样本操作放进 `numel` 循环。单次操作虽数学正确，但在大 shape 下会变成算法错误。

| 反模式 | 旧行为 | 正确替换 | 设计审查问题 |
| --- | --- | --- | --- |
| 伪规约 | 每个元素做一次短 DMA + `ReduceSum(1)` | 一个 UB tile 只做 CopyIn → square → CopyOut | 该分支真的需要 Reduce 吗？规约长度是否可能为 1/0？ |
| 按元素切核 | 每核按逻辑元素边界写出 | 按 32B block 切核，最后一个有效核独占全局尾块 | 相邻核是否会写同一个 32B GM 块？ |
| 标量 partial 工作区 | 每核写一个 4B fp32 partial | 每核固定 32B slot，汇总时按 slot stride 读取 | workspace 的地址间距是否至少等于 DMA 最小写粒度？ |
| 每轮依赖全局停顿 | raw TBuf 复用只依赖偶然时序，或无区分插入 barrier | 明确 MTE2→V、V→MTE3、MTE3→复用依赖；先正确、再细化流水 | 本轮写回尚未完成时，下一轮是否覆盖同一 UB？ |
| 反复验证错误产物 | 用工作区包、旧 OPP 或不同 OS 包验证，然后外推提交物 | 由最终 `.run` 解包/安装/加载，再跑同一用例 | 日志中的 `.so`、kernel、动态源码和包 SHA 是否来自同一产物？ |

一个实用的复杂度审查公式是：

```text
总 DMA 次数 = 每 tile 的 DMA 次数 × ceil(totalElements / tileElements)
总 Vector/Reduce 次数 = 每 tile 的指令次数 × tile 数
```

若 `tileElements=1` 只是因为把“没有规约”伪装为 `R=1`，应在 Host 分支处消除该路径；不要先尝试用更多核、更多 barrier 或更大的 workspace 掩盖它。

### 10.3 可复用的四份契约

对任意 Ascend C 算子，在实现前写出并在 Host、Kernel、UT 中共享下列契约；它们比零散的 if/else 更能防止问题回归。

1. **语义—算法契约**：每个属性组合映射到唯一算法类别。例如 `axis=[] → elementwise`，`reduceElements=0 && outputElements>0 → zero-fill`，普通规约才进入 AR/ARA/multi-axis。
2. **单位—范围契约**：所有字段标明单位（元素、字节、32B block、行、tile）；shape 乘法、元素偏移、字节长度和 API 窄类型转换均在 Host 做 checked 64 位计算。
3. **所有权—同步契约**：每个 GM 输出区间、workspace slot 和全核 barrier 都有唯一写者/参与者定义。短 DMA 的所有权按 32B，而不是按 `float`/`half` 元素。
4. **源码—产物—运行时契约**：源码、kernel binary、动态源码 staging、`.run`、安装目录和实际加载路径可逐项追溯，并具有可比较的 SHA 或 `cmp` 证据。

建议把这四份契约作为设计评审的固定小节，并把关键字段放入 TilingData 注释。例如 `noReduceTotalElements` 是元素、`noReduceTileElements` 是按 dtype 对齐后的元素数、workspace partial slot 是 32B。

### 10.4 推荐的实现顺序（避免在错误层修补）

1. 读取规格并枚举语义分区：空 axis、全规约、空规约、连续/非连续 axis、keep_dims。
2. 为每个语义分区估算数据移动与算术复杂度；若无规约路径出现 `ReduceSum(1)`，在此阶段阻断。
3. 以 DMA 最小粒度规划 GM 所有权，再由所有权推导 `blockDim`、每核 begin/end 和最后尾块归属。
4. 以实际 `InitBuffer` 总和反推 tile 上限；Host 与 Kernel 不得各自维护不一致的 UB 预算。
5. 只在数据流和所有权确定后加入队列/事件/barrier；含 `SyncAll()` 的模式单独声明“全核参与”。
6. 为每个分区创建 Host UT、small NPU 功能用例和一个能暴露规模/对齐问题的边界用例。
7. 由发布脚本生成包并验证包内动态源码，再进入隔离安装和真机 smoke。

### 10.5 防呆门禁：应自动化，而不是靠记忆

| 阶段 | 必须自动检查 | 失败时应阻断什么 |
| --- | --- | --- |
| Tiling UT | `axis=[]` 必为 NO_REDUCE；空规约输出非空必为 zero-fill；超大 shape 乘法溢出必须拒绝 | 语义错路由、未初始化输出、截断偏移 |
| Kernel 审查 | NO_REDUCE 路径 `ReduceSum` 次数为 0；每个跨核短写有 32B 唯一所有者；每个 raw TBuf 复用有依赖 | 短 DMA/伪规约、workspace 竞争、偶发错误 |
| Barrier 审查 | 对每个 `SyncAll()` 列出启动核集合、早退条件和到达次数 | 死锁/挂起 |
| 构建 | 清除 CANN 的 kernel-source copy timestamp；比较 `op_kernel/` 与 dynamic staging 的 `.cpp/.h/tiling_data.h` | 本地 binary 与安装后动态源码不一致 |
| 包 | 对最终 `.run` 解包或比较 CPack staging；检查动态路径、注册 JSON、vendor 和源码哈希 | 提交包路径错误、旧源码被交付 |
| 运行 | 日志确认加载的 opapi/opmaster/kernel 来自当前隔离安装目录 | 残留 OPP、同名内置算子或旧包污染 |
| 性能 | 对每条语义路径记录 tile 数、DMA 次数预估和真实 task 数据 | 把“能跑”误判为“可扩展” |

### 10.6 最小回归矩阵模板

以下矩阵不依赖 SquareSumV1 的具体 shape，可作为“属性语义 × 物理边界 × 发布物”的最小骨架：

| 维度 | 必选用例 | 目标 |
| --- | --- | --- |
| 语义 | empty axis、全规约、单轴、非连续多轴、空规约 | 不同数学语义不共用错误路径 |
| 数据类型 | fp16、fp32、bf16（若声明支持） | Cast、累加精度和 API 覆盖 |
| DMA | 32B 对齐、非对齐尾块、`blockCount=4095/4096` | 长度/stride/上限正确 |
| 并发 | 少于核数、非整除分核、仅最后核有尾块、workspace 相邻 slot | 工作所有权和 barrier 协议正确 |
| 规模 | 最小、UB 临界、地址/元素乘法溢出拒绝、目标大 shape | 防止复杂度或偏移随规模失控 |
| 发布 | 源码构建、最终 `.run`、隔离安装后的 smoke | 防止工件漂移 |

### 10.7 本次复盘的行动结论

- 对“两个版本都失败”的问题，优先找两版共享的语义分支和发布链路，不要先追逐新增优化路径。
- 对“单测小 shape 正确、大 shape 失败”，先数 DMA/循环次数并审查 64 位偏移和 32B 所有权，而不是先调精度阈值。
- 对“报告通过、提交失败”，默认怀疑测试对象不是同一包；先比较 `.run` SHA、动态源码 SHA 和运行时加载路径。
- 对“所有 UT 同时失败”，先检查测试注册/加载身份；只有测试已真正进入 tiling/kernel，断言结果才可用于判断算法回归。

上述规则的核心是：**语义分支在 Host 一次判定，数据按 tile 批量处理，GM 按 DataBlock 定义所有权，发布物而不是工作区作为验收对象。**

## 11. 官方资料入口

- Ascend C 官方仓库：`/home/liyc/asc-devkit/README.md`
- DataCopyPad API：`/home/liyc/asc-devkit/docs/api/context/DataCopyPad(ISASI).md`
- 赛题优化建议：`/home/liyc/hw-S9/S9挑战赛910B软硬件深度协同优化建议.md`

## 12. SquareSumV1 修复与评测闭环（2026-07-25）

### 当前已验证状态与证据边界

- 当前隔离 OPP 下，`npu_acceptance_test.py` 得到 fp16/fp32 评分路径 44/44、BF16 3/3、非法输入 4/4；这证明本地当前包通过，不等同外部平台 Case4 已回归。
- Key4（mode 4）fp16/fp32 已做直接 1000 次和完整 wrapper 后紧接 Key4 各 100 次压力；mode 5、mode 6 跨 4 GiB、mode 7 空规约均有专项回归。
- mssanitizer 已覆盖 Key4 三 dtype 和 mode 5，未发现越界、未对齐访问或多核覆盖。工具要求目录不可组写：项目目录已收紧为 0750；profiler/sanitizer 子目录还要处理 inherited default ACL，不能修改共享父目录。
- 外部历史回执仍是 Case1/2/3/5 Pass、Case4 Run failed。在取得新外部回执前，所有文档只能写“本地修复通过”。

### 当前实现的关键事实

1. mode 4 Host 固定 `usedCoreNum=1` / `SetBlockDim(1)`；workspace 申请包含 16 MiB framework reserve，Kernel 用 `GetUserWorkspace(workspace)` 取用户区。中间 fp32 采用每标量 32B slot，层间使用 `PipeBarrier<PIPE_ALL>()`，不在该单核编译路径中保留 `SyncAll()`。
2. mode 5 保留多核 `SyncAll()`，但 partial workspace 也按完整 32B ownership slot 规划；含全核 barrier 的分支必须在任何 `myRows==0` 早退之前 dispatch。
3. `axis=[]` 是 mode 6 的 tiled elementwise square；规约轴长度为零而输出非空是 mode 7 zero-fill。两者按 32B DataBlock 分核并使用 64 位元素/字节范围检查。
4. `GetPhyAddr(offset)` 的 `offset` 是元素偏移；不要为“修复大地址”将其改成字节偏移。

### 可直接复用的当前验证命令

```bash
cd /home/liyc/hw-S9/case_910b_SquareSumV1/SquareSumV1
source /home/ma-user/Ascend/cann-8.5.0/set_env.sh
source op_project/custom_squaresumv1/npu_opp.F2ERJn/vendors/customize/bin/set_env.bash
export ASCEND_RT_VISIBLE_DEVICES=0
python3 npu_acceptance_test.py
```

性能采集使用同一 wrapper：每例 30 个目标 task，过滤 `aclnnMul`，取第 11–30 个 SquareSumV1 task 的 P50。原始 42 workload 总览在 `perf_eval_20260725_3/score_matrix_pipe/`；深度 profile 归档为 `docs/perf/round_006/`（大 ARA）和 `docs/perf/round_007/`（mode 4）。报告为项目根目录的 `20260725-3算子性能评测和瓶颈分析报告.md`。

### 当前性能基线与下一步

| 口径 | 当前结果 | 决策 |
|---|---:|---|
| 42 workload 稳定窗口 P50 合计 | 762.250 µs | 后续回归基线，不能与外部五题总分直接比较 |
| 大 ARA fp16 `(2024,3000), axis=0` | 50.364 µs，CV 1.64%，38 核 | 无单一 bound，接近 MTE2 主导；优先搜索 tile/流水重叠 |
| mode 4 fp16 非连续多轴 | 68.774 µs，CV 1.74%，1 核 | MTE2/MTE3/Scalar 混合；先设计可验证多核 stage 协议，再谈恢复性能 |
| 最大评分风格单例 | ARA row-split fp16 `(4,10000,100), axis=1` 120.284 µs | 优先优化 mode 3 chunk/tile，保持 FP32 累加与 DMA 边界 |

没有修复前后二进制的同机成对三轮基线，因此当前结论是“未证明整体提升”，不是性能回退的严格归因。任何性能候选必须先跑本节正确性/安全门禁，再以同一矩阵至少三轮独立采集决定是否合入。
