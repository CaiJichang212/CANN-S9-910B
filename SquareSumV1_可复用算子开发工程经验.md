# Ascend C 算子开发工程经验（可复用）

## 修订记录

| 版本 | 内容 | 日期 |
| --- | --- | --- |
| v1.0 | 基于 SquareSumV1（Ascend 910B / CANN 8.5.0）的问题定位、修复、上板回归与性能采集沉淀 | 2026-07-21 |
| v1.1 | 补充私有 L0 注册、提交包闭环、设备映射、证据分级及 910B 多级规约经验 | 2026-07-24 |
| v1.2 | 补充提交评分器的发布身份契约、源码包验证方法及 SquareSumV1 兼容性回归 | 2026-07-24 |

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
- fp16/bf16 的平方与累加应提升至 fp32；避免 half 规约中间结果溢出或 bf16 不支持的向量算术。
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
- 中间 workspace 用两个对齐的 FP32 stage ping-pong，而不是 32B/标量 staging 或逐标量 DMA。Host 必须精确计算每个 stage 的元素数、字节数与对齐。
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

## 10. 官方资料入口

- Ascend C 官方仓库：`/home/liyc/asc-devkit/README.md`
- DataCopyPad API：`/home/liyc/asc-devkit/docs/api/context/DataCopyPad(ISASI).md`
- 赛题优化建议：`/home/liyc/hw-S9/S9挑战赛910B软硬件深度协同优化建议.md`
