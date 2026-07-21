# Ascend C 算子开发工程经验（可复用）

## 修订记录

| 版本 | 内容 | 日期 |
| --- | --- | --- |
| v1.0 | 基于 SquareSumV1（Ascend 910B / CANN 8.5.0）的问题定位、修复、上板回归与性能采集沉淀 | 2026-07-21 |

## 1. 推荐开发闭环

1. **先固化规格**：明确 dtype、shape、axis、空 tensor、NaN/Inf、精度阈值与输出 shape 责任边界。
2. **按数据布局设计 TilingKey**：连续尾轴规约、非尾轴规约、不连续多轴规约分别实现，不用单一路径覆盖全部场景。
3. **先做最小可运行路径**：完成 Host Tiling、Kernel、ACLNN 接口、安装包和真实调用链，再扩展性能分支。
4. **每个复杂路径有针对性 NPU 用例**：Simulator/Mock 只能辅助，二维 DMA、Pattern Reduce、非对齐尾块和跨流水同步必须上板验证。
5. **正确性稳定后再优化**：性能修改只动已证明的热点，并保留每次 profile 和回归证据。

## 2. DataCopyPad 的核心约束

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

## 3. UB 容量、对齐和类型转换

- UB 预算要同时包含输入、fp32 Cast/Mul 工作区、累加器、输出、Reduce 临时区和队列的 buffer 倍数。
- `DataCopyPad` 可能在 UB 侧占用完整的 32B 尾块；输入缓冲和 fp32 工作缓冲都要为这部分容量留余量。
- fp16/bf16 的平方与累加应提升至 fp32；避免 half 规约中间结果溢出或 bf16 不支持的向量算术。
- `Cast` 的 count 不能为满足“看似对齐”而超过目标 buffer 实际容量。优先按已分配容量传入精确 count；若 API 确有对齐要求，则在 Host 侧同步扩容。
- 任何 `int64_t → uint32_t/uint16_t` 的 DMA 参数转换都应有显式上界来源，例如 shape 约束、UB 预算或 `4095` 上限。

## 4. 流水同步：队列与原始 TBuf 不可混淆

`TQue` 的 `EnQue/DeQue` 自带阶段同步；直接通过 `TBuf.Get()` 获得的原始 LocalTensor 则没有这层保护。

对复用同一 TBuf 的路径，至少检查三类依赖：

1. **MTE2 → Vector**：`DataCopyPad` 完成后，Vector 读取前需要对应同步。
2. **Vector → MTE3**：计算/类型转换结果写回 GM 前，需要等待 Vector 完成。
3. **MTE3 → 下一轮复用**：CopyOut 后，下一 tile 的 `Duplicate`、Cast 或 CopyIn 复用该 UB 前，需要等待 MTE3 读完源 buffer。

当使用原始 TBuf 且没有更精确的事件同步方案时，`PipeBarrier<PIPE_ALL>` 是安全的诊断与保守实现方式。确认正确性后，再依据 profile 评估能否替换为更细粒度同步；不要为了减少 barrier 牺牲数据依赖正确性。

## 5. 规约算子的分支策略

| 布局 | 推荐路径 | 原因 |
| --- | --- | --- |
| 归约轴位于最内层（AR） | 整行 CopyIn → fp32 square → ReduceSum | 内存连续，HBM 流量最低 |
| AR 超出 UB | 按列/R chunk 分载，fp32 标量累加 | 保持连续访问，避免超 UB |
| 中间轴（ARA） | 2D DataCopyPad 重排后沿 R 累加 | 需明确 GM/UB stride 和行 pitch |
| 大 R 或 blockCount 超限 | R 分块，fp32 向量累加器跨块累加 | 同时满足 DMA 编码上限与 UB 容量 |
| 不连续多轴 | 从内向外逐层规约，workspace 保存 fp32 中间结果 | 每层降低后续数据量，逻辑更可验证 |

对于布局敏感的 Pattern Reduce API，不要只以 Simulator 结果为依据。若真实 NPU 出现不稳定或错误，使用清晰的 `Add` 累加回退路径先保证正确性，再单独评估替代方案的性能收益。

## 6. 测试与定位方法

### 用例矩阵

每个 TilingKey 至少应覆盖：

- fp16、fp32、bf16（若支持）；
- 32B 对齐与非对齐的最后维；
- 最小规约长度、临界 UB 大小、超过 DMA blockCount 上限；
- `keep_dims` 的两种取值；
- 不连续多轴、负轴；
- 全零、NaN、正负 Inf；
- 每种 CopyIn/CopyOut 的最后一行或最后一 tile。

### 定位顺序

1. 先确认调用的确是新安装的自定义 `libcust_opapi.so` 与新构建的 Python 扩展，避免 ABI/旧包误判。
2. 用最小 shape 复现后，逐步只增加一个变量：非对齐、R 长度、A0 长度、dtype、axis。
3. 错误只在 NPU 出现时，优先审查 DataCopyPad 参数、UB 对齐、Pattern API 布局约束与 pipe 同步，而不是先怀疑数学公式。
4. 对 Run failed，先核对 DMA 参数范围、GM 有效长度、UB 总预算和所有 raw TBuf 的复用时序。

## 7. 性能采集与决策

- 使用与比赛一致的调用方式、预热次数、采样区间和统计量；记录 shape、卡号、包版本与 CSV 路径。
- 小算子常受固定 launch、标量和 DMA 开销限制。若 profile 没有任一流水线长期接近饱和，激进重写主路径的风险通常高于收益。
- 正确性修复也可能改善性能，例如消除非法 DMA、错误同步造成的异常等待；但必须以同口径的中位数复测确认。
- 不要为可见样例硬编码 Tiling。性能分支的条件必须由 dtype、shape、UB 预算和 API 硬约束推导，确保隐藏形状可泛化。

## 8. 交付前检查清单

- [ ] Host Tiling 与 Kernel 对齐、tile 大小、buffer 容量的推导一致。
- [ ] 所有 `DataCopyPad` 的 `blockLen`、stride 单位和 `blockCount` 都按官方文档核对。
- [ ] raw TBuf 的 MTE2→V、V→MTE3、MTE3→复用依赖均有同步。
- [ ] 回归覆盖每个 TilingKey 的典型、非对齐和边界用例，并在真实 NPU 通过。
- [ ] `git diff --check`、编译、安装、实际加载路径均确认无误。
- [ ] 提交包由项目指定脚本生成，且 `.run` 产物时间晚于最后一次源码修改。
- [ ] 性能结果注明采集口径，不将同类回归误写成隐藏测评已通过。

## 9. 官方资料入口

- Ascend C 官方仓库：`/home/liyc/asc-devkit/README.md`
- DataCopyPad API：`/home/liyc/asc-devkit/docs/api/context/DataCopyPad(ISASI).md`
- 赛题优化建议：`/home/liyc/hw-S9/S9挑战赛910B软硬件深度协同优化建议.md`

