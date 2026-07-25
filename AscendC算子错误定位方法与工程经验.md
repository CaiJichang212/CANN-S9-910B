# Ascend C 算子错误定位方法与工程经验

## 1. 文档目的

本文总结 Ascend C 自定义算子在“本地测试通过、提交后隐藏用例 `Run failed`”场景下的定位方法。重点不是罗列所有可能故障，而是建立一条可复现、可证伪、能落到 Host Tiling 与 Kernel 具体指令的证据链。

本文以 SquareSumV1 的排查过程为实例来源，但方法可复用于逐元素、归约、转置及包含 workspace/多核同步的 Vector 算子。

## 2. 三条基本原则

### 2.1 交付物优先于工作区源码

评分器执行的是提交包中的 `.run`、Host `.so` 和 Kernel `.o`，不是当前目录里“看起来已经修好”的源码。必须先回答：

1. 被测试包和正式提交包的 SHA256 是否一致；
2. 运行时实际加载了哪个 vendor、哪个 `opmaster`、哪个 Kernel 二进制；
3. 测试环境的 OS、CANN、SoC、安装方式是否和评分环境一致；
4. 报告中的测试矩阵是否覆盖了隐藏用例可能触发的语义分支。

只要二进制哈希不同，本地“全部通过”就只能证明另一个构建产物，不能证明提交物。

### 2.2 把事实、推断和待确认分开

建议每条结论显式标注证据等级：

| 等级 | 含义 | 示例 |
|---|---|---|
| A：直接事实 | 正式二进制在真实 NPU 上可重复触发，且有 plog/错误码 | 正式 Kernel 报 `MTE instruction DDR address out of range` |
| B：强代码证据 | Host 参数可确定路由到某 Kernel 分支，地址/同步关系可静态推导 | `axis=[]` 被映射为 `R=1` 的逐行归约 |
| C：差分推断 | 多次提交的共同失败与共同代码路径吻合 | 两包都失败，优先审计未变的 mode 0，而非新加的 mode 5 |
| D：待确认 | 缺少官方输入或日志，只能提出假设 | 官方 Case4 究竟是超时还是 MTE 越界 |

结论强度不能超过证据强度。复现了同类错误，不等于已经拿到隐藏 Case 的原始输入。

### 2.3 先分类，再深入

`Run failed` 不是一个根因。先通过退出码、调用耗时和 plog 将其分为：

- Host/API 失败：参数校验、Shape 推导、Tiling、Kernel 查找或注册失败；
- Kernel 崩溃：AICore error、MTE 非法地址、UB 越界、指令参数非法；
- Kernel 卡死：多核 `SyncAll()` 参与核不一致、队列 EnQue/DeQue 不配对；
- 超时：算法复杂度错误、单核串行、逐元素 DMA、极端循环次数；
- 精度失败：输出产生但超出误差门限；
- 资源失败：HBM/workspace 分配失败、UB 预算不成立。

分类后再选择工具；否则容易把性能问题、精度问题和崩溃问题混为一谈。

## 3. 一条推荐的定位流程

### 3.1 固化现场

至少保存以下信息：

```text
提交包路径、SHA256
解包后 vendor/opmaster/kernel 路径及 SHA256
CANN/驱动/固件/SoC/OS
输入 shape、dtype、axis、keep_dims
API 返回码、进程退出码、运行时长
plog 路径和故障时间窗口
```

常用只读命令：

```bash
sha256sum <package.run>
file <package.run>
ldd <opmaster.so>
rg -n "AIC ERROR|MTE|out of range|deadlock|error code|kernel_name" <plog>
```

不要覆盖原始日志；后续过滤结果应另存，并保留时间戳和进程号。

### 3.2 用提交差分缩小范围

如果两个版本只有部分实现不同，而同一 Case 都失败：

1. 列出两包共同的 Host 路由和 Kernel 分支；
2. 将只存在于新包的分支降为次要嫌疑；
3. 比较失败 Case 以外的性能变化，判断新分支是否真的被命中；
4. 对共同路径优先构造边界用例。

差分法不能单独证明根因，但能显著减少无方向的代码阅读。本项目两次提交都只有 Case4 失败，因此第二次才新增的 cooperative reduce-all 和重写后的多轴路径，不能自然解释第一次的同一失败；共同的空轴路由应优先审计。

### 3.3 建立“语义到指令”的路径表

不要只看 Kernel。对每类输入写出完整链路：

```text
API 属性
  -> Shape/axis 规范化
  -> Coalesce 后的 A1/R/A0
  -> tilingMode、blockDim、workspace
  -> 每核任务区间
  -> UB Buffer
  -> DataCopy/Vector/Reduce/Sync
  -> GM 输出区间
```

建议建立最小路由矩阵：

| 语义 | 典型输入 | 应走路径 | 特别检查 |
|---|---|---|---|
| 无规约 | `axis=[]` | elementwise square | 不应调用 `ReduceSum(1)` |
| 尾轴规约 | `axis=[-1]` | AR | R 边界、尾块 |
| 非尾轴规约 | `axis=[0]` | ARA | 行距、A0 tile、RA scratch |
| 全规约 | axis 覆盖全部维度 | cooperative reduce | partial slot、全核同步 |
| 非连续多轴 | `axis=[1,3]` | 分层/重排归约 | workspace layout、层间同步 |
| 空规约集合 | 规约轴长度为 0 | zero-fill 或规格定义路径 | 输出非空时必须写值 |

### 3.4 从边界值反推隐藏 Case

隐藏用例通常覆盖可见用例缺失的分支，而不一定是随机大 Shape。优先枚举：

- `axis=[]`、全轴、负轴、非连续多轴、重复轴和越界轴；
- rank 0、rank 1、最大 rank；
- 0、1、7/8、15/16、31/32、4095/4096 等对齐和 API 上限边界；
- 总元素数跨越 `INT32_MAX`、4 GiB 字节偏移及设备容量附近；
- 核数大于工作项数、存在空闲核；
- fp16、bf16、fp32 的不同 Buffer 和 Cast 路径；
- `keep_dims=true/false`。

每个用例只改变一个主变量，才能把异常归因到特定边界。

## 4. Host Tiling 审计方法

### 4.1 Shape 乘法必须可检查

Shape 维度乘积、字节数、workspace 大小和 GM 偏移均应使用 `int64_t/uint64_t`，并在乘法前检查溢出：

```cpp
if (dim != 0 && total > limit / dim) {
    return ge::GRAPH_FAILED;
}
total *= dim;
```

仅把最终变量声明为 64 位不够；中间表达式、API 参数窄化和 Kernel 侧索引也必须逐一核对。

### 4.2 Host 与 Kernel 的 UB 预算逐项对账

把 Kernel 中所有 `InitBuffer` 列成表，按最大并存时刻求和，而不是凭经验扣一个“安全余量”：

| Buffer | 数据类型 | 元素数/字节数 | 生命周期 | 是否双缓冲 |
|---|---|---:|---|---|
| 输入队列 | T | tile × sizeof(T) | CopyIn 到 Compute | 是/否 |
| FP32 计算区 | fp32 | tile × 4 | Cast/Mul/Reduce | 否 |
| accumulator | fp32 | 对齐后列数 × 4 | 跨 chunk | 否 |
| Reduce scratch | byte | API 查询值 | Reduce 期间 | 否 |
| 输出队列 | T/fp32 | 对齐后输出 | Compute 到 CopyOut | 是/否 |

最低 32B 的临时区也必须计入。Host 模型和 Kernel 分配模型应由同一组字段驱动。

### 4.3 TilingData 是契约，不是便笺

对每个字段核对：

- 单位是元素、字节还是 32B block；
- 有符号/无符号及位宽是否一致；
- Host 设置后 Kernel 是否真的消费；
- mode 切换时是否残留旧字段；
- `blockDim`、`usedCoreNum` 和同步参与核数是否一致。

Host 计算了复杂的子 Tiling，但 Kernel 没有使用，是常见的“纸面安全、实际失配”。

## 5. Kernel 内存与同步审计

### 5.1 GM 地址区间证明

对每次 GM 访问证明：

```text
0 <= baseOffset
0 <= transferBytes
baseOffset + transferBytes <= tensor/workspace valid bytes
```

同时检查：

- 元素偏移和字节偏移是否混用；
- `int64_t` 是否在传入指令/API 时被截成 `int32_t/uint32_t`；
- `blockLen`、`blockCount`、stride 是否超过 API 限制；
- 尾块 `DataCopyPad` 是否只在全局最后一个所有者核处理；
- workspace 基址是否包含系统 workspace 偏移约定。

### 5.2 32B DataBlock 的跨核所有权

MTE 写 GM 的最小事务粒度与短写行为需要按 32B 设计。即使逻辑上每核只写一个 fp32，若地址分别为 `workspace[0]、workspace[1]...`，多个核仍可能写同一个 32B DataBlock。

稳妥做法是：

- 每核 partial slot 至少独占并按 32B 对齐；或
- 由单核汇总后紧凑写回；或
- 按完整 DataBlock 切分输出，每个 block 只有一个核负责。

“逻辑元素不重叠”不等于“DMA 事务不竞争”。

### 5.3 `SyncAll()` 必须全体同构

对每个 `SyncAll()` 检查：

1. 所有启动核是否必定到达；
2. 到达次数和顺序是否相同；
3. 是否有 `myRows == 0`、`blockIdx != 0` 等提前返回；
4. 循环次数是否依赖每核不同的数据范围；
5. `blockDim` 是否等于协议规定的参与核数。

需要全核同步的 mode 应在空工作量判断之前 dispatch；空闲核可以不做计算，但必须参与相同同步协议。

### 5.4 队列和流水线配对

逐路径核对：

- 每次 `AllocTensor` 都能 `FreeTensor`；
- `EnQue/DeQue` 次数一致；
- raw `TBuf.Get()` 的复用有必要的 PipeBarrier；
- MTE2、Vector、MTE3 的依赖方向正确；
- 发生早退时没有遗留队列对象或跳过同步。

## 6. 正式包复现的工程方法

### 6.1 隔离 OPP 污染

机器上残留的同名 vendor 或 CANN 自带同名算子，可能让测试加载错误的 `opmaster`。应在干净进程中：

1. 解包正式 `.run`；
2. 设置最小化的 OPP/vendor 搜索路径；
3. 记录动态库加载轨迹；
4. 在 `aclInit` 后确认或显式加载正式包的 Host 库；
5. 从 plog 中核对最终选中的 Kernel 路径和 tiling key。

若日志指向 `/usr/local/Ascend/.../vendors/<其他vendor>`，该次结果不能归因于待测包。

### 6.2 真实 NPU ST 与 Mock/Host UT 分工

- Host UT：验证 Shape、axis 规范化、mode、UB/workspace 计算和错误返回；
- Kernel Mock/CPU 仿真：验证分支、索引和基本数值逻辑；
- Simulator：辅助发现越界、流水和性能风险；
- 真实 NPU ST：确认真实 DMA、同步、二进制加载和运行时行为。

只有真实 NPU 调用正式安装包，才能闭环 `Run failed`。Mock ST 通过不能替代这一门禁。

## 7. 常见误区

1. 报告测试包与提交包哈希不同，却用报告结论证明提交物。
2. 只测试可见 Shape，没有覆盖空 axis、空维、全规约和超大扁平长度。
3. 看到 `Run failed` 就先优化性能，没有先查 plog。
4. 只查 Kernel 数值逻辑，不核对 Host Tiling 路由。
5. 仅证明元素区间不重叠，忽略多个核共享 32B DataBlock。
6. 在 `SyncAll()` 前允许部分核提前返回。
7. 用当前源码编译的新包复现，却没有验证正式提交包。
8. 环境中存在同名注册或残留 vendor，仍把错误日志归因于目标包。
9. 一个极端复现输入本身超过设备容量，却直接宣称它就是官方隐藏用例。
10. 把“没有崩溃”当作“性能可接受”；逐元素 DMA 可能只在大 Shape 上超时。

## 8. 最小回归与验收矩阵

修复后至少执行以下门禁：

| 层级 | 必测内容 | 通过标准 |
|---|---|---|
| Host UT | 所有 mode、边界 Shape、溢出、workspace/UB | 路由与计算符合设计，非法输入明确失败 |
| 精度 ST | dtype × axis 类别 × keep_dims × 对齐边界 | 满足评分精度标准，无未初始化输出 |
| 稳定性 ST | 30～100 次重复、空闲核、同步路径 | 无卡死、AICore error、MTE error |
| 大 Shape ST | 超过 32 位元素/字节边界前后的可分配 Shape | 地址正确；不可分配输入在 Host 明确失败 |
| 性能 ST | 冷/热启动区分，每例多次 launch | 无逐元素 DMA 灾难，P50/P95 稳定 |
| 正式包验收 | 安装提交 `.run` 后重复上述关键用例 | 加载路径、Kernel SHA 与提交物一致 |

## 9. 推荐的证据记录模板

```markdown
### 现象
- 输入：shape=..., dtype=..., axis=..., keep_dims=...
- 结果：返回码/超时/精度数据

### 交付物
- run SHA256：...
- opmaster 路径与 SHA256：...
- kernel 路径与 SHA256：...

### Host 路由
- normalizedAxis：...
- A1/R/A0：...
- tilingMode/blockDim/workspace：...

### Kernel 证据
- 访问/同步位置：文件:行号
- plog 原文：...

### 结论等级
- 已证实事实：...
- 高置信推断：...
- 待确认：...

### 修复与回归
- 代码策略：...
- 新增用例：...
- 正式包验证：...
```

## 10. 方法总结

高质量定位应形成闭环：提交物身份一致、输入语义明确、Host 路由可复算、Kernel 地址和同步可证明、真实 NPU 日志可对应、修复后正式包可回归。任何一环缺失，都应诚实地保留为“待确认”，而不是用经验猜测替代证据。
