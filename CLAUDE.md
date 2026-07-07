# case_910b — S9 Ascend C 算子挑战赛工程

> 本文件是本仓库（`/home/liyc/hw-S9/case_910b`）的项目级开发指南。算子开发的通用方法论与踩坑记录见上层文档与 memory，本文件只讲「这个仓库怎么用」。

## 1. 项目定位

- **赛题**：S9 Ascend C 算子性能挑战赛，5 个算子（Concat / Greater / IndexAdd / Transpose / SquareSumV1），每个算子独立排名。
- **目标**：在精度通过的前提下，5 个 case 的 AICore 耗时总和最小（单位 µs）。当前聚焦 **Concat**。
- **评分**：5 case 总耗时排名；要求 5 个 case 全部通过且总耗时 ≤ 500 µs（排行榜头部约 500 µs 量级）。
- **当前分支**：`dev-concat-0630`（仅在此分支开发 Concat，勿混用其他分支代码）。

## 2. 环境前提（必须满足，否则无法计分）

| 项 | 值 |
|----|----|
| CANN | 社区版 **8.5.0**（`/usr/local/Ascend/cann-8.5.0`） |
| SoC / 计算单元 | **ascend910b**（`CMakePresets.json` 的 `ASCEND_COMPUTE_UNIT`、`op_host` 的 `AddConfig("ascend910b")`） |
| 硬件 | 昇腾 910B4-1，**每卡 20 个 AICore**（`npu-smi info -t common -i 0` 的 `Aicore Count`） |
| OS | openEuler / Euler 2.10（容器 `cann850`，**已在容器内**，无需再 `docker exec`） |
| 架构 | aarch64 |
| Python | 3.11 |

> ⚠️ 910B 单卡物理 AICore = **20**。`blockDim`（`SetBlockDim`）上限应取 20，超过会分波串行（多 wave）增加开销。Greater 参考实现正确用了 20；**Concat host 当前误用 `MAX_AIV_NUM=48`，属待修性能 bug**。

## 3. 目录结构

```
case_910b/
├── op/                         # 算子工程根（开发 + 打包源）
│   ├── ConcatCustom.json       # 算子 IR 定义（msopgen 输入）
│   └── CustomOp/               # ★ 当前正在开发的 Concat 算子工程
│       ├── CMakePresets.json    # 构建预设（ascend910b）
│       ├── build.sh             # 编译入口 → build_out/custom_opp_*.run
│       ├── op_host/             # host 侧：tiling + infershape + op 注册
│       │   ├── concat.cpp
│       │   └── concat_tiling.h
│       └── op_kernel/           # kernel 侧：算子核函数
│           └── concat.cpp
├── Concat/                     # Concat 测试与性能采集脚手架（pybind 调用样例）
│   ├── test_op.py              #   评测样例（模板只含 case1，5 个 case 由评测系统注入）
│   ├── run.sh                  #   一键跑测 + msprof 采性能
│   ├── get_time.py             #   从 op_summary*.csv 取 AICore 中位耗时
│   ├── setup.py                #   编译安装 custom_ops_lib whl
│   └── extension/custom_op.cpp #   pybind 入口：EXEC_NPU_CMD(aclnnConcat, ...)
├── Greater/  IndexAdd/  Transpose/  SquareSumV1/   # 其余 4 算子的脚手架（结构同 Concat/）
├── build_and_pack.sh           # ★ Concat 一键构建 + 打包 → Concat_0630.zip
└── .gitignore                  # 排除 build_out/ dist/ *.run PROF_*/ 等
```

- `op/CustomOp/` 是**唯一需要改源码**的地方（`op_host/` + `op_kernel/`）。
- `Concat/` 等 `<Op>/` 目录是**测试与性能采集脚手架**，评测系统会替换其中的 `test_op.py`。
- `libcust_opapi.so`（`vendors/customize/op_api/lib/`）在 `libopapi.so` 之前被解析，故自定义 kernel **覆盖内置 aclnn**。量内置 baseline 时需去掉该 `LD_LIBRARY_PATH`。

## 4. 开发循环（改一次源码 → 跑测 → 量性能）

```bash
# 1) 编译算子工程，产出 .run 并安装到 vendors/customize
cd /home/liyc/hw-S9/case_910b/op/CustomOp
bash build.sh                       # → build_out/custom_opp_*.run，会自动安装

# 2) 编译安装 pybind whl（首次或改了 extension/custom_op.cpp 才需要）
cd /home/liyc/hw-S9/case_910b/Concat
bash run.sh 1                       # 传 1 会重建并安装 whl

# 3) 跑某个 case 并采集性能
bash run.sh <N>                     # N∈{1..5}，内部走 msprof，取 op_summary 中位数
```

- `run.sh` 流程：清 `PROF*` → `msprof --application="python3 test_op.py <N>"` → `get_time.py` 解析 `op_summary*.csv` 的 `Task Duration(us)`，**过滤掉 `aclnnMul`**（预热占位算子），取索引 [10,30) 中位数。
- 性能单位 **µs**，5 case 耗时求和为总分。
- 精度阈值：fp16 1‰、fp32 1e-4、int8/int32 无误差。

## 5. 打包提交

```bash
cd /home/liyc/hw-S9/case_910b
bash build_and_pack.sh              # → Concat_0630.zip（内含 Concat_0630_zip/{op_host,op_kernel,custom_opp_*.run}）
```

- **`.run` 必须与提交源码一致**（最后一次上板版本）：打包前 `build.sh` 重新构建。
- 提交结构：`op_host/` + `op_kernel/` + `custom_opp_*.run` 同级目录打 zip。
- 备选：从 `case_910b/` 跑 `bash /home/liyc/hw-S9/zip_op.sh <Name>_zip`（`../op` 解析到 `/home/liyc/hw-S9/op`，需先把 CustomOp 拷到那里）。
- **泛化要求**：tiling 不得针对已知用例定制（否则 0 分）。dtype / dim / 非对齐 / 多输入分派必须通用。

## 6. Concat 算子设计要点与当前状态

### 6.1 数据模型

每个输入 `i` 的连续内存视为 `[beforeDimSize, inputCatLen[i], afterDimSize]`，输出为 `[beforeDimSize, totalCatLen, afterDimSize]`，`afterDimSize` 维内存连续。输入来自 `torch.split` 的 view，已在 `extension/custom_op.cpp` 里 `.contiguous()`，故均为连续。

- `beforeDimSize` = dim 之前各维乘积；`afterDimSize` = dim 之后各维乘积；`totalCatLen` = 各输入 `catLen` 之和；`inputCatOffset[i]` = 前缀和。
- 支持任意 dim、任意维数、4 dtype（fp32/fp16/int32/int8）、最多 64 路输入。

### 6.2 当前实现的问题（待优化）

1. **多余的 UB→UB 拷贝（性能杀手）**：`CopyBatch` 里 GM→UB（`inQueue`）后，又用 `DataCopy(yLocal, xLocalDeq, alignedBytes)` 把数据从 inQueue 搬到 outQueue，再 UB→GM。纯访存算子不需要 Vector 单元参与，这一步是纯开销，且把流水串成 MTE2→V→MTE3。
2. **TILE 过小**：`TILE_BYTES=8192`（8KB），910B UB 192KB（可用 ~184KB）。小 tile → DMA 建链开销大、带宽利用率低。
3. **核切分按 `beforeDimSize` 行**：当 `dim=0`（`beforeDimSize=1`）或 beforeDim 很小时退化为单核/少核，严重欠载。应按**扁平输出字节区间**在 32B 对齐边界切分，与 beforeDim 解耦。
4. **`MAX_AIV_NUM=48` 与 20 核不符**：见第 2 节，应改 20。
5. **无对齐快路径**：全程 `DataCopyPad`（慢），即便完全对齐。应对齐走 `DataCopy`、仅尾部非对齐走 `DataCopyPad`。
6. **Case5 运行失败**：评测报 `Run failed!`（case1-4 通过，case5 未出耗时）。需本地构造覆盖大 shape / dim=0 / 中间 dim / 各 dtype / 非对齐 / 多输入的压力用例复现并定位（见第 8 节排查清单）。

### 6.3 当前评测得分

| Case | 结果 | 耗时(µs) |
|------|------|---------|
| 1 | Pass | 30.032 |
| 2 | Pass | 41.94 |
| 3 | Pass | 116.08 |
| 4 | Pass | 130.832 |
| 5 | **Run failed** | — |

case1-4 合计 ~319 µs（已 < 500），但 case5 失败导致整题无效。修复 case5 并控制其耗时即可达成目标。

## 7. 910B / CANN 8.5 关键约束（写 kernel 前过一遍）

- **UB**：192KB（`__NPU_ARCH__==2201`），可用 ~184KB；按 dtype 精算各 buffer 占用，超 UB 会**运行时崩溃**（非编译期）。
- **AICore**：20/卡，`blockDim ≤ 20`，按数据量自适应。
- **DataCopy 对齐**：count 须 32B 对齐（理想 256B）；非对齐用 `DataCopyPad`。
- **DataCopyExtParams**：首字段 `blockCount` 是 `uint16_t`，**花括号初始化会 narrowing 报错**，用成员赋值。
- **DataCopyPad**：GM→UB 需 4 参（含 `DataCopyPadExtParams<T>`），UB→GM 只需 3 参。
- **tiling 字段**：全用 `uint32_t`（host `TilingDef` 可能插 padding，与 kernel 侧 `#pragma pack(1)` POD 布局一致）；数组字段 host 用局部数组 + `set_xxx(arr)`，kernel 用 `GET_TILING_DATA`（**不要** include host 的 `*_tiling.h`）。
- **dtype 一致性策略**：Concat 是纯搬运，kernel 用 `uint8_t` 视角统一处理所有 dtype，靠 `dtypeSize` 区分字节数即可（无需 dtype 分派）。

## 8. Case5 失败排查清单（按优先级）

1. **UB 越界**：tile / buffer 字节数超 UB 上限（大 tile 或 buffer 计数错）。
2. **`SetBlockDim(48)` 超核数**：极端小 shape 时 `usedCoreNum` 逻辑边界，或大 shape 分波导致超 `run.sh` 的 `timeout 180`（报 `timed out`）。
3. **offset / 字节计算越界**：大 shape 下 `beforeDimSize*totalCatLen*afterDimSize` 量级，检查 host 与 kernel 是否全程 `uint64`。
4. **极端 shape 路径**：1D 输入、`dim=0`、`beforeDimSize=1`、`afterDimSize` 巨大、输入数达上限。
5. **非对齐尾部**：`alignedBytes = (bytes+31)&~31` 对小 tail 的 UB 读写是否越界。

复现方法：在 `Concat/test_op.py` 的 `case_data` 加自定义 case → `bash run.sh <N>` → 看 `PROF_*/` 日志与报错。

## 9. 参考资料位置

- Ascend C 官方实现（910B 真实 dtype/模式支持，比文档可靠）：`/home/liyc/asc-devkit/impl/basic_api/dav_c220/`
- 官方 API 头：`/home/liyc/asc-devkit/include/basic_api/kernel_struct_data_copy.h`（`DataCopyExtParams` 等字段）
- 样例（DataCopyPad 用法等）：`/home/liyc/hw-S9/samples/operator/ascendc/`
- 上层方法论与踩坑：`/home/liyc/hw-S9/AscendC算子开发经验教训.md`、memory `ascendc-910b-gotchas`
- Greater 成功参考实现：`/home/liyc/hw-S9/case_910b/Greater/`（已通过、精度 OK、~6% 超内置，可借流水/对齐快路径/核切分模式）

## 10. 提交前检查清单

- [ ] `ASCEND_COMPUTE_UNIT=ascend910b`、`AddConfig("ascend910b")`？
- [ ] `blockDim ≤ 20`，按数据量自适应（已修 `MAX_AIV_NUM`）？
- [ ] TILE 按 dtype 贴满 UB（精算 buffer，无越界）？
- [ ] 对齐走 `DataCopy`，非对齐尾部走 `DataCopyPad`？
- [ ] 纯搬运路径无多余 UB→UB Vector 拷贝？
- [ ] tiling 字段全 `uint32_t`、数组用 `set_`、kernel 用 `GET_TILING_DATA`？
- [ ] 5 case 全通过 + 精度（fp16 1‰ / fp32 1e-4 / int8·int32 无误差）？
- [ ] 打包前 `.run` 重新构建、与源码一致？tiling 通用无定制？
