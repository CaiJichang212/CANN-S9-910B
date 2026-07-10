# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> 本文件是本工作目录（`/home/liyc/hw-S9/case_910b_IndexAdd`）的项目级开发指南。算子开发通用方法论与踩坑见上层文档 `/home/liyc/hw-S9/AscendC算子开发经验教训.md` 与 memory，本文件只讲「这个项目怎么用、IndexAdd 怎么做」。

## 1. 项目定位与分支纪律

- **本目录是 git worktree**：主仓 `/home/liyc/hw-S9/case_910b`（`.git` 指向其 `worktrees/case_910b_IndexAdd`）。在此目录提交即提交到 **`dev-index-add-0707`** 分支。
- **唯一任务**：用 **Ascend C** 开发 **IndexAdd** 算子，做到精度通过且 AICore 耗时最小。IndexAdd 是 S9 赛题 5 个算子之一，各算子独立排名。
- **分支纪律（重要）**：**只在本分支开发，不要引入其他分支的代码**。主仓 `case_910b` 当前在 `dev-concat-0630`（Concat），另有 `dev-greater-0703`（Greater）等分支。可参考上层方法论文档与官方实现，但 IndexAdd 的 `op_host/op_kernel` 须在本分支从零编写，不得从其他分支 cherry-pick / 复制其他算子的 kernel 代码。
- **本分支当前状态**：仅含 5 个算子的 **pybind 调用 / 测试脚手架**（`Concat/`、`Greater/`、`IndexAdd/`、`SquareSumV1/`、`Transpose/`，最近一次提交 `8d076a4`）。**Ascend C 算子工程（`op_host/` + `op_kernel/`）尚未创建**，需用 `msopgen` 生成并开发。

## 2. 环境前提（不满足无法计分）

| 项 | 值 |
|----|----|
| CANN | 社区版 **8.5.0**（`/usr/local/Ascend/cann-8.5.0`） |
| SoC / 计算单元 | **ascend910b**（`CMakePresets.json` 的 `ASCEND_COMPUTE_UNIT`、`op_host` 的 `AddConfig("ascend910b")`） |
| 硬件 | 昇腾 910B4-1，**单卡 20 个 AICore**，UB **192KB**（可用 ~184KB） |
| OS | **openEuler / Euler 2.10**（容器 `cann850`，**已在容器内**，无需再 `docker exec`；OS 必须为 Euler 2.10，否则无法计分） |
| 架构 / Python | aarch64 / 3.11 |

> 910B 单卡物理 AICore = **20**。`SetBlockDim` 上限取 20，超过会分波串行（多 wave）反而降速。

## 3. 目录结构与当前状态

```
case_910b_IndexAdd/                # = 本 worktree（dev-index-add-0707）
├── IndexAdd/                     # ★ IndexAdd 测试与性能采集脚手架（pybind 调用样例）
│   ├── test_op.py                #   评测样例（模板仅含 case1，其余 case 由评测系统注入）
│   ├── run.sh                    #   一键跑测 + msprof 采性能（run.sh 1 重建 whl）
│   ├── get_time.py               #   从 op_summary*.csv 取 AICore 中位耗时
│   ├── setup.py                  #   编译安装 custom_ops_lib whl
│   ├── extension/custom_op.cpp   #   pybind 入口：EXEC_NPU_CMD(aclnnIndexAdd, ...)
│   └── common/pytorch_npu_helper.hpp
├── Concat/ Greater/ SquareSumV1/ Transpose/   # 其余 4 算子脚手架（结构同 IndexAdd/）
└── op/                           # ★ 待创建：Ascend C 算子工程
    ├── IndexAddCustom.json        #   算子 IR 定义（msopgen 输入，op 字段可命名 "IndexAddCustom"）
    └── CustomOp/                  #   算子工程（op 名注册为 "IndexAdd" → 生成 aclnnIndexAdd）
        ├── CMakePresets.json      #   ASCEND_COMPUTE_UNIT=ascend910b
        ├── build.sh               #   编译入口 → build_out/custom_opp_*.run（自动安装到 vendors/customize）
        ├── op_host/               #   host 侧：tiling + infershape + op 注册
        │   ├── indexadd.cpp
        │   └── indexadd_tiling.h
        └── op_kernel/             #   kernel 侧：算子核函数
            └── indexadd.cpp
```

- `IndexAdd/` 是**测试与性能采集脚手架**，评测系统会替换/注入其中的 `test_op.py` 真实用例；**唯一需要改源码的是 `op/CustomOp/`**（`op_host/` + `op_kernel/`）。
- `libcust_opapi.so`（`vendors/customize/op_api/lib/`，`run.sh` 把它加到 `LD_LIBRARY_PATH` 最前）在 `libopapi.so` 之前被解析，故自定义 kernel **覆盖内置 `aclnnIndexAdd`**。量内置 baseline 时需去掉该路径。

## 4. IndexAdd 算子规格

参考算子 `torch.index_add`；pybind 入口 `extension/custom_op.cpp` 调用 `EXEC_NPU_CMD(aclnnIndexAdd, input, index, source, dim, result)`（参数顺序：self, index, source, dim, output）。

| 角色 | 参数 | 形状 | dtype | 取值范围 |
|------|------|------|-------|---------|
| INPUT | self | (...,N4,N3,N2,N) ND | float32, bfloat16, float16, int32, int8 | N∈[1,10000], N2∈[1,10000], N3∈[1,1000], N4∈[1,1000] |
| INPUT | index | (M) | int32 | M∈[1,8000] |
| INPUT | source | (...,M,...) | 同 self | source 的 dim 维长度 == M，其余各维 == self 对应维 |
| ATTR | dim | int | — | 默认 0 |
| OUTPUT | output | 同 self | 同 self | — |

**语义**：`output = copy(self)`，再沿 dim 维做 scatter-add：对每个 `i∈[0,M)`，`output[..., index[i], ...] += source[..., i, ...]`。

**数据视图**（设计 kernel 的基础）：把内存按 dim 拆成三段 `[beforeDimSize, dimLen, afterDimSize]`，`afterDimSize` 维内存连续。
- self / output：`[beforeDimSize, dimLen, afterDimSize]`
- source：`[beforeDimSize, M, afterDimSize]`
- index：`[M]`（int32）
- 每个 `(row, i)` 的 scatter 是一段连续的 `afterDimSize` 个元素：`output[row, index[i], :] += source[row, i, :]`

**两大难点**（S9 五算子中最难，⭐⭐⭐⭐⭐）：
1. **随机 / 不规则访存**：`index[i]` 决定写入位置，访问模式完全随机，命中缓存差。
2. **index 可重复**：同一输出位置可能被多次累加，直接写 HBM 会产生 **WAW（写后写）冲突**，必须保证累加而非覆盖。

**必覆盖的泛化场景**（评测用例随机生成）：5 dtype（含 bfloat16）、任意 dim、非 32 对齐、index 重复、1D ~ 5D、大 shape、`dim=0`（beforeDimSize=1）、`afterDimSize` 巨大。

## 5. 开发循环（改一次源码 → 跑测 → 量性能）

```bash
# 0) 生成算子工程脚手架（op 名用内置名 "IndexAdd"，build 后自动生成 aclnnIndexAdd 覆盖内置）
#    输入 JSON 放 /home/liyc 下（放 /tmp 报安全错）
msopgen gen -i op/IndexAddCustom.json -f pytorch -c ai_core-ascend910b -out op/CustomOp -lan cpp

# 1) 改两处 SoC（msopgen 默认是 ascend910，必须改 ascend910b）：
#    - op/CustomOp/CMakePresets.json: ASCEND_COMPUTE_UNIT = "ascend910b"
#    - op/CustomOp/op_host/*.cpp: AddConfig("ascend910b")

# 2) 编译算子工程，产出 .run 并安装到 vendors/customize
cd /home/liyc/hw-S9/case_910b_IndexAdd/op/CustomOp
bash build.sh                       # → build_out/custom_opp_*.run，会自动安装

# 3) 编译安装 pybind whl（首次或改了 extension/custom_op.cpp 才需要，传 1 重建）
cd /home/liyc/hw-S9/case_910b_IndexAdd/IndexAdd
bash run.sh 1                       # 传 1 会重建并安装 whl

# 4) 跑某个 case 并采集性能
bash run.sh <N>                     # N∈{1..5}，内部走 msprof，取 op_summary 中位数
```

- **op 注册名必须是 `IndexAdd`**（`class IndexAdd : public OpDef` + `OP_ADD(IndexAdd)`），这样 build 产出 `aclnnIndexAdd`，覆盖 `torch_npu` 内置同名算子。msopgen 的 JSON `op` 字段可叫 `IndexAddCustom`（仅工程名），不影响。
- `run.sh` 流程：清 `PROF*` → `timeout 180 msprof --application="python3 test_op.py <N>"` → `get_time.py` 解析 `op_summary*.csv` 的 `Task Duration(us)`，**过滤掉 `aclnnMul`**（预热占位算子，见 `extension/custom_op.cpp` 里的 30 轮循环），取索引 [10,30) 中位数。
- 性能单位 **µs**，pass 标准 = AICore 时间 ≤ 基线。
- `test_op.py` 的 `case_data` **只有 case1**（int8, [32,128], dim=0）作为模板，case2-5 由评测系统注入。**本地自测时务必在 `case_data` 自行添加覆盖全 dtype / dim / 非对齐 / index 重复 的用例**（建议多次自验，赛题明示「测试用例数据随机生成，无法确保自验证通过用例一定得分」）。
- **量内置 baseline**：从 `LD_LIBRARY_PATH` 去掉 `vendors/customize/op_api/lib/`，回退到 `libopapi.so` 的内置 `aclnnIndexAdd`，同条件对比才公平。

## 6. IndexAdd 设计与优化要点

**两段式 kernel**：
1. **bulk copy self → output**：output 是 `empty_like(self)` 的全新 buffer，必须整体填充。可沿用纯访存算子的扁平字节区切分（按输出总字节均匀切给 ≤20 核，每核写互不重叠的 32B 区间），对齐走 `DataCopy`、尾部非对齐走 `DataCopyPad`，单队列双缓冲 `TQue<..., BUFFER_NUM=2>` 让 CopyIn(MTE2) 与 CopyOut(MTE3) 流水重叠。
2. **scatter-add**：对每个 `(row, i)`，把 source 的一段连续 `afterDimSize` 元素加到 `output[row, index[i], :]`（随机写）。

**重复 index 处理（核心）**——任选其一或组合：
- **排序 + UB 内累加合并再单次写回**（推荐）：将 `(index, source_slice)` 按 index 排序，同 index 的 source 先在 UB 中累加，再对每个输出位置**只写一次**，避免对同一 HBM 地址的多次读写与 WAW 冲突。
- **按 index 值域分桶**：把 index 值域切成多个桶，每桶在独立 UB 块内处理，减少排序开销。
- **原子加回退**：若场景允许，可直接对 HBM 原子加，逻辑简单但性能较低（910B 通用类型的 HBM 原子操作能力需查文档确认，谨慎）。

**性能调优杠杆（按收益排序）**：
1. TILE 贴满 UB（192KB / 可用 ~184KB）：按 dtype 精算各 buffer 占用（含 source tile、index tile、累加 tile、work buffer），双缓冲下 `2×TILE ≤ 184KB`，取 256 的倍数。
2. 对齐快路径 `DataCopy` + 仅尾部 `DataCopyPad`，全程 `DataCopyPad` 有额外开销。
3. `blockDim` 用满 20 核，按数据量自适应 `min(20, ceil(total/阈值))`。
4. 双缓冲流水。
5. scatter 路径尽量把连续 `afterDimSize` 向量成段处理，减少散粒度随机写。

**dtype 分派注意**：`Add` 支持 half/float/int16/int32；**int8 / bfloat16 可能需要 `Cast`**（bf16→float、int8→half 做加法再视情况转回），具体支持矩阵以 `/home/liyc/asc-devkit/impl/basic_api/dav_c220/` 的 `ASCENDC_ASSERT(SupportType<...>)` 为准（比文档可靠）。bf16 标量广播有 MTE2→V 竞态坑（见经验教训 §2.6），优先全 Vector 路径。

## 7. 910B / CANN 8.5 关键约束（写 kernel 前过一遍）

- **UB**：192KB（`__NPU_ARCH__==2201`），可用 ~184KB；超 UB 会**运行时崩溃**（非编译期）。按 dtype 精算 buffer。
- **AICore**：20/卡，`blockDim ≤ 20`。
- **DataCopy 对齐**：count 须 32B 对齐（理想 256B）；非对齐用 `DataCopyPad`。`DataCopyPad`：GM→UB 需 4 参（含 `DataCopyPadExtParams<T>`），UB→GM 只需 3 参。
- **`DataCopyExtParams`**：首字段 `blockCount` 是 `uint16_t`，**花括号初始化会 narrowing 报错**，用成员赋值。
- **Compare**：int32 **仅支持 `CMPMODE::EQ`**；int8/bf16 不能直接作 src（需 Cast）。
- **Select（bitmask 版）**：语义是 `dst = bit?src0:src1`（bit 置位→src0），**与文档另一个重载相反**，实测确认；标量重载语义又不同且无性能收益，慎用。
- **`Max`/`Min`/`Duplicate`** 的 count 是 `const int32_t&`（`Compare`/`Select`/`Cast` 是 `uint32_t`）；统一用 `uint32_t`，在 `Max`/`Duplicate` 处 `static_cast<int32_t>()`。`Duplicate` 不支持 int8/uint8。
- **dtype 分派用 `if constexpr`**：分派函数须做成 `template<typename CT>`，分支条件用 `IsSameType<CT, ...>`（依赖 CT），否则 discarded 分支仍做语义检查而报错。
- **tiling 字段全用 `uint32_t`**（host `TilingDef` 可能插 padding，与 kernel 侧 `#pragma pack(1)` POD 布局一致）；数组字段 host 用局部数组 + `tiling.set_xxx(arr)`，kernel 用 `GET_TILING_DATA(t, tiling)`（**不要** include host 的 `*_tiling.h`）。
- **`Compare`/`Cast` 的 count 须 256B 对齐**（`count*sizeof(T) % 256 == 0`），否则静默漏算尾部。
- 大 shape 下字节/偏移计算全程 `uint64` 防溢出。

## 8. 精度与性能

- **精度标准（赛题权威）**：fp16 千分之一、fp32 万分之一、int8 / int32 **无误差**。`test_op.py` 模板里的 `verify_result` 对非 fp32 一律用 1e-3（偏松），仅供模板参考，**以赛题标准为准**。
- **性能**：`run.sh` 走 msprof，`get_time.py` 取 `op_summary` 中位数（µs）；pass = AICore 时间 ≤ 基线。
- 调优前先量 baseline（当前实现 + 内置实现），算带宽利用率，判断 DMA-bound 还是 vec-bound，再对症。

## 9. 打包提交

- **交付物**：`op_host/` + `op_kernel/` + `custom_opp_*.run` 同级目录打 zip。**必须用主办方提供的 `zip_op.sh`**，否则成绩无效。
- 从本目录运行：`bash /home/liyc/hw-S9/zip_op.sh <name>`。脚本从 `../op/<name>/` 读取 `op_host/`、`op_kernel/`、`build_out/custom_opp_*.run`，staging 到 `./<name>_zip/` 后打 `<name>.zip`（`../op` 解析到 `/home/liyc/hw-S9/op`，**需先把算子工程放到 `/home/liyc/hw-S9/op/<name>/` 或拷贝过去**）。
- **`.run` 必须与提交源码一致**（最后一次上板版本）：打包前重新 `bash build.sh`，确保 `.run` 来自当前 `op_host/op_kernel`；不一致则成绩无效。
- **泛化要求**：tiling 不得针对已公布用例定制（否则 0 分）。dtype / dim / 非对齐 / index 重复 分派必须通用，不能 hardcode shape。
- 建议加 `.gitignore` 排除 `build_out/`、`dist/`、`*.run`、`PROF_*/`、`__pycache__/`（本分支当前无 `.gitignore`，可参考 `/home/liyc/hw-S9/case_910b/.gitignore`）。

## 10. 参考资料位置

- **Ascend C 官方实现（910B 真实 dtype/模式支持，比文档可靠）**：`/home/liyc/asc-devkit/impl/basic_api/dav_c220/`
- 官方 API 头（`DataCopyExtParams` 等字段）：`/home/liyc/asc-devkit/include/basic_api/kernel_struct_data_copy.h`
- 官方文档：`/home/liyc/asc-devkit/docs/api/context/`（面向多架构，910B 以 dav_c220 impl 为准）
- 样例（`DataCopyPad` / scatter 用法等）：`/home/liyc/hw-S9/samples/operator/ascendc/`
- **方法论与踩坑**：`/home/liyc/hw-S9/AscendC算子开发经验教训.md`（Greater 全流程沉淀，含可复用检查清单与 API 速查表）
- **IndexAdd 优化策略**：`/home/liyc/hw-S9/S9挑战赛910B软硬件深度协同优化建议.md` §3（排序合并 / 分桶 / 原子加回退）
- 赛题 / 规则 / 环境：`/home/liyc/hw-S9/S9挑战性能赛题.md`、`评分规则.md`、`开发环境.md`、`调用样例说明.txt`
- CANNBot 工作流（首次响应须加载）：`/home/liyc/AGENTS.md` → `/ops-registry-invoke-workflow`

## 11. 提交前检查清单

- [ ] `ASCEND_COMPUTE_UNIT=ascend910b`、`AddConfig("ascend910b")`？
- [ ] op 注册名为 `IndexAdd`（产出 `aclnnIndexAdd` 覆盖内置）？
- [ ] `blockDim ≤ 20`，按数据量自适应？
- [ ] TILE 按 dtype 贴满 UB（精算 buffer，无越界，2×TILE ≤ 184KB）？
- [ ] bulk copy self→output 覆盖全输出；对齐 `DataCopy`、尾部 `DataCopyPad`？
- [ ] **重复 index 正确累加**（排序合并 / 分桶），无 WAW 丢值？
- [ ] dtype 分派为 `template<CT>` + `if constexpr`；int8/bf16 的 Cast 路径查过 dav_c220？
- [ ] tiling 字段全 `uint32_t`、数组用 `set_`、kernel 用 `GET_TILING_DATA`？
- [ ] `DataCopyExtParams` 用成员赋值；`Max`/`Duplicate` count 加 `int32_t` cast？
- [ ] 泛化：不 hardcode shape/dim/dtype；tiling 对未知用例通用？
- [ ] 精度自测覆盖 5 dtype + 非对齐 + index 重复 + 各 dim + 边界（1 元素 / 大 shape）？
- [ ] msprof 量 AICore 时间并对比内置 baseline，≤ 基线？
- [ ] 打包前 `.run` 重新构建、与源码一致；用 `zip_op.sh` 打包？
