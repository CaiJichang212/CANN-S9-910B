# case_910b_Transpose 项目指南（CLAUDE.md）

> 本文件指导在 **dev-transpose-0707 分支** 上使用 Ascend C 开发 **Transpose（torch.permute）** 算子。
> **重要：只能使用本分支（dev-transpose-0707）的代码，禁止引用/合并其他分支（Concat/Greater/IndexAdd/SquareSumV1）的算子实现。** 跨分支参考仅限阅读 `samples` 与 `asc-devkit` 官方资料。
>
> 状态说明（2026-09-01）：本文包含早期从零开发计划，部分“Host/Kernel 待生成”描述
> 已经过期。当前唯一发布源码根为 `Transpose/`；最新导航、官方反馈边界和 Git/存储
> 策略以 [`README.md`](README.md)、[`docs/INDEX.md`](docs/INDEX.md) 和
> [`perf/README.md`](perf/README.md) 为准。

---

## 1. 任务目标

在 Ascend 910B（DAV_2201）上用 Ascend C 实现高性能 Transpose 算子，覆盖全 dtype、全 shape（含非对齐）的通用 permute，精度达标且 AICore 耗时优于基线，最终打包成 `.zip` 提交评分。

- **参考算子**：`torch.permute(input, dims)`（[文档](https://docs.pytorch.org/docs/2.5/generated/torch.permute.html#torch-permute)）
- **本质**：纯数据重排（地址重映射），无计算，**访存带宽受限**，主用 MTE/DMA 搬运。

---

## 2. 赛题规格（Transpose）

| 项 | 规格 |
|----|------|
| **输入** `inputs` | tensor，shape `(..., N5, N4, N3, N2, N)`，最多 5D |
| **属性** `dims` | `list_int`，长度 = 输入维数，是 `[0..ndim-1]` 的一个排列 |
| **输出** `output` | tensor，shape 由 `dims` 重排得到，**dtype 与输入一致**，格式 **ND** |
| **支持 dtype** | `float32, float16, int32, int8`（注意：与 Greater 不同，**不含 bfloat16**） |
| **shape 取值范围** | N∈[1,10000]，N2∈[1,10000]，N3∈[1,1000]，N4∈[1,1000]，N5∈[1,1000] |
| **特征** | N~N5 均可能为**非 32 的整倍数**，必须处理非对齐场景 |

> 模板 `test_op.py` 当前仅 `case1`：`fp16 (128,256)`，`dims=(1,0)`（2D 转置）。开发时须自行扩展 case 覆盖 5D / 各 dtype / 非对齐，**但 tiling 逻辑必须通用，禁止针对已知用例定制化**（否则 0 分）。

---

## 3. 硬件与开发环境

| 项 | 值 |
|----|------|
| **SoC / 计算单元** | `ascend910b`（DAV_2201） |
| **NPU** | 昇腾 910B4-1，**AICore 数 = 20**（`npu-smi info -t common -i 0` 实测） |
| **UB 容量** | 192KB（`__NPU_ARCH__==2201`），可用 ~184KB（`TMP_UB_OFFSET`） |
| **HBM 峰值带宽** | ~1.5 TB/s |
| **CANN** | 社区版 8.5.0（`/usr/local/Ascend/cann-8.5.0`） |
| **OS** | openEuler / Euler（**必须**，否则无法计分） |
| **容器** | `cann850`（**当前已在容器内**，CANN 操作无需再 `docker exec`） |
| **Python / 工具链** | Python 3.11、GCC 12.3.1、`ccec`(clang15)、`msopgen`、`msprof`、`msobjdump` |

详细环境信息见 `/home/liyc/hw-S9/开发环境.md`。

### 910B 关键约束（Transpose 相关）
- **不支持 NDDMA**（高维 DMA 是 950/A3 专属），只能用传统 `DataCopy` + `DataCopyParams`（stride 搬运）。
- **向量 `Transpose` API 仅支持 `int16_t/uint16_t/half`**（见 `impl/basic_api/dav_c220/kernel_operator_vec_transpose_intf_impl.h` 的 `SupportType` assert），**不支持 fp32/int32/int8**。故通用 permute 路径应走 `DataCopy`+`DataCopyParams`，向量 Transpose 仅可作为 fp16 且两维均为 16 整倍时的 2D 特例加速。
- `DataCopy` 对齐要求：count 须 32B（一块）对齐，理想 256B；非对齐用 `DataCopyPad`。
- 多核：`blockDim` 用满 20 核，按数据量自适应 `min(20, ceil(total/合理块))`。

---

## 4. 项目结构与当前状态

```
case_910b_Transpose/                 # 仓库根（git: dev-transpose-0707）
└── Transpose/                       # 算子工程目录
    ├── setup.py                     # pybind 打包（NpuExtension → custom_ops whl）
    ├── run.sh                       # 测试入口：装whl + msprof + get_time.py 判定
    ├── test_op.py                   # 精度测试（EXEC_NPU_CMD 调 aclnnTranspose）
    ├── get_time.py                  # 解析 op_summary*.csv 取 AICore 中位耗时
    ├── extension/custom_op.cpp     # pybind 入口（当前调用内置 aclnnTranspose，待替换为自定义）
    └── common/pytorch_npu_helper.hpp  # EXEC_NPU_CMD / aclTensor 转换（勿改）
```

**当前缺口**：尚无 `op_host/` 与 `op_kernel/`（需用 `msopgen` 生成脚手架后开发）。`custom_op.cpp` 现在跑的是**内置 `aclnnTranspose`**（含一段 `aclnnMul` 占位 warmup，`get_time.py` 会过滤掉 `aclnnMul` 行）。

### 调用与覆盖机制（关键）
- 测试通过 `EXEC_NPU_CMD(aclnnTranspose, input, dims, result)` 调用算子。
- `pytorch_npu_helper.hpp` 中 `GetOpApiFuncAddr` **先解析 `libcust_opapi.so`，再解析 `libopapi.so`**。
- `run.sh` 把 `$ASCEND_OPP_PATH/vendors/customize/op_api/lib/` 前置于 `LD_LIBRARY_PATH`，因此**自定义 kernel 编译安装后即覆盖内置 `aclnnTranspose`**，无需改动 `custom_op.cpp` 的调用代码。
- 自定义算子名注册为 `aclnnTranspose`（用内置 op 名生成工程即可自动同名覆盖）。

---

## 5. 构建与测试流程

### 5.1 生成工程脚手架（msopgen）
```bash
# op.json 放在 /home/liyc 下（不可放 /tmp，会报安全错）
msopgen gen -i /home/liyc/transpose.json -f pytorch \
    -c ai_core-ascend910b -out ./Transpose -lan cpp
```
生成 `build.sh` / `CMakeLists.txt` / `cmake/` / `op_host/` / `op_kernel/`。`op.json` 中算子名用 `"Transpose"`（内置同名，build 自动生成覆盖内置 aclnn）。

### 5.2 必改项（否则编译/部署错误）
1. `CMakePresets.json`：`ASCEND_COMPUTE_UNIT` 默认 `ascend910` → 改 **`ascend910b`**。
2. `op_host/*.cpp`：`AddConfig("ascend910")` → 改 **`"ascend910b"`**。
3. `op_host` 的 `InferShape`：按 `dims` 重排输出 shape（**不是**复制输入 shape）。
4. `InferDataType`：输出 dtype 跟输入（fp32/fp16/int32/int8），**不涉及 bool**。

### 5.3 编译并部署 .run
```bash
cd Transpose && bash build.sh
# 产出 build_out/custom_opp_ubuntu_aarch64.run
bash build_out/custom_opp_ubuntu_aarch64.run   # 安装到 vendors/customize
```

### 5.4 跑精度+性能（pybind 链路）
```bash
cd Transpose
bash run.sh 1   # 1 = 重新编译并安装 custom_ops whl；之后可 bash run.sh <caseN>
```
`run.sh` 流程：装 whl → `msprof --application="python3 test_op.py <case>"` → `get_time.py` 取耗时 → 与 `time_base` 比较（模板里 `time_base=9999999999999`，本地只判非 0；**真实基线由评分系统判定**）。

### 5.5 性能度量（get_time.py 规则）
- 解析 `PROF_*/prof_device_*/summary/op_summary*.csv` 的 `Task Duration(us)`。
- **跳过 `Op Name` 含 `aclnnMul` 的行**（占位 warmup 噪声）。
- 取 `time_use_list[10:30]` 的**中位数**（`test_op.py` 内 `round=30` 次循环）。
- 量内置基线对比时：从 `LD_LIBRARY_PATH` 去掉 `vendors/customize/op_api/lib/`，回退 `libopapi.so`，同条件 msprof。

---

## 6. 精度与性能标准

| dtype | 精度阈值（官方） |
|-------|------------------|
| fp16  | 千分之一（1‰） |
| fp32  | 万分之一（1‱） |
| int8  | 无误差 |
| int32 | 无误差 |

> 模板 `test_op.py` 的 `verify_result`：fp32 用 `rtol=atol=1e-4`，其余（fp16）用 `rtol=atol=1e-2`，`loss=1e-3`（按 `numel*loss` 容忍错误数）。本地通过 ≠ 一定得分（用例随机生成），建议多次自验。

**性能**：`msprof` 统计 AICore 执行时间，**≤ 基线**即通过。排名按各用例耗时总和，前 10 名得分（100/90/.../10）。

---

## 7. Transpose 优化策略（910B 可用）

> NDDMA 不可用 → 基于 `DataCopy` + `DataCopyParams` 实现跨 stride 搬运。

| 策略 | 说明 |
|------|------|
| **DataCopyParams 跨 stride 搬运** | 用 `blockCount`/`blockLen`/`srcStride`/`dstStride` 模拟维度重排，把不连续的输入块搬成连续输出（或反之） |
| **合并不变维度** | 分析 `dims`：前缀/后缀未移动的维度作为 batch 循环跳过，只对真正置换的内核维度做 stride copy |
| **取最长连续段** | 若最内层 `dims[-1]==ndim-1`（未移动），输出最内层连续，按行 copy；若最内层被移动（如 2D `(1,0)`），则按列 stride 抽取 |
| **对齐/非对齐分支** | base 与 n 均 256B 对齐 → `DataCopy`；否则 `DataCopyPad`（`DataCopyExtParams` + `DataCopyPadExtParams<T>`）处理 tail |
| **TILE 贴满 UB** | 按 dtype 精算各 buffer 占用，TILE 取 256 倍数贴满 ~184KB；超 UB 会**运行时崩溃**（非编译期） |
| **多核切分** | blockDim ≤ 20，按数据量自适应，保证负载均衡 |
| **双缓冲流水** | `TQue<..., BUFFER_NUM=2>`，CopyIn/Compute/CopyOut 重叠（纯搬运算子 Compute 极轻，重点是 CopyIn↔CopyOut 流水） |
| **dtype 分派** | fp16 走 half、fp32 走 float、int32/int8 走对应类型；用 `template<typename T>` + `if constexpr` 依赖 T 的分支 |

### 关键代码骨架（思路，非定稿）
```cpp
// 把 permute 归约为「连续输出 ← stride 输入」的 DataCopy
for (uint32_t b = blockIdx; b < totalBlocks; b += blockDim) {
    // 计算 src/dst 起始偏移与每块 stride
    DataCopyParams params;          // blockCount/blockLen/srcStride/dstStride（成员赋值）
    DataCopy(ubBuf, xGm[srcOff], params);   // HBM→UB（带 stride）
    DataCopy(yGm[dstOff], ubBuf, params);   // UB→HBM
}
```

---

## 8. 910B / Ascend C API 陷阱（Transpose 重点）

> 完整踩坑记录见 `/home/liyc/hw-S9/AscendC算子开发经验教训.md`（基于 Greater 沉淀，多数通用）。

1. **`DataCopyExtParams` 首字段 `blockCount` 是 `uint16_t`**：花括号初始化 `{1, n, 0, 0, 0}` 触发 narrowing 编译报错。改用**成员赋值**：
   ```cpp
   DataCopyExtParams p; p.blockCount=...; p.blockLen=n*sizeof(T);
   p.srcStride=0; p.dstStride=0; p.rsv=0;
   DataCopyPadExtParams<T> pad; pad.isPad=true; pad.leftPadding=0;
   pad.rightPadding=0; pad.paddingValue=(T)0;
   ```
   `DataCopyPad`：GM→UB 需 4 参（含 `DataCopyPadExtParams<T>`），UB→GM 只需 3 参。
2. **`DataCopyParams`（对齐路径）字段均为 `uint16_t`**：`blockCount/blockLen/srcStride/dstStride`，stride 单位是 32B 块，注意溢出与单位换算。
3. **向量 `Transpose` API dtype 受限**：仅 `int16/uint16/half`；fp32/int32/int8 不可用，必须走 DataCopy 路径。
4. **if constexpr dtype 分派**：分派函数须做成 `template<typename CT>`，分支条件依赖 `CT`（如 `IsSameType<CT, int32_t>`），否则 discarded 分支仍做语义检查导致编译报错。
5. **tiling 字段统一 `uint32_t`**：host 用 `TILING_DATA_FIELD_DEF_ARR` 生成的 `set_xxx(arr)/get_xxx()`，不能 `[]`；所有字段 4 字节对齐，host 与 kernel `#pragma pack(1)` 布局一致。大尺寸（totalSize）转 uint32（用例 < 4e9 安全）。
6. **kernel 侧用 `GET_TILING_DATA(t, tiling)`**：不要 `#include "xxx_tiling.h"`（host 头不在 kernel include 路径）。入口函数把字段逐个传给 kernel 类 `Init`。
7. **`DataCopy` count 须 32B 对齐**，理想 256B；非对齐 tail 用 `DataCopyPad`。
8. **多核 blockIdx 与 GetBlockIdx**：用 `GetBlockNum()/GetBlockIdx()` 切分，注意尾块处理。

---

## 9. 交付与打包

**交付物**：`op_host/` + `op_kernel/` + `build_out/custom_opp_*.run` 三者，用主办方 `zip_op.sh` 打成 `.zip`。

```bash
# 1) 打包前务必重新 build.sh，确保 .run 与当前源码一致（否则成绩无效）
cd Transpose && bash build.sh
# 2) 按 zip_op.sh 期望的 staging 布局准备（op_dir=../op/${op_name}）
#    将 op_host/ op_kernel/ build_out/custom_*.run 放到 ../op/Transpose_zip/
# 3) 打包
bash /home/liyc/hw-S9/zip_op.sh Transpose_zip
# 产出 Transpose_zip.zip，内含 Transpose_zip/{op_host, op_kernel, custom_*.run}
```
- `.run` 必须是**最后一次上板版本**，与提交源码一致。
- 目录内**勿放无关文件**。
- 打包脚本：`/home/liyc/hw-S9/zip_op.sh`（`op_dir="../op/${op_name}"`，故需从合适目录运行，staging 到 `../op/Transpose_zip/`）。

---

## 10. 开发流程建议

1. **先读规格与环境**：本文件 + `/home/liyc/hw-S9/评分规则.md` + `/home/liyc/hw-S9/开发环境.md` + `/home/liyc/hw-S9/S9挑战赛910B软硬件深度协同优化建议.md`。
2. **查 API 真实签名/dtype 支持**（优先级）：
   1. `/home/liyc/asc-devkit/impl/basic_api/dav_c220/`（910B 实际实现，看 `ASCENDC_ASSERT(SupportType<...>)`）
   2. `/home/liyc/asc-devkit/docs/api/context/`（`DataCopy.md`/`DataCopyPad(ISASI).md`/`Transpose-39.md`）
   3. `/home/liyc/asc-devkit/include/basic_api/kernel_struct_data_copy.h`（结构体字段）
   4. `/home/liyc/hw-S9/samples/operator/ascendc/`（调用模式样例）
3. **msopgen 生成脚手架** → 改 SoC 为 ascend910b → 写 `InferShape`/`InferDataType`/Tiling。
4. **增量验证**：先 fp16 + 2D `(1,0)`（case1）跑通精度，再扩 5D / 4 dtype / 非对齐 / 边界（1 元素、大 shape）。
5. **性能调优**：先 msprof 量当前与内置基线、算带宽利用率，判断 DMA-bound，再 TILE 贴满 UB + 多核 + 双缓冲。
6. **打包前**重 build，确保 .run 一致。

---

## 11. 复用检查清单（开工前过一遍）

- [ ] 分支 = `dev-transpose-0707`，未混入其他分支算子代码？
- [ ] SoC = `ascend910b`（CMakePresets + AddConfig）？
- [ ] 输出 dtype 跟输入（fp32/fp16/int32/int8），InferShape 按 dims 重排？
- [ ] tiling 字段全 `uint32_t`，数组用 `set_`，kernel 用 `GET_TILING_DATA`？
- [ ] dtype 分派函数是 `template<CT>`，分支依赖 CT？
- [ ] `DataCopyExtParams`/`DataCopyParams` 用成员赋值，不花括号？
- [ ] 对齐路径用 `DataCopy`，非对齐用 `DataCopyPad` + 256B 取整？
- [ ] 向量 `Transpose` API 仅用于 half/int16 且两维 16 整倍的特例，通用路径走 DataCopy？
- [ ] TILE 按 dtype 贴满 UB（精算 buffer），blockDim ≤ 20 自适应？
- [ ] 精度测试覆盖 4 dtype + 2D~5D + 非对齐 + 边界？
- [ ] msprof 量 AICore 时间并对比内置 baseline？
- [ ] tiling 通用，无针对已知用例的定制化？
- [ ] 打包前 `.run` 重新构建，与源码一致，`zip_op.sh` 打包？

---

## 12. 关键路径速查

| 用途 | 路径 |
|------|------|
| 仓库根 | `/home/liyc/hw-S9/case_910b_Transpose` |
| 算子工程 | `Transpose/`（op_host/op_kernel 待生成） |
| 赛题 | `/home/liyc/hw-S9/S9挑战性能赛题.md` |
| 评分规则 | `/home/liyc/hw-S9/评分规则.md` |
| 开发环境 | `/home/liyc/hw-S9/开发环境.md` |
| 优化建议 | `/home/liyc/hw-S9/S9挑战赛910B软硬件深度协同优化建议.md` |
| 调用样例说明 | `/home/liyc/hw-S9/调用样例说明.txt` |
| 经验教训 | `/home/liyc/hw-S9/AscendC算子开发经验教训.md` |
| pybind 依赖安装 | `/home/liyc/hw-S9/init_pybind.sh` |
| 打包脚本 | `/home/liyc/hw-S9/zip_op.sh` |
| Ascend C 官方仓 | `/home/liyc/asc-devkit`（impl/`dav_c220` 为 910B 实现基准） |
| DataCopy 结构体 | `/home/liyc/asc-devkit/include/basic_api/kernel_struct_data_copy.h` |
| CANN 工具 | `/usr/local/Ascend/cann-8.5.0`（`bin/msopgen`、`tools/msprof` 等） |
| AddCustom 样例 | `/home/liyc/hw-S9/samples/operator/ascendc/0_introduction/1_add_frameworklaunch` |
