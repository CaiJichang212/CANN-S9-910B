# Transpose 算子开发交接文档（给 Coding Agent）

> **本文档用于指导在新 910B 服务器（NPU 正常可用）上继续 Transpose 算子开发。**
> 项目：`case_910b_Transpose`（分支 `dev-transpose-0707`）
> 工作目录：`~/case_910b_Transpose/Transpose`
> 目标平台：Ascend 910B（DAV_2201），CANN 8.5.0，20 AICore
> 最后更新：2026-07-10

---

## 〇、给 Agent 的快速指令（先读这段）

**当前算子已开发到「代码完成、可编译、未经上板验证」阶段。** 之前因旧服务器 NPU 设备卡死（崩溃内核遗留 AICore hang），无法跑精度/性能测试。新服务器 NPU 正常，你的首要任务是：

1. **搭好环境**（见第三节）→ **build + 装包** → **跑 `run_all.sh` 验证 12 个 case 的精度**。
2. 根据精度结果**修 bug**（大概率有几处，详见第五节「待验证/高风险点」）。
3. 精度全过后，**msprof 量性能对比内置 baseline**，优化慢路径（非 half 2D 转置、任意 permute）。
4. 达标后**打包 zip 提交**。

**绝对约束**：只能用 `dev-transpose-0707` 分支代码，禁止合并其他分支（Concat/Greater/IndexAdd/SquareSumV1）的算子实现。tiling 必须通用，**禁止针对已知 case 定制化**（否则 0 分）。

---

## 一、当前开发阶段

### 已完成
| 模块 | 状态 | 文件 |
|------|------|------|
| Host（TilingFunc/InferShape/InferDataType/OpDef） | ✅ 完成 | `op_host/transpose.cpp`、`op_host/transpose_tiling.h` |
| Kernel（三路径搬运） | ✅ 完成（**待上板验证**） | `op_kernel/transpose.cpp` |
| 构建 | ✅ 全 4 dtype 变体编译通过，`.run` 可生成 | `build.sh` |
| 测试用例 | ✅ 12 个 case（1→12，覆盖 4 dtype/2D-5D/非对齐/边界/任意 permute） | `test_op.py` |
| 批量自测脚本 | ✅ 新增 | `run_all.sh` |
| SoC 配置 | ✅ `ascend910b`（CMakePresets + AddConfig） | — |

### 未完成（你的任务）
- ❌ **上板精度验证**（核心，之前设备卡死没跑成）
- ❌ **修精度 bug**（有几处高风险点，详见第五节）
- ❌ **性能对比基线 + 优化慢路径**
- ❌ **打包提交**（`zip_op.sh`）

---

## 二、内核设计（必须读懂才能继续）

任意 permute 被归约为「**按输出行搬运**」：
- `W` = 输出末维 = `inShape[dims[ndim-1]]`
- `S` = 源步长 = `inStride[dims[ndim-1]]`（元素）
- 外层输出维（0..ndim-2）每个组合对应输出一行，源基址由 `DecodeRow(row)` 混合基解码得到。

### 三条路径（host 在 `TilingFunc` 里按 dims 选 mode）

| mode | 路径 | host 触发条件 | kernel 实现 | 性能 |
|------|------|--------------|-------------|------|
| 0 | **COPY 连续** (`ProcessCopy`→`ProcessCopyRow`→`CopyTileContiguous`) | `S==1`（dims 末位 identity，如 `(1,0,2)`、identity permute） | 连续 GM→UB→GM，UB 贴满 80KB | 高效 ✅ |
| 0 | **COPY 步长兜底** (`CopyTileStrided`) | `S>1` 且非末两维交换（任意 permute，如 case11 `(1,2,0)`、case12 `(2,0,1)`） | 逐元素 `xGm.GetValue(步长S)`→`ub.SetValue`→连续写 | **正确但慢** ⚠️ |
| 1 | **TRANSPOSE** (`ProcessTranspose`→`TransposeBlk`→`TransposeUB`) | 末两维相邻交换 + 前缀 identity（`dims=[0..,n-1,n-2]`，2D 转置，可带 batch） | 16×16 分块：紧凑读→**half 硬件 `Transpose()`(vtranspose)** / 其余 dtype 逐元素 `TransposeGeneric`→紧凑写 | half 高效 ✅；其余 **正确但慢** ⚠️ |

### 关键设计决策
- **half 走 vtranspose**：host 固定 `tileM=tileN=16`。因 16 个 half=32B=1 dataBlock，GM→UB 读时 `dstStride=0` 天然紧凑成连续 16×16，满足 `Transpose(dst,src)` 的输入要求。用 `if constexpr (sizeof(DTYPE_X)==sizeof(half))` 守卫，非 half 不实例化该调用。
- **其余 dtype 走 `TransposeGeneric`**：逐元素 `GetValue`/`SetValue`，dtype 无关、可证明正确，但慢。这是**已知性能债**，精度过后必须优化（见第六节）。
- **非 half 的 tile 更大**：host 给 fp32/int32/int8 用 16 倍数大 tile（贴满 64KB），摊薄逐元素开销。

---

## 三、环境搭建（新服务器首次必做）

### 3.1 Python 依赖（构建链 + 运行）
```bash
pip3 install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip3 install torch_npu==2.5.1
pip3 install numpy==1.24.0 pybind11==2.13.1 expecttest pyyaml decorator scipy attrs psutil
```
> **坑**：`opc`（算子编译器）依赖 decorator/scipy/attrs/psutil/pyyaml，缺哪个就报哪个，层层嵌套。一次装齐。
> **坑**：whl 构建（`setup.py` import torch_npu）会触发设备初始化，若设备异常设 `export TORCH_DEVICE_BACKEND_AUTOLOAD=0` 绕过。

### 3.2 确认设备正常
```bash
npu-smi info                      # 应正常显示 20 AICore，无 "device is used"
python3 -c "import torch,torch_npu; print(torch.npu.is_available(), torch.npu.device_count())"
```
> 若报 `drvRet=87 / device is used (-8020)`，是设备被崩溃内核污染，需 `npu-smi set -t device-reset -i 0` 或重启。

---

## 四、标准构建/测试流程

```bash
cd ~/case_910b_Transpose/Transpose

# 1) 构建内核（产出 .run）—— 改了 op_kernel/op_host 后必做
bash build.sh
# 产出 build_out/custom_opp_openEuler_aarch64.run

# 2) 安装自定义算子（覆盖内置 aclnnTranspose）
bash build_out/custom_opp_openEuler_aarch64.run

# 3) 构建 pybind whl（改了 extension/custom_op.cpp 才需重做）
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python3 setup.py build bdist_wheel
pip3 install dist/custom_ops*.whl --force-reinstall

# 4) 批量精度验证（12 个 case）
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
bash run_all.sh

# 5) 单 case 精度+性能（msprof）
bash run.sh 1      # caseN：1=重新装whl；之后可 bash run.sh <N>
```

### 性能度量规则（`get_time.py`）
- 解析 `PROF_*/prof_device_*/summary/op_summary*.csv` 的 `Task Duration(us)`
- **跳过 `Op Name` 含 `aclnnMul` 的行**（占位 warmup 噪声）
- 取 `time_use_list[10:30]` 中位数（`test_op.py` 内 `round=30` 次循环）
- `run.sh` 的 `time_base=9999999999999`（本地只判非 0，**真实基线由评分系统判定**）

### 量内置基线对比
从 `LD_LIBRARY_PATH` 去掉 `vendors/customize/op_api/lib/`，回退 `libopapi.so`，同条件 msprof。

---

## 五、待验证 / 高风险点（上板后优先排查）

> 这些是**逻辑层面已手推验证、但未经上板实测**的点，按崩溃/错误概率排序。上板后若某 case 崩溃或出错，**优先查这里**。

### 🔴 高风险 1：half vtranspose 路径（case1/5/9，主路径）
- `Transpose(dst, src)` 在 `TransposeBlk` 里被调用。源 `src` 从 `srcQue(VECIN)` DeQue，目的 `dstUB` 从 `dstQue(VECOUT)` AllocTensor。
- **风险点**：vtranspose 对 UB 布局/对齐有隐含要求。当前依赖「16 half=32B=1block，dstStride=0 天然紧凑」。若 vtranspose 仍崩，检查：
  - `srcQue`/`dstQue` 的 `InitBuffer` 大小是否足够（当前 `ubElems=max(srcBytes,dstBytes)/dtypeSize`，half 时 = 256 元素=512B/buffer）。
  - 是否需要显式 `SetFlag/WaitFlag` 同步（当前靠 TQue 的 EnQue/DeQue 自动同步 MTE2→V→MTE3）。
  - 尾块（mh<16 或 nw<16）时 vtranspose 读到 UB 未初始化区（垃圾值），但输出只写有效区——逻辑上安全，**但需实测确认不崩**。

### 🟡 中风险 2：非 half 2D 转置逐元素路径（case2/3/4/6/7）
- `TransposeGeneric` 用 `GetValue`/`SetValue` 逐元素，从 `src(VECIN)` 读、写 `dstUB(VECOUT)`。
- **风险点**：LocalTensor 的 `GetValue`/`SetValue` 在 S pipe，与 EnQue/DeQue 的跨 pipe 同步可能需显式 event。若结果错乱或崩溃，考虑加 `SetFlag<HardEvent::V_S>`/`WaitFlag` 或改用 UB→UB `DataCopy(DataCopyParams)`。

### 🟡 中风险 3：任意 permute 步长兜底（case11/12）
- `CopyTileStrided` 用 `xGm.GetValue()`（GM 直接读，AICPU 往返）逐元素。慢但应正确。
- **风险点**：`GetValue` 对 int8/fp32 的类型转换；`blockCount≤4095` 的 tile 切分（已 clamp 到 4095）。

### 🟢 低风险 4：COPY 连续路径（case8 identity、S==1 的 permute）
- `CopyTileContiguous` 是标准 DataCopyPad 连续搬运，最稳。

### 关键 API 约束速查（踩过的坑，别再踩）
| 约束 | 说明 |
|------|------|
| **`Transpose()` 仅 half/int16/uint16** | 见 `impl/basic_api/dav_c220/kernel_operator_vec_transpose_intf_impl.h` 的 `SupportType` assert。release 构建断言被编译掉（`#if ASCENDC_CPU_DEBUG`），非 half 不报错但产出乱数据。已用 `if constexpr` 守卫。 |
| **DataCopyPad stride 单位** | GM 侧=**字节**；UB 侧(VECIN/VECOUT)=**32B dataBlock（必须 32B 整倍）**。语义=相邻块 GAP（前块尾到后块头）。 |
| **blockCount(uint16_t) ∈ [1,4095]** | 不是 65535！大 tile 要切分。 |
| **ccec 无 `<algorithm>`** | 禁用 `std::max/std::min`，用三元运算符（host 侧 `op_host` 可用 std）。 |
| **DataCopyExtParams 用成员赋值** | 花括号 `{1,n,0,0,0}` 触发 narrowing。`p.blockCount=...; p.blockLen=...; ...` |
| **DataCopyPad 参数数** | GM→UB 需 4 参（含 `DataCopyPadExtParams<T>`）；UB→GM 只需 3 参。 |
| **tiling 字段全 uint32_t** | host 用 `set_xxx(arr)/get_xxx()`，kernel 用 `GET_TILING_DATA(t,tiling)`，不 include host 头。 |
| **910B 无 NDDMA** | 只能 `DataCopy`+`DataCopyParams`/`DataCopyPad` stride 搬运，无高维 DMA。 |

---

## 六、待优化（精度过后）

### 6.1 非 half 2D 转置性能（case2/3/4/6/7）
当前 `TransposeGeneric` 逐元素，慢。优化方向：
- **UB→UB `DataCopy(DataCopyParams)` 块转置**：fp32/int32 按 8 元素=1 block 做 8×8 块转置；int8 按 32 元素=1 block。
- 或参考内置 `aclnnTranspose` 的 `v35/transpose_*.h`（`tensor_move`/`n_last`/`cut_one_axis` 路径），位于 `/usr/local/Ascend/cann-8.5.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/ascendc/transpose/`。
- 或用 `scatter_vnchwconv` 指令（内置 Transpose/TransDataTo5HD 底层用它）。

### 6.2 任意 permute 步长性能（case11/12）
`CopyTileStrided` 逐元素 `GetValue` 慢。可改用 `DataCopyPad` blockCount=curLen, blockLen=dtypeSize, srcStride=(S-1)*dtypeSize 一次 strided 读多元素（注意 UB dstStride 单位是 32B block，布局会稀疏，需配套写出）。

### 6.3 多核负载均衡
mode=1 的 `blockDim` 当前由 `numRows`（COPY 几何）推算，应改用 `totalTiles` 推算更好均衡。见 `op_host/transpose.cpp:118`。

### 6.4 TILE/双缓冲
- COPY 路径 `copyTileLen` 贴满 80KB（已做）。
- TRANSPOSE 路径 half 固定 16×16（小 buffer），可考虑更大 tile（如 16×256）循环 16×16 子块，提升 GM 读复用。

---

## 七、上下文信息（Agent 必须知道的）

### 7.1 关键路径
| 用途 | 路径 |
|------|------|
| 算子工程 | `~/case_910b_Transpose/Transpose` |
| 项目指南（赛题/规格/约束/打包） | `~/case_910b_Transpose/CLAUDE.md`（**必读**） |
| 经验教训（Greater 沉淀，通用） | `~/AscendC算子开发经验教训.md` |
| 优化建议 | `~/S9挑战赛910B软硬件深度协同优化建议.md` |
| 评分规则 | `~/评分规则.md` |
| 打包脚本 | `~/zip_op.sh` |
| Ascend C 官方仓 | `/home/liyc/asc-devkit`（`impl/basic_api/dav_c220/` 为 910B 实现基准） |
| DataCopyPad 文档 | `/home/liyc/asc-devkit/docs/api/context/DataCopyPad(ISASI).md` |
| DataCopy 结构体 | `/home/liyc/asc-devkit/include/basic_api/kernel_struct_data_copy.h` |
| vtranspose 实现 | `/usr/local/Ascend/cann-8.5.0/aarch64-linux/asc/impl/basic_api/dav_c220/kernel_operator_vec_transpose_impl.h` |
| 内置 transpose 参考 | `/usr/local/Ascend/cann-8.5.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/ascendc/transpose/` |
| CANN 工具 | `/usr/local/Ascend/cann-8.5.0`（`bin/msopgen`、`tools/msprof`） |
| AddCustom 样例 | `~/samples/operator/ascendc/0_introduction/1_add_frameworklaunch` |

### 7.2 调用与覆盖机制
- 测试通过 `EXEC_NPU_CMD(aclnnTranspose, input, dims, result)` 调用。
- `pytorch_npu_helper.hpp` 的 `GetOpApiFuncAddr` **先解析 `libcust_opapi.so`（自定义），再 `libopapi.so`（内置）**。
- `run.sh` 把 `vendors/customize/op_api/lib/` 前置 `LD_LIBRARY_PATH` → 自定义 kernel 安装后**自动覆盖内置 aclnnTranspose**，无需改 `custom_op.cpp`。

### 7.3 赛题规格
- dtype：fp16/fp32/int32/int8（**不含 bfloat16**）
- shape：最多 5D，N/N2∈[1,10000]，N3/N4/N5∈[1,1000]，**可能非 32 整倍数**
- 精度阈值：fp16 千分之一、fp32 万分之一、int8/int32 零误差

### 7.4 测试用例清单（test_op.py）
| case | dtype | shape | dims | 路径 |
|------|-------|-------|------|------|
| 1 | fp16 | (128,256) | (1,0) | TRANSPOSE-vtranspose |
| 2 | fp32 | (37,53) | (1,0) | TRANSPOSE-generic |
| 3 | int8 | (100,33) | (1,0) | TRANSPOSE-generic |
| 4 | int32 | (256,512) | (1,0) | TRANSPOSE-generic |
| 5 | fp16 | (12,64,97) | (0,2,1) | TRANSPOSE-vtranspose+batch |
| 6 | fp32 | (3,5,37,53) | (0,1,3,2) | TRANSPOSE-generic+batch |
| 7 | int8 | (2,3,4,17,9) | (0,1,2,4,3) | TRANSPOSE-generic+batch |
| 8 | fp16 | (64,128) | (0,1) | COPY(identity) |
| 9 | fp16 | (33,100) | (1,0) | TRANSPOSE-vtranspose |
| 10 | fp16 | (1,1) | (1,0) | TRANSPOSE(边界) |
| 11 | fp16 | (7,11,13) | (1,2,0) | COPY-strided(任意permute) |
| 12 | fp32 | (2,19,23) | (2,0,1) | COPY-strided(任意permute) |

---

## 八、已解决的关键问题（历史记录，避免重犯）

1. **崩溃根因**：原内核对所有 dtype 调 `Transpose()`(vtranspose)，仅 half 支持 → 用 `if constexpr` 守卫。
2. **stride 单位错**：UB 侧错把字节当 32B block → 按 `DataCopyPad2D` 样例重写。
3. **batch 循环丢失**：`ProcessTranspose` 漏前缀 batch 维 → `totalTiles=transBatch*nTiles*mTiles`，按平铺索引解 `(b,ti,tj)`，`matBase=b*transM*transN`。
4. **S>1 任意 permute 兜底缺失**：host 设 mode=0 但 kernel 按连续读 → COPY 路径按 `S_==1/S_>1` 分流。
5. **`std::max` 不可用**：ccec 无 `<algorithm>` → 三元运算符。
6. **构建依赖链**：decorator/scipy/attrs/psutil/pyyaml 逐个补齐。
7. **whl 构建设备初始化**：`TORCH_DEVICE_BACKEND_AUTOLOAD=0` 绕过。

---

## 九、验收检查清单（提交前过一遍）

- [ ] `run_all.sh` 12 case 精度全过
- [ ] msprof 量 AICore 时间 ≤ 内置 baseline（各 case）
- [ ] 分支 = `dev-transpose-0707`，无其他分支代码混入
- [ ] SoC = `ascend910b`，dtype 跟输入，InferShape 按 dims 重排
- [ ] tiling 字段全 uint32_t，kernel 用 `GET_TILING_DATA`
- [ ] tiling 通用，无针对已知 case 定制化
- [ ] TILE 按 dtype 贴满 UB（精算 buffer，不超 184KB）
- [ ] blockDim ≤ 20 自适应
- [ ] 打包前重新 `bash build.sh`，`.run` 与源码一致
- [ ] `zip_op.sh` 打包，目录无无关文件

---

## 十、建议执行顺序

1. 搭环境（第三节）→ 确认 `npu-smi` + torch_npu 正常
2. `bash build.sh && bash build_out/custom_opp_*.run` → 装自定义算子
3. 构建 whl + `bash run_all.sh` → 看哪些 case 过/不过
4. **修精度 bug**：按第五节高风险点排查（half vtranspose > 非 half 逐元素 > 步长兜底）
5. 精度全过 → msprof 量性能 + 对比内置 baseline
6. 优化慢路径（第六节：非 half 转置 > 任意 permute > 多核均衡）
7. 达标 → `bash build.sh`（确保一致）→ `zip_op.sh` 打包提交
