# Transpose 算子开发总结

> 项目：`case_910b_Transpose`（分支 `dev-transpose-0707`，工作树 `/home/liyc/hw-S9/case_910b_Transpose`）
> 目标平台：Ascend 910B（DAV_2201），CANN 8.5.0，20 AICore
> 算子：Transpose（`torch.permute`），支持 fp16/fp32/int32/int8、2D~5D、非对齐 shape
> 文档日期：2026-07-10

---

## 一、算子实现进度

### 当前阶段：内核代码完成、可编译安装，**精度/性能上板验证受阻于设备卡死**

| 模块 | 状态 | 说明 |
|------|------|------|
| `op_host/transpose.cpp`（TilingFunc/InferShape/InferDataType/OpDef） | ✅ 完成 | 按 dims 重排输出 shape、dtype 跟输入、SoC=ascend910b、三路径模式选择 |
| `op_host/transpose_tiling.h` | ✅ 完成 | 三路径统一 tiling 字段（uint32_t，数组用 set_） |
| `op_kernel/transpose.cpp` | ✅ 完成（待上板验证） | 三路径搬运，half 走 vtranspose |
| 构建（`build.sh`） | ✅ 通过 | 全 4 个 dtype 变体编译通过，`.run` 生成 |
| 安装（custom op） | ✅ 完成 | `libcust_opapi.so` 安装到 `vendors/customize`，覆盖内置 `aclnnTranspose` |
| pybind whl（`setup.py`） | ✅ 完成 | `dist/custom_ops-1.0-cp311-...whl` 已构建（需 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`） |
| 测试用例（`test_op.py`） | ✅ 完成 | 1 → 12 个 case（4 dtype × 2D~5D × 非对齐 × 边界 × 任意 permute） |
| 精度上板验证 | ❌ 阻塞 | NPU 设备卡死，无法跑 |
| 性能（msprof）对比基线 | ❌ 阻塞 | 同上 |
| 打包（`zip_op.sh`） | ⬜ 待办 | 上板验证通过后做（需重新 build 确保一致） |

### 三条搬运路径（内核）

| mode | 路径 | 触发条件 | 实现 | 性能 |
|------|------|---------|------|------|
| 0 | **COPY 连续** | 输出末维源端连续（S==1，如 dims 末位 identity） | 连续 GM→UB→GM，UB 贴满 | 高效 |
| 0 | **COPY 步长兜底** | 任意 permute 且 S>1（如 case11 `(1,2,0)`） | 逐元素 `GetValue` 步长读 + 连续写 | 正确，偏慢 |
| 1 | **TRANSPOSE** | 末两维交换 + 前缀 identity（2D 转置） | 16×16 分块：紧凑读 → half 硬件 vtranspose / 其余逐元素 UB 转置 → 连续写 | half 高效；其余正确但慢 |

---

## 二、遇到的问题与解决方案

### 问题 1：内核崩溃（case1 128×256 fp16 2D 转置）

- **现象**：上板即崩，`exception_info` dump 显示 UB 垃圾值（`0xa5a5a5a5` 填充模式），`vendors/customize` 为空（未成功跑完）。
- **根因**：原内核在 `TransposeBlk16` 中**对全部 dtype 无条件调用 `Transpose(dst, src)`（vtranspose 指令）**。但查阅 `impl/basic_api/dav_c220/kernel_operator_vec_transpose_intf_impl.h`，该 API 的 `ASCENDC_ASSERT(SupportType<PrimT<T>, int16_t, uint16_t, half>())` 表明**仅 half/int16/uint16 支持**。release 构建下断言被编译掉（`#if ASCENDC_CPU_DEBUG`），fp32/int32/int8 会经 uint16 reinterpret 产出乱数据甚至越界崩溃。
- **解决**：
  1. 用 `if constexpr (sizeof(DTYPE_X) == sizeof(half))` 守卫 `Transpose()` 调用，非 half 走 `TransposeGeneric`（逐元素，dtype 无关）。
  2. half 路径：host 固定 `tileM=tileN=16`，使 16×16 块天然紧凑（16 个 half = 32B = 1 dataBlock，dstStride=0），满足 vtranspose 的「连续 16×16」要求。
  3. 修正 UB 侧 stride 单位（见问题 4）。

### 问题 2：环境缺失 torch/torch_npu

- **现象**：`import torch` 报 `No module named 'torch'`，但之前能跑（环境被清过）。
- **解决**：`pip3 install torch==2.5.1`(CPU, aarch64, cp311) + `torch_npu==2.5.1` + numpy/pybind11/expecttest。网络可用。

### 问题 3：构建缺 Python 依赖

- **现象**：`opc`（算子编译器）构建时报 `No module named 'decorator'/'scipy'/'attr'/'psutil'/'yaml'`，真实 ccec 错误被掩盖。
- **解决**：逐个 `pip3 install decorator scipy attrs psutil pyyaml`。构建工具链依赖较隐蔽，**报错信息层层嵌套，需逐层装齐才能看到真正的 ccec 编译错误**。

### 问题 4：DataCopyPad 的 stride 单位陷阱

- **现象**：原内核 UB 侧 stride 直接用字节，但实际 UB 侧单位是 **32B dataBlock**。
- **根因**：查 `DataCopyPad(ISASI).md` 与 `DataCopyPad2D` 样例确认：
  - **GM 侧** `srcStride/dstStride` 单位 = **字节**；
  - **UB 侧（VECIN/VECOUT）** `srcStride/dstStride` 单位 = **32B dataBlock（必须 32B 整倍）**；
  - stride 语义 = 相邻块「前块尾到后块头」的 **GAP**（非总跨度）；
  - `blockCount(uint16_t) ∈ [1, 4095]`（不是 65535！）。
- **解决**：重写 stride 计算，UB 行间距用 `Align32(...)/32` 块数表达；逐元素兜底路径 dstStride=0（每元素占 1 block）。

### 问题 5：`ProcessTranspose` 丢失 batch 循环

- **现象**：重写时漏掉前缀 batch 维遍历，3D+ 输入（如 `case5 (12,64,97)`）输出错误。
- **解决**：`totalTiles = transBatch_ * nTiles * mTiles`，按平铺索引 `tb` 解出 `(b, ti, tj)`，`matBase = b * transM * transN`。

### 问题 6：S>1 任意 permute 回退缺失

- **现象**：host 对非末两维交换的 permute（如 `(1,2,0)`）设 mode=0 但 S>1，原 kernel 按连续读 → 错误。
- **解决**：COPY 路径按 `S_==1` / `S_>1` 分流：S>1 走 `CopyTileStrided`（逐元素 `xGm.GetValue(步长S)` → `ub.SetValue` → 连续写），正确兜底任意 permute。

### 问题 7：ccec 不支持 `std::max`

- **现象**：`error: no member named 'max' in namespace 'std'`。
- **解决**：kernel 内禁用 `<algorithm>`，用三元运算符替代 `std::max`。

### 问题 8：whl 构建触发设备初始化失败

- **现象**：`setup.py` import torch_npu 触发 runtime 初始化，设备不可用时 whl 构建失败。
- **解决**：`export TORCH_DEVICE_BACKEND_AUTOLOAD=0` 跳过 backend 自动加载，whl 可离线构建。

### 问题 9：陈旧构建产物导致 param json 与 op store 不一致

- **现象**：`invalid input nums[1], which should be equal to input nums[2] in op store`。
- **解决**：`rm -rf build_out` 全新构建。

---

## 三、反复出现的问题

### 🔴 反复问题 1：NPU 设备卡死（最大阻塞）

- **现象**：`npu-smi` 报 `dcmi model initialized failed, device is used. ret=-8020`；`aclrtGetDeviceCount` 返回 `507899`（`drvRet=87`）。无进程占用设备（`ps`/`/proc/fd` 查无），但驱动拒绝访问。
- **成因**：崩溃内核遗留 AICore hang，驱动状态不健康，**不会自愈**。
- **现状**：自动模式拦截 `npu-smi set -t device-reset`，需人工重置设备或重启容器/主机。
- **教训**：**kernel 崩溃会污染设备状态长达数小时**，开发期频繁崩溃成本极高 → 应先用 CPU_DEBUG/仿真验证再上板，减少崩溃次数。

### 🔴 反复问题 2：构建依赖链隐蔽

- `opc` 报错层层嵌套（decorator → scipy → attrs → psutil → yaml），每次只暴露一层。
- **教训**：新环境首次构建前，应一次性装齐构建依赖，避免反复试错。

### 🟡 反复问题 3：错误信息被日志噪声淹没

- ccec 真实错误被大量 `platform_info`/`deprecated` WARNING 淹没，需 `grep -iE "error:|\.cpp:[0-9]+"` 精确过滤。

---

## 四、开发过程梳理（时间线）

1. **现状盘点**：读 CLAUDE.md / AGENTS.md → 发现内核已写一半但 case1 崩溃、custom op 未安装、环境缺 torch。
2. **根因定位**：读 `Transpose` API 实现（dav_c220）→ 确认 vtranspose dtype 限制；读 `DataCopyPad` 文档 → 确认 stride 单位约定。
3. **参考对照**：查阅内置 `aclnnTranspose`（`transpose.cpp` + `v35/transpose_*.h`）的 tensor_move/n_last 等成熟路径，确认「输出驱动逐行搬运」是正确通用范式。
4. **内核重写**：三路径（COPY 连续 / COPY 步长兜底 / TRANSPOSE），half 用 vtranspose + `if constexpr` 守卫，其余逐元素保正确。
5. **Host tiling 调整**：half 固定 16×16 tile；其余 16 倍数大 tile；三路径统一 tiling 字段。
6. **构建排障**：装 Python 依赖、清陈旧产物、修 `std::max`、设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。
7. **逻辑验证**：手推 case1 / case11 / case6 的地址与 stride、越界、混合基解码（**无法上板，仅逻辑层面**）。
8. **测试扩展**：test_op.py 1→12 case，新增 `run_all.sh`。
9. **阻塞**：设备卡死，上板验证停滞。

---

## 五、开发建议

### 架构与设计

1. **通用正确路径与硬件加速路径分离**：先写 dtype 无关、可证明正确的通用兜底（如逐元素），再用 `if constexpr` 按 dtype/对齐条件叠加硬件加速分支（vtranspose）。避免「一开始就全走硬件 API」导致 dtype 不兼容崩溃。
2. **优先参考内置 op 实现**：CANN 内置 `aclnnTranspose`（`opp/built-in/.../transpose/v35/*.h`）的多路径（tensor_move/n_last/cut_one_axis/gather）是经过验证的范式，新算子应先读懂内置再动笔，避免重复踩坑。
3. **归约为「输出驱动逐行搬运」**：任意 permute 都可归约为「按输出行搬运」，源基址由外层维 stride 加权得到（`DecodeRow`）。这一范式统一、可证明正确，是通用 permute 的稳健基线。

### API 使用

4. **精确掌握 DataCopyPad 的 stride 单位**：GM=字节、UB=32B dataBlock；语义是 GAP（块尾到下块头）；`blockCount≤4095`。写代码前先查 `DataCopyPad(ISASI).md` + `DataCopyPad2D` 样例对照，**不要凭直觉写 stride**。
5. **硬件指令 dtype 限制用 `if constexpr` 守卫**：vtranspose 仅 half/int16/uint16、`Transpose-39` 仅 A3、ND2ND_B16 有尺寸约束。用编译期 `sizeof`/`IsSameType` 分支，确保非支持 dtype 不实例化该调用。
6. **UB 行布局用 32B 对齐**：`Align32(rowBytes)`，行间距用块数（`/32`）表达，避免 stride 非 32B 整倍。
7. **kernel 内禁用 `<algorithm>`/`std::max`**：ccec 内核编译环境精简，用三元运算符。

### 验证策略

8. **先 CPU_DEBUG/仿真验证再上板**：设备一旦被崩溃内核污染需数小时人工恢复，成本极高。应先用 CPU_DEBUG 模式跑通精度，确认无崩溃再上板。
9. **增量验证**：先跑 S==1 连续路径（最简单）→ 再 2D 转置（half vtranspose）→ 再非 half 转置 → 再任意 permute。每步独立验证，缩小问题范围。
10. **测试用例覆盖矩阵**：4 dtype × {2D,3D,4D,5D} × {对齐,非对齐} × {末两维交换,任意 permute,identity} × 边界(1×1)。int8/int32 必须严格零误差。

### 环境与工具链

11. **新环境首构建前一次装齐依赖**：`pip3 install decorator scipy attrs psutil pyyaml numpy pybind11 expecttest`，避免 `opc` 层层报错。
12. **whl 离线构建**：`export TORCH_DEVICE_BACKEND_AUTOLOAD=0` 绕过 torch_npu 设备初始化，设备不可用时仍能构建 pybind whl。
13. **构建错误过滤**：`grep -iE "error:|\.cpp:[0-9]+:[0-9]+|Kernel Compilation"` 过滤噪声，定位真实 ccec 错误。

### 待优化（设备恢复后）

14. **非 half 2D 转置性能**：当前逐元素 UB 转置慢，需用 UB→UB strided `DataCopy(DataCopyParams)` 做 8×8(fp32/int32)/32×32(int8) 块转置，或参考内置 `scatter_vnchwconv`。
15. **多核负载均衡**：mode=1 的 `blockDim` 当前由 `numRows`（COPY 几何）推算，应改用 `totalTiles` 推算以更好均衡。

---

## 六、当前交付物状态

```
op_kernel/transpose.cpp   364 行  ✅ 编译通过
op_host/transpose.cpp     252 行  ✅
op_host/transpose_tiling.h          ✅
test_op.py                12 case  ✅
run_all.sh                         ✅ 批量精度自测
build_out/custom_opp_openEuler_aarch64.run  ✅ 已安装
dist/custom_ops-1.0-cp311-...whl           ✅
```

**设备恢复后立即可执行的验证流程**：
```bash
cd /home/liyc/hw-S9/case_910b_Transpose/Transpose
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
bash run_all.sh     # 批量验证 12 case 精度
bash run.sh 1       # case1 精度 + 性能(msprof 对比基线)
```
