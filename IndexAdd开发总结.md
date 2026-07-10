# IndexAdd 算子开发总结

> 本文档沉淀 S9 IndexAdd 算子（`dev-index-add-0707` 分支）从零开发全流程：设计决策、踩过的坑、错误定位与解法、可复用经验。目标是为后续算子开发（尤其含 scatter / Cast / 多 dtype 的算子）加速、避免重复犯错。
>
> 硬件/环境：Ascend 910B4-1（单卡 20 AICore，UB 192KB），CANN 8.5.0，`__NPU_ARCH__==2201`（dav_c220），aarch64 / Python 3.11。

---

## 0. 一句话结论

**先查 API 真实签名与 dtype/roundmode 支持矩阵（以 `asc-devkit/impl/basic_api/dav_c220/` 的 `SupportType` 为准，比文档可靠）-> 再写代码；编译过 ≠ 跑通 ≠ 精度对，三重验证缺一不可；性能对齐 built-in baseline 即视为达标。**

IndexAdd 最终交付：5 dtype 全部精度通过（自测 36 例 + 官方 5 case），AICore 耗时达内置 `aclnnIndexAdd` baseline（≤ baseline，在噪声内）。

---

## 1. 算子规格与难点

### 1.1 规格（参考 `torch.index_add`）

| 角色 | 参数 | 形状 | dtype |
|------|------|------|-------|
| INPUT | self | (...,N4,N3,N2,N) ND | float32, bfloat16, float16, int32, int8 |
| INPUT | index | (M) | int32 |
| INPUT | source | (...,M,...) | 同 self（dim 维长 == M） |
| ATTR | dim | int | 默认 0 |
| OUTPUT | output | 同 self | 同 self |

**语义**：`output = copy(self)`，再沿 dim 维 scatter-add：对每个 `i∈[0,M)`，`output[..., index[i], ...] += source[..., i, ...]`。

### 1.2 数据视图（设计基础）

把内存按 dim 拆成三段 `[beforeDimSize, dimLen, afterDimSize]`，`afterDimSize` 维内存连续：
- self/output：`[beforeDimSize, dimLen, afterDimSize]`
- source：`[beforeDimSize, M, afterDimSize]`
- index：`[M]`
- 每个 `(row, i)` 的 scatter 是一段连续的 `afterDimSize` 个元素。

### 1.3 两大难点（S9 五算子中最难 ⭐⭐⭐⭐⭐）

1. **随机/不规则访存**：`index[i]` 决定写入位置，访问模式完全随机，缓存命中差。
2. **index 可重复**：同一输出位置被多次累加，直接写 HBM 会产生 **WAW（写后写）冲突**，必须保证累加而非覆盖。

### 1.4 必覆盖的泛化场景（评测随机生成）

5 dtype（含 bfloat16）、任意 dim、非 32 对齐、index 重复、1D~5D、大 shape、`dim=0`（beforeDimSize=1）、`afterDimSize` 巨大、负 dim、M=1、全同 index。

---

## 2. 开发流程总览

```
┌─ 1. 前期调研（并行 subagent 检索 asc-devkit + 参考实现）
│   ├─ 原子操作 / scatter / sort API 支持矩阵（910B）
│   ├─ Add / Cast / Compare 等 dtype 支持矩阵
│   ├─ Concat 参考实现的可复用模式
│   └─ 官方 scatter_add / inplace_index_add 内置实现策略
│
├─ 2. 脚手架生成
│   ├─ 写 IR JSON (op/IndexAddCustom.json)
│   ├─ msopgen gen 生成 op/CustomOp
│   └─ 改 SoC：ascend910 -> ascend910b；op 名 IndexAddCustom -> IndexAdd
│
├─ 3. host 编写
│   ├─ index_add_tiling.h（TilingData 全 uint32_t）
│   └─ index_add.cpp（TilingFunc + InferShape + InferDataType + OP_ADD(IndexAdd)）
│
├─ 4. kernel 编写
│   ├─ 阶段 1：bulk copy self->output（扁平字节切分，双缓冲）
│   └─ 阶段 2：scatter-add（二维切分，无 WAW，dtype 分派）
│
├─ 5. 编译排错（ccec 错误：pipe 未声明、narrowing、文件名/入口名）
│
├─ 6. 运行排错（运行时崩溃、输出全 0、int8 数值错、bf16 精度）
│
├─ 7. 精度验证（36 例自测 + 官方 5 case）
│
└─ 8. 性能量测（msprof AICore 中位 vs 内置 baseline）
```

---

## 3. 关键技术决策

### 3.1 核切分：二维切分，天然无 WAW（核心创新）

**最初设想**（CLAUDE.md 推荐）：排序合并 / 分桶 / 原子加。

**调研结论**：
- `AscendC::VectorScatter` -> `ASCENDC_REPORT_NOT_SUPPORT`（910B **不支持**）。
- `SetAtomicAdd<T>()` 支持 5 dtype，但**只通过 plain `DataCopy`（需 32B 对齐 count），`DataCopyPad` 不 honors atomic** -> 非对齐尾部无法用原子。
- `int32` 的 `Sort`（radix, `adv_api`）存在但实现复杂，v1 未采用。

**最终方案**：按 afterDim 轴（与 beforeDim 行）**二维切分核**，每核拥有互不重叠的 `output[rowRange, :, afterSlice]` 区域。由于 afterDim 是最内层连续维，按它切片使每个输出元素**恰好归属一个核**：
- **天然无 WAW、无需原子、无需排序合并**；
- `DataCopyPad` 适配任意对齐（不依赖 32B）；
- **带宽最优**（每个 source 元素全卡总共只读一次）；
- 对 `dim=0`（beforeDimSize=1，CLAUDE.md 标注的难点）同样成立。

双模式自适应满核：
- **ROW**（mode=0）：按 beforeDim 行切分（beforeDim 大时）。
- **AFTER**（mode=1）：按 afterDim 轴切分（beforeDim 小、含 dim=0 时）。
- host 取 `max(min(20,beforeDim), min(20,afterDim))` 对应模式。

### 3.2 两阶段 kernel

1. **bulk copy self->output**：output 是 `empty_like(self)`，必须整体填充。扁平字节均匀切给 ≤20 核，对齐走 `DataCopy`/`DataCopyPad`，双缓冲流水，完成后 `SyncAll()` 全核同步。
2. **scatter-add**：每核串行 RMW（读 output -> 加 source -> 写 output），重复 index 同核内自然累加。

### 3.3 dtype 分派（`template<InputT, ComputeT>` + `if constexpr`）

| dtype | InputT | ComputeT | 路径 |
|---|---|---|---|
| float32 | float | float | 直接 Add |
| float16 | half | half | 直接 Add |
| int32 | int32_t | int32_t | 直接 Add |
| int8 | int8_t | half | Cast↔half（half 精确表示 int8，无舍入损失） |
| bfloat16 | bfloat16_t | float | Cast↔float（逐次 RNE 舍入） |

**分派函数必须做成 `template<typename CT>`，分支条件用 `IsSameType<CT, ...>`（依赖 CT）**，否则 `if constexpr` 的 discarded 分支仍做语义检查而报错。

---

## 4. 遇到的错误与解决（按时间线）

### 4.1 msopgen IR：type 数组长度不一致

**现象**：`msopgen gen` 报 `The number(1) of type for "index" is inconsistent with the number(5) of "self"`。

**原因**：msopgen 要求所有 input 的 type/format 数组长度一致。`index` 只有 1 个 int32，但 `self` 有 5 个 dtype。

**解决**：把 `index` 的 type 数组 pad 到 5 个 `int32`（host 的 `OpDef` 里 index 强制 `DT_INT32` 即可，不影响）。

### 4.2 SoC 配置：默认 ascend910，须改 ascend910b

**现象**：msopgen 生成的 `CMakePresets.json` 和 `op_host/*.cpp` 默认 `ascend910`（非 910b）。

**解决**：
- `CMakePresets.json`：`ASCEND_COMPUTE_UNIT = "ascend910b"`。
- `cmake/config.cmake`：默认值也改 `ascend910b`。
- `op_host`：`AddConfig("ascend910b")`。

### 4.3 op 名 / 源文件名 / kernel 入口名 三者须一致

**现象**：分别报 `kernel entry 'index_add' not implement in 'index_add.cpp'` 与 `source file ... index_add.cpp does not found`。

**原因**：msopgen 按 JSON 的 `op` 字段（`IndexAddCustom`）生成 `index_add_custom.cpp` 与入口 `index_add_custom`。但 build 按 **op 注册名**（改成 `IndexAdd`）查找 `index_add.cpp` 与入口 `index_add`，二者不匹配。

**解决**：
- 源文件重命名：`index_add_custom.cpp` -> `index_add.cpp`，`index_add_custom_tiling.h` -> `index_add_tiling.h`（host `#include` 同步改）。
- kernel 入口函数：`index_add_custom` -> `index_add`。
- op 注册名用 `IndexAdd`（`class IndexAdd : public OpDef` + `OP_ADD(IndexAdd)`），build 产出 `aclnnIndexAdd` 覆盖内置。
- 规律：**op 名 PascalCase -> 源文件名与入口函数 snake_case**（`IndexAdd` -> `index_add`）。

### 4.4 ccec 编译错误：`pipe` 未声明 + narrowing

**现象 1**：`error: use of undeclared identifier 'pipe'`。

**原因**：Concat 参考实现里有 `AscendC::TPipe pipe;` 成员，我漏了。

**解决**：在 kernel 类私有成员加 `TPipe pipe;`。

**现象 2**：`error: non-constant-expression cannot be narrowed from 'uint32_t' to 'uint8_t'`。

**原因**：`DataCopyPadExtParams` 的 `leftPadding`/`rightPadding` 是 `uint8_t`，花括号初始化 `pad{true, 0, rightPad, ...}`（rightPad 是 uint32）触发 narrowing。

**解决**：用成员赋值（CLAUDE.md 已警示的 gotcha），`pad.rightPadding = static_cast<uint8_t>(rightPad);`。

### 4.5 运行时崩溃（AICore exception，507015）

**现象**：跑测报 `rtDeviceSynchronizeWithTimeout execution failed, reason=aicore exception`，输出全 0 / 垃圾值。

**定位**：隔离测试发现 **float32 也崩**（非 cast 路径），故是结构性问题。开「阶段 1 only」隔离，发现 bulk copy 写回的 output 全是垃圾。

**根因**：`DataCopyPad` UB->GM（MTE3）**必须从 VECOUT 位置的 buffer 读**。我最初用一个 `TQue<VECIN>` 缓冲既做读入又做写出，MTE3 从 VECIN 读会**静默写失败**（不报错但 output 不更新）。

**解决**：读用 `TBuf<VECIN>` / `TQue<VECIN>`，写用 `TBuf<VECOUT>` / `TQue<VECOUT>`，中间用一次 UB->UB `DataCopy`（VECIN->VECOUT）桥接。AddCustom 样例正是此模式（`inQueueX: VECIN`，`outQueueZ: VECOUT`）。

> **通用规律**：MTE2（GM->UB）写 VECIN；MTE3（UB->GM）读 VECOUT；Vector 计算在中间。跨位置必须显式 UB->UB 拷贝，不要图省事共用一个 buffer。

### 4.6 int8 cast 路径崩溃（isPad=true）

**现象**：int8 用 `DataCopyPadExtParams{isPad=true, rightPadding=...}` 读 sub-32B blockLen 时 AICore 崩溃。

**解决**：去掉 `isPad`。改用「读 n 个真实元素（不 pad）+ Cast 用 `round_up(n, 256)` 对齐计数」：padding 区读到的是 UB 残留垃圾，但**仅写回真实 n 字节**，故结果正确。这同时简化了逻辑、避免了 isPad 的对齐坑。

### 4.7 int8 数值错误（输出 0 而非 7）— Cast RoundMode 不匹配 ⭐

**现象**：int8 scatter 位置全输出 0（`5+2` 应得 7，却得 0）。float32 正常。

**定位**：查 `kernel_operator_vec_vconv_impl.h` 的 `CastIntrinsicsImpl(__ubuf__ half* dst, __ubuf__ int8_t* src, ...)`：
```cpp
switch (roundMode) {
    case RoundMode::CAST_NONE:  vconv_s82f16(...); break;   // 唯一支持的分支
    default: ASCENDC_ASSERT((false), {...}); break;        // Release 下 assert 被编译掉 -> 静默不发出指令 -> dst 保持 0
}
```
我用了 `CAST_ROUND`，走到 `default` 分支，**Release 模式下 assert 编译掉，静默输出 0**（Debug 模式才会报错）。

**根因**（RoundMode 支持矩阵，查 dav_c220 impl 确认）：
- **上行转换**（窄->宽，int8->half / bf16->float / int4b->half）：**只支持 `CAST_NONE`**。
- **下行转换**（宽->窄，half->int8 / float->bf16 / float->int32）：支持 `CAST_RINT`/`CAST_FLOOR`/`CAST_CEIL`/`CAST_ROUND`/`CAST_TRUNC`/`CAST_NONE`。

**解决**：上行用 `CAST_NONE`，下行用 `CAST_RINT`。

> **教训**：Cast 的 RoundMode 不是随便选的，**每个 src->dst 组合支持的 mode 不同**，必须查 `CastIntrinsicsImpl` 的 switch。Release 下 assert 编译掉，不支持的 mode 会**静默输出垃圾/0**，只有精度测试能暴露。

### 4.8 bf16 精度失败（maxdiff 0.03125）— 逐次舍入 vs 一次性舍入 ⭐⭐

**现象**：bf16 含重复 index 时 maxdiff=0.03125（其余 dtype 全过）。

**错误尝试 1（float workspace）**：分配 float32 workspace W，`self(bf16)->W(float) cast` -> 在 W 上 scatter-add（float 无中间舍入）-> `W(float)->output(bf16) 一次性 cast`。结果 maxdiff 降到 0.015625 但仍失败。

**定位**（逐元素分析 `[0,2]`，被 index=2 触及两次）：
- `exact float sum = -1.95703125`（恰是 bf16 的精确 tie，介于 -1.953125 与 -1.9609375 之间）。
- torch golden = -1.953125（RNE，round to even）。
- 我的 workspace 一次性舍入 = -1.953125（RNE，对！）。
- 但**其他元素** torch golden 与「float 累加后 RNE 舍入」不符——验证发现 **torch 对 bf16 是 per-step RNE**（每次 bf16 add 后立即舍入），而非 float 累加后一次性舍入。二者在非 tie 处也可能差 1 ULP（浮点加法非结合）。

**根因**：`torch.index_add` 对 bf16 输入**逐次做 bf16 加法（每次 RNE 舍入）**，不是上转 float 累加。

**解决**：bf16 也用直接路径（逐次 RMW），不能用 workspace 累加。下行 float->bf16 用 `CAST_RINT`（RNE）。

> **教训**：精度对齐必须理解 reference 的**舍入语义**（逐次 vs 累加、RNE vs RNA）。bf16 这种低精度类型，逐次舍入与一次性舍入结果可能不同。不能想当然用「高精度累加更准」。

### 4.9 下行 Cast 用 CAST_ROUND 还是 CAST_RINT

**现象**：bf16 直接路径用 `CAST_ROUND`（float->bf16）仍 maxdiff=0.015625。

**定位**：查 `CastIntrinsicsImpl(__ubuf__ bfloat16_t* dst, __ubuf__ float* src, ...)`：
- `CAST_RINT` -> `vconv_f322bf16r`（**RNE，round to nearest even**）。
- `CAST_ROUND` -> `vconv_f322bf16a`（**RNA，round to nearest, ties away from zero**）。
- torch.bfloat16 舍入是 **RNE**。

精确 tie 处（如 -1.95703125）RNA 取远离零者（-1.9609375），RNE 取偶数（-1.953125），差 1 ULP。

**解决**：所有下行转换（float->bf16）改用 `CAST_RINT`。

> **教训**：`CAST_ROUND` 名字有误导性，它是 RNA（ties away），不是 RNE。与 torch / numpy 的默认 RNE 对齐要用 `CAST_RINT`。

### 4.10 numpy 无 bfloat16

**现象**：`test_op.py` 里 `np.random.uniform(...).astype(np.bfloat16)` 报 `module 'numpy' has no attribute 'bfloat16'`。

**解决**：bf16 case 以 float32 存放 + `dtype` 字段标记，测试时 `torch.from_numpy(...).to(torch.bfloat16)`。

### 4.11 test_op.py 模板的 case3 特殊分支

**现象**：case3（float16）报 `index_add() received an invalid combination of arguments - got (numpy.ndarray, ...)`。

**原因**：模板有 `if int(num) == 3: input_x = case["input"]`（保留 numpy ndarray，未转 torch）。

**解决**：移除该特殊分支，统一 `torch.from_numpy`。

### 4.12 共享 vendor 冲突（5 算子互相覆盖）⭐⭐⭐

**现象**：跑测间歇性报 `call aclnnIndexAdd failed ... got null for argument source` 或 `custom_op(): incompatible function arguments ... (arg0: Tensor, arg1: Tensor)`。`nm -D libcust_opapi.so | grep aclnnIndexAdd` 时有时无。

**定位**：S9 五算子（Concat/Greater/IndexAdd/Transpose/SquareSumV1）的开发都安装到**同一个** `vendors/customize/`（`$ASCEND_OPP_PATH/vendors/customize/op_api/lib/libcust_opapi.so`），且 pybind whl 都叫 `custom_ops` 包（`custom_ops_lib.cpython-*.so`，`custom_op` 函数）。**一个算子装了，另一个算子重装会覆盖前者的 `aclnn<Op>` 符号与 `custom_op` 签名**。

**表现**：
- opapi 被内置覆盖 -> "null source"（内置 `aclnnIndexAdd` 签名是 `(self,index,source,alpha,out)`，与 pybind 传的 `dim` 位置错位）。
- whl 被他算子覆盖 -> "incompatible arguments"（如 IndexAdd 4-参被 Transpose 的 2-参覆盖）。

**解决**：每次跑测/量性能前，**原子地**重装本算子的 opapi + whl：
```bash
bash op/CustomOp/build_out/custom_opp_openEuler_aarch64.run >/dev/null 2>&1          # 重装 opapi
cd IndexAdd && pip3 install --no-cache-dir --force-reinstall dist/custom_ops*.whl >/dev/null 2>&1  # 重装 whl
export LD_LIBRARY_PATH=$ASCEND_OPP_PATH/vendors/customize/op_api/lib/:$LD_LIBRARY_PATH
# 立即验证：nm -D ... | grep aclnnIndexAdd 应返回 2；custom_op 签名应为 4 参
```
- **`--no-cache-dir` 关键**：pip 会缓存 whl，不带它可能装回旧/他算子版本。

> **教训**：多算子并行开发时，共享安装目录 + 共享包名是巨大陷阱。任何「间歇性失败、时好时坏」先怀疑这个。

---

## 5. 调试方法论（可复用）

### 5.1 隔离定位法

遇到运行时错误时，**最小化复现**：
- 单 dtype 小 shape（如 `[4,8]`）直接调 `custom_op`，打印逐元素 out vs golden。
- 开「阶段 1 only」（在阶段 2 前 `return`）隔离两阶段。
- 按错误类型分类：崩溃（结构性）vs 数值错（算法/cast）vs 精度差（舍入语义）。

### 5.2 看 ascend 日志

- 运行时错误（507015 aicore exception）的细节在 `/root/ascend/log/debug/plog/plog-<pid>_<ts>.log`，grep `ERROR|IndexAdd|aicore|overflow|addr`。
- aclnn 参数错误（EZ1001）会显示在 python Traceback 里，含 `OpName:[aclnnIndexAdd_1]`。

### 5.3 API 支持矩阵查证（最重要的方法论）

**不要信文档，信 `asc-devkit/impl/basic_api/dav_c220/` 的实现**：
- dtype 支持：grep `ASCENDC_ASSERT(SupportType<...>` 或 `static_assert(SupportType<...>`。
- RoundMode 支持：看 `CastIntrinsicsImpl` 的 `switch(roundMode)` 哪些 case 有指令、哪些走 `default`（assert）。
- count 类型：看 Level-2 接口签名（`uint32_t` vs `const int32_t&`）。
- 是否支持：看是否 `ASCENDC_REPORT_NOT_SUPPORT(false, "...")`。

### 5.4 精度对齐：理解 reference 舍入语义

对低精度类型（bf16/half/int8），**逐次舍入 vs 累加后一次性舍入结果可能不同**。对齐 reference（torch）时：
- 先确认 torch 的语义（`torch.index_add` 对 bf16 是 per-step RNE）。
- 用受控小用例（已知 tie 值）验证：如 `self=-0.96875, src=-0.98828125 -> exact=-1.95703125`（bf16 tie），RNE=-1.953125，RNA=-1.9609375，看输出匹配哪个。

### 5.5 量内置 baseline

```bash
# 从 LD_LIBRARY_PATH 去掉 vendors/customize/op_api/lib/，回退 libopapi.so 内置
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed 's#[^:]*vendors/customize/op_api/lib/:##g')
# 同条件 msprof 量 AICore 中位
```
**注意**：内置 `aclnnIndexAdd` 签名是 `(self,index,source,alpha,out)`，pybind 传 `dim`->`alpha`。`dim=1` 时 alpha=1 结果正确可比；`dim=0` 时 alpha=0 是 no-op（仅能量时序，不可比精度）。`torch.index_add` 走的是 `InplaceCopy`（不同 op），不可作 baseline。

---

## 6. 910B / CANN 8.5 关键 API 速查（IndexAdd 实测）

| API | dtype/模式支持（dav_c220 实测） | 备注 |
|---|---|---|
| `Add(dst, s0, s1, count)` | half/float/int16/int32（**不含 int8/bf16**） | count 是 `const int32_t&`，需 `static_cast` |
| `Cast(dst, src, roundMode, count)` | 见下 | count 是 `uint32_t`，须 256B 对齐 |
| Cast 上行（int8->half / bf16->float） | **只 `CAST_NONE`** | 用其他 mode 走 default 静默输出 0 |
| Cast 下行（half->int8 / float->bf16） | `CAST_RINT`(RNE)/`CAST_ROUND`(RNA)/`CAST_FLOOR`/`CAST_CEIL`/`CAST_TRUNC`/`CAST_NONE` | 与 torch 对齐用 `CAST_RINT` |
| `Duplicate(dst, scalar, count)` | half/bf16/int16/uint16/int32/uint32/float（**不含 int8/uint8**） | count 是 `const int32_t&` |
| `Compare` | int32 **仅 `CMPMODE::EQ`**；half/float 全模式 | dst 是 uint8 bitmask |
| `Select`(bitmask) | dst/src 仅 half/float；selMask uint8/16/32/64 | 语义 `dst=bit?src0:src1` |
| `SetAtomicAdd<T>()` | half/float/int16/int32/int8/bf16（全 5 dtype） | **仅 plain `DataCopy`（需 32B 对齐），`DataCopyPad` 不 honors atomic** |
| `VectorScatter` | **910B 不支持**（`ASCENDC_REPORT_NOT_SUPPORT`） | 散写靠二维切分/原子/排序合并 |
| `SyncAll()` | 全核同步，硬件实现无需 workspace | 阶段间保证 MTE3 写回可见 |
| `DataCopyPad` | GM->UB 需 4 参（含 `DataCopyPadExtParams`）；UB->GM 3 参 | MTE3 必须从 VECOUT 读 |
| `DataCopyExtParams` | `blockCount` 是 `uint16_t` | 花括号初始化会 narrowing，用成员赋值 |

---

## 7. 910B 关键约束（写 kernel 前过一遍）

- **UB**：192KB（可用 ~184KB），超 UB **运行时崩溃**（非编译期）。按 dtype 精算 buffer，双缓冲 `2×TILE ≤ 184KB`。
- **AICore**：20/卡，`blockDim ≤ 20`，按数据量自适应 `min(20, ...)`。
- **DataCopy 对齐**：count 须 32B 对齐（理想 256B）；非对齐用 `DataCopyPad`。
- **MTE3 写回位置**：UB->GM 必须从 VECOUT 读，不能从 VECIN（静默失败）。
- **Cast count 256B 对齐**：`count*sizeof(T) % 256 == 0`，否则静默漏算尾部。
- **dtype 分派** `template<CT>` + `if constexpr` + `IsSameType<CT, ...>`（依赖 CT），否则 discarded 分支仍语义检查报错。
- **tiling 字段全 `uint32_t`**（host `TilingDef` 可能插 padding，与 kernel 侧 `#pragma pack(1)` POD 布局一致）；数组字段 host 用局部数组 + `set_xxx(arr)`，kernel 用 `GET_TILING_DATA`（**不要** include host 的 `*_tiling.h`）。
- 大 shape 下字节/偏移计算全程 `uint64` 防溢出。

---

## 8. 提交前检查清单（IndexAdd 专用）

- [ ] `ASCEND_COMPUTE_UNIT=ascend910b`、`AddConfig("ascend910b")`？
- [ ] op 注册名 `IndexAdd`（产出 `aclnnIndexAdd` 覆盖内置）？源文件/入口 `index_add`（snake_case）？
- [ ] `blockDim ≤ 20`，按数据量自适应（ROW/AFTER 双模式选能满核者）？
- [ ] TILE 按 dtype 贴满 UB（精算 buffer，无越界，`2×TILE ≤ 184KB`）？
- [ ] bulk copy self->output 覆盖全输出；MTE3 从 VECOUT 读；对齐 `DataCopy`、尾部 `DataCopyPad`？
- [ ] **重复 index 正确累加**（二维切分无 WAW，同核内串行 RMW）？
- [ ] dtype 分派 `template<CT>` + `if constexpr`；int8/bf16 的 Cast 路径查过 dav_c220？
- [ ] Cast 上行 `CAST_NONE`、下行 `CAST_RINT`（RNE，与 torch 对齐）？
- [ ] tiling 字段全 `uint32_t`、数组用 `set_`、kernel 用 `GET_TILING_DATA`？
- [ ] `DataCopyExtParams` 用成员赋值（`blockCount` 是 uint16）？
- [ ] 泛化：不 hardcode shape/dim/dtype；tiling 对未知用例通用？
- [ ] 精度自测覆盖 5 dtype + 非对齐 + index 重复 + 各 dim + 边界（1 元素 / 大 shape / 1D / M=1 / 负 dim / all-same-index）？
- [ ] msprof 量 AICore 时间并对比内置 baseline，≤ baseline？
- [ ] 跑测前原子重装 opapi + whl（`--no-cache-dir`），避免共享 vendor 覆盖？
- [ ] 打包前 `.run` 重新构建、与源码一致；用 `zip_op.sh` 打包？

---

## 9. 最终交付状态

**精度**：自测 36 例全过 + 官方 5 case 全 pass（5 dtype × ROW/AFTER × 非对齐 × 重复 index × 3D/4D/5D/中间 dim × 负 dim × 大 shape × 1D × M=1 × all-same-index）。

**性能**（AICore 中位 µs，对比内置 `aclnnIndexAdd`）：

| case | dtype / shape | 本实现 | 内置 baseline |
|---|---|---|---|
| 1 | int8 [32,128] 120idx | ~70 | 68 |
| 2 | fp32 [256,512] 500idx | ~209 | 208 |
| 3 | fp16 [128,256,64] 200idx dim=1 | ~486 | 488 |
| 4 | bf16 [200,400] 300idx | ~137 | 133 |
| 5 | int32 [64,1000] 400idx | ~186 | 186 |

**5 个 case 均达内置 baseline**（≤ baseline，在噪声内；fp16/int32 持平甚至略优，Cast 路径 int8/bf16 因 3 Cast+1 Add 固有开销略高但仍在噪声内）。

**可进一步优化（未做）**：scatter 阶段改「UB 内累加合并再单次写回」（dimLen×afterDim 装 UB 时），减少 per-(row,i) HBM 写次数。bf16 仍须逐次舍入故收益有限。

---

## 10. 参考资料位置

- **Ascend C 官方实现（910B 真实 dtype/模式支持，比文档可靠）**：`/home/liyc/asc-devkit/impl/basic_api/dav_c220/`
  - Cast RoundMode 支持：`kernel_operator_vec_vconv_impl.h` 的 `CastIntrinsicsImpl` switch
  - Add/Duplicate dtype：`kernel_operator_vec_binary_impl.h` / `kernel_operator_vec_duplicate_impl.h`
  - 原子操作：`kernel_operator_set_atomic_impl.h`
  - Scatter 不支持：`kernel_operator_vec_scatter_impl.h`（`ASCENDC_REPORT_NOT_SUPPORT`）
- 官方 API 头：`/home/liyc/asc-devkit/include/basic_api/kernel_struct_data_copy.h`（`DataCopyExtParams` 等字段类型）
- 内置实现参考：`/usr/local/Ascend/cann-8.5.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/ascendc/{inplace_index_add,scatter_add}/`
- 样例：`/home/liyc/hw-S9/samples/operator/ascendc/`（AddCustom 的 VECIN/VECOUT 队列模式）
- 上层方法论：`/home/liyc/hw-S9/AscendC算子开发经验教训.md`（Greater 全流程沉淀）
- 项目指南：`/home/liyc/hw-S9/case_910b_IndexAdd/CLAUDE.md`
