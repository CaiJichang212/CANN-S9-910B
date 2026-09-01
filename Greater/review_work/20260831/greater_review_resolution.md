# Greater最终检视收敛记录

## 最终身份

- Host：`a47810d0b0d6bd52cf7e4d0f5dc6270d7a3e1f02258c4a9df2b6bb1772978abb`
- TilingData：`f96c9e9d643a69ab37c4d3e59c4bc8b6c3761030b55136ac47aa7a7c5fcc79f1`
- Kernel：`1b7f6964c53c4dc58b8471a216144de7460011e9544d780da59b4439fcffde7f`

## 阻断项闭环

| 根因 | 原条例投影 | 最终处理 |
|---|---|---|
| Host输入与数值契约 | TOPK-7、REDLINE-2/3/6、CPPSEC-1.1/1.2/1.3/3.1、CPPGEN-1.1 | 判空；rank≤8；非负维；广播兼容；dtype一致；checked multiply；uint32可表示性；raw tiling容量；SetBlockDim/SetOutputDataType状态 |
| P2总UB超限 | TIL-2、CPPSEC-3.2及原API-6误投影 | 以DAV_2201的184KiB用户区建模；删除死Buffer；按dtype/方向条件分配；batch cap；编译期static_assert；5 dtype可达边界5/5 PASS |
| u32 ceil-div回绕 | REDLINE-4 | RoundUp与total block ceil-div在首操作数处提升到uint64，并对RoundUp结果钳位 |
| P2 stream错误线性寻址 | 精度穿刺发现 | Host/Kernel镜像检查stream outer stride连续性，不满足时回退ComputeBases |

## 复核结论

- `TIL-1`：证据豁免。Kernel按blockDim均分；直接扩大generic核数有实测回退，最终按路径/工作量使用20或40核，94/94 PASS且共同79项0 material回退。
- `PREC-1`：最终PASS。队列路径使用EnQue/DeQue；resident/scalar batch有显式事件；blocked scalar覆盖前有V到MTE2同步；Copy/Brcb后有V PipeBarrier；工程默认自动同步覆盖generic GetValue链。
- `API-6`：原归类撤销。Alloc/Free及EnQue/DeQue已配对；UB容量属于TIL-2而非API-6。
- `PERF-1/PERF-6`：原本是后续性能假设。本轮已用整批Copy/Brcb和blocked scalar消除主要短行逐row/重复scalar开销；大inner resident仍是后续可选路线，不阻断本次提交。
- `CPPGEN-4.3`、const引用、参数顺序和文档格式项：不影响运行契约，未为风格引入ABI或大范围机械改动。

最终快速定向复检覆盖TOPK-7、REDLINE-2/3/4/6/9、CPPSEC-3.1/3.2、API-3/4/10、PREC-1和TIL-2，均为PASS。最终构建、Host 5/5、UB 5/5、mixed 40/40、sweep 85/85、full94 94/94及s8 12项代表集全部通过。
