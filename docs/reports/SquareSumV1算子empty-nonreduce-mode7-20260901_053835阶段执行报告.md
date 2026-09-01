# SquareSumV1 empty-nonreduce-mode7 阶段执行报告

## 结论

本地接受，等待 s8 干净打包。当前组合同时包含两层 dense mode 4 性能优化、mode 4/mode 6 输出同步修复和非规约零维 no-work 路由。

## 身份

- 发布源码聚合 SHA256：`146589376522de3510013f32a2eef60e835951c4d1f538c3f8987dd7e44f4a17`。
- dev `.run` SHA256：`b3e7f2f2fd56fff52ba348a3b618007eefcd4a802e6d01f45153cebb47f6942a`。
- Commit：`9ebc145744ad630fe9125f6be4d440046dc7dc48`，工作树 dirty。

## 门禁结果

- Host UT 106/106。
- Mock ST：L0 117/117、L1 462/462、L2 9/9、边界 26/26。
- Real ST：L0 106/106、L1 361/361；超资源上限 case 明确计为 skip。
- 评分矩阵 44/44、BF16 4/4、非法输入 4/4。
- mode 6 随机回归累计 20/20；mode 4/空 tensor 定向 3/3；rank-8 与 Case4 压力通过。
- 三 dtype Kernel 与性能证据包哈希一致，继承 6 轮 A/B 和三 dtype memcheck。

## 边界

以上均为本地物理卡 7、CANN 8.5 证据。历史官方四个包的 Case4 仍是 `Run failed`；当前包尚未提交，不能宣称官方目标已达成。
