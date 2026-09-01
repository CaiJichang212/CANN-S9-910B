# SquareSumV1 mode6-output-v-mte3-sync 阶段执行报告

## 结论

候选解决 mode 6 BF16 随机脏值，但因父版本的 dense mode 4 仍有输出可见性缺陷，完整包拒绝并进入下一单变量候选。

## 证据

- `.run` SHA256：`53542684a15f04eb8820b5ce5bb6580488b7ea62190fc56721d75469a686e477`。
- Kernel 动态源码 SHA256：`4948ec4ac45389e5437e426f890880e686806a43c528fcefc48e4a8e6a0e9fa1`。
- 原 4 条 BF16 mode 6 失败 case 在 5 个新进程中累计 20/20 PASS。
- L0 Real ST 106/106 PASS；L1 Real ST 358/361 PASS。
- L1_270、L1_307 及第三条同类 case 均指向 dense mode 4 layer 输出的 Vector→MTE3 依赖缺失。
