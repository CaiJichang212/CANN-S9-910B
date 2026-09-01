# Greater large-inner-full-resident-capped 阶段经验总结

- 大-inner完整outer广播适合“inner切片×outer分区”二维切分；单核resident slice跨行复用可稳定降低重复MTE2，且正反广播、5 dtype均有效。
- resident复用与启核策略必须联合建模。只有输出至少覆盖每个AIV一个dtype TILE时才突破generic cap，可消除低工作量5D回退，同时不损失大目标40核收益。
- 共同79不含新目标，只能作为旧工作区间回归集；full94包含目标但压力项权重任意。目标收益、集合总和和逐项长尾必须分别判定。
- 单轮未命中路径的临界material异常不能忽略，也不能立即归因于代码；用相邻边界和反序配对复测后，保留原异常并以多轮结果判断。
- v1拒绝后恢复父源码精确hash，再从同一官方父版本建立v2，避免失败候选成为隐式parent。

适用边界：当前DAV_2201/CANN 8.5实现、项目dtype TILE和20/40核策略；这些常量不能迁移为通用平台参数。
