# SquareSumV1 非规约零维 mode 7 路由修复方案

## 根因

非连续多轴由 `CoalesceAxis` 以 `totalRows=-1` 标记。若输入的 0 维不属于规约轴，输出也为 0 元素，但 Host 仍进入 mode 4 layer 构造；对应 layer 的 `a0Length=0`、`tileA0Len=0`，最终可达 `CeilDiv(0,0)`。

## 单变量修改

保留“规约轴本身为 0”时按 surviving output 元素显式 zero-fill 的现有逻辑。在其之后检查输入总元素数：若为 0，说明零维只存在于非规约轴，输出必为 0 元素，调用 `BuildElementwiseTiling(..., 0, EMPTY_REDUCE_MODE)`，生成 mode 7、`totalWorkItems=0`、`BlockDim=1` 的 no-work tiling。

非零输入和所有已有 mode 0-6 路由不变，因此无需新的非零性能 A/B；需要 UT、Real ST 和最终包 smoke 证明空 tensor 行为。
