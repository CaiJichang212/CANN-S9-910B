#!/usr/bin/env python3
"""Generate the final Chinese bottleneck report from validated CSV artifacts."""

from __future__ import annotations

import csv
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[2]
REPORT = ROOT / "docs/性能瓶颈分析/Concat算子性能测试与瓶颈分析报告-20260830.md"


def read_csv(name: str) -> List[Dict[str, str]]:
    with (HERE / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def f(value: object, digits: int = 3) -> str:
    try:
        return ("{:,.%df}" % digits).format(float(value))
    except (TypeError, ValueError):
        return str(value)


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def group_rows(rows: Sequence[Dict[str, str]], dimension: str) -> List[Dict[str, str]]:
    return [row for row in rows if row["dimension"] == dimension]


def main() -> None:
    required = (
        "calibration.csv", "calibration_summary.csv", "case_manifest.csv",
        "tiling_model.csv", "latency_case_summary.csv", "latency_round_case_summary.csv",
        "latency_round_totals.csv", "latency_group_summary.csv", "latency_rankings.csv",
        "latency_profile_validation.csv", "historical_sanity.csv", "deep_case_summary.csv",
        "deep_profile_validation.csv", "deep_sample_status.csv",
    )
    missing = [name for name in required if not (HERE / name).is_file()]
    if missing:
        raise SystemExit("missing report inputs: {}".format(", ".join(missing)))
    correctness = (HERE / "correctness/correctness_full.log").read_text()
    marker = "CORRECTNESS_COMPLETE fixed=48 generated=300 contracts=4 repeat10_controls=11"
    if marker not in correctness:
        raise SystemExit("correctness completion marker is absent")

    calibration = read_csv("calibration.csv")
    calibration_summary = read_csv("calibration_summary.csv")
    models = read_csv("tiling_model.csv")
    latency = read_csv("latency_case_summary.csv")
    round_groups = read_csv("latency_round_case_summary.csv")
    round_totals = read_csv("latency_round_totals.csv")
    grouped = read_csv("latency_group_summary.csv")
    rankings = read_csv("latency_rankings.csv")
    history = read_csv("historical_sanity.csv")
    deep = read_csv("deep_case_summary.csv")
    samples = read_csv("deep_sample_status.csv")
    latency_validation = read_csv("latency_profile_validation.csv")
    deep_validation = read_csv("deep_profile_validation.csv")

    by_case = {row["case"]: row for row in latency}
    model_by_case = {row["case"]: row for row in models}
    deep_by_case = {row["case"]: row for row in deep}
    local_sum = sum(float(row["p50_us"]) for row in latency)
    median_case = statistics.median(float(row["p50_us"]) for row in latency)
    slowest = max(latency, key=lambda row: float(row["p50_us"]))
    lowest_bw = min(latency, key=lambda row: float(row["effective_bidirectional_gbps"]))
    bounds = Counter(row["bound"] for row in deep)
    row_paths = sum(row["predicted_split_path"] == "row" for row in models)
    col_paths = len(models) - row_paths
    history_ratios = [float(row["current_over_historical"]) for row in history]
    sample_nonzero = any(int(row["nonzero_task_cyc_rows"]) > 0 for row in samples)
    profiler_mte_fields = any(int(row["profiler_mte_instruction_fields_available"]) for row in deep)

    lines: List[str] = []
    lines.append("# Concat 算子性能测试与瓶颈分析报告（2026-08-30）")
    lines.append("")
    lines.append("## 1. 结论")
    lines.append("")
    lines.append(
        "本轮在不修改 `op/CustomOp/` 的前提下，完成了官方提交基线的隔离重建、"
        "bitwise 正确性门禁、三卡校准、6 轮 92-case 延迟采集，以及 10 个锚点的三卡七组指标深采集。"
        "构建前后三份实现源码均与 `Concat_20260722_102940_zip` 逐字节一致。")
    lines.append("")
    lines.append("- 正确性：48 个固定/宽行 case、三个种子各 100 个随机 case、四种 dtype 的混合 `(0,)` 契约 case 全部通过；11 个高风险控制各连续重复 10 次通过。")
    lines.append("- Profile 完整性：6 份延迟 profile 均为 `92 × 30 = 2760` 条 Concat task；21 份 metric profile 与 3 份 sample profile 均为 `10 × 30 = 300` 条。")
    lines.append("- 本地 92-case 跨轮 case P50 之和为 **{} us**，单 case P50 中位数为 **{} us**。这只是诊断矩阵汇总，不是官方成绩。".format(f(local_sum, 3), f(median_case, 3)))
    lines.append("- 最慢 case 为 `{}`（{} us）；最低逻辑双向有效带宽为 `{}`（{} GB/s）。".format(slowest["case"], f(slowest["p50_us"]), lowest_bw["case"], f(lowest_bw["effective_bidirectional_gbps"])))
    lines.append("- 按本项目 AIV-only 分类规则，10 个深采集锚点的主 Bound 分布：{}。大数据与碎片路径的观测与 MTE2/小 DMA 发射受限一致；tiny 路径观测到较高 Scalar ratio 或无单一深 Bound。具体成本来源仍需候选隔离 A/B。".format(", ".join("{}={}".format(key, value) for key, value in sorted(bounds.items()))))
    lines.append("- 优先优化方向：单输入直通、低行数超宽非对齐数据的安全并行、小任务 Tiling/TPipe 固定开销、256 路小 DMA 发射、对齐块的直接 DataCopy 快路。任何实现候选都必须做单变量 A/B，不能从本地 shape 反推隐藏 Case5。")
    lines.append("")

    lines.append("## 2. 测试对象与证据边界")
    lines.append("")
    lines.append("测试对象是官方已提交且 5/5 正确的 `Concat_20260722_102940_zip` 对应源码，官方基线为 564.6515 us。私有 `.run` 和 wheel 均从当前源码重新构建；实际加载的 `libcust_opapi.so` 路径见 `metadata/runtime_load.txt`。原始 PROF 与私有 OPP 位于 `.gitignore` 排除的 `raw/`、`private/`，可由脚本重建。")
    lines.append("")
    lines.append("性能矩阵由 42 个固定 case、6 个宽行 case、seed `20260721` 的 12 个随机 case 和 32 个正交微基准构成。报告把 `scoring_proxy` 与 `robustness_or_diagnostic` 分开汇总；后者用于发现实现边界，不用于推断赛题隐藏 shape。")
    lines.append("")
    lines.append("PyTorch 的 Concat 契约要求非空输入除拼接维外形状一致，同时允许任意输入是 `(0,)` 的一维空张量；本轮四种注册 dtype 均覆盖了连续两个 `(0,)` 输入。参考 [PyTorch `torch.cat` 文档](https://docs.pytorch.org/docs/2.5/generated/torch.cat.html)。")
    lines.append("")

    lines.append("## 3. 环境、隔离与校准")
    lines.append("")
    lines.append("任务容器由已停止的 `cann850` 快照提交而来，使用 CANN 8.5.0，并只读挂载离线 Python 3.9.10 / Torch 2.5.1 / Torch NPU 2.5.1.post1 环境。物理 NPU 5/6/7 映射到逻辑 0/1/2；物理 4 因运行中的 `jinyr_vllm_new` 明确映射而不使用。每次 msprof 前均保存 `npu-smi` 快照并断言 5–7 无进程。")
    lines.append("")
    lines.append("快照内 CANN OPP 元数据为 `root:root 750`。为保持安装权限不变，CANN 编译、NPU 执行和 msprof 由隔离容器 root 运行；私有 OPP/wheel、全部离线解析、CSV 与报告由宿主 UID 9002 生成。容器无网络，且只映射三张任务卡。身份分工见 `metadata/execution_identity.txt`。")
    lines.append("")
    lines.append(markdown_table(
        ("物理卡", "逻辑卡", "case", "尝试", "P50(us)", "CV", "组内阈值", "纳入"),
        ((row["physical_device"], row["logical_device"], "tiny" if row["case"] == "rank1_int32_exact" else "large",
          row["attempt"], f(row["p50_us"]), f(row["cv_pct"], 2) + "%", row["within_card_limit_pct"] + "%",
          "是" if int(row["included"]) else "否") for row in calibration)))
    lines.append("")
    lines.append(markdown_table(
        ("校准 case", "纳入卡数", "卡间 P50 极差", "阈值", "结果"),
        ((row["case"], row["included_cards"], f(row["cross_card_spread_pct"], 2) + "%",
          row["cross_card_limit_pct"] + "%", "Pass" if int(row["pass"]) else "Fail") for row in calibration_summary)))
    lines.append("")
    lines.append("4 号卡未参与使跨卡覆盖从可选的四卡降为三卡，但本轮所有正式数据都来自固定的 5–7 三卡闭环；未将 4 号卡历史数据混入聚合。")
    lines.append("")

    lines.append("## 4. 正确性门禁")
    lines.append("")
    lines.append(markdown_table(
        ("层级", "覆盖", "结果"),
        (
            ("固定 + 宽行", "42 + 6", "48/48 bitwise Pass"),
            ("随机", "seed 20260721 / 20260830 / 91000007，各 100，rank 1–7", "300/300 bitwise Pass"),
            ("契约", "fp16/fp32/int32/int8，非空 rank2 中混合连续 `(0,)`", "4/4 Pass"),
            ("重复稳定性", "255/256 路、四 dtype 碎片、连续空、>64 KiB、超宽非对齐、特殊位模式", "11 组 × 10 次 Pass"),
        )))
    lines.append("")
    lines.append("浮点按 int16/int32 位视图逐 bit 比较，整数精确比较；任一失败都会阻止后续性能采集。完整日志为 `correctness/correctness_full.log`。")
    lines.append("")

    lines.append("## 5. 采集与统计口径")
    lines.append("")
    lines.append("- 延迟：6 个完整轮次，设备顺序 5→6→7→5→6→7；每轮使用归档的确定性乱序。每 case 30 task，丢弃第 1 条，统计其余 29 条 P50/P95/均值/CV/极值/AIV 时间/BlockDim。")
    lines.append("- 深采集：10 个锚点在三卡各采 `PipeUtilization`、`ArithmeticUtilization`、`Memory`、`MemoryL0`、`MemoryUB`、`L2Cache`、`ResourceConflictRatio` 和 sample-based。每个 metric/card 只启动一个 batch profile。")
    lines.append("- `Task Duration`：按 [CANN 8.5 OpBasicInfo](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/optool/atlasopdev_16_0098.html) 定义包含 task 调度、设备执行和响应结束，不等同于纯 AIV 执行时间；报告同时保留 `aiv_time(us)`。")
    lines.append("- 有效带宽：`2 × 输出字节 / Task Duration`，代表 task 级逻辑读+写双向有效吞吐；与 profiler 的 `aiv_main_mem_*_bw` 字段分别保留，不互相替代。")
    lines.append("- `msprof` 输出不提供流水图气泡时间，因此 Bound 只按各 pipeline busy ratio 判定，气泡数值不适用。")
    lines.append("")

    lines.append("## 6. Step 1：Tiling 理论建模与实测核对")
    lines.append("")
    lines.append("本轮逐行复现 Host `ChooseSplit`、Kernel 列交集定位、`SubmitStridedPiece` 与 `SubmitLinearRange`。92 个 case 中模型选择 row 路径 {} 个、column 路径 {} 个；每份 profiler 的 BlockDim 均与模型一致，否则解析器会失败。".format(row_paths, col_paths))
    lines.append("")
    lines.append("UB 为一个 `TQueBind<VECIN,VECOUT,1>`，两个 64 KiB slot，总计 128 KiB；一次二维搬运的行数上限为 `min(65536 / AlignUp32(pieceBytes), 4095)`。最小对齐块为 32 B，所以实际最大行数为 2048，4095 永远不能成为当前实现的主动上限。")
    lines.append("")
    rows_micro = [model_by_case["micro_rows_{:04d}".format(value)] for value in (2047, 2048, 2049, 4094, 4095, 4096)]
    lines.append(markdown_table(
        ("before 行数", "预测核数", "列块(B)", "SubmitTile", "最大 rows/tile", "跨 tile 片段", "4095 主动"),
        ((row["before_dim"], row["predicted_used_cores"], row["col_block_bytes"], row["submit_tiles"],
          row["max_rows_per_tile"], row["tile_transition_piece_count"], "是" if int(row["block_count_4095_active"]) else "否")
         for row in rows_micro)))
    lines.append("")

    lines.append("## 7. Step 2：卡间流水")
    lines.append("")
    lines.append("Concat 是单卡纯搬运算子，无 HCCS/SHMEM/集合通信，也无跨卡同步；本步骤不适用，跳过。三卡仅用于复测环境稳定性，不构成卡间流水。")
    lines.append("")
    lines.append("## 8. Step 3：核间流水")
    lines.append("")
    lines.append("Kernel 无跨核同步、原子依赖或核间通信；各核处理互斥输出范围。故不做核间同步流水分析，只通过 BlockDim、切分路径和跨核工作模型检查负载。")
    lines.append("")

    lines.append("## 9. Step 4：单核流水与 Bound 诊断")
    lines.append("")
    lines.append("[CANN Community Edition 8.5 PipeUtilization](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/devaids/optool/atlasopdev_16_0099.html) 定义各 ratio 的字段语义，但不提供本报告采用的 80%/70% 主 Bound 阈值。该阈值是本项目用于一致归类的诊断规则，不是 CANN 通用验收标准。Concat 为 AIV-only，实际 parser 只分类 MTE2、VEC、MTE3、Scalar 和无明显单一 Bound。")
    lines.append("")
    lines.append(markdown_table(
        ("锚点", "P50(us)", "核", "Scalar", "MTE2", "MTE3", "主 Bound", "有效双向 GB/s", "模型 DMA 次数", "平均 B/DMA"),
        ((row["case"], f(row["task_duration_us"]), f(row["block_dim"], 0),
          f(float(row["scalar_ratio"]) * 100, 1) + "%", f(float(row["mte2_ratio"]) * 100, 1) + "%",
          f(float(row["mte3_ratio"]) * 100, 1) + "%", row["bound"], f(row["effective_bidirectional_gbps"], 2),
          f(row["model_submit_tiles"], 0), f(row["model_avg_logical_bytes_per_dma"], 1)) for row in deep)))
    lines.append("")
    if sample_nonzero:
        lines.append("sample-based `task_cyc` 本轮存在非零值，但非零只是必要条件；仍需验证 task/core 映射和频率口径后才能用于逐核时长。主结论仍由 task-based 指标给出。")
    else:
        lines.append("三张卡的 sample-based `aicore.db` 中 `task_cyc` 均为 0，只能判定本轮该字段无有效数据；未继续定位是工具、配置、版本还是平台行为。本报告不据此推断逐核时长、频率或负载均衡。")
    if profiler_mte_fields:
        lines.append("本轮 Memory CSV 提供了 MTE 指令数字段；报告同时保留 profiler 实测与模型值。")
    else:
        lines.append("本轮 Memory CSV 未导出 MTE 指令数字段；表中的 DMA 次数均明确标为 Kernel 模型 `SubmitTile` 次数，不冒充硬件计数。")
    lines.append("")
    lines.append("### Memory、L2 与资源冲突交叉检查")
    lines.append("")
    lines.append(markdown_table(
        ("锚点", "主存读 GB/s", "主存写 GB/s", "L2 读命中", "L2 写命中", "bankgroup", "bank", "resource"),
        ((row["case"], f(row["main_mem_read_gbps"], 3), f(row["main_mem_write_gbps"], 3),
          f(row["l2_read_hit_pct"], 1) + "%", f(row["l2_write_hit_pct"], 1) + "%",
          f(row["vec_bankgroup_conflict_ratio"], 3), f(row["vec_bank_conflict_ratio"], 3),
          f(row["vec_resource_conflict_ratio"], 3)) for row in deep)))
    lines.append("")
    lines.append("三个已采集 Vector 冲突字段在全部锚点均为 0，因此当前数据不支持 UB bank/resource conflict 是主因；这不等于排除了所有 UB 或同步问题。L2 命中随访问规模与分片变化明显；`score_shape_2024x3000_fp32` 与超宽单行的读命中较低，但 Concat 是一次性流式读写，低命中本身不能证明启用更强缓存策略会受益。主存带宽列保留 profiler 原始字段口径，不能与按逻辑字节计算的双向有效带宽直接相加。")
    lines.append("")

    lines.append("## 10. 92-case 延迟与分组")
    lines.append("")
    for dimension, title in (("dtype", "dtype"), ("input_bucket", "输入数"), ("size_bucket", "输出规模"),
                             ("alignment", "行对齐"), ("split_path", "切分路径"), ("scope", "规格层级")):
        lines.append("### 按{}汇总".format(title))
        lines.append("")
        rows = group_rows(grouped, dimension)
        lines.append(markdown_table(
            ("组", "case", "P50 和(us)", "case P50 中位(us)", "聚合有效 GB/s"),
            ((row["group"], row["case_count"], f(row["local_p50_sum_us"]),
              f(row["case_p50_median_us"]), f(row["aggregate_effective_bidirectional_gbps"], 2)) for row in rows)))
        lines.append("")

    lines.append("### 最慢、最高波动与最低带宽")
    lines.append("")
    for kind, title in (("slowest_p50", "最慢 P50"), ("highest_across_round_cv", "最高跨轮 CV"),
                        ("lowest_effective_bandwidth", "最低有效带宽")):
        selected = [row for row in rankings if row["ranking"] == kind][:10]
        lines.append("**{}**".format(title))
        lines.append("")
        lines.append(markdown_table(("排名", "case", "值", "单位", "层级"),
                                    ((row["rank"], row["case"], f(row["value"]), row["unit"], row["scope"]) for row in selected)))
        lines.append("")

    lines.append("### 微基准转折")
    lines.append("")
    micro_groups = (
        ("输入数曲线", ["micro_inputs_{:03d}".format(value) for value in (1, 2, 4, 8, 16, 32, 64, 128, 256)]),
        ("固定 46,200 B 核扩展", ["micro_cores_{:02d}".format(value) for value in (1, 2, 3, 5, 7, 11, 20, 40)]),
        ("对齐边界", ["micro_align_{:03d}b".format(value) for value in (31, 32, 33, 511, 512, 513)]),
        ("64 KiB 分片边界", ["micro_piece_{:05d}b".format(value) for value in (65535, 65536, 65537)]),
        ("二维 DMA 行数边界", ["micro_rows_{:04d}".format(value) for value in (2047, 2048, 2049, 4094, 4095, 4096)]),
    )
    for title, names in micro_groups:
        lines.append("**{}**".format(title))
        lines.append("")
        lines.append(markdown_table(
            ("case", "输入数", "输出 B", "行 B", "路径/核", "P50(us)", "SubmitTile", "B/DMA", "有效 GB/s"),
            ((name, model_by_case[name]["input_count"], model_by_case[name]["output_bytes"],
              model_by_case[name]["output_row_bytes"], "{}/{}".format(model_by_case[name]["predicted_split_path"], model_by_case[name]["predicted_used_cores"]),
              f(by_case[name]["p50_us"]), model_by_case[name]["submit_tiles"],
              f(model_by_case[name]["avg_logical_bytes_per_dma"], 1), f(by_case[name]["effective_bidirectional_gbps"], 2))
             for name in names)))
        lines.append("")

    lines.append("## 11. 历史 sanity check 与官方目标")
    lines.append("")
    lines.append("共享 39 case 相对 2026-07-21 历史采集的 P50 比值中位数为 {}，范围 {}–{}。由于日期、卡号、温度、容器和 profiler 运行均不同，这只用于发现数量级异常，不是严格 A/B，也不用于评价当前源码相对历史实现的收益。".format(f(statistics.median(history_ratios), 4), f(min(history_ratios), 4), f(max(history_ratios), 4)))
    lines.append("")
    lines.append(markdown_table(
        ("官方 case", "基线(us)", "占比", "目标解释"),
        (
            ("Case1", "10.6000", "1.88%", "隐藏 shape，不映射本地 case"),
            ("Case2", "32.6205", "5.78%", "隐藏 shape，不映射本地 case"),
            ("Case3", "18.0405", "3.19%", "隐藏 shape，不映射本地 case"),
            ("Case4", "104.0925", "18.43%", "第二优化目标"),
            ("Case5", "399.2980", "70.72%", "第一优化目标"),
            ("总计", "564.6515", "100%", "官方反馈，非本地 92-case 和"),
        )))
    lines.append("")
    lines.append("从 564.6515 us 降至 300 us 需要减少 **264.6515 us**。若 Case1–4 完全不变，Case5 必须由 399.2980 us 降至不高于 **134.6465 us**，即至少改善 264.6515 us（约 66.28%）。该算术不包含任何隐藏 shape 推断。")
    lines.append("")

    lines.append("## 12. 证据排序的优化建议与 A/B 门槛")
    lines.append("")
    lines.append("1. **单输入直通。** 当前通用路径仍执行 TensorList 定位、队列与输入循环；历史 P2/P2.1 的 Identity 局部结果证明该方向可能有效，但历史总体方案不自动继承。预期指标：单输入 case 的 Scalar ratio、模型/实测 DMA 发射和 P50 下降。门槛：固定 6 轮中至少 5/6 更快，单输入 P50 至少改善 `max(1 us, 5%)`，92-case 总和不回退超过 1%，全部 bitwise 门禁通过。")
    lines.append("2. **低行数、超宽、非 32 B 行的安全并行。** row fallback 在 before=1/8 时并行度不足，是超宽数据吞吐受限的重要来源。候选必须按 32 B 写所有权或完全独立的连续 span 切分，不能让两个核写同一数据块。预期指标：BlockDim/有效带宽上升、P50 降低且非对齐尾部 bitwise 稳定。门槛：六个宽行 case 中至少 5/6 改善，两个 before=1 锚点至少改善 10%，重复 10 次无写冲突，其他 scoring proxy 无 material 回退。")
    lines.append("3. **小任务 TilingData/TPipe 固定开销。** tiny 锚点用于判断 Scalar/启动占比；历史 P0 的紧凑 Tiling 有局部收益，但 256 路出现回退，因此只能逐项移植。预期指标：tiny Scalar ratio 与 Task-AIV gap 下降。门槛：tiny P50 改善 `max(0.5 us, 8%)`，输入数 64/255/256 三档均不回退超过 `max(1 us, 2%)`。")
    lines.append("4. **256 路小 DMA 发射。** 优先减少重复前缀定位、合并物理连续且语义连续的小片段，或为长 TensorList 建立低成本 checkpoint；不能声称减少 DMA，除非 profiler 字段或严格 Kernel 模型均证明。历史 P3 checkpoint 曾改善专项 fp32/int8，却未通过全局 5 轮门禁。门槛：四 dtype 碎片控制至少 3/4 改善、无一项回退超过 `max(2 us, 2%)`，92-case 总和至少 4/6 轮改善。")
    lines.append("5. **对齐 DataCopy 快路。** 31/32/33 B 与 511/512/513 B 微基准用于判断 DataCopyPad 参数开销和列切分转折；候选可在严格对齐、连续且无 padding 时使用更直接的搬运形式。门槛：32/512 B 两点改善至少 5%，相邻非对齐点 bitwise 正确且 P50 不回退超过 2%，BlockDim 仍与 Host 模型一致。")
    lines.append("")
    lines.append("P2 的宽泛 Tiny/FlatSpan 路由已被历史五轮数据证伪，P3 checkpoint 也因全局门禁失败撤回；后续不得把这些整体方案原样恢复。可复用的是局部假设和单变量实验设计，不是历史结论。")
    lines.append("")

    lines.append("## 13. 可追溯性与限制")
    lines.append("")
    lines.append("- 源码、包、Kernel `.o`、运行库与环境 SHA-256：`metadata/*_sha256.txt`、`metadata/environment.txt`。")
    lines.append("- 容器 PID/namespace/设备映射、总线、温度与进程：`metadata/host_pre_state.txt`、`host_post_state.txt`、各 `device_gate_*.txt`。")
    lines.append("- case 与顺序：`case_manifest.csv`、`round_orders/round_*.txt`。")
    lines.append("- 正确性：`correctness/correctness_full.log`。")
    lines.append("- 延迟：`latency_task_samples.csv`、`latency_round_case_summary.csv`、`latency_case_summary.csv`、`latency_group_summary.csv`。")
    lines.append("- 深度指标：`deep_metric_values.csv`、`deep_card_summary.csv`、`deep_case_summary.csv`、`deep_sample_status.csv`。")
    lines.append("- 原始 PROF 和私有 OPP 不提交仓库，但 `scripts/host_run_all.sh` 可从当前源码重建。")
    lines.append("- 官方隐藏输入与采集产物仍不可见；本报告只能筛选普适优化假设，最终收益必须由官方 5-case 提交反馈确认。")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print("wrote report {}".format(REPORT))


if __name__ == "__main__":
    main()
