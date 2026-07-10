#!/usr/bin/env python3
"""S5 Case Mapper — SquareSumV1

将 S2P2_cases.json 的抽象参数组合映射为具体的 tensor 构造配置（shape + dtype）。
纯计算模块，无 torch / torch_npu 依赖。

运行时加载 S2P1_operator_model.json 获取 shape_mapping（唯一真相来源）。
"""

import ast
import copy
import heapq
import json
import math
import operator
import os
import random

# ---------------------------------------------------------------------------
# 运行时加载 — 禁止硬编码 shape_mapping / operator_model
# ---------------------------------------------------------------------------

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_BASE_DIR, "S2P1_operator_model.json")) as f:
    _data = json.load(f)

SM = _data["shape_mapping"]
MODEL = {"inputs": _data["inputs"], "outputs": _data["outputs"]}


# ---------------------------------------------------------------------------
# 数学工具
# ---------------------------------------------------------------------------

def _prime_factors(n: int) -> list:
    """返回质因数列表（降序）。如 49 → [7, 7]，128 → [2,2,2,2,2,2,2]。"""
    if n <= 0:
        return [n] if n == 0 else []
    primes = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            primes.append(d)
            n //= d
        d += 1
    if n > 1:
        primes.append(n)
    primes.sort(reverse=True)
    return primes


def _balanced_decompose(num: int, parts: int) -> tuple:
    """将 num 分解为 parts 个整数因子，满足 math.prod(result) == num，各因子尽量接近。

    算法：① 质因数分解；② 初始化 parts 个桶为 [1]*parts；
    ③ 最小堆，每次弹出最小桶乘以最大质因数。

    允许因子=1。不允许 num=0 调用（parts>0 时返回全 0 元组做防御）。
    """
    if parts <= 0:
        return ()
    if num == 0:
        return tuple([0] * parts)
    if num == 1:
        return tuple([1] * parts)

    primes = _prime_factors(num)
    buckets = [1] * parts
    heapq.heapify(buckets)
    for p in primes:
        smallest = heapq.heappop(buckets)
        heapq.heappush(buckets, smallest * p)
    return tuple(sorted(buckets))


def _balanced_decompose_nontrivial(num: int, parts: int) -> tuple:
    """同 _balanced_decompose 但要求每个因子 > 1。

    若质因数不足 parts 个，返回含 1 的降级结果。
    """
    if parts <= 0:
        return ()
    if num == 0:
        return tuple([0] * parts)

    primes = _prime_factors(num)
    if len(primes) < parts:
        return _balanced_decompose(num, parts)

    buckets = [1] * parts
    heapq.heapify(buckets)
    for p in primes:
        smallest = heapq.heappop(buckets)
        heapq.heappush(buckets, smallest * p)
    return tuple(sorted(buckets))


# ---------------------------------------------------------------------------
# AST 白名单 — 用于 when 表达式和 derived.expr 求值
# ---------------------------------------------------------------------------

_AST_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_AST_CMP_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

_AST_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
}


def _eval_ast(node, context: dict):
    """AST 白名单求值器。违规表达式抛 ValueError。"""
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body, context)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in context:
            return context[node.id]
        if node.id in ("True", "False", "None"):
            return {"True": True, "False": False, "None": None}[node.id]
        raise ValueError(f"unknown name: {node.id}")
    if isinstance(node, ast.BinOp):
        op_func = _AST_BIN_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"unsupported binop: {type(node.op).__name__}")
        return op_func(_eval_ast(node.left, context), _eval_ast(node.right, context))
    if isinstance(node, ast.UnaryOp):
        op_func = _AST_UNARY_OPS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"unsupported unaryop: {type(node.op).__name__}")
        return op_func(_eval_ast(node.operand, context))
    if isinstance(node, ast.Compare):
        left = _eval_ast(node.left, context)
        for op, comparator in zip(node.ops, node.comparators):
            op_func = _AST_CMP_OPS.get(type(op))
            if op_func is None:
                raise ValueError(f"unsupported cmpop: {type(op).__name__}")
            right = _eval_ast(comparator, context)
            if not op_func(left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.BoolOp):
        values = [_eval_ast(v, context) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(values)
        else:
            return any(values)
    if isinstance(node, ast.Subscript):
        value = _eval_ast(node.value, context)
        if isinstance(node.slice, ast.Slice):
            lo = _eval_ast(node.slice.lower, context) if node.slice.lower else None
            hi = _eval_ast(node.slice.upper, context) if node.slice.upper else None
            step = _eval_ast(node.slice.step, context) if node.slice.step else None
            return value[lo:hi:step]
        idx = _eval_ast(node.slice, context)
        return value[idx]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_ast(e, context) for e in node.elts)
    if isinstance(node, ast.List):
        return [_eval_ast(e, context) for e in node.elts]
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "len":
            return len(_eval_ast(node.args[0], context))
        raise ValueError(f"unsupported call: {ast.dump(node)}")
    raise ValueError(f"unsupported AST node: {type(node).__name__}")


def _eval_expr(expr: str, context: dict):
    """编译并安全求值表达式字符串。"""
    tree = ast.parse(expr, mode="eval")
    return _eval_ast(tree, context)


def _eval_when(expr: str, case: dict) -> bool:
    """按 schema AST 白名单求值 when 表达式。"""
    try:
        result = _eval_expr(expr, dict(case))
        return bool(result)
    except (ValueError, SyntaxError, TypeError, KeyError, IndexError):
        return False


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def _resolve_param(case: dict, param_name: str, shape_params: dict):
    """按 case[param] > group_defaults[group] > default 优先级解析参数。"""
    if param_name in case:
        return case[param_name]
    sp = shape_params.get(param_name, {})
    group = case.get("_group")
    if group and "group_defaults" in sp and group in sp["group_defaults"]:
        return sp["group_defaults"][group]
    if "default" in sp:
        return sp["default"]
    return None


def _sample_attr(case: dict, name: str, spec: dict, rng: random.Random):
    """采样或直接取 attrs 属性值。"""
    param = spec.get("param", name)
    if param in case:
        val = case[param]
        coerce = spec.get("coerce")
        if coerce == "float":
            val = float(val)
        elif coerce == "int":
            val = int(val)
        elif coerce == "bool":
            val = bool(val)
        return val

    sampling = spec.get("sampling")
    if sampling is None:
        return spec.get("default")

    strategy = sampling.get("strategy", "default")
    if strategy == "log_uniform":
        lo, hi = sampling["range"]
        return 10 ** rng.uniform(math.log10(lo), math.log10(hi))
    elif strategy == "uniform":
        lo, hi = sampling["range"]
        return rng.uniform(lo, hi)
    elif strategy == "choice":
        return rng.choice(sampling["values"])
    elif strategy == "default":
        return spec.get("default", sampling.get("default"))
    return spec.get("default")


# ---------------------------------------------------------------------------
# ndim 约束
# ---------------------------------------------------------------------------

def _apply_ndim_constraints(lo: int, hi: int, constraints: list, case: dict):
    """对 ndim 范围应用 tensor_constraints 过滤。返回 (lo, hi)。"""
    for c in constraints:
        when_expr = c.get("when", "True")
        if _eval_when(when_expr, case):
            lo = max(lo, c.get("min", lo))
            hi = min(hi, c.get("max", hi))
    if lo > hi:
        lo = hi
    return lo, hi


# ---------------------------------------------------------------------------
# derived.expr 中用到的 Tensor 代理对象
# ---------------------------------------------------------------------------

class _TensorProxy:
    def __init__(self, shape):
        self.shape = tuple(shape)


# ---------------------------------------------------------------------------
# 输出 shape 推导
# ---------------------------------------------------------------------------

def _compute_output_shape(expr, inputs, attrs, dtype_str, sm):
    """根据 derived.expr 和已构造的 inputs 推导输出 shape。"""
    ctx = {}
    for tname, tspec_dict in inputs.items():
        if tspec_dict is not None:
            ctx[tname] = _TensorProxy(tspec_dict["shape"])
            ctx[f"{tname}_ndim"] = len(tspec_dict["shape"])
    ctx["len"] = len
    for k, v in attrs.items():
        ctx[k] = v
    try:
        result_shape = _eval_expr(expr, ctx)
        if isinstance(result_shape, (list, tuple)):
            result_shape = list(result_shape)
        else:
            result_shape = [result_shape]
    except (ValueError, TypeError):
        # fallback: 直接切片
        input_shape = inputs.get("input", {}).get("shape", [])
        if len(input_shape) >= 2:
            result_shape = input_shape[:-2] + input_shape[-1:]
        else:
            result_shape = input_shape
    return result_shape


# ---------------------------------------------------------------------------
# 核心：map_case
# ---------------------------------------------------------------------------

def map_case(case: dict, rng: random.Random, sm: dict = None, model: dict = None) -> dict:
    """核心映射：单个 case → tensor 构造配置。"""
    if sm is None:
        sm = SM
    if model is None:
        model = MODEL

    # --- 1. 解析 dtype ---
    dtype_str = case[sm["dtype"]["param"]]

    # --- 2. 解析 attrs ---
    shape_params = sm.get("shape_params", {})
    attrs = {}
    for name, spec in sm.get("attrs", {}).items():
        attrs[name] = _sample_attr(case, name, spec, rng)

    # --- 3. 解析 shape_params（resolved 值缓存）---
    resolved_shape_params = {}
    for pname in shape_params:
        val = _resolve_param(case, pname, shape_params)
        if val is not None:
            resolved_shape_params[pname] = val

    # --- 4. 确定各 tensor 的 ndim ---
    ndim_cfg = sm["ndim"]
    base_lo, base_hi = ndim_cfg["range"]
    tensor_constraints = ndim_cfg.get("tensor_constraints", {})

    # 预计算所有 tensor 的 ndim 约束范围
    tensor_ndim_ranges = {}
    for tname in sm.get("inputs", {}):
        constraints = tensor_constraints.get(tname, [])
        lo, hi = _apply_ndim_constraints(base_lo, base_hi, constraints, case)
        tensor_ndim_ranges[tname] = (lo, hi)

    # 找主 tensor（第一个 decompose 且有 leading 的输入）
    input_specs = sm.get("inputs", {})
    main_tensor = None
    for tname, tspec in input_specs.items():
        if tspec.get("rule") == "decompose":
            decomp = tspec.get("decompose", {})
            if "leading" in decomp:
                main_tensor = tname
                break
    if main_tensor is None:
        for tname, tspec in input_specs.items():
            if tspec.get("rule") == "decompose":
                main_tensor = tname
                break

    # 主 tensor ndim
    main_ndim = None
    if main_tensor:
        lo, hi = tensor_ndim_ranges[main_tensor]
        main_spec = input_specs[main_tensor]
        align_target = main_spec.get("align_trailing_with")
        if align_target and align_target in tensor_ndim_ranges:
            tlo, _ = tensor_ndim_ranges[align_target]
            lo = max(lo, tlo)
        decomp = main_spec.get("decompose", {})
        if "leading" in decomp:
            leading_param = decomp["leading"]["param"]
            leading_val = _resolve_param(case, leading_param, shape_params)
            if leading_val is not None and leading_val > 1:
                trailing_len = len(decomp.get("trailing", []))
                lo = max(lo, trailing_len + 1)
        if lo > hi:
            lo = hi
        ndim_range = list(range(lo, hi + 1))
        main_ndim = rng.choice(ndim_range) if ndim_range else hi

    # 其余 decompose tensor ndim
    tensor_ndims = {}
    if main_tensor:
        tensor_ndims[main_tensor] = main_ndim
    for tname, tspec in input_specs.items():
        if tname == main_tensor or tspec.get("rule") != "decompose":
            continue
        lo, hi = tensor_ndim_ranges.get(tname, (base_lo, base_hi))
        align_target = tspec.get("align_trailing_with")
        if align_target and align_target in tensor_ndim_ranges:
            tlo, _ = tensor_ndim_ranges[align_target]
            lo = max(lo, tlo)
        if main_ndim is not None:
            hi = min(hi, main_ndim)
        decomp = tspec.get("decompose", {})
        if "leading" not in decomp:
            tensor_ndims[tname] = len(decomp.get("trailing", []))
        else:
            if lo > hi:
                lo = hi
            ndim_range = list(range(lo, hi + 1))
            tensor_ndims[tname] = rng.choice(ndim_range) if ndim_range else hi

    # --- 5. 构造 inputs ---
    inputs = {}

    def _build_decompose(tname, tspec):
        decomp = tspec.get("decompose", {})
        ndim = tensor_ndims.get(tname, base_lo)

        # trailing shape
        trailing_shape = []
        for t_elem in decomp.get("trailing", []):
            t_param = t_elem["param"]
            t_val = _resolve_param(case, t_param, shape_params)
            if t_val is None:
                t_val = shape_params.get(t_param, {}).get("default", 1)
            trailing_shape.append(int(t_val))

        # leading shape
        parts_expr = decomp.get("parts_expr", "ndim - len(trailing)")
        context = {"ndim": ndim, "trailing": tuple(trailing_shape), "len": len}
        leading_dims_count = _eval_expr(parts_expr, context)

        if leading_dims_count == 0:
            leading_shape = ()
        else:
            leading_cfg = decomp.get("leading")
            if leading_cfg:
                product = _resolve_param(case, leading_cfg["param"], shape_params)
                if product is None:
                    product = shape_params.get(leading_cfg["param"], {}).get("default", 1)
                product = int(product)
                strategy = leading_cfg.get("strategy", "balanced")
                if strategy == "balanced_nontrivial":
                    leading_shape = _balanced_decompose_nontrivial(product, leading_dims_count)
                    if any(f <= 1 for f in leading_shape):
                        ndim_fb = tspec.get("ndim_fallback")
                        if ndim_fb is not None:
                            merged = math.prod(trailing_shape) if trailing_shape else 1
                            new_trailing = [merged]
                            new_parts = ndim - ndim_fb
                            if new_parts <= 0:
                                leading_shape = ()
                                trailing_shape = new_trailing
                            else:
                                leading_shape = _balanced_decompose(product, new_parts)
                                trailing_shape = new_trailing
                else:
                    leading_shape = _balanced_decompose(product, leading_dims_count)
            else:
                leading_shape = ()

        shape = tuple(leading_shape) + tuple(trailing_shape)
        return {"shape": list(shape), "dtype": dtype_str}

    for tname, tspec in input_specs.items():
        rule = tspec.get("rule")
        if rule == "decompose":
            inputs[tname] = _build_decompose(tname, tspec)
        elif rule == "sync_with":
            target = tspec["sync_with"]
            inputs[tname] = {"shape": list(inputs[target]["shape"]),
                             "dtype": inputs[target]["dtype"]}
        elif rule == "fixed":
            inputs[tname] = {"shape": list(tspec["shape"]),
                             "dtype": tspec.get("dtype", dtype_str)}
        elif rule == "optional":
            opt_param = tspec["optional_param"]
            if case.get(opt_param):
                swp = tspec["shape_when_present"]
                if swp.get("rule") == "decompose":
                    inputs[tname] = _build_decompose(tname, {"rule": "decompose",
                                                              "decompose": swp.get("decompose", {})})
                elif swp.get("rule") == "fixed":
                    inputs[tname] = {"shape": list(swp["shape"]),
                                     "dtype": swp.get("dtype", dtype_str)}
            else:
                inputs[tname] = None

    # --- 6. 构造 outputs ---
    outputs = {}
    for oname, ospec in sm.get("outputs", {}).items():
        rule = ospec.get("rule")
        if rule == "same_as":
            target = ospec["same_as"]
            src = inputs.get(target) if target in inputs else outputs.get(target)
            if src is None:
                outputs[oname] = None
            else:
                outputs[oname] = {"shape": list(src["shape"]), "dtype": src["dtype"]}
        elif rule == "derived":
            derived = ospec.get("derived", {})
            expr = derived.get("expr", "")
            dtype_override = derived.get("dtype_override")
            result_shape = _compute_output_shape(expr, inputs, attrs, dtype_str, sm)
            out_dtype = dtype_override if dtype_override else dtype_str
            outputs[oname] = {"shape": result_shape, "dtype": out_dtype}
        elif rule == "fixed":
            outputs[oname] = {"shape": list(ospec.get("shape", [])),
                              "dtype": ospec.get("dtype", dtype_str)}

    # --- 7. 组装返回值 ---
    params = {**case}
    params.pop("source", None)
    params.pop("reason", None)
    params.update(resolved_shape_params)
    params.update(attrs)

    return {
        "params": params,
        "tensors": {"inputs": inputs, "outputs": outputs}
    }


# ---------------------------------------------------------------------------
# validate_config
# ---------------------------------------------------------------------------

def validate_config(cfg: dict, case: dict, sm: dict = None, model: dict = None) -> list:
    """动态校验，返回错误列表（空 = pass）。"""
    if sm is None:
        sm = SM
    if model is None:
        model = MODEL

    errors = []
    inputs = cfg["tensors"]["inputs"]
    outputs = cfg["tensors"]["outputs"]

    # 输入 dtype ∈ 合法值列表
    for inp_spec in model["inputs"]:
        name = inp_spec["name"]
        if name in inputs and inputs[name] is not None:
            valid_dtypes = inp_spec["dtype"]["values"]
            if inputs[name]["dtype"] not in valid_dtypes:
                errors.append(f"input '{name}' dtype '{inputs[name]['dtype']}' not in {valid_dtypes}")

    # 输入 ndim ∈ 范围
    for inp_spec in model["inputs"]:
        name = inp_spec["name"]
        if name in inputs and inputs[name] is not None:
            rank_min = inp_spec["rank"]["min"]
            rank_max = inp_spec["rank"]["max"]
            ndim = len(inputs[name]["shape"])
            if ndim < rank_min or ndim > rank_max:
                errors.append(f"input '{name}' ndim {ndim} not in [{rank_min}, {rank_max}]")

    # align_trailing_with 校验
    for tname, tspec in sm.get("inputs", {}).items():
        align_target = tspec.get("align_trailing_with")
        if align_target and tname in inputs and inputs[tname] is not None:
            src_shape = inputs[tname]["shape"]
            tgt_dict = inputs.get(align_target)
            if tgt_dict is not None:
                tgt_shape = tgt_dict["shape"]
                n = len(src_shape)
                if n <= len(tgt_shape):
                    if src_shape != list(tgt_shape[-n:]):
                        errors.append(
                            f"align_trailing_with: {tname}.shape {src_shape} != "
                            f"{align_target}.shape[-{n}:] {tgt_shape[-n:]}")
                else:
                    errors.append(
                        f"align_trailing_with: {tname}.ndim {len(src_shape)} > "
                        f"{align_target}.ndim {len(tgt_shape)}")

    # tensor_constraints 校验
    tensor_constraints = sm.get("ndim", {}).get("tensor_constraints", {})
    for tname, constraints in tensor_constraints.items():
        if tname in inputs and inputs[tname] is not None:
            ndim = len(inputs[tname]["shape"])
            for c in constraints:
                when_expr = c.get("when", "True")
                if _eval_when(when_expr, case):
                    if "min" in c and ndim < c["min"]:
                        errors.append(
                            f"tensor_constraints: {tname}.ndim {ndim} < min {c['min']}")
                    if "max" in c and ndim > c["max"]:
                        errors.append(
                            f"tensor_constraints: {tname}.ndim {ndim} > max {c['max']}")

    return errors


# ---------------------------------------------------------------------------
# 批量加载
# ---------------------------------------------------------------------------

def load_mapped_configs(cases_file: str, model_file: str, seed: int = 42) -> list:
    """批量映射：加载 JSON → 逐 case 映射 → 校验。"""
    with open(cases_file) as f:
        cases = json.load(f)
    rng = random.Random(seed)
    results = []
    all_errors = []

    for i, case in enumerate(cases):
        cfg = map_case(case, rng, SM, MODEL)
        errors = validate_config(cfg, case, SM, MODEL)
        if errors:
            all_errors.extend([f"case {i:05d} ({case.get('_group', '?')}): {e}" for e in errors])
        results.append({
            "id": f"case{i:05d}",
            "params": cfg["params"],
            "tensors": cfg["tensors"]
        })

    if all_errors:
        print(f"VALIDATION ERRORS ({len(all_errors)}):")
        for e in all_errors:
            print(f"  {e}")
    else:
        print(f"All {len(results)} cases passed validation.")

    return results


# ---------------------------------------------------------------------------
# 网络 case 映射 — 直接从 explicit shape/dim/keepdim 构造
# ---------------------------------------------------------------------------

def map_network_case(net_config: dict, idx: int) -> dict:
    """将网络配置（explicit shape + dim + keepdim）映射为 tensor 构造配置。"""
    cfg = net_config["config"]
    shape = list(cfg["shape"])
    dim = cfg["dim"]
    dtype = cfg["dtype"]
    keepdim = cfg.get("keepdim", False)

    # 规范化 dim 为列表
    if isinstance(dim, int):
        dim_list = [dim]
    else:
        dim_list = list(dim)

    # 规范化负索引
    ndim = len(shape)
    norm_dims = sorted(set((d + ndim) if d < 0 else d for d in dim_list))

    # 计算 output shape
    out_shape = []
    for i in range(ndim):
        if i in norm_dims:
            if keepdim:
                out_shape.append(1)
        else:
            out_shape.append(shape[i])

    # axis 推断：reduce 最内层（-1 或连续尾部）
    # 对于 SquareSumV1，axis 在 shape_mapping 中是隐式的（rLength = reduce axis）
    # 但网络 case 的 axis 直接来自 dim
    # 从 dim 推断 axis 值
    axis = list(dim_list)  # 保留原始 dim（可能含负索引）

    params = {
        "dtype": dtype,
        "keep_dims": keepdim,
        "axis": axis,
        "shape": shape,
    }
    # 不透传 source/reason（禁止规则 6.1.9）

    return {
        "id": f"network{idx:05d}",
        "params": params,
        "tensors": {
            "inputs": {
                "input": {"shape": shape, "dtype": dtype}
            },
            "outputs": {
                "result": {"shape": out_shape, "dtype": dtype}
            }
        }
    }


def load_network_configs(low_configs_file: str) -> list:
    """批量映射网络用例。"""
    with open(low_configs_file) as f:
        configs = json.load(f)
    results = []
    for i, cfg in enumerate(configs):
        results.append(map_network_case(cfg, i))
    print(f"Mapped {len(results)} network cases.")
    return results


# ---------------------------------------------------------------------------
# 空 tensor 补全
# ---------------------------------------------------------------------------

def supplement_empty_cases(sm: dict, path_cases: list, seed: int = 42) -> list:
    """从 path case 模板生成空 tensor 变体，最大化维度多样性。"""
    if not path_cases:
        return []

    shape_params = sm.get("shape_params", {})
    input_specs = sm.get("inputs", {})

    # Phase 1: 场景发现
    scenarios = []
    for tname, tspec in input_specs.items():
        if tspec.get("rule") != "decompose":
            continue
        decomp = tspec.get("decompose", {})
        leading_cfg = decomp.get("leading")
        if leading_cfg:
            scenarios.append({
                "name": f"{tname}_{leading_cfg['param']}",
                "tensor": tname,
                "region": "leading",
                "param": leading_cfg["param"]
            })
        for i, t_elem in enumerate(decomp.get("trailing", [])):
            scenarios.append({
                "name": f"{tname}_{t_elem['param']}",
                "tensor": tname,
                "region": "trailing",
                "param": t_elem["param"],
                "trailing_idx": i
            })

    template = copy.deepcopy(path_cases[0])
    empty_cases = []
    seen_shapes = set()

    def _update_outputs(case_data):
        inputs_d = case_data["tensors"]["inputs"]
        outputs_d = case_data["tensors"]["outputs"]
        for oname, ospec in sm.get("outputs", {}).items():
            rule = ospec.get("rule")
            if rule == "same_as":
                target = ospec["same_as"]
                src = inputs_d.get(target)
                if src is not None:
                    outputs_d[oname] = {"shape": list(src["shape"]), "dtype": src["dtype"]}
            elif rule == "derived":
                derived = ospec.get("derived", {})
                expr = derived.get("expr", "")
                dtype_override = derived.get("dtype_override")
                ctx = {}
                for tn, ts in inputs_d.items():
                    if ts is not None:
                        ctx[tn] = _TensorProxy(ts["shape"])
                        ctx[f"{tn}_ndim"] = len(ts["shape"])
                ctx["len"] = len
                try:
                    result_shape = _eval_expr(expr, ctx)
                    if isinstance(result_shape, (list, tuple)):
                        result_shape = list(result_shape)
                    else:
                        result_shape = [result_shape]
                except (ValueError, TypeError):
                    inp_shape = inputs_d.get("input", {}).get("shape", [])
                    if len(inp_shape) >= 2:
                        result_shape = inp_shape[:-2] + inp_shape[-1:]
                    else:
                        result_shape = inp_shape
                out_dtype = dtype_override if dtype_override else \
                    inputs_d.get("input", {}).get("dtype", "float16")
                outputs_d[oname] = {"shape": result_shape, "dtype": out_dtype}

    # Phase 2: 逐场景生成 single_zero 变体
    for sc in scenarios:
        variant = copy.deepcopy(template)
        tname = sc["tensor"]
        inp = variant["tensors"]["inputs"].get(tname)
        if inp is None:
            continue
        shape = inp["shape"]
        decomp = input_specs.get(tname, {}).get("decompose", {})
        trailing = decomp.get("trailing", [])

        if sc["region"] == "trailing":
            leading_count = len(shape) - len(trailing)
            dim_idx = leading_count + sc["trailing_idx"]
            if dim_idx < len(shape):
                new_shape = list(shape)
                new_shape[dim_idx] = 0
                inp["shape"] = new_shape
        elif sc["region"] == "leading":
            leading_count = len(shape) - len(trailing)
            new_shape = list(shape)
            for i in range(leading_count):
                if i < len(new_shape):
                    new_shape[i] = 0
            inp["shape"] = new_shape

        variant["params"][sc["param"]] = 0
        _update_outputs(variant)

        shape_key = json.dumps(variant["tensors"]["inputs"], sort_keys=True)
        if shape_key not in seen_shapes:
            seen_shapes.add(shape_key)
            empty_inputs = [tn for tn, ts in variant["tensors"]["inputs"].items()
                            if ts is not None and 0 in ts.get("shape", [])]
            group_name = "_".join(empty_inputs) + "_empty" if empty_inputs else "empty"
            variant["id"] = f"case_empty_{sc['name']}"
            variant["params"]["_group"] = group_name
            empty_cases.append(variant)

    # all_zero 兜底
    all_zero = copy.deepcopy(template)
    for tname, tspec in all_zero["tensors"]["inputs"].items():
        if tspec is not None:
            tspec["shape"] = [0] * len(tspec["shape"])
    _update_outputs(all_zero)
    empty_inputs = [tname for tname, tspec in all_zero["tensors"]["inputs"].items()
                    if tspec is not None]
    all_zero["params"]["_group"] = "_".join(empty_inputs) + "_empty"
    all_zero["id"] = "case_empty_all_zero"
    empty_cases.append(all_zero)

    return empty_cases


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    cases_file = os.path.join(_BASE_DIR, "S2P2_cases.json")
    model_file = os.path.join(_BASE_DIR, "S2P1_operator_model.json")
    output_file = os.path.join(_BASE_DIR, "S5_mapped_cases_path.json")

    results = load_mapped_configs(cases_file, model_file, seed=42)

    with open(output_file, "w") as f:
        json.dump({"cases": results}, f, ensure_ascii=False)

    print(f"Wrote {len(results)} cases to {output_file}")
    return results


if __name__ == "__main__":
    main()
