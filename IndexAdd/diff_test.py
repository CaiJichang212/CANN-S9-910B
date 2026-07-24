"""IndexAdd custom OPP differential matrix.

Run after installing this operator and rebuilding ``custom_ops_lib``:

    NPU_DEVICE=6 python3 diff_test.py

The cases deliberately exercise both tiling paths, ranks 1--6, non-32B
vectors, scalar scatter, duplicate ordering, BF16 rounding, FP16 magnitude
loss, and integer overflow/exact accumulation.
"""
import os

import torch
import torch_npu  # noqa: F401

import custom_ops_lib


DEVICE = int(os.environ.get("NPU_DEVICE", "0"))


def _source_shape(shape, dim, index_len):
    result = list(shape)
    result[dim] = index_len
    return result


def _float_values(shape, dtype, low=-1.0, high=1.0):
    return (torch.rand(shape, dtype=torch.float32) * (high - low) + low).to(dtype)


def _int_values(shape, dtype, low, high):
    return torch.randint(low, high, shape, dtype=dtype)


def _case(name, shape, dim, dtype, index, self_data, source_data):
    return {
        "name": name,
        "shape": shape,
        "dim": dim,
        "dtype": dtype,
        "index": torch.tensor(index, dtype=torch.int32),
        "self": self_data,
        "source": source_data,
    }


def build_cases():
    cases = []
    # FP32 unaligned, M=1, and repeated positions cover owner bucketing.
    shape = (17, 9)
    cases.append(_case("fp32_unaligned_m1", shape, 0, torch.float32, [3],
                       _float_values(shape, torch.float32, -1000, 1000),
                       _float_values(_source_shape(shape, 0, 1), torch.float32, -1000, 1000)))
    shape = (2, 7, 5)
    cases.append(_case("fp32_mid_dim_repeat", shape, 1, torch.float32, [6, 0, 6, 3, 6],
                       _float_values(shape, torch.float32),
                       _float_values(_source_shape(shape, 1, 5), torch.float32)))
    # 256B is the atomic threshold for FP32 (64 elements).
    shape = (11, 64)
    cases.append(_case("fp32_atomic_threshold", shape, 0, torch.float32, [10, 1, 10, 1],
                       _float_values(shape, torch.float32),
                       _float_values(_source_shape(shape, 0, 4), torch.float32)))
    # FP16 keeps every half update; the magnitude gap catches unintended FP32
    # accumulation across occurrences.
    shape = (3, 5, 17, 3)
    fp16_src = torch.full(_source_shape(shape, 2, 4), 0.125, dtype=torch.float16)
    fp16_src[0, 0, 0, 0] = 512.0
    cases.append(_case("fp16_rank4_magnitude_gap", shape, 2, torch.float16, [0, 0, 16, 0],
                       _float_values(shape, torch.float16, -1, 1), fp16_src))
    # BF16 repeated ties must round after each occurrence, not once at the end.
    shape = (2, 3, 5, 7, 9)
    bf16_src = torch.full(_source_shape(shape, 3, 5), 0.00390625, dtype=torch.bfloat16)
    cases.append(_case("bf16_rank5_tie_repeat", shape, 3, torch.bfloat16, [1, 1, 1, 0, 1],
                       torch.full(shape, 1.0, dtype=torch.bfloat16), bf16_src))
    # 6D tail dimension has K=1 and is the scalar scatter boundary.
    shape = (2, 2, 2, 2, 2, 13)
    cases.append(_case("fp32_rank6_scalar_tail", shape, 5, torch.float32, [12, 0, 12, 4],
                       _float_values(shape, torch.float32),
                       _float_values(_source_shape(shape, 5, 4), torch.float32)))
    # Integer paths: int32 must be exact; int8 must preserve established RINT
    # overflow behavior under repeated updates.
    shape = (9, 7)
    cases.append(_case("int32_exact_repeat", shape, 0, torch.int32, [2, 2, 8, 2],
                       _int_values(shape, torch.int32, -1000, 1000),
                       _int_values(_source_shape(shape, 0, 4), torch.int32, -1000, 1000)))
    shape = (8, 5)
    cases.append(_case("int8_overflow_owner", shape, 0, torch.int8, [4, 4, 4, 1],
                       torch.full(shape, 120, dtype=torch.int8),
                       torch.full(_source_shape(shape, 0, 4), 20, dtype=torch.int8)))
    # Rank-1 also confirms negative dim normalization at the public interface.
    shape = (31,)
    cases.append(_case("fp32_rank1_negative_dim", shape, -1, torch.float32, [0, 30, 0, 14],
                       _float_values(shape, torch.float32),
                       _float_values(_source_shape(shape, 0, 4), torch.float32)))
    return cases


def compare(actual, expected, dtype):
    actual = actual.cpu()
    if dtype in (torch.int8, torch.int32):
        assert torch.equal(actual, expected), "integer result differs"
        return
    rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (1e-3, 1e-3)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=atol, equal_nan=True)


def main():
    torch.npu.set_device(DEVICE)
    for item in build_cases():
        expected = torch.index_add(item["self"], item["dim"], item["index"], item["source"])
        actual = custom_ops_lib.custom_op(item["self"].npu(), item["index"].npu(),
                                          item["source"].npu(), item["dim"])
        torch.npu.synchronize()
        compare(actual, expected, item["dtype"])
        print("PASS", item["name"], flush=True)


if __name__ == "__main__":
    main()
