#!/usr/bin/env python3
"""Run the fixed historical 39-case matrix in 30-task groups."""

from test_matrix import CASES, generated_cases, run_case


CASE_NAMES = """rank1_fp16_dim0_zero rank1_int32_exact rank2_fp32_dim0_zero rank2_int8_last_unaligned
rank3_int32_middle rank3_fp16_last_zero rank3_int32_middle_aligned fp16_boundary_lengths_unaligned_row
fp16_middle_axis_2d fp32_before_dim_over_4095 single_input_large_row_fallback rank6_fp16_dim3
rank6_fp32_negative_axis rank7_int32_axis0 score_shape_2024x3000_fp32 fragmented_256_fp16
fragmented_256_fp32 fragmented_256_int8 s9_fp16_last_axis_10000 s9_fp32_axis0_10000
s9_int32_rank4_axis1_1000 s9_int8_rank4_axis2_1000 s9_fp16_rank5_axis3_999 input_count_8_fp16
input_count_64_int32 fp16_special_values_bitwise fp32_special_values_bitwise generated_00_rank5_float32
generated_01_rank3_int32 generated_02_rank4_float16 generated_03_rank4_int8 generated_04_rank1_int8
generated_05_rank6_int32 generated_06_rank7_int8 generated_07_rank2_float32 generated_08_rank1_int8
generated_09_rank3_float16 generated_10_rank2_float16 generated_11_rank7_int32""".split()


def main() -> None:
    known = {case.name: case for case in CASES + generated_cases(12, 20260721)}
    for case_name in CASE_NAMES:
        run_case(known[case_name], repeat=1)


if __name__ == "__main__":
    main()
