# Screening device availability

The pre-run host `npu-smi` snapshot reported no active compute process on
physical devices 5, 6, and 7. Docker ownership inspection then found the
unrelated running container `zxc_vllm` mapped both `/dev/davinci5` and
`/dev/davinci6` with `ASCEND_VISIBLE_DEVICES=5,6`.

Project policy treats a device mapped into any running container as occupied,
even when `npu-smi` shows no active kernel process. The P3 run therefore did
not start profiling on devices 5 or 6 and did not stop or modify that unrelated
container. Physical device 7 was isolated in `concat-p3-p7` and used serially.

Consequences:

- formal three-card screening remained incomplete;
- `screening/` contains the one planned card7 round;
- `screening_card7_confirm/` contains a separate AB/BA/AB rejection diagnostic;
- no single-card result was promoted to a three-card acceptance result.
