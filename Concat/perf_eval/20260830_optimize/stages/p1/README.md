# P1 local decision

P1-128 passed the local gate and was accepted by the official hidden evaluation:
5/5 Pass, 562.35 us total. The detailed report is
`docs/20260830-1-Concat_P1_Identity评测报告.md`.

- Threshold winner: 128 KiB minimum work per Identity core.
- Identity threshold sweep: 50.044 us baseline to 21.640 us, 2.313x.
- Full matrix: 4/6 faster rounds, median 1786.662 us to 1771.057 us.
- All five Identity cases were faster in 6/6 rounds.
- No paired material regression on the 87 default-path cases.
- Final artifact: `Concat_20260830_121239.zip`.
- Official S8 package: `Concat_20260830_204619_zip`, status `official_accepted`.
