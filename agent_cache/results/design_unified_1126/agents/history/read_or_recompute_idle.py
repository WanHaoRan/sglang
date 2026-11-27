#!/usr/bin/env python3
"""Side check of one sentence of DESIGN.md section 2, found while reading the records for the history file.

The draft says: on an idle FA3 box (c = 1, mu = 1, kappa = 0) recomputing a prefix of L tokens always beats reading
it, because gamma L < lags + L / beta with gamma = 0.037 to 0.044 ms, 1 / beta = 0.047 ms and 0.9 s of lags.

The records give other inputs for an idle box:
- the 0.9 s is the read planner's fixed seed, 0.5 s + 0.044 s + 0.36 s, taken under dense load; at sparse load the
  two lags are about 0.03 s and 0.02 s (ORACLE_PLAN.md section 2.4; results/design_review_1123/facts_calib.md,
  "engine read-time model"), so about 0.094 s with the 0.044 s term;
- an uncontended read runs at 18.9 to 25.7 K tokens a second (facts_calib.md, "read thread rate"), 2.1 GB/s is the
  device's ceiling for reads and writes together;
- FA3 prefill is quoted as 0.037 to 0.044 ms a token (the workflow's constants) and as 0.042 to 0.047 ms
  (OPEN_ITEMS.md section 7).

Recompute wins at every length only if gamma <= the read's time per token. Otherwise the read wins beyond
L* = lag / (gamma - 1 / rate). This prints L* over the quoted ranges (inf: recompute wins at every length).
The third lag adds the 0.4 s between a read's landing and admission that the review measured under load
(results/design_review_1123/critique.md, "max(GPU turn, own landing) + 0.4 s"); the draft's K_read leaves it out.

Run: nice -n 10 python3 -I -B read_or_recompute_idle.py > read_or_recompute_idle.txt
"""
GAMMAS_MS = (0.037, 0.042, 0.044, 0.047)          # prefill, ms per token
READ_RATES = (18_900, 21_400, 25_700)             # tokens per second; 21,400 is the 2.1 GB/s ceiling
LAGS_S = (0.094, 0.494, 0.904)                    # sparse-load lags; the same plus 0.4 s to admission; the dense-load seed


def crossover_tokens(gamma_ms, rate, lag_s):
    per_token_s = gamma_ms / 1000 - 1 / rate
    return float("inf") if per_token_s <= 0 else lag_s / per_token_s


def main():
    print("read wins on an idle box for prefixes longer than L* tokens (inf: recompute wins at every length)")
    print(f"{'gamma ms/token':>15} {'read tokens/s':>14} {'read ms/token':>14} {'L* at 0.094 s':>14} {'L* at 0.494 s':>14} {'L* at 0.904 s':>14}")
    for gamma_ms in GAMMAS_MS:
        for rate in READ_RATES:
            cells = [crossover_tokens(gamma_ms=gamma_ms, rate=rate, lag_s=lag_s) for lag_s in LAGS_S]
            shown = ["inf" if cell == float("inf") else f"{cell:,.0f}" for cell in cells]
            print(f"{gamma_ms:>15.3f} {rate:>14,} {1000 / rate:>14.4f} {shown[0]:>14} {shown[1]:>14} {shown[2]:>14}")
    length = 25_000
    print(f"\na {length:,}-token history (the mean returning prompt is about 25.5 K tokens), sparse-load lag 0.094 s:")
    for gamma_ms, rate in ((0.037, 18_900), (0.044, 21_400), (0.044, 25_700), (0.047, 25_700)):
        read_s = 0.094 + length / rate
        recompute_s = gamma_ms / 1000 * length
        print(f"  gamma {gamma_ms} ms, read {rate:,} tokens/s: read {read_s:.2f} s, recompute {recompute_s:.2f} s")


if __name__ == "__main__":
    main()
