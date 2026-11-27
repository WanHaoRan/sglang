# Policy comparison on the simulator (2026-11-26)

`kvsim` (calibrated on campaigns 34, 32 and 28: `CALIBRATION.md`; refereed: `referee/REFEREE_NOTES.md`) run on the
four settings of the GPU comparison, each arm paired with `ec_full` on the requests both sent before the cut (mean
over seeds, range over seeds; negative = faster than the current design). Switch names are the simulator's
(`CALIBRATION.md` section 6); `F4B` was added afterwards (`kvsim_policy.py`, `LATER_SWITCHES`) and is not part of
`all`.

Settings: A = 128 users, 160 GB, 2 h (campaign 35); B = the same with the SSD cleaner at 50/40 % (campaign 38);
C = 128 users, 112 GB (campaign 37); D = 64 users, 88 GB, 1 h (campaign 36).

What the referee says the simulator can carry: the sign and rough size of effects of 0.5 s or more that hold in every
seed; not differences under 0.5 s as statements about the GPU, not anything that acts through refused reads, pending
writes or the write hold (P1, P2, R2, R3), not SSD-tier rules beyond their sign, and settings B, C and D have no GPU
anchor yet.

## On the referee's patched simulator, five seeds (`policy_runs/p1.out`)

`referee/kvsim_compare_patched.py`, seeds 0 to 4; `ec_full` simulated: 11.16, 11.82, 11.91 and 0.61 s. A star marks
a sign that is the same in every seed.

| Switch | What it changes | A | B | C | D |
|---|---|---|---|---|---|
| R4 | SSD pages in use stay young; ended sessions' pages first | -0.25* | -0.94* | -0.14* | 0.00 |
| R1 | no write for an ended session's last turn | -0.03 | -0.03 | -0.17 | -0.07 |
| P2 | reads wait for writes only when room is short | +0.01 | +0.03 | -0.07 | -0.02 |
| P1 | host room for a read taken in index order | +0.06 | +0.09 | -0.36 | -0.05 |
| F1 | landing quantile for held reads | +0.03 | +0.02 | -0.04 | 0.00 |
| F3 | read time before arrival counts the device's line | +0.26 | +0.33 | +1.02* | -0.01 |
| F2 + F3 | the same with the stall weighed by c | +0.04 | +0.03 | +0.40* | +0.01 |
| F4B | read or recompute, restricted form | +0.22* | +0.16* | +0.40 | +0.10* |
| F5 | lookups that find nothing are free | +0.02 | +0.04 | +0.03 | 0.00 |
| R5 | host eviction by the full index | +0.13 | +0.06 | +0.33 | +0.03 |
| PR1 | lambda read off the eviction frontier | +0.05* | +0.12* | -0.11 | -0.13 |
| F1 F2 F3 F5 R1 R3 R4 PR1 | the first build, without read-or-recompute | -0.22* | -0.93* | -0.50* | -0.14 |
| F5 R1 R4 | the store changes without Fetch or Price | -0.21* | -0.94* | -0.15 | -0.05 |

Seed 0 (the GPU campaigns' users), with the interval over users: the first build -0.28 [-0.40, -0.17] in A,
-0.77 [-0.94, -0.61] in B, -1.02 [-1.49, -0.58] in C, -0.19 [-0.26, -0.13] in D; R4 alone -0.26, -0.82, -0.04, 0.00.

## The load from host memory (`policy_runs/p2.out`, `policy_runs/compare_load.py`)

The simulator charges 0.85 us of GPU time per token loaded from host memory (22 ms a request; the logs give 40 to
53 ms of blocked pass). `fa3_nocopy` sets that charge to zero: the upper bound of what decoding beside the copy
can give (the held turn's own delay, about 30 ms, is not modelled). Patched simulator, seeds 0 to 4.

| Arm | A | B | C | D |
|---|---|---|---|---|
| the current design with a copy that costs no GPU time | -2.61* (-3.92 to -1.23) | -2.38* (-3.47 to -1.35) | -1.50* (-1.82 to -1.17) | -0.07* |
| seed 0 | -3.14 [-3.44, -2.84] | -2.93 [-3.24, -2.63] | -1.54 [-1.94, -1.16] | -0.04 [-0.09, 0.00] |
| the first build | -0.22* | -0.93* | -0.50* | -0.14 |
| the first build with a copy that costs no GPU time | -2.84* (-4.33 to -1.39) | -3.52* (-5.59 to -1.73) | -3.04* (-4.39 to -1.42) | -0.18* |
| seed 0 | -3.36 [-3.69, -3.05] | -3.78 [-4.19, -3.41] | -4.09 [-4.69, -3.50] | -0.22 [-0.30, -0.16] |

Mean TTFT of `ec_full`: 11.16, 11.82, 11.91, 0.61 s; with the free copy 8.91, 9.83, 10.65, 0.54 s; throughput in A
136.5 to 141.0 requests a minute. The two changes add up: the copy acts on GPU time per request, the first build on
what is recomputed.

## First pass on the unpatched simulator, three seeds (`policy_runs/m1.out` to `m4.out`)

The numbers below were drawn before the referee's patch. R4 and F3 keep their size on the patched copy; P2's
gain in C (-0.40 s here) does not (-0.07 s, range -0.62 to +0.82), so nothing is claimed for P2.

## One switch at a time (seconds of mean TTFT against `ec_full`)

| Switch | What it changes | A | B | C | D |
|---|---|---|---|---|---|
| F1 | landing quantile for held reads | -0.03 | +0.03 | +0.01 | +0.10 |
| F2 | stall weighed by the measured c | 0.00 | 0.00 | +0.09 | +0.12 |
| F3 | read time before arrival counts the device's line | +0.37 | +0.42 | +1.58 | +0.06 |
| F2 + F3 | | -0.06 | -0.02 | +0.56 | +0.15 |
| F4 | read or recompute, draft v1's rule | +5.91 | +5.79 | +18.84 | +1.22 |
| F4B | read or recompute, revised (only with no ready turn waiting) | +0.17 | +0.17 | +0.59 | +0.12 |
| F5 | lookups that find nothing are free | +0.01 | 0.00 | +0.20 | +0.04 |
| R1 | no write for an ended session's last turn | -0.02 | -0.05 | -0.24 | +0.05 |
| R2 | write order by eviction order | +0.01 | +0.02 | +0.16 | +0.20 |
| R4 | SSD pages in use stay young; ended sessions' pages first | -0.23 | -0.90 | -0.17 | 0.00 |
| R5 | host eviction by the full index | +0.10 | +0.13 | +0.56 | +0.19 |
| P1 | host room for a read taken in index order | +0.09 | +0.15 | +0.09 | +0.04 |
| P2 | reads wait for writes only when room is short | -0.04 | +0.04 | -0.40 | +0.12 |
| PR1 | lambda read off the eviction frontier | +0.03 | +0.06 | -0.11 | 0.00 |
| all (draft v1) | | +13.2 | +11.9 | +29.5 | +2.7 |

Run-to-run noise of the simulator with the same users (5 ms of send jitter): 0.03 to 0.17 s. Sign the same in
every seed: R4 in A, B and C; R1 in C; P2 in C; F3 and F4 (worse) everywhere.

## Sets

| Set | A | B | C | D |
|---|---|---|---|---|
| F1 F2 F3 F5 R1 R3 R4 PR1 (what the engine builds, without read-or-recompute) | -0.20 | -0.86 | -0.34 | -0.03 |
| the same + P2 | -0.17 | -0.89 | -0.59 | -0.03 |
| the same + P1 + P2 | -0.19 | -0.85 | -0.55 | -0.02 |
| the same + P1 + P2 + R2 | -0.21 | -0.88 | -0.60 | -0.02 |
| the same + R5 | -0.20 | -0.88 | -0.29 | -0.02 |
| the same + F4B | 0.00 | -0.70 | -0.04 | +0.08 |
| everything but F4 | -0.25 | -0.87 | -0.54 | -0.03 |
| R1 R4 P2 | -0.21 | -0.87 | -0.07 | +0.06 |
| F5 R1 R4 | -0.18 | -0.90 | -0.23 | +0.03 |
| F5 R1 R4 P1 P2 (no Fetch or Price change) | -0.25 | -0.89 | -0.13 | +0.10 |

## The first build over five seeds (`policy_runs/m4.out`)

Mean over seeds 0 to 4 (range), and seed 0 alone with its interval over users (seed 0 is the GPU campaigns' users).

| Set | A | B | C | D |
|---|---|---|---|---|
| F1 F2 F3 F5 R1 R3 R4 PR1 | -0.22 (-0.36 to -0.12) | -0.91 (-1.51 to -0.46) | -0.57 (-1.57 to -0.19) | -0.07 (-0.27 to +0.01) |
| seed 0 | -0.21 [-0.34, -0.09] | -0.83 [-0.99, -0.66] | -0.19 [-0.48, +0.10] | -0.00 [-0.02, +0.02] |
| the same + P2 | -0.19 (-0.27 to -0.09) | -0.91 (-1.47 to -0.43) | -0.76 (-1.77 to -0.28) | -0.06 (-0.24 to +0.01) |
| seed 0 | -0.18 [-0.31, -0.06] | -0.89 [-1.07, -0.71] | -0.51 [-0.80, -0.22] | -0.01 [-0.03, +0.01] |

The sign is the same in every seed in A, B and C.

## Reading

1. The largest term is not a cache policy: the GPU time a prefill pass spends waiting on the copy from host memory.
   With that time given back the current design is 2.4 to 2.6 s faster at 128 users (1.5 s on the tight host pool),
   in every seed. The simulator charges half of what the logs measure for it, and a hold cannot give all of it back,
   so the GPU run decides the size.
2. Among the cache rules the gain is Retain on the SSD (R4): the cleaner's order. It is the only single switch whose
   sign holds in every seed where the cleaner runs, and it grows with the time the cleaner works (0.25 s in A,
   0.94 s in B). The simulator keeps the SSD per node, so the page-level replay (`agents/ev_ssd/`) is the better
   size: 1.3 to 1.7 s of the last 20 minutes in A, 2.7 to 3.7 s at half the volume (at N = 128).
3. Read or recompute by price loses in every setting, in the draft's form (rent charged on the device's line,
   mu = 1 + ready turns: 3,400 recomputes by choice, +5.8 to +18.8 s) and in the restricted form (+0.1 to +0.4 s).
   It stays off.
4. F3 alone is harmful, as the algebra said (a longer read time makes the start rule read early at saturation
   unless the stall weight falls with it); with F2 it is neutral except on the tight host pool.
5. The Fetch and Price changes (F1, F2, F3, PR1) gain nothing by themselves, but the first build as a whole is the
   only set that is faster in all four settings: the store changes without them gain 0.15 s instead of 0.50 s on
   the tight host pool.
6. Nothing is claimed for P1, P2, R2 and R3: they act through refused reads and pending writes, which the simulator
   does not model well enough (the referee's patch moved P2's gain in C from -0.40 s to -0.07 s).
7. The full index for host eviction (R5) changes nothing measurable: the engine's return-chance order with a 30 s
   horizon is already the index at one horizon.
8. Expected on the GPU, on the campaigns' users (seed 0): the hold alone up to -3.1 s in campaign 39 against
   campaign 35's `ec_full` (run-to-run spread measured there: +0.01 s [-0.09, +0.11]); the first build about -0.3 s
   in campaign 35, -0.8 s in campaign 38, -1.0 s in campaign 37 and -0.2 s in campaign 36; both together up to -3.4,
   -3.8, -4.1 and -0.2 s.
