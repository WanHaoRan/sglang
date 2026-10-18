# Campaign 19 — the original HiCache baseline and oracle v2 + next-use at x10 gaps, on the redesigned client (2026-10-18)

Experiment 2 of the request recorded in `../campaign18/DECISIONS.md` ("2) gap 10, QPS 0.1, original wc + oracle v2
(like campaign 17)"). It runs right after campaign 18, from the same launcher
(`frozen/oracle_scripts_staging/c18_c19.sh`, copied as `../campaign18/launcher.sh`; its log is
`../campaign18/launcher.log`), and differs from campaign 18 only in the gap scale.

"Oracle v2 (like campaign 17)" is read as campaign 17's oracle arm, `three_tier_wc_oracle_v2_nu` (oracle v2 with
next-use eviction), the same arm as campaign 18's.

## Arms and workload

As campaign 18 (arms, engine, hints, L3 wipe, acceptance rules, stop and resume), with `GAP=10`:
`USERS=768 TURNS=50 ARRIVAL=0.1 GAP=10 C=128 SEED=0 ORACLE_HINTS=1 CLIENT_TIMEOUT=18000` on the 262K trace. A returning
turn follows its predecessor after the recorded gap x 10 (median ~7 s, mean ~14 s; campaign 17), at most 1,200 s.

Not comparable turn for turn with campaign 17: the redesigned client has no mid-session start, no 30-50 turn draw and
a fixed cap of 128 users in session instead of the backlog-driven cap (`../../REPLAY_REDESIGN.md` section 7.2).

## What to expect (estimates)

- With sessions of ~10-20 min, ~60-100 users are in session at a time when service is fast (0.1/s x session length),
  below the cap of 128; the cap binds only when service slows. Campaign 17 at x10 was GPU-bound while arrivals lasted
  (users after the first 128 offered ~3 turns/s against ~2.4 the server could serve), so expect congestion between
  ~1 h and the last arrival (~7,633 s) here too, and admission waits once 128 users are in session.
- Campaign 17's baseline stormed at x10 because paced retries spent each request's lookup budget polling for new
  messages that are never on the SSD, after which history lost while queued was recomputed (`../../ORACLE_FINDINGS.md`
  section 8.5). The redesigned client does not change that server behaviour, so the same failure may recur if queue waits
  reach tens of seconds.
- One user has five 1,200 s gaps at x10 (campaign 17's u722; same conversation and arrival at seed 0), so no arm ends
  before ~14,200 s; each arm likely ends between that and the 5 h timeout.

## Running it

Launched by campaign 18's command once campaign 18 holds both arms; a relaunch of that command resumes here.
