"""Replay a run's returning turns through a two-tier model to bound what next-use eviction
can gain over LRU before spending GPU time (agent_cache/ORACLE_V2_PLAN.md section 4.7).

  python3 next_use_replay.py --client results/<boot>/client_<...>/client.jsonl --l1-tokens 600000 \
      --host-room-tokens 318000 578000 1250000 [--horizon-s 300] [--out <csv>]

Model: each session's context after a turn (prompt + reply tokens) is one unit. When a turn ends its context enters
L1 (an LRU of finished contexts, capacity --l1-tokens; running requests are outside it). An L1 eviction moves the
context to the host-only room (capacity --host-room-tokens; the write-through mirror of L1 is not part of that room).
When the session's next turn starts, the context is found in L1 (device hit), in the host room (host hit) or nowhere
(an SSD read of the whole context). The host room evicts by LRU (least recently ended) or by next use: ended sessions
first, then sessions whose next turn is beyond --horizon-s (farthest first), then the rest (farthest first). The next
arrival of every session is known from the record, which is what the next-turn hint gives the engine.

Reports, per (room, policy): the share of returning turns served by each tier and the SSD gigabytes read (6 MiB per
64-token page); the real run's client-side tier shares are printed for comparison.
"""
import argparse
import collections
import csv
import json
import os

BYTES_PER_TOKEN = 6 * 2**20 / 64


def load_sessions(path: str) -> dict:
    rows = [json.loads(l) for l in open(path)]
    turns = [r for r in rows if r["kind"] == "turn" and "t_send" in r and r.get("ttft") is not None]
    by_conv = collections.defaultdict(list)
    for r in turns:
        by_conv[r["conv"]].append(r)
    sessions = {}
    for conv, rs in by_conv.items():
        rs.sort(key=lambda r: r["turn"])
        seq = []
        for k, r in enumerate(rs):
            nxt = rs[k + 1]["t_send"] if k + 1 < len(rs) else None
            seq.append(dict(start=r["t_send"], end=r["t_send"] + r["latency"], ctx=r["prompt_tokens"] + r["completion_tokens"],
                            next_start=nxt, cached=r["cached_tokens"], details=r.get("cached_details") or {}, prompt=r["prompt_tokens"]))
        sessions[conv] = seq
    return sessions


def real_shares(sessions: dict) -> dict:
    counts = collections.Counter()
    for seq in sessions.values():
        for turn in seq[1:]:
            d = turn["details"]
            if (d.get("storage") or 0) > 0:
                counts["ssd"] += 1
            elif (d.get("host") or 0) > 0:
                counts["host"] += 1
            elif turn["cached"] >= 0.5 * turn["prompt"]:
                counts["device"] += 1
            else:
                counts["cold"] += 1
    n = sum(counts.values()) or 1
    return {k: v / n for k, v in counts.items()}


def replay(sessions: dict, l1_tokens: int, room_tokens: int, policy: str, horizon_s: float) -> dict:
    events = []
    for conv, seq in sessions.items():
        for k, turn in enumerate(seq):
            if k > 0:
                events.append((turn["start"], 0, conv, k))
            events.append((turn["end"], 1, conv, k))
    events.sort()
    l1: dict = {}    # conv -> (ctx, t_end)
    host: dict = {}  # conv -> (ctx, t_end, next_use)
    l1_used = room_used = 0
    counts = collections.Counter()
    ssd_tokens = 0

    def host_victim(now: float):
        def key(item):
            conv, (ctx, t_end, next_use) = item
            if policy == "lru":
                return (0, t_end)
            if next_use is None:
                return (0, t_end)
            if next_use - now > horizon_s:
                return (1, -next_use)
            return (2, -next_use)
        return min(host.items(), key=key)[0]

    for t, kind, conv, k in events:
        turn = sessions[conv][k]
        if kind == 0:  # the session returns
            if conv in l1:
                counts["device"] += 1
                l1_used -= l1.pop(conv)[0]
            elif conv in host:
                counts["host"] += 1
                room_used -= host.pop(conv)[0]
            else:
                counts["ssd"] += 1
                ssd_tokens += sessions[conv][k - 1]["ctx"]
            continue
        # the turn ended: its context is on device
        l1[conv] = (turn["ctx"], t)
        l1_used += turn["ctx"]
        while l1_used > l1_tokens and len(l1) > 1:
            victim = min(l1, key=lambda c: l1[c][1])
            ctx, t_end = l1.pop(victim)
            l1_used -= ctx
            # the victim's next arrival: the next turn of the session after the turn that ended at t_end
            seq = sessions[victim]
            idx = next((i for i, tr in enumerate(seq) if tr["end"] == t_end), None)
            next_use = seq[idx]["next_start"] if idx is not None else None
            host[victim] = (ctx, t_end, next_use)
            room_used += ctx
            while room_used > room_tokens and host:
                loser = host_victim(t)
                room_used -= host.pop(loser)[0]
    n = sum(counts.values()) or 1
    return dict(returns=n, device=counts["device"] / n, host=counts["host"] / n, ssd=counts["ssd"] / n,
                ssd_gb=ssd_tokens * BYTES_PER_TOKEN / 1e9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--client", required=True)
    ap.add_argument("--l1-tokens", type=int, default=600_000)
    ap.add_argument("--host-room-tokens", type=int, nargs="+", default=[318_000, 578_000, 1_250_000])
    ap.add_argument("--horizon-s", type=float, default=300.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    sessions = load_sessions(a.client)
    real = real_shares(sessions)
    print(f"{len(sessions)} sessions, {sum(len(s) - 1 for s in sessions.values())} returning turns; real run tiers: "
          + ", ".join(f"{k} {v:.1%}" for k, v in sorted(real.items())))
    rows = []
    print(f"{'host room':>10} {'policy':>8} {'device':>8} {'host':>8} {'ssd':>8} {'SSD GB':>9}")
    for room in a.host_room_tokens:
        for policy in ("lru", "next_use"):
            r = replay(sessions, a.l1_tokens, room, policy, a.horizon_s)
            rows.append(dict(host_room_tokens=room, policy=policy, **r))
            print(f"{room:>10,} {policy:>8} {r['device']:>8.1%} {r['host']:>8.1%} {r['ssd']:>8.1%} {r['ssd_gb']:>9,.0f}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
