"""Audit scratch (read-only): what became of turns whose staged read was released (ec_release) or bumped (ec_bump):
were they later served with cached_details.storage == 0 from pages still in host memory (the log miscount named under
'Known and left'), and how far that can move 'share of SSD reads used' (sum of cached_details.storage / tokens read).
  nice -n 10 python3 -I -B release_fate.py label=run ..."""
import collections, datetime as dt, json, os, re, sys
RES = "/home/wanhr/sglang/agent_cache/results"
EVT = re.compile(r"^\[(.{23})\] HICACHE_EVT (ec_release|ec_bump|s2h_io|ntp_arrive) rid=\S+?-c(\d+)-t(\d+)(.*)")
PAGES = re.compile(r"pages=(\d+)")
PAGE = 64
stamp = lambda s: dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=dt.timezone.utc).timestamp()
for arg in sys.argv[1:]:
    name, run = arg.split("=")
    client = sorted(d for d in os.listdir(f"{RES}/{run}") if d.startswith("client_"))[-1]
    turns = {}
    for line in open(f"{RES}/{run}/{client}/client.jsonl"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("kind") == "turn" and r.get("ttft") is not None:
            turns[(r["conv"], r["turn"])] = r
    rel, reads, arrive = [], collections.defaultdict(list), {}
    for line in open(f"{RES}/{run}/server.log", errors="replace"):
        m = EVT.match(line)
        if not m:
            continue
        t, kind, key = stamp(m.group(1)), m.group(2), (int(m.group(3)), int(m.group(4)))
        if kind in ("ec_release", "ec_bump"):
            rel.append((t, kind, key))
        elif kind == "s2h_io":
            reads[key].append((t, int(PAGES.search(m.group(5)).group(1))))
        else:
            arrive.setdefault(key, t)
    read_turns = sum(1 for k in turns if k[1] > 0 and reads.get(k))
    tok_read = sum(p * PAGE for k in turns if k[1] > 0 for _, p in reads.get(k, ()))
    tok_used = sum(int((turns[k].get("cached_details") or {}).get("storage") or 0) for k in turns if k[1] > 0)
    print(f"\n{name} {run}: returning turns recorded {sum(1 for k in turns if k[1] > 0)}, of them reading (s2h_io) {read_turns}; "
          f"tokens read {tok_read:,}, served as storage hits {tok_used:,} = {100 * tok_used / max(1, tok_read):.1f} % used")
    for kind in ("ec_release", "ec_bump"):
        ev = [(t, k) for t, kk, k in rel if kk == kind]
        if not ev:
            continue
        last = {}
        for t, k in ev:
            last[k] = t          # a turn's last release/bump
        fate, hidden_tok = collections.Counter(), 0
        for k, t in last.items():
            r = turns.get(k)
            before = sum(p for at, p in reads.get(k, ()) if at <= t) * PAGE
            after = sum(p for at, p in reads.get(k, ()) if at > t) * PAGE
            if r is None:
                fate["turn not recorded (never came, or after the client stopped)"] += 1
                continue
            cd = r.get("cached_details") or {}
            sto, host, rec = int(cd.get("storage") or 0), int(cd.get("host") or 0), max(r.get("reply_reprefilled") or 0, 0)
            if after > 0:
                fate["read again after it (storage hit %s)" % ("> 0" if sto > 0 else "= 0")] += 1
            elif sto > 0:
                fate["no later read, storage hit > 0"] += 1
            elif rec > 64:
                fate["no later read, storage hit 0, recomputed > 64 tokens"] += 1
            elif host > 0:
                fate["no later read, storage hit 0, host hit > 0, recomputed <= 64: served by the released pages (the miscount)"] += 1
                hidden_tok += before
            else:
                fate["no later read, storage hit 0, host hit 0, recomputed <= 64 (all on the GPU)"] += 1
        print(f"  {kind}: {len(ev)} events on {len(last)} turns")
        for f, n in fate.most_common():
            print(f"     {n:4d}  {f}")
        print(f"     tokens of released reads then served as host hits: {hidden_tok:,} -> 'used' would be {100 * (tok_used + hidden_tok) / max(1, tok_read):.1f} % "
              f"instead of {100 * tok_used / max(1, tok_read):.1f} % (upper bound: counts the whole released read)")
