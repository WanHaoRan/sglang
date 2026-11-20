"""Campaign 28 or 29: what load a lower arrival rate or user cap gives, from the trace and the measured GPU costs.
In the container, from agent_cache/scripts/:
  python3 -B ../results/campaign28/checks/load_model.py ../results/campaign28 [capacity requests/s | model] [max context] > ../results/campaign28/checks/load_model.out
"model" takes 0.87 of the rate at which this model fills the GPU pool (it gave 1.44 against a measured 1.25 at 65,536);
a max context overrides the campaign's MAX_CONTEXT, to see what the longer sessions would ask. Two more options as
key=value: exclude=<harness> leaves that harness's users out, gap_cap=<s> overrides GAP_CAP.
Uses the client's own loader and arrival draw (as arrival_set.py). From the users that arrive: turns a user, the
recorded wait before a turn (capped at GAP_CAP), prompt, new and output tokens, and the context a session holds
averaged over its life. GPU costs are campaign 28's measurements (checks/decode_scale.out): a decode step of
11.2 ms + 0.18 ms a running request + 0.0357 ms a thousand context tokens, and 0.158 ms a prefilled token.
Then, for a requests-per-second level: the arrival rate that asks it, the running requests and decode time it
needs (Little's law on the step-time fit), the users in session, and the KV those sessions hold against the
host pool. A model, not a run: it leaves out queueing from bursts. Stdlib and the client modules only."""
import random
import sys
import types

sys.path.insert(0, ".")
import replay_programs  # noqa: E402

opts = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
sys.argv = [a for a in sys.argv if "=" not in a]
cdir = sys.argv[1]
settings = dict(line.strip().split("=", 1) for line in open(f"{cdir}/settings.env") if "=" in line)
args = types.SimpleNamespace(trace=settings["TRACE"], wait_field="wait", split_idle=None, split_compaction=False,
                             max_context=int(sys.argv[3]) if len(sys.argv) > 3 else int(settings["MAX_CONTEXT"]),
                             sample="shuffle", seed=int(settings["SEED"]), trees=0)
header, trees, dropped = replay_programs.load_trees(args)
rate, duration, gap_cap = float(settings["RATE"]), float(settings["DURATION"]), float(settings["GAP_CAP"])
rng, t, n = random.Random(f"{args.seed}:arrivals"), 0.0, 0
while t < duration and n < len(trees):   # run_arrivals' own draw
    n += 1
    t += rng.expovariate(rate)
users = [x for x in trees[:n] if (x.get("system") or ("?",))[0] != opts.get("exclude")]
gap_cap = float(opts.get("gap_cap", gap_cap))
B = header.get("block_tokens") or 0
STEP_MS, PER_REQ_MS, PER_KTOKEN_MS, PREFILL_MS = 11.2, 0.18, 0.0357, 0.158
KV_BYTES = 98304   # per token: 48 layers x 2 x 4 KV heads x 128 x 2 bytes
POOL_TOKENS = 668096


def prompt(r):
    return (r["keep"] + len(r["new"])) * B + int(r.get("tail") or 0) if "new" in r else r["prefix"] + r["append"]


def new_tokens(r, k):
    if k == 0:
        return prompt(r)
    return len(r["new"]) * B + int(r.get("tail") or 0) if "new" in r else r["append"]


turns = waits = outs = prompts = news = 0
ctx_time = life = 0.0
for tree in users:
    for st in tree["streams"]:
        rs = st["requests"]
        for k, r in enumerate(rs):
            turns += 1
            w = min(float(r["wait"] or 0), gap_cap) if k else 0.0
            waits += w
            outs += r["out"]
            prompts += prompt(r)
            news += new_tokens(r, k)
            # The context sits in the cache through this turn's reply and the wait before the next turn.
            nxt = min(float(rs[k + 1]["wait"] or 0), gap_cap) if k + 1 < len(rs) else 0.0
            ctx_time += (prompt(r) + r["out"]) * (nxt + 10.0)
            life += nxt + 10.0
per_user = turns / len(users)
think, out, ctx, new = waits / turns, outs / turns, prompts / turns, news / turns
held = ctx_time / life
print(f"max context {args.max_context:,}: {len(trees)} sessions kept, {dropped} left out; options {opts or 'none'}")
print(f"{settings['TRACE'].split('/')[-1]}, the {len(users)} users that arrive: {per_user:.1f} turns a user; a turn waits {think:.1f} s (mean, capped at {gap_cap:.0f} s), "
      f"sends {ctx / 1e3:.1f} K prompt tokens ({new / 1e3:.2f} K new) and gets {out:.0f} output tokens; a session holds {held / 1e3:.1f} K tokens averaged over its life")
prefill_s = new * PREFILL_MS / 1e3


def running(x):
    """Running requests that serve x requests/s: R = x * out * step(R) / (1 - x * prefill), step linear in R."""
    if x * prefill_s >= 1:
        return float("inf")
    a = x * out / (1e3 * (1 - x * prefill_s))
    slope = PER_REQ_MS + PER_KTOKEN_MS * (ctx + out / 2) / 1e3
    return STEP_MS * a / (1 - slope * a) if slope * a < 1 else float("inf")


lo, hi = 0.01, 5.0
for _ in range(60):   # the rate at which the running requests fill the GPU pool
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if running(mid) * (ctx + out / 2) < POOL_TOKENS else (lo, mid)
model_cap = lo
cap = float(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "model" else 0.87 * model_cap
print(f"GPU time a request: prefill {prefill_s:.2f} s; the pool of {POOL_TOKENS:,} tokens fills at {model_cap:.2f} requests/s by this model "
      f"({60 * model_cap:.0f} a minute, {model_cap / per_user:.4f} users/s); capacity used below: {cap:.2f} requests/s")
print(f"{'share of capacity':>17s} | {'requests/s':>10s} {'users/s':>8s} | {'running':>7s} {'decode s':>8s} {'ms a token':>10s} | {'users in session':>16s} | "
      f"{'KV of the sessions':>18s} {'host GB to hold it all':>22s} {'GB for half':>11s} | users in 2 h | session KV past a host pool of 160, 100, 70 GB")
for share in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
    x = share * cap
    r = running(x)
    step = STEP_MS + (PER_REQ_MS + PER_KTOKEN_MS * (ctx + out / 2) / 1e3) * r
    decode = out * step / 1e3 / (1 - x * prefill_s)
    session = per_user * (think + decode + 1.0)
    in_session = x / per_user * session
    kv = in_session * held
    print(f"{100 * share:16.0f}% | {x:10.2f} {x / per_user:8.4f} | {r:7.1f} {decode:8.1f} {1e3 * decode / out:10.1f} | {in_session:16.0f} | "
          f"{kv / 1e6:15.2f} M {kv * KV_BYTES / 2**30:22.0f} {kv * KV_BYTES / 2**30 / 2:11.0f} | "
          f"{7200 * x / per_user:5.0f} | {100 * max(0.0, kv - 160 * 2**30 / KV_BYTES) / kv:5.0f}% {100 * max(0.0, kv - 100 * 2**30 / KV_BYTES) / kv:5.0f}% {100 * max(0.0, kv - 70 * 2**30 / KV_BYTES) / kv:5.0f}%")
