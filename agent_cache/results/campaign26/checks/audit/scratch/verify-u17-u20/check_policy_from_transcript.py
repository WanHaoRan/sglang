"""Check next_turn_policy.read_start / worth_holding against brute force and the room model's solver."""
import importlib.util, math, random, sys
sys.path.insert(0, "/sgl-workspace/sglang/agent_cache/scripts")
spec = importlib.util.spec_from_file_location("ntpol", "/sgl-workspace/sglang/python/sglang/srt/mem_cache/next_turn_policy.py")
pol = importlib.util.module_from_spec(spec); sys.modules["ntpol"] = pol; spec.loader.exec_module(pol)
import room_model as rm

def brute_cost(remaining, start, read_s, wait_s, rent):
    c = 0.0
    for g in remaining:
        if g <= start:
            c += max(0.0, read_s - wait_s)
        else:
            land, due = start + read_s, g + wait_s
            c += max(0.0, land - due) + rent * max(0.0, due - land)
    return c / len(remaining)

random.seed(1)
worst = 0.0; n_none = n_now = n_later = 0
for trial in range(3000):
    p50 = math.exp(random.uniform(math.log(0.5), math.log(120)))
    p10 = p50 * math.exp(-random.uniform(0.02, 3.0)); p90 = p50 * math.exp(random.uniform(0.02, 3.0))
    d = pol.GapDistribution.from_quantiles(p10=p10, p50=p50, p90=p90)
    sup = d.support()
    assert list(sup) == sorted(sup)
    idle = random.choice([0.0, 0.0, random.uniform(0, 2 * p50)])
    rem = pol.remaining_gaps(support=sup, idle_s=idle)
    read_s = random.uniform(0.3, 8.0); wait_s = random.choice([0.0, 0.0, random.uniform(0, 6)])
    rent = random.choice([1e-6, 5e-6, 5e-5, 1e-4, 1e-3]) * random.choice([8000, 32000, 64000, 128000])
    got = pol.read_start(remaining=rem, read_s=read_s, wait_s=wait_s, rent_per_s=rent)
    at_arr = max(0.0, read_s - wait_s)
    if not rem or at_arr == 0:
        assert got is None; n_none += 1; continue
    grid = sorted(set([0.0] + [max(0.0, g - (read_s - wait_s)) for g in rem] + list(rem) + [x * max(rem) / 400 for x in range(401)]))
    costs = [brute_cost(rem, s, read_s, wait_s, rent) for s in grid]
    bmin = min(costs)
    if got is None:
        assert at_arr <= bmin + 1e-9, (trial, at_arr, bmin); n_none += 1
    else:
        c = brute_cost(rem, got, read_s, wait_s, rent)
        assert c <= bmin + 1e-9 and c < at_arr, (trial, c, bmin, at_arr)
        worst = max(worst, c - bmin); n_now += got == 0; n_later += got > 0
    # worth_holding vs brute force over release times
    stall = at_arr
    keep = pol.worth_holding(remaining=rem, stall_at_arrival_s=stall, rent_per_s=rent)
    best = max((stall * sum(1 for g in rem if g <= rho) - rent * sum(min(g, rho) for g in rem)) for rho in rem)
    assert keep == (best > 0), (trial, keep, best)
print("read_start vs brute force: ok; at-arrival", n_none, "now", n_now, "later", n_later, "worst excess", worst)

# against the room model's solver on its own support (100 points, capped 1200 s), delta = 0
agree = 0; tot = 0
for trial in range(400):
    p50 = math.exp(random.uniform(math.log(0.5), math.log(120)))
    p10 = p50 * math.exp(-random.uniform(0.05, 3.0)); p90 = p50 * math.exp(random.uniform(0.05, 3.0))
    job = dict(p10=p10, p50=p50, p90=p90, r_pred=random.uniform(0.3, 7.0), W_pred=random.choice([0.0, random.uniform(0, 4)]), tokens=random.choice([8000, 32000, 64000, 128000]))
    lam = random.choice([5e-6, 2e-5, 5e-5, 1e-4])
    want = rm.expected_cost_start(job, lam=lam, form="split", delta=0.0)
    xs, ws = rm.support(job, "split")
    got = pol.read_start(remaining=tuple(sorted(xs)), read_s=job["r_pred"] + rm.JIT_MARGIN_S, wait_s=job["W_pred"], rent_per_s=lam * job["tokens"])
    tot += 1
    if (want is None) == (got is None) and (want is None or abs(want - got) < 1e-6):
        agree += 1
    else:
        cw = None if want is None else brute_cost(sorted(xs), want, job["r_pred"] + 1, job["W_pred"], lam * job["tokens"])
        cg = None if got is None else brute_cost(sorted(xs), got, job["r_pred"] + 1, job["W_pred"], lam * job["tokens"])
        print("differs:", want, got, "costs", cw, cg, "at arrival", max(0, job["r_pred"] + 1 - job["W_pred"]))
print(f"room model solver agreement: {agree}/{tot}")

# the deck's q table: with a sharp entry and no queue, the read lands where the stall risk is about q
for tokens, q in ((8000, 0.29), (32000, 0.62), (64000, 0.77), (128000, 0.87)):
    lam = 5e-5; rent = lam * tokens
    print(tokens, "q =", round(rent / (1 + rent), 3), "(deck", q, ")")
d = pol.GapDistribution.from_quantiles(p10=17.5, p50=20.0, p90=22.6)
for tokens in (8000, 32000, 64000, 128000):
    read_s = tokens * 98304 / 2.11e9
    s = pol.read_start(remaining=d.support(), read_s=read_s, wait_s=0.0, rent_per_s=5e-5 * tokens)
    risk = None if s is None else 1 - d.survival(s + read_s) / d.survival(s) if d.survival(s) > 0 else None
    print(f"sharp key 20 s: {tokens} tokens read {read_s:.2f} s -> start {s}, stall risk at start {risk}")
wide = pol.GapDistribution.from_quantiles(p10=0.85, p50=22.0, p90=58.0)
print("pip-like 64K:", pol.read_start(remaining=wide.support(), read_s=3.05, wait_s=0.0, rent_per_s=5e-5 * 64000))
pt = pol.GapDistribution.point(27.7)
print("exact 27.7 s, read 2.0:", pol.read_start(remaining=pt.support(), read_s=2.0, wait_s=0.0, rent_per_s=3.2), "(want 25.7)")
print("chance_within sharp at idle 60:", d.chance_within(idle_s=60, horizon_s=30), "wide at idle 60/300:", wide.chance_within(idle_s=60, horizon_s=30), wide.chance_within(idle_s=300, horizon_s=30))
c = pol.PriceController(price=5e-5)
c.observe(now=0.0, idle_tokens=0); c.refuse(30000); c.observe(now=5.0, idle_tokens=10000)
print("price step:", c.step(room_tokens=44000))
