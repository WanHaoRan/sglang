"""Test (d): the kept-turn margin and the shared set.
Mean TTFT per arm on the shared returning turns of LEAD5, ALL11, CUT7 and the two rivals with the margin at 0, 300 and
600 s, and with the shared set restricted to complete sessions: sessions whose every turn (0 .. turns-1 of the client's
conv_start record) is kept in every arm of the set.
  nice -n 19 python3 -I -B td_margin.py"""
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

rank = C.order_rank()
D = {a: C.load(a) for a in C.RUNS if a != "order"}
READ = {a: {} for a in D}
for a in D:
    for t, c, n, pages, tok, ms in D[a]["s2h"]:
        READ[a][(c, n)] = READ[a].get((c, n), 0) + pages

print("turn-count field: conv_start.turns against turns recorded for sessions that ended uncapped (per arm: sessions ended uncapped, of them with all turns recorded)")
for a in D:
    n_rec = {}
    for t in D[a]["turns"]:
        n_rec[t["conv"]] = n_rec.get(t["conv"], 0) + 1
    done = [c for c, e in D[a]["conv_end"].items() if not e.get("capped")]
    full = sum(1 for c in done if n_rec.get(int(c), 0) == D[a]["conv_start"][c]["turns"] == D[a]["conv_end"][c]["turns_done"])
    print(f"  {a:14s} {len(done):4d} {full:4d}")

print("\nlate records: per arm, turns sent in the last 300 s and the last 600 s before the last send; longest ttft and latency anywhere; turns with latency over 300 s")
for a in D:
    ts = [t for t in D[a]["turns"] if t["ttft"] is not None]
    last = max(t["t_send"] for t in ts)
    print(f"  {a:14s} last send {last:7.1f} s | last 300 s: {sum(1 for t in ts if t['t_send'] >= last - 300):4d} turns mean ttft {C.mean(t['ttft'] for t in ts if t['t_send'] >= last - 300):6.2f} | "
          f"last 600 s: {sum(1 for t in ts if t['t_send'] >= last - 600):4d} | max ttft {max(t['ttft'] for t in ts):6.1f} max latency {max(t['latency'] for t in ts):6.1f} | latency over 300 s: {sum(1 for t in ts if t['latency'] > 300)}")


def run(names, title, extra=None):
    print(f"\n{title}")
    for margin in (0.0, 300.0, 600.0):
        B = {a: C.by_key(C.kept(D[a], margin)) for a in names}
        keys = C.shared_keys(B, names, rank)
        # complete sessions: every turn 0..turns-1 kept in every arm
        convs = set(k[0] for k in keys)
        complete = set()
        for c in convs:
            ok = True
            for a in names:
                cs = D[a]["conv_start"].get(str(c))
                if cs is None or any((c, i) not in B[a] for i in range(cs["turns"])):
                    ok = False
                    break
            if ok:
                complete.add(c)
        kc = [k for k in keys if k[0] in complete]
        for label, ks in ((f"margin {margin:3.0f} s, shared turns", keys), (f"margin {margin:3.0f} s, complete sessions only", kc)):
            m = {a: C.mean(B[a][k]["ttft"] for k in ks) for a in names}
            ep = [k for k in ks if C.in_episode(rank, k)]
            calm = [k for k in ks if not C.in_episode(rank, k)]
            line = f"  {label:42s} {len(ks):6d} turns {len(set(k[0] for k in ks)):4d} sessions | " + " ".join(f"{a} {m[a]:.3f}" for a in names)
            print(line)
            me = {a: C.mean(B[a][k]["ttft"] for k in ep) for a in names}
            mc = {a: C.mean(B[a][k]["ttft"] for k in calm) for a in names}
            print(f"      episodes {len(ep):5d}: " + " ".join(f"{me[a]:.3f}" for a in names) + f" | elsewhere {len(calm):5d}: " + " ".join(f"{mc[a]:.3f}" for a in names) + f" (range {max(mc.values()) - min(mc.values()):.3f})")
            if extra:
                extra(m, B, ks)


def r3(m, B, ks):
    out = []
    for base in ("baseline", "base_rep"):
        for a in ("nofc_noend", "hintfree", "hintfree_ctl"):
            out.append(f"{a} from {base} {100 * (m[base] - m[a]) / (m[base] - m['jit_exact']):.1f} %")
    print("      share of the way to jit_exact: " + "; ".join(out))


def r2(m, B, ks):
    rn = sum(1 for k in ks if READ["nofc_noend"].get(k, 0) > 0)
    rf = sum(1 for k in ks if READ["nofc"].get(k, 0) > 0)
    print(f"      nofc_noend - nofc {m['nofc_noend'] - m['nofc']:+.3f} s; reading turns {rn} against {rf} ({100 * (rn / rf - 1):+.1f} %)")


def r4(m, B, ks):
    print(f"      jit_p50_d_cut - jit_p50_ctl {m['jit_p50_d_cut'] - m['jit_p50_ctl']:+.3f}; jit_p50 - jit_p50_ctl {m['jit_p50'] - m['jit_p50_ctl']:+.3f}; "
          f"lead of nofc_noend, ec_r13, jit_exact over jit_p50_d_cut {m['jit_p50_d_cut'] - m['nofc_noend']:+.3f} {m['jit_p50_d_cut'] - m['ec_r13']:+.3f} {m['jit_p50_d_cut'] - m['jit_exact']:+.3f}")


def r1(m, B, ks):
    rn = sum(1 for k in ks if READ["nofc_noend"].get(k, 0) > 0)
    rf = sum(1 for k in ks if READ["nofc"].get(k, 0) > 0)
    print(f"      order: {' < '.join(sorted(m, key=m.get))}; nofc_noend - nofc {m['nofc_noend'] - m['nofc']:+.3f}; reading turns nofc_noend {rn} against nofc {rf} ({100 * (rn / rf - 1):+.1f} %)")


run(C.LEAD5, "R1/R2: LEAD5 (jit_exact ec_r13 ec_r1 nofc nofc_noend)", r1)
run(["nofc", "nofc_noend"], "R2: the two rivals alone", r2)
run(C.ALL11, "R3: ALL11 (" + " ".join(C.ALL11) + ")", r3)
run(C.CUT7, "R4: CUT7 (" + " ".join(C.CUT7) + ")", r4)
