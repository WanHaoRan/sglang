"""The closed loop's accounting on two runs of the same users: does the change in TTFT follow from the change in
GPU time per request?

    python3 identity_check.py <client.jsonl of A> <client.jsonl of B> [--bin S] [--to S]

Per window of the client's clock, for each run: users in session N and requests in flight F (the client's own
"lanes" samples), completions a second X, mean response time R and mean TTFT of the requests sent in the window, and
the mean time from first token to the end D. Two checks:
  Little's law in the window: F / X against R (they agree when the window is long against R).
  The identity TTFT = N / X - Z' - D with Z' the think time per request (N - F) / X: so TTFT = F / X - D.
  Predicted change of TTFT from A to B = change of F / X - change of D, against the measured change.
"""
import argparse
import json
import statistics as st


def load(path):
    reqs, lanes = [], []
    with open(path) as f:
        for line in f:
            if '"kind":"req"' in line:
                r = json.loads(line)
                if r.get("phase") == "profile" and r.get("ttft") is not None and not r.get("error"):
                    reqs.append(r)
            elif '"kind":"lanes"' in line:
                lanes.append(json.loads(line))
    return reqs, lanes


def window(reqs, lanes, lo, hi):
    sent = [r for r in reqs if lo <= r["t_send"] < hi]
    done = sum(1 for r in reqs if lo <= r["t_send"] + r["latency"] < hi)
    ln = [x for x in lanes if lo <= x["t"] < hi]
    if not sent or not ln or not done:
        return None
    return dict(
        n=st.mean(x["live"] for x in ln),
        f=st.mean(x["in_flight"] for x in ln),
        x=done / (hi - lo),
        r=st.mean(r["latency"] for r in sent),
        ttft=st.mean(r["ttft"] for r in sent),
        d=st.mean(r["latency"] - r["ttft"] for r in sent),
        sent=len(sent),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--bin", type=float, default=600.0)
    ap.add_argument("--to", type=float, default=6600.0)
    a = ap.parse_args()
    ra, la = load(a.a)
    rb, lb = load(a.b)
    print("window (min) | run: users N, in flight F, done/s X, F/X, response R, TTFT, decode D | TTFT change: measured, from F/X - D")
    lo = 0.0
    tot = [0.0, 0.0, 0]
    while lo < a.to:
        hi = min(lo + a.bin, a.to)
        wa, wb = window(ra, la, lo, hi), window(rb, lb, lo, hi)
        if wa and wb:
            pa, pb = wa["f"] / wa["x"] - wa["d"], wb["f"] / wb["x"] - wb["d"]
            meas, pred = wb["ttft"] - wa["ttft"], pb - pa
            n = min(wa["sent"], wb["sent"])
            tot[0] += meas * n; tot[1] += pred * n; tot[2] += n
            print(f"{lo / 60:4.0f}-{hi / 60:<4.0f} | A: N {wa['n']:5.1f} F {wa['f']:5.1f} X {wa['x']:.3f} F/X {wa['f'] / wa['x']:6.2f} R {wa['r']:6.2f} TTFT {wa['ttft']:6.2f} D {wa['d']:5.2f}"
                  f" | B: N {wb['n']:5.1f} F {wb['f']:5.1f} X {wb['x']:.3f} F/X {wb['f'] / wb['x']:6.2f} R {wb['r']:6.2f} TTFT {wb['ttft']:6.2f} D {wb['d']:5.2f}"
                  f" | {meas:+6.2f} s, {pred:+6.2f} s")
        lo = hi
    if tot[2]:
        print(f"weighted by requests: measured {tot[0] / tot[2]:+.2f} s, from the accounting {tot[1] / tot[2]:+.2f} s")


if __name__ == "__main__":
    main()
