"""What each rule of the unified design did in a run, next to the current design: one table per setting.

    python3 mechanisms.py <setting folder under compare_unified> [--bins 0,28,50,70,90,110]

Reads the setting's manifest.txt (label, run folder, programs folder) and, per arm, server.log and client.jsonl.
Per window of the client's clock (minutes; requests by send time, server lines by time stamp):
  rec %     prompt tokens recomputed although the client's history says they were cached before, share of prompt
  ssd rd    SSD tokens read into host memory (s2h_io), millions
  ssd wr    tokens written to the SSD (h2s_io), millions
  skip      writes of an ended session's last turn skipped (h2s_skip), and their tokens in millions
  failed    SSD page reads that failed (pages the cleaner had deleted)
  clean     cleaner passes, groups deleted, groups kept by the re-check (l3_kept)
  ends      sessions ended by their hint (ntp_end explicit=1) / forgotten (explicit=0) / ended otherwise (current design)
  touch     l3_touch lines (refresh of pages in use), l3_retire lines
  held      prefill batches held for their copy (load_hold)
  prices    medians of mu, c and lam over the window's prices lines
"""
import argparse
import calendar
import collections
import json
import os
import re
import statistics as st

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
STAMP = re.compile(r"^\[(\d{4})-(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d\.\d+)\] ")
EVT = re.compile(r"HICACHE_EVT (\S+) ?(.*)$")
CLEAN = re.compile(r"NIXL L3 cleanup: deleted (\d+) groups")


def stamp(m):
    y, mo, d, h, mi, s = m.groups()
    return calendar.timegm((int(y), int(mo), int(d), int(h), int(mi), 0)) + float(s)


def fields(text):
    out = {}
    for tok in text.split():
        k, _, v = tok.partition("=")
        out[k] = v
    return out


def client(path):
    t0 = None
    reqs = []
    with open(path) as f:
        for line in f:
            if t0 is None and '"kind":"run"' in line:
                t0 = json.loads(line)["t_start"]
            if '"kind":"req"' not in line:
                continue
            r = json.loads(line)
            if r.get("phase") == "profile" and r.get("ttft") is not None and not r.get("error"):
                reqs.append(r)
    return t0, reqs


def server(path, t0, edges):
    nb = len(edges) - 1
    acc = [collections.Counter() for _ in range(nb)]
    prices = [collections.defaultdict(list) for _ in range(nb)]

    def bin_of(t):
        m = (t - t0) / 60.0
        for i in range(nb):
            if edges[i] <= m < edges[i + 1]:
                return i
        return None

    with open(path, errors="replace") as f:
        for line in f:
            m = STAMP.match(line)
            if not m:
                continue
            body = line[m.end():]
            if "HICACHE_EVT" not in body and "NIXL L3 cleanup" not in body and "failed to retrieve page" not in body:
                continue
            b = bin_of(stamp(m))
            if b is None:
                continue
            a = acc[b]
            if "failed to retrieve page" in body:
                a["failed"] += 1
                continue
            c = CLEAN.search(body)
            if c:
                a["passes"] += 1
                a["deleted"] += int(c.group(1))
                continue
            e = EVT.search(body)
            if not e:
                continue
            name, rest = e.group(1), e.group(2)
            if name == "s2h_io":
                a["ssd_rd"] += int(fields(rest).get("tokens", 0))
            elif name == "h2s_io":
                a["ssd_wr"] += int(fields(rest).get("tokens", 0))
            elif name == "h2s_skip":
                a["skip"] += 1
                a["skip_tok"] += int(fields(rest).get("tokens", 0))
            elif name == "l3_kept":
                a["kept"] += int(fields(rest).get("groups", 0))
            elif name == "ntp_end":
                a["end_explicit" if fields(rest).get("explicit") == "1" else "end_forgot"] += 1
            elif name == "l3_touch":
                a["touch"] += 1
            elif name == "l3_retire":
                a["retire"] += 1
            elif name == "load_hold":
                a["held"] += 1
            elif name == "nue_register" and "next_use_ms=ended" in rest:
                a["end_registered"] += 1
            elif name == "prices":
                f = fields(rest)
                for k in ("mu", "c", "lam"):
                    try:
                        prices[b][k].append(float(f[k]))
                    except (KeyError, ValueError):
                        pass
    return acc, prices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("setting")
    ap.add_argument("--bins", default="0,28,50,70,90,110")
    x = ap.parse_args()
    edges = [float(v) for v in x.bins.split(",")]
    folder = os.path.join(RES, "compare_unified", x.setting)
    arms = [line.split() for line in open(os.path.join(folder, "manifest.txt")) if line.strip()]
    for label, run, prog in arms:
        if label.endswith("_c34"):
            continue
        t0, reqs = client(os.path.join(RES, run, prog, "client.jsonl"))
        acc, prices = server(os.path.join(RES, run, "server.log"), t0, edges)
        print(f"== {label}: {run}")
        print("  minutes  | TTFT s | rec %  | ssd rd M | ssd wr M | skip (M tok) | failed | clean: passes, deleted, kept | ends: hint, forgot, reg | touch, retire | held | mu, c, lam")
        for i in range(len(edges) - 1):
            lo, hi = edges[i] * 60, edges[i + 1] * 60
            w = [r for r in reqs if lo <= r["t_send"] < hi]
            lost = sum(max(0, (r.get("ideal_cached") or 0) - (r.get("cached_tokens") or 0)) for r in w)
            prompt = sum(r.get("prompt_tokens") or 0 for r in w)
            a, p = acc[i], prices[i]
            med = lambda k: f"{st.median(p[k]):.3g}" if p[k] else "-"  # noqa: E731
            print(f"  {edges[i]:3.0f}-{edges[i + 1]:<4.0f} | {st.mean(r['ttft'] for r in w) if w else float('nan'):6.2f} | "
                  f"{100 * lost / prompt if prompt else 0:5.2f}  | {a['ssd_rd'] / 1e6:8.1f} | {a['ssd_wr'] / 1e6:8.1f} | "
                  f"{a['skip']:5d} ({a['skip_tok'] / 1e6:4.1f}) | {a['failed']:6d} | {a['passes']:2d}, {a['deleted']:6d}, {a['kept']:5d} | "
                  f"{a['end_explicit']:4d}, {a['end_forgot']:4d}, {a['end_registered']:4d} | {a['touch']:4d}, {a['retire']:4d} | "
                  f"{a['held']:5d} | {med('mu')}, {med('c')}, {med('lam')}")


if __name__ == "__main__":
    main()
