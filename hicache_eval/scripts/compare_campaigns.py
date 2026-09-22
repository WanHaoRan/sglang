"""Side-by-side: old H100 campaigns vs a rerun, Qwen3-8B on the nixl backend. Read-only on the old dirs.

  RESULTS=<new campaign dir> python3 compare_campaigns.py
  optional env: OLD_C1 / OLD_C2 / OLD_C3 (old campaign dirs), OLD_LABEL, NEW_LABEL, COMPARE_OUT (output dir),
                CAMPAIGN=8b|c2 (default: c2 when RESULTS holds exp1_70b or exp1_32b, else 8b)

  8b: Qwen3-8B, campaign 1 (Exp 0/1 + backend A/B) and campaign 3 (Exp 2/3/4)
  c2: Llama-3.3-70B-AWQ and Qwen3-32B-FP8 with fp8_e5m2 KV, campaign 2 (Exp 0/1 + the 70B p_measure sweep)

Writes $RESULTS/COMPARISON.md and $RESULTS/comparison.csv. Every statistic is computed the same way
on both sides, from the raw per-probe files, with the definitions the old analysis used:
  Exp 1  fit = np.polyfit(L, per-L median TTFT, 1); rate = 147456/slope/2**30 GiB/s (archive/analyze_nixl.py)
  Exp 2  loaded points use rep<2 (rep index is confounded with queue depth, HANDOFF §3); an L3 row
         counts only if cached_storage == 16320; a control row only if it hit no tier
  Exp 3  bytes from the hicache_backup counters and the files on disk (archive/exp3.py)
Missing stages are skipped, so this can run while the campaign is still in progress.
"""
import json
import math
import os

import numpy as np
import pandas as pd

BASE = "/sgl-workspace/sglang/hicache_eval/results"
NEW = os.environ["RESULTS"]
C1 = os.environ.get("OLD_C1", f"{BASE}/20260907_203035")
C2 = os.environ.get("OLD_C2", f"{BASE}/20260908_llama70b_awq_fp8kv")
C3 = os.environ.get("OLD_C3", f"{BASE}/20260908_nixl_exp234")
OUT_DIR = os.environ.get("COMPARE_OUT", NEW)
MODE = os.environ.get("CAMPAIGN") or ("c2" if any(os.path.isdir(f"{NEW}/exp1_{k}") for k in ("70b", "32b")) else "8b")
# key -> (display name, KV bytes per token with fp8_e5m2 KV)
C2_MODELS = {"70b": ("Llama-3.3-70B-AWQ", 163840), "32b": ("Qwen3-32B-FP8", 131072)}
OLD_LABEL = os.environ.get("OLD_LABEL", "H100 PCIe, virtio disk (old)")
NEW_LABEL = os.environ.get("NEW_LABEL", "A100 SXM4, local NVMe (new)")
KV = 147456
FULL_HIT = 16320

md, flat = [], []


def ex(p):
    return os.path.exists(p)


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "-"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.{nd}f}"
    return str(v)


def ratio(new, old):
    try:
        return new / old if old else float("nan")
    except Exception:
        return float("nan")


def table(title, rows, note=None, nd=3):
    """rows: (metric, old, new[, unit]) -> markdown table with a new/old column; also fills the flat csv."""
    md.append(f"\n### {title}\n")
    md.append("| metric | old | new | new / old |")
    md.append("|---|---:|---:|---:|")
    for r in rows:
        name, o, n = r[0], r[1], r[2]
        unit = r[3] if len(r) > 3 else ""
        num = all(isinstance(x, (int, float, np.integer, np.floating)) and x == x for x in (o, n))
        md.append(f"| {name}{' (' + unit + ')' if unit else ''} | {fmt(o, nd)} | {fmt(n, nd)} | "
                  f"{fmt(ratio(n, o), 2) + 'x' if num and o else '-'} |")
        flat.append({"section": title, "metric": name, "unit": unit, "old": o, "new": n,
                     "new_over_old": ratio(n, o) if num and o else None})
    if note:
        md.append(f"\n{note}")


# ------------------------------------------------------------------ setup ---
def startup(path):
    out = {}
    if not ex(path):
        return out
    for line in open(path, errors="ignore"):
        if "max_total_num_tokens=" in line and "device_pool" not in out:
            out["device_pool"] = int(line.split("max_total_num_tokens=")[1].split(",")[0])
        if "host pool:" in line:
            out["host_pool"] = int(line.split("host pool:")[1].split("tokens")[0])
        if "'attention_backend': '" in line and "attn" not in out:
            out["attn"] = line.split("'attention_backend': '")[1].split("'")[0]
    return out


def server_log_facts(path):
    """Weight-kernel path and KV dtype as the server logged them."""
    out = {}
    if not ex(path):
        return out
    for line in open(path, errors="ignore"):
        low = line.lower()
        if "kernel" not in out and any(k in low for k in ("marlin kernel", "awq_marlin kernel", "deepgemm jit")):
            out["kernel"] = line.split("] ", 1)[-1].strip()[:110]
        if "kv_dtype" not in out and "KV Cache is allocated" in line and "dtype:" in line:
            out["kv_dtype"] = line.split("dtype:")[1].split(",")[0].strip()
        if "attn" not in out and "'attention_backend': '" in line:
            out["attn"] = line.split("'attention_backend': '")[1].split("'")[0]
    return out

# ------------------------------------------------------------------ Exp 0 ---
def compare_exp0(po, pn, title, old_dir_note=""):
    if not (ex(po) and ex(pn)):
        return
    o, n = json.load(open(po)), json.load(open(pn))
    steps = [("step1_cold", "1 cold recompute"), ("step3_L1", "3 L1 hit"), ("step4_L2", "4 L2 hit"),
             ("step5_L3", "5 L3 hit, page cache dropped"), ("step6_L3_pagecache", "6 L3 hit, page cache warm")]
    rows = [(f"TTFT {lbl}", o.get(k, {}).get("ttft_s"), n.get(k, {}).get("ttft_s"), "s") for k, lbl in steps]
    table(title, rows,
          note="Attribution (device/host/storage tokens) new run: " + "; ".join(
              f"{lbl.split()[0]}: {n.get(k, {}).get('cached_device')}/{n.get(k, {}).get('cached_host')}/"
              f"{n.get(k, {}).get('cached_storage')}" for k, lbl in steps) +
          f". L3 files after backup: old {o.get('step2_backup', {}).get('l3_files_after')}"
          f"{old_dir_note}, new {n.get('step2_backup', {}).get('l3_files_after')}. "
          "n=1 and the first long prefill after boot pays kernel warm-up: use Exp 1 for timings.")


# ------------------------------------------------------------------ Exp 1 ---
def exp1(d):
    p = f"{d}/ttft_by_tier.csv"
    if not ex(p):
        return None, None
    t = pd.read_csv(p)
    t = t[t.ttft_s.notna()]
    hit = (t.cached_device.fillna(0) + t.cached_host.fillna(0) + t.cached_storage.fillna(0)) > 0
    t = t[~((t.tier == "recompute") & hit)]
    med = t.groupby(["tier", "L"]).ttft_s.median().unstack(0)
    fits = {}
    for tier in med.columns:
        s = med[tier].dropna()
        if len(s) >= 2:
            slope, icept = np.polyfit(s.index, s.values, 1)
            fits[tier] = (slope, icept)
    return med, fits


def compare_exp1(old_dir, new_dir, kv, prefix=""):
    mo, fo = exp1(old_dir)
    mn, fn = exp1(new_dir)
    if mo is None or mn is None:
        return None
    rows = []
    for tier in ["recompute", "L1", "L2", "L3"]:
        if tier in fo and tier in fn:
            unit = "tok/s" if tier == "recompute" else "GiB/s"
            val = (lambda f: 1 / f[0]) if tier == "recompute" else (lambda f: kv / f[0] / 2 ** 30)
            name = "P = marginal prefill rate" if tier == "recompute" else f"{tier} delivered rate (fit)"
            rows.append((name, val(fo[tier]), val(fn[tier]), unit))
            rows.append((f"{tier} fit intercept", fo[tier][1], fn[tier][1], "s"))
    if "recompute" in fo and "recompute" in fn:
        rows.insert(1, ("recompute bar b*P", kv / fo["recompute"][0] / 2 ** 30, kv / fn["recompute"][0] / 2 ** 30, "GiB/s"))
    for a, b, name in [("L2", "L1", "host to device, differential (L2 - L1 slopes)"),
                       ("L3", "L2", "disk read, differential (L3 - L2 slopes)")]:
        if all(k in f for f in (fo, fn) for k in (a, b)):
            # A difference of two fitted slopes is only a transfer rate when the slopes are clearly apart. For a
            # slow model the layer-wise host load overlaps with compute, so L2 - L1 understates the copy time.
            d = lambda f: kv / (f[a][0] - f[b][0]) / 2 ** 30 if f[a][0] >= 1.5 * f[b][0] else float("nan")
            rows.append((name, d(fo), d(fn), "GiB/s"))
    for tier in ["L2", "L3"]:
        if all(k in f for f in (fo, fn) for k in (tier, "recompute")):
            rows.append((f"{tier} rate / recompute bar", fo["recompute"][0] / fo[tier][0], fn["recompute"][0] / fn[tier][0], "x"))

            def be(f):
                ds = f[tier][0] - f["recompute"][0]
                return (f["recompute"][1] - f[tier][1]) / ds if ds < 0 else float("nan")
            rows.append((f"{tier} beats recompute above L (fit)", be(fo), be(fn), "tokens"))
    table(f"{prefix}Exp 1: tier cost fits over 7 lengths (512 to 32512), medians of n=3", rows,
          note="A tier pays only when its delivered rate exceeds the recompute bar (rate / bar > 1). "
               "`-` for the break-even means the tier's slope is not below recompute's, so it never wins at any length.")
    for tier in ["recompute", "L1", "L2", "L3"]:
        if tier in mo.columns and tier in mn.columns:
            table(f"{prefix}Exp 1: median TTFT, {tier}",
                  [(f"L={int(L)}", mo[tier].get(L), mn[tier].get(L), "s") for L in mn.index if L in mo.index], nd=4)
    if "L3" in mn.columns and "recompute" in mn.columns:
        table(f"{prefix}Exp 1: L3 hit TTFT / recompute TTFT (below 1 = L3 wins)",
              [(f"L={int(L)}", mo["L3"].get(L) / mo["recompute"].get(L), mn["L3"].get(L) / mn["recompute"].get(L), "x")
               for L in mn.index if L in mo.index], nd=2)
    return fo, fn


# ------------------------------------------------------------------ Exp 2 ---
def exp2(paths, control):
    fr = [pd.read_csv(p) for p in paths if ex(p)]
    if not fr:
        return None
    d = pd.concat(fr, ignore_index=True)
    d = d[d.control.isin([control, control + "_recompute"])].copy()
    if d.empty:
        return None
    for c in ("cached_storage", "cached_device", "cached_host"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["is_ctl"] = d.control.str.endswith("_recompute")
    hit_any = (d.cached_storage.fillna(0) + d.cached_device.fillna(0) + d.cached_host.fillna(0)) > 0
    d["valid"] = np.where(d.is_ctl, ~hit_any, d.cached_storage == FULL_HIT)
    return d


def exp2_point(d, R):
    g = d[d.requested_rps == R]
    sel = g if R == 0 else g[g.rep < 2]
    l3 = sel[(~sel.is_ctl) & sel.valid].ttft_s
    rc = sel[sel.is_ctl & sel.valid].ttft_s
    return {"L3": l3.mean() if len(l3) else float("nan"), "rec": rc.mean() if len(rc) else float("nan"),
            "valid_L3": int(g[(~g.is_ctl)].valid.sum()), "n_L3": int((~g.is_ctl).sum()),
            "bad_ctl": int((g.is_ctl & ~g.valid).sum()),
            "achieved": g.achieved_rps.max(), "wgbps": g.write_gbps_actual.max(),
            "iow": sel[~sel.is_ctl].iostat_w_mbps.mean(), "ior": sel[~sel.is_ctl].iostat_r_mbps.mean()}


def paired(d, rates):
    g = d[d.requested_rps.isin(rates) & (d.rep < 2) & d.valid]
    a = g[~g.is_ctl].set_index(["requested_rps", "rep"]).ttft_s
    b = g[g.is_ctl].set_index(["requested_rps", "rep"]).ttft_s
    diff = (a - b).dropna()
    if len(diff) < 2:
        return None
    m, s = diff.mean(), diff.std(ddof=1)
    h = 1.96 * s / math.sqrt(len(diff))
    return m, m - h, m + h, len(diff)


def run_8b():


    so, sn = startup(f"{C1}/nixl/exp1_nixl/startup_facts.txt"), startup(f"{NEW}/exp1_nixl/startup_facts.txt")
    if not sn:
        sn = startup(f"{NEW}/ab_nixl/startup_facts.txt")
    md.append("# Old vs new: HiCache evaluation, Qwen3-8B, nixl backend\n")
    md.append(f"- **old**: {OLD_LABEL}; Exp 0/1 + A/B from `{os.path.basename(C1)}`, Exp 2/3/4 from `{os.path.basename(C3)}`")
    md.append(f"- **new**: {NEW_LABEL}; `{os.path.basename(NEW)}`")
    md.append("- same image, same sglang commit under `python/`, same server flags and client parameters; "
              "differences forced by the box are listed in `DEVIATIONS.md`")
    if so or sn:
        table("Setup", [("device pool (L1)", so.get("device_pool"), sn.get("device_pool"), "tokens"),
                        ("host pool (L2)", so.get("host_pool"), sn.get("host_pool"), "tokens"),
                        ("attention backend", so.get("attn"), sn.get("attn"))])


    compare_exp0(f"{C1}/nixl/exp0_nixl/exp0_results.json", f"{NEW}/exp0_nixl/exp0_results.json",
                 "Exp 0: tier attribution, one 4096-token prompt, n=1", " (old driver looked at the wrong dir)")

    # ------------------------------------------------------------- backend A/B ---
    for side in ("nixl", "file"):
        po, pn = f"{C1}/backend_ab/{side}.json", f"{NEW}/backend_ab/{side}.json"
        if ex(po) and ex(pn):
            o, n = json.load(open(po)), json.load(open(pn))
            table(f"Backend A/B, {side} backend, 4096-token probe", [
                ("L3 hit TTFT", o.get("L3_ttft_s"), n.get("L3_ttft_s"), "s"),
                ("L3 read rate (hit bytes / TTFT)", o.get("L3_read_GBps"), n.get("L3_read_GBps"), "GiB/s"),
                ("L3 write rate (backup bytes / drain)", o.get("L3_write_GBps"), n.get("L3_write_GBps"), "GiB/s"),
                ("backup drain", o.get("backup_drain_s"), n.get("backup_drain_s"), "s"),
                ("cold recompute TTFT", o.get("cold_recompute_ttft_s"), n.get("cold_recompute_ttft_s"), "s")])
    pa, pb = f"{NEW}/backend_ab/nixl.json", f"{NEW}/backend_ab/file.json"
    if ex(pa) and ex(pb):
        a, b = json.load(open(pa)), json.load(open(pb))
        oa, ob = json.load(open(f"{C1}/backend_ab/nixl.json")), json.load(open(f"{C1}/backend_ab/file.json"))
        table("Backend A/B, nixl over file", [
            ("L3 read rate ratio nixl/file", ratio(oa.get("L3_read_GBps"), ob.get("L3_read_GBps")),
             ratio(a.get("L3_read_GBps"), b.get("L3_read_GBps")), "x")], nd=1)


    compare_exp1(f"{C1}/nixl/exp1_nixl", f"{NEW}/exp1_nixl", KV)

    bo = exp2([f"{C3}/exp2_qwen8b_rerun/interference.csv"], "baseline")
    bn = exp2([f"{NEW}/exp2_qwen8b/interference.csv"], "baseline")
    if bo is not None and bn is not None:
        rates = sorted(set(bn.requested_rps) & set(bo.requested_rps))
        rows, notes = [], []
        for R in rates:
            o, n = exp2_point(bo, R), exp2_point(bn, R)
            tag = "all reps" if R == 0 else "rep<2"
            rows += [(f"R={R:g}: L3 hit TTFT, {tag}", o["L3"], n["L3"], "s"),
                     (f"R={R:g}: paired recompute TTFT, {tag}", o["rec"], n["rec"], "s"),
                     (f"R={R:g}: speed-up recompute / L3", ratio(o["rec"], o["L3"]), ratio(n["rec"], n["L3"]), "x"),
                     (f"R={R:g}: achieved write-load rate", o["achieved"], n["achieved"], "req/s"),
                     (f"R={R:g}: committed write rate", o["wgbps"], n["wgbps"], "GiB/s")]
            notes.append(f"R={R:g}: full L3 hits old {o['valid_L3']}/{o['n_L3']}, new {n['valid_L3']}/{n['n_L3']}; "
                         f"contaminated controls old {o['bad_ctl']}, new {n['bad_ctl']}")
        table("Exp 2: L3 hit latency under write-through load, 16384-token probe, wait_complete", rows,
              note="Validity per point. " + ". ".join(notes) + ".")
        r0o = bo[(bo.requested_rps == 0) & (bo.rep == 0) & bo.valid]
        r0n = bn[(bn.requested_rps == 0) & (bn.rep == 0) & bn.valid]
        if len(r0o) and len(r0n):
            g = lambda d, c: d[d.is_ctl == c].ttft_s.mean()
            table("Exp 2: truly idle point (R=0, rep 0 only: later reps overlap the previous control's write-through)", [
                ("L3 hit TTFT", g(r0o, False), g(r0n, False), "s"), ("paired recompute TTFT", g(r0o, True), g(r0n, True), "s"),
                ("speed-up recompute / L3", ratio(g(r0o, True), g(r0o, False)), ratio(g(r0n, True), g(r0n, False)), "x")])
        po_, pn_ = paired(bo, [0.5, 1, 2, 4]), paired(bn, [0.5, 1, 2, 4])
        if po_ and pn_:
            table("Exp 2: pooled paired difference L3 - recompute, R in {0.5, 1, 2, 4}, rep<2", [
                ("mean difference (negative = L3 wins)", po_[0], pn_[0], "s"),
                ("95% CI low", po_[1], pn_[1], "s"), ("95% CI high", po_[2], pn_[2], "s"), ("pairs", po_[3], pn_[3])])

    to = exp2([f"{C3}/exp2_qwen8b/interference.csv"], "timeout_policy")
    tn = exp2([f"{NEW}/exp2_qwen8b/interference.csv"], "timeout_policy")
    wo = exp2([f"{C3}/exp2_qwen8b/interference.csv"], "baseline")          # the un-paired sweep the 1.53x came from
    if to is not None and tn is not None and bn is not None:
        med = lambda d: d[(d.requested_rps == 0) & (~d.is_ctl) & d.valid].ttft_s.median()
        rows = [("idle L3 hit TTFT, timeout policy, median", med(to), med(tn), "s"),
                ("idle L3 hit TTFT, wait_complete, median", med(wo) if wo is not None else None, med(bn), "s"),
                ("wait_complete / timeout at idle", ratio(med(wo), med(to)) if wo is not None else None, ratio(med(bn), med(tn)), "x")]
        for R in sorted(set(tn.requested_rps) - {0}):
            o, n = exp2_point(to, R), exp2_point(tn, R)
            rows.append((f"R={R:g}: L3 hit TTFT, timeout policy, rep<2 (full hits only)", o["L3"], n["L3"], "s"))
            rows.append((f"R={R:g}: full L3 hits out of 3", o["valid_L3"], n["valid_L3"]))
        table("Exp 2: timeout vs wait_complete prefetch policy", rows,
              note="Under load the timeout deadline (1 s + 0.25 s per 1024 tokens, about 5 s for this probe) can fire "
                   "before the read completes; such rows are partial hits and are excluded from the TTFT means.")

    # --------------------------------------------------------------- Exp 3 / 4 ---
    po, pn = f"{C3}/exp3_qwen8b/amplification.csv", f"{NEW}/exp3_qwen8b/amplification.csv"
    if ex(po) and ex(pn):
        o = pd.read_csv(po).drop_duplicates("condition").set_index("condition")
        n = pd.read_csv(pn).drop_duplicates("condition").set_index("condition")
        rows = []
        for c in [c for c in o.index if c in n.index]:
            rows += [(f"{c}: written below the GPU", o.written_L3_bytes[c] / 2 ** 30, n.written_L3_bytes[c] / 2 ** 30, "GiB"),
                     (f"{c}: bytes on L3 disk", o.l3_bytes_on_disk[c] / 2 ** 30, n.l3_bytes_on_disk[c] / 2 ** 30, "GiB"),
                     (f"{c}: tokens read back from L3", o.read_L3_tokens[c], n.read_L3_tokens[c], "tokens"),
                     (f"{c}: client duration", o.duration_s[c], n.duration_s[c], "s"),
                     (f"{c}: drain after the run", o.drain_s[c], n.drain_s[c], "s"),
                     (f"{c}: generated tokens", o.generation_tokens[c], n.generation_tokens[c], "tokens")]
        table("Exp 3 / 4: write amplification, bench_multiturn (16 clients x 4 rounds)", rows, nd=2)
        sav = []
        for c, lbl in [("wts100", "write_through_selective, host 100 GB"), ("wts30", "write_through_selective, host 30 GB"),
                       ("oracle100", "strip-thinking oracle, host 100 GB"), ("oracle30", "strip-thinking oracle, host 30 GB")]:
            ref = "wt100" if c.endswith("100") else "wt30"
            if all(k in d.index for d in (o, n) for k in (c, ref)):
                s = lambda d: 100 * (d.written_L3_bytes[c] - d.written_L3_bytes[ref]) / d.written_L3_bytes[ref]
                sav.append((f"{lbl}: bytes written vs write_through", s(o), s(n), "%"))
        if sav:
            table("Exp 3 / 4: savings against plain write_through", sav, nd=1)


def run_c2():
    md.append("# Old vs new: HiCache evaluation, 70B and 32B with fp8_e5m2 KV, nixl backend\n")
    md.append(f"- **old**: {OLD_LABEL}; `{os.path.basename(C2)}`")
    md.append(f"- **new**: {NEW_LABEL}; `{os.path.basename(NEW)}`")
    md.append("- same image, same sglang commit under `python/`, same server flags and client parameters; "
              "differences forced by the box are listed in `DEVIATIONS.md`")
    for key, (name, kv) in C2_MODELS.items():
        od, nd_ = f"{C2}/exp1_{key}", f"{NEW}/exp1_{key}"
        if not ex(f"{nd_}/ttft_by_tier.csv"):
            continue
        so, sn = startup(f"{od}/startup_facts.txt"), startup(f"{nd_}/startup_facts.txt")
        lo, ln = server_log_facts(f"{od}/server.log"), server_log_facts(f"{nd_}/server.log")
        table(f"{name}: setup", [("device pool (L1)", so.get("device_pool"), sn.get("device_pool"), "tokens"),
                                 ("host pool (L2)", so.get("host_pool"), sn.get("host_pool"), "tokens"),
                                 ("KV bytes per token", kv, kv, "B"),
                                 ("attention backend", lo.get("attn") or so.get("attn"), ln.get("attn") or sn.get("attn")),
                                 ("KV cache dtype", lo.get("kv_dtype"), ln.get("kv_dtype")),
                                 ("weight kernel path", lo.get("kernel"), ln.get("kernel"))])
        compare_exp0(f"{C2}/exp0_{key}/exp0_results.json", f"{NEW}/exp0_{key}/exp0_results.json",
                     f"{name}: Exp 0: tier attribution, one 4096-token prompt, n=1")
        compare_exp1(od, nd_, kv, prefix=f"{name}: ")
    # The 70B also has a pure-recompute sweep on a server without HiCache; on the H100 it agreed with Exp 1 to 0.1 %.
    (_, fo), (_, fn) = exp1(f"{C2}/p_measure"), exp1(f"{NEW}/p_measure")
    (_, eo), (_, en) = exp1(f"{C2}/exp1_70b"), exp1(f"{NEW}/exp1_70b")
    if fo and fn and "recompute" in fo and "recompute" in fn:
        rows = [("P from p_measure (no HiCache)", 1 / fo["recompute"][0], 1 / fn["recompute"][0], "tok/s")]
        if eo and en and "recompute" in eo and "recompute" in en:
            rows += [("P from the Exp 1 server (write_through on)", 1 / eo["recompute"][0], 1 / en["recompute"][0], "tok/s"),
                     ("Exp 1 P / p_measure P", fo["recompute"][0] / eo["recompute"][0], fn["recompute"][0] / en["recompute"][0], "x")]
        table("Llama-3.3-70B-AWQ: recompute with and without HiCache write-through", rows,
              note="A ratio below 1 on the new box means write-through to the slower disk is perturbing prefill.")


def write_out():
    os.makedirs(OUT_DIR, exist_ok=True)
    name = "COMPARISON.md"
    open(os.path.join(OUT_DIR, name), "w").write("\n".join(md) + "\n")
    pd.DataFrame(flat).to_csv(os.path.join(OUT_DIR, "comparison.csv"), index=False)
    print("\n".join(md))
    print(f"\nwrote {OUT_DIR}/{name} and comparison.csv ({len(flat)} rows)")


if __name__ == "__main__":
    run_c2() if MODE == "c2" else run_8b()
    write_out()
