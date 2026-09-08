"""Single-request probe: one JSON line with TTFT and per-tier cache attribution.

Streams /v1/chat/completions with max_tokens=1 so the measurement isolates
prefill. --enable-cache-report makes the server return per-tier cached-token
counts, which is how each sample is attributed to L1/L2/L3.
"""
import argparse
import json
import sys
import time

import requests

sys.path.insert(0, "/sgl-workspace/sglang/hicache_eval/scripts")
import hcommon  # noqa: E402


DETAIL_KEYS = ("cached_tokens_details", "sglext_cached_tokens_details")


def find_details(obj, path=""):
    """Locate per-tier cached-token counts wherever the build puts them.

    The field is opt-in (return_cached_tokens_details) and at this commit lands
    as a top-level sglext_cached_tokens_details rather than inside usage, so
    search the whole payload instead of guessing one path.
    """
    if isinstance(obj, dict):
        for k in DETAIL_KEYS:
            v = obj.get(k)
            if isinstance(v, dict) and ("device" in v or "storage" in v):
                return v, f"{path}.{k}".lstrip(".")
        for k, v in obj.items():
            d, p = find_details(v, f"{path}.{k}".lstrip("."))
            if d:
                return d, p
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            d, p = find_details(v, f"{path}[{i}]")
            if d:
                return d, p
    return {}, None


def probe(base, body, target, seed, meta, enable_thinking=False, timeout=1200):
    payload = {
        "model": hcommon.MODEL,
        "messages": [{"role": "user", "content": body}],
        "max_tokens": 1,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "return_cached_tokens_details": True,
    }
    t0 = time.perf_counter()
    ttft = None
    usage = None
    details_seen = []
    with requests.post(
        f"{base}/v1/chat/completions", json=payload, stream=True, timeout=timeout
    ) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode() if isinstance(raw, bytes) else raw
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data.strip() == "[DONE]":
                break
            obj = json.loads(data)
            ch = obj.get("choices") or []
            if ttft is None and ch and (ch[0].get("delta") or {}).get("content") is not None:
                ttft = time.perf_counter() - t0
            if obj.get("usage"):
                usage = obj["usage"]
            d, dp = find_details(obj)
            if d:
                details_seen.append((d, dp))
    latency = time.perf_counter() - t0
    if ttft is None:
        ttft = latency
    usage = usage or {}
    details, details_path = details_seen[-1] if details_seen else ({}, None)
    ptd = usage.get("prompt_tokens_details") or {}
    rec = {
        "target_tokens": target,
        "seed": seed,
        "prompt_tokens": usage.get("prompt_tokens"),
        "ttft_s": ttft,
        "latency_s": latency,
        "cached_tokens": ptd.get("cached_tokens", usage.get("cached_tokens")),
        "cached_device": details.get("device"),
        "cached_host": details.get("host"),
        "cached_storage": details.get("storage"),
        "details_path": details_path,
        "ts": time.time(),
    }
    rec.update(meta)
    return rec, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--len", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default=None, help="jsonl to append to")
    ap.add_argument("--base", default=hcommon.BASE)
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--meta", default="{}", help="JSON dict merged into the record")
    ap.add_argument("--print-usage", action="store_true")
    ap.add_argument("--repeat", type=int, default=1, help="send N times, log the last")
    a = ap.parse_args()

    body = hcommon.build_prompt(a.len, a.seed, a.enable_thinking)
    meta = json.loads(a.meta)
    rec = usage = None
    for _ in range(a.repeat):
        rec, usage = probe(a.base, body, a.len, a.seed, meta, a.enable_thinking)
    if a.print_usage:
        print(json.dumps(usage, indent=2), file=sys.stderr)
    line = json.dumps(rec)
    print(line)
    if a.out:
        with open(a.out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
