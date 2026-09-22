"""Shared helpers for the HiCache evaluation: exact-length prompt construction,
metrics scraping, cache-state control."""
import glob
import json
import os
import random
import subprocess
import time
import urllib.request

BASE = os.environ.get("BASE", "http://127.0.0.1:30000")
# Must match env.sh's default, or a script run without L3_DIR exported silently points at the wrong
# device: l3_stats() would report 0 files and drop_page_cache() would fadvise nothing.
L3_DIR = os.environ.get("L3_DIR", "/mnt/ssd/hicache_l3")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
KV_BYTES_PER_TOKEN = int(os.environ.get("KV_BYTES_PER_TOKEN", 147456))

_TOK = None
_WORDS = None


def tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer

        _TOK = AutoTokenizer.from_pretrained(MODEL)
    return _TOK


def single_token_words():
    """Words that are exactly one token when preceded by a space.

    Using these makes prompt length linear in word count, so hitting an exact
    target is a search over N rather than a per-token fixup.
    """
    global _WORDS
    if _WORDS is None:
        tok = tokenizer()
        out = []
        for s, tid in tok.get_vocab().items():
            w = tok.convert_tokens_to_string([s]).strip()
            if len(w) >= 3 and w.isascii() and w.isalpha():
                if len(tok.encode(" " + w, add_special_tokens=False)) == 1:
                    out.append(w)
        out.sort()
        _WORDS = out
    return _WORDS


def templated_len(body, enable_thinking=False):
    tok = tokenizer()
    text = tok.apply_chat_template(
        [{"role": "user", "content": body}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return len(tok.encode(text, add_special_tokens=False))


def build_prompt(target_tokens, seed, enable_thinking=False):
    """Body whose templated length is exactly target_tokens.

    Distinct seeds pick a distinct first word, so two prompts diverge at token
    one and can never share a radix prefix.
    """
    words = single_token_words()
    rng = random.Random(seed)
    n = max(1, target_tokens - 24)
    body = " ".join(rng.choice(words) for _ in range(n))
    cur = templated_len(body, enable_thinking)
    guard = 0
    while cur != target_tokens and guard < 200:
        guard += 1
        delta = target_tokens - cur
        n = max(1, n + delta)
        rng = random.Random(seed)
        body = " ".join(rng.choice(words) for _ in range(n))
        cur = templated_len(body, enable_thinking)
    if cur != target_tokens:
        raise RuntimeError(f"could not hit {target_tokens} (got {cur})")
    return body


# ----------------------------------------------------------------- metrics --
def scrape(path=None):
    with urllib.request.urlopen(BASE + "/metrics", timeout=20) as r:
        text = r.read().decode()
    if path:
        with open(path, "w") as f:
            f.write(text)
    return text


def parse_metrics(text):
    """Sum each metric family across its label sets."""
    out = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        head, _, val = line.rpartition(" ")
        name = head.split("{", 1)[0].strip()
        try:
            out[name] = out.get(name, 0.0) + float(val)
        except ValueError:
            pass
    return out


def parse_metrics_labeled(text, family):
    """Per-label-set values for one metric family, keyed by the label string."""
    out = {}
    for line in text.splitlines():
        if not line.startswith(family):
            continue
        head, _, val = line.rpartition(" ")
        if not head.startswith(family):
            continue
        labels = head[len(family):]
        try:
            out[labels] = float(val)
        except ValueError:
            pass
    return out


def metrics_delta(before_text, after_text):
    b, a = parse_metrics(before_text), parse_metrics(after_text)
    return {k: a.get(k, 0.0) - b.get(k, 0.0) for k in set(a) | set(b)}


# ------------------------------------------------------------ cache control --
def flush_cache(timeout_s=None):
    """Flush L1+L2. Leaves the L3 files on disk.

    Default wait is 120 s (HICACHE_FLUSH_TIMEOUT overrides): the flush is gated on the
    L3 backup queue draining, which takes minutes on a disk slower than the old box's.

    The endpoint 400s while requests are running or waiting, so pass a timeout
    and let it wait for quiescence rather than racing it.
    """
    if timeout_s is None:
        timeout_s = float(os.environ.get("HICACHE_FLUSH_TIMEOUT", 120.0))
    req = urllib.request.Request(
        f"{BASE}/flush_cache?timeout={timeout_s}", method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s + 60) as r:
            return r.read().decode()
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"flush_cache failed ({e.code}): {body}") from None


def drop_page_cache():
    """Preferred: global drop_caches. Falls back to fadvise on the L3 files."""
    subprocess.run(["sync"], check=False)
    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("3\n")
        return "drop_caches"
    except Exception:
        pass
    for p in glob.glob(os.path.join(L3_DIR, "**", "*.bin"), recursive=True):
        try:
            fd = os.open(p, os.O_RDONLY)
            os.fsync(fd)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except OSError:
            pass
    return "fadvise"


def l3_stats():
    """Count every regular file in the store, not just *.bin.

    The file backend names pages "<key>.bin" in one flat directory; nixl shards
    into xx/ subdirectories and writes "<key>_<model>_<tp>_k" / "_v" with no
    extension. Globbing *.bin silently reported 0 files for nixl.
    """
    files = [f for f in glob.glob(os.path.join(L3_DIR, "**", "*"), recursive=True)
             if os.path.isfile(f)]
    return {"files": len(files), "bytes": sum(os.path.getsize(f) for f in files)}


def cached_meminfo_kb():
    for line in open("/proc/meminfo"):
        if line.startswith("Cached:"):
            return int(line.split()[1])
    return -1


def wait_backup_drain(poll_s=2.0, stable_s=10.0, max_wait_s=600.0):
    """Block until hicache_backup_tokens_total stops moving.

    A burst keeps draining after the client is done; flushing before it settles
    would attribute a partial L3 write to the next phase.
    """
    key = "sglang:hicache_backup_tokens_total"
    t0 = time.time()
    last, stable_since = None, None
    while time.time() - t0 < max_wait_s:
        cur = parse_metrics(scrape()).get(key, 0.0)
        if last is not None and cur == last:
            if stable_since is None:
                stable_since = time.time()
            elif time.time() - stable_since >= stable_s:
                return cur
        else:
            stable_since = None
        last = cur
        time.sleep(poll_s)
    return last


def wait_until_flushable(max_wait_s=3600.0, poll_s=5.0, verbose=False):
    """Poll flush_cache until it succeeds, and return how long that took.

    flush_cache is gated on Scheduler.is_fully_idle(), which for HiCache
    additionally requires ongoing_write_through / ongoing_load_back /
    ongoing_prefetch / ongoing_backup to all be empty
    (scheduler.py:4778-4785). hicache_backup_tokens_total counts *enqueued*
    backups and goes flat long before the files land, so polling the counter
    is not a drain test -- polling the flush itself is.
    """
    import urllib.error

    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        try:
            req = urllib.request.Request(f"{BASE}/flush_cache?timeout=0", method="POST")
            with urllib.request.urlopen(req, timeout=60) as r:
                r.read()
            return time.time() - t0
        except urllib.error.HTTPError:
            if verbose:
                print(f"  waiting for backup drain... {time.time()-t0:.0f}s "
                      f"files={l3_stats()['files']}", flush=True)
            time.sleep(poll_s)
    raise RuntimeError(f"cache never became flushable within {max_wait_s}s")
