"""Shared loaders for the SSD-tier replay of campaign 34 (label ev_ssd). Stdlib only, read-only on the records.

Page identity. The client builds each prompt from the trace's block ids (16 tokens a block, tokens a function of
(scope, block id)), plus a tail of < 16 tokens unique to the request. The server's page key is a chained hash of
64-token pages, so page j of a prompt (blocks 4j..4j+3) is identified here by a running hash over
(scope, block 0 .. block 4j+3). Every full page of a prompt is made of whole blocks (16 r + tail < 64), so prompt
pages are shared exactly where block prefixes are. Pages past the prompt's last full page hold the tail and the
generated output: unique to the request.
"""
import gzip, json, re, struct, hashlib
from array import array
from datetime import datetime, timezone

RES = "/home/wanhr/sglang/agent_cache/results"
HERE = RES + "/design_unified_1126/agents/ev_ssd"
ARMS = {  # campaign34/manifest.txt
    "jit_p50": ("20261123_073130_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H160", "programs_073338_jit_p50"),
    "ec_full": ("20261123_093604_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H160", "programs_093918_ec_full"),
    "jit_exact": ("20261123_114134_three_tier_wc_oracle_v2_nu_owned_custom_L1nat_H160", "programs_114453_jit_exact"),
}
PAGE = 64
BLOCK = 16
BPP = PAGE // BLOCK  # blocks per page


def client_path(arm):
    run, prog = ARMS[arm]
    return f"{RES}/{run}/{prog}/client.jsonl"


def evt_path(arm):
    return f"{HERE}/extracts/{arm}.evt"


def load_trees(path=HERE + "/extracts/trees_c34.jsonl.gz"):
    trees = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        header = json.loads(f.readline())
        for line in f:
            t = json.loads(line)
            trees[t["tree"]] = t
    return header, trees


class Interner:
    """8-byte page hash -> small int id."""

    def __init__(self):
        self.ids = {}

    def get(self, h):
        i = self.ids.get(h)
        if i is None:
            i = self.ids[h] = len(self.ids)
        return i

    def __len__(self):
        return len(self.ids)


class TreeChains:
    """Block lists and page-id chains of one tree's requests (BlockPrompts' rule: base[:keep] + new)."""

    def __init__(self, tree, interner):
        self.tree = tree
        self.scope = str(tree.get("scope") or tree["tree"])
        self.seed = hashlib.blake2b(("scope|" + self.scope).encode(), digest_size=8).digest()
        self.interner = interner
        self._blocks = {}
        self._pages = {}   # (s, k) -> (array('I') of page ids, list of 8-byte hashes)

    def blocks(self, s, k):
        got = self._blocks.get((s, k))
        if got is None:
            # iterative resolution of base references
            stack = [(s, k)]
            while stack:
                cs, ck = stack[-1]
                if (cs, ck) in self._blocks:
                    stack.pop()
                    continue
                r = self.tree["streams"][cs]["requests"][ck]
                base = r.get("base")
                if base is None:
                    self._blocks[(cs, ck)] = list(r["new"])
                    stack.pop()
                elif tuple(base) in self._blocks:
                    self._blocks[(cs, ck)] = self._blocks[tuple(base)][: r["keep"]] + list(r["new"])
                    stack.pop()
                else:
                    stack.append(tuple(base))
            got = self._blocks[(s, k)]
        return got

    def pages(self, s, k):
        """(page ids, page hashes) of the full pages of the request's block list."""
        got = self._pages.get((s, k))
        if got is not None:
            return got
        r = self.tree["streams"][s]["requests"][k]
        blocks = self.blocks(s, k)
        n_pages = len(blocks) // BPP
        base = r.get("base")
        ids = array("I")
        hashes = []
        start = 0
        if base is not None and tuple(base) in self._pages:
            b_ids, b_hashes = self._pages[tuple(base)]
            start = min(r["keep"] // BPP, len(b_ids), n_pages)
            ids.extend(b_ids[:start])
            hashes.extend(b_hashes[:start])
        elif base is not None:
            b_ids, b_hashes = self.pages(*base)
            start = min(r["keep"] // BPP, len(b_ids), n_pages)
            ids.extend(b_ids[:start])
            hashes.extend(b_hashes[:start])
        prev = hashes[-1] if hashes else self.seed
        get = self.interner.get
        pack = struct.pack
        for j in range(start, n_pages):
            b = blocks[j * BPP:(j + 1) * BPP]
            prev = hashlib.blake2b(prev + pack("<4q", *b), digest_size=8).digest()
            hashes.append(prev)
            ids.append(get(prev))
        got = self._pages[(s, k)] = (ids, hashes)
        return got


_TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?:\.(\d+))?\]")
_KV = re.compile(r"(\w+)=([^\s,)]+)")


def epoch(line):
    m = _TS.match(line)
    if not m:
        return None
    d = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    if m.group(2):
        d += float("0." + m.group(2))
    return d


def load_client(arm):
    """(t_start, list of request dicts in file (completion) order, plays, run args)."""
    reqs, plays, t_start, args = [], [], None, None
    with open(client_path(arm)) as f:
        for line in f:
            if '"kind":"req"' in line:
                r = json.loads(line)
                reqs.append(r)
            elif '"kind":"run"' in line and t_start is None:
                d = json.loads(line)
                t_start, args = d["t_start"], d["args"]
            elif '"kind":"play"' in line:
                plays.append(json.loads(line))
    return t_start, reqs, plays, args


_RTS = re.compile(r"ReqTimeStats\(rid=([^,]+), input_len=(\d+), cached_input_len=(\d+), output_len=(\d+).*?"
                  r"queue_duration=([\d.]+)ms, forward_duration=([\d.]+)ms, entry_time=([\d.]+)")
_CLN = re.compile(r"NIXL L3 cleanup: deleted (\d+) groups / (\d+) files \(([\d.]+) GiB\) in ([\d.]+)s, "
                  r"initial_hot=\{[^:]+: '([\d.]+)%'\} final=\{[^:]+: '([\d.]+)%'\}")
_FAIL = re.compile(r"Prefetch operation (\S+) failed to retrieve page ([0-9a-f]+)")
_OK = re.compile(r"HiCache prefetch success req=(\S+) completed=(\d+) matched=(\d+) loaded=(\d+) occupied=(\d+)")
_SHR = re.compile(r"prefix shrank before admission req=(\S+) lookup_match=(\d+) current_match=(\d+)")


def load_events(arm):
    """Parse the extract into {event name: list of dicts with 't' (epoch s)}; numbers are ints where they parse."""
    ev = {}
    with open(evt_path(arm)) as f:
        for line in f:
            t = epoch(line)
            if t is None:
                continue
            i = line.find("HICACHE_EVT ")
            if i >= 0:
                rest = line[i + 12:]
                name, _, tail = rest.partition(" ")
                d = {"t": t}
                for k, v in _KV.findall(tail):
                    try:
                        d[k] = int(v)
                    except ValueError:
                        try:
                            d[k] = float(v)
                        except ValueError:
                            d[k] = v
                ev.setdefault(name, []).append(d)
                continue
            m = _RTS.search(line)
            if m:
                ev.setdefault("req_stats", []).append({"t": t, "rid": m[1], "input_len": int(m[2]), "cached": int(m[3]),
                                                        "output_len": int(m[4]), "queue_ms": float(m[5]),
                                                        "forward_ms": float(m[6]), "entry": float(m[7])})
                continue
            m = _CLN.search(line)
            if m:
                ev.setdefault("cleanup", []).append({"t": t, "groups": int(m[1]), "files": int(m[2]), "gib": float(m[3]),
                                                      "secs": float(m[4]), "pct0": float(m[5]), "pct1": float(m[6])})
                continue
            m = _FAIL.search(line)
            if m:
                ev.setdefault("read_fail", []).append({"t": t, "rid": m[1], "hash": m[2]})
                continue
            m = _OK.search(line)
            if m:
                ev.setdefault("prefetch_ok", []).append({"t": t, "rid": m[1], "completed": int(m[2]), "matched": int(m[3]),
                                                          "loaded": int(m[4]), "occupied": int(m[5])})
                continue
            m = _SHR.search(line)
            if m:
                ev.setdefault("shrank", []).append({"t": t, "rid": m[1], "lookup_match": int(m[2]),
                                                     "current_match": int(m[3])})
    return ev
