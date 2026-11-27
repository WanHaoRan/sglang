"""Who shares whose pages: for the sessions one run played, how many prompt blocks a session takes from
another session's first prompt (a shared head) and from another session's later turns (a fork), and how many
of those were taken after the owning session had ended (R4: pages that zeroing an ended session's files at its
end would have hit while still wanted).

Usage: python3 -I -B fork_share.py <trace.programs.jsonl.gz> <client.jsonl>
Block ids are chained (equal ids at a position mean equal prefixes), and sharing is within one scope, so a
(scope, block id) pair names one page prefix. A session is a stream instance of a play. "Owner" is the session
that sent the block first in this run.
"""
import gzip
import json
import re
import sys
from collections import defaultdict


def main(trace_path, client_path):
    sent = defaultdict(list)       # tree id -> [(t_send, play, stream, k, inst, t_done, is_end)]
    for line in open(client_path):
        if '"kind":"req"' not in line:
            continue
        r = json.loads(line)
        if r.get("ttft") is None:
            continue
        is_end = (r.get("hint") or {}).get("reply_key") == "_end"
        sent[r["tree"]].append((r["t_send"], r["play"], r["stream"], r["k"], r["inst"],
                                r["t_send"] + (r.get("latency") or 0.0), is_end))
    want = set(sent)
    trees = {}
    head = re.compile(r'"tree":"([0-9a-f]+)"')
    with gzip.open(trace_path, "rt", encoding="utf-8") as f:
        header = json.loads(f.readline())
        for line in f:
            m = head.search(line[:200])
            if m is None or m.group(1) not in want:
                continue
            trees[m.group(1)] = json.loads(line)
            if len(trees) == len(want):
                break
    B = header["block_tokens"]
    print(f"trace {trace_path}: block {B} tokens; trees played {len(want)}, found {len(trees)}")

    # every sent request in send order
    events = []
    for tree_id, lst in sent.items():
        for t_send, play, s, k, inst, t_done, is_end in lst:
            events.append((t_send, tree_id, play, s, k, inst, t_done, is_end))
    events.sort()
    memo = {}

    def blocks(tree_id, s, k):
        got = memo.get((tree_id, s, k))
        if got is None:
            r = trees[tree_id]["streams"][s]["requests"][k]
            base = r.get("base")
            got = list(r["new"]) if base is None else blocks(tree_id, *base)[: r["keep"]] + list(r["new"])
            memo[(tree_id, s, k)] = got
        return got

    owner = {}                    # (scope, block) -> (inst, after the owner's first prompt)
    first_len = {}                # inst -> blocks of its first prompt
    seen_by = defaultdict(set)    # inst -> block keys already counted for it
    end_at = {}                   # inst -> time its explicit last turn finished
    own_first = own_later = 0
    head_shared = fork_shared = 0
    head_after_end = fork_after_end = 0
    fork_same_play = fork_other_play = 0
    per_inst_fork = defaultdict(int)
    inst_play = {}
    n_first = n_first_shared_blocks = n_first_blocks = 0
    for t_send, tree_id, play, s, k, inst, t_done, is_end in events:
        tree = trees.get(tree_id)
        if tree is None:
            continue
        scope = str(tree.get("scope") or tree["tree"])
        inst_play[inst] = play
        bl = blocks(tree_id, s, k)
        if inst not in first_len:
            first_len[inst] = len(bl)
            n_first += 1
            n_first_blocks += len(bl)
        mine = seen_by[inst]
        for i, b in enumerate(bl):
            key = (scope, b)
            if key in mine:
                continue
            mine.add(key)
            o = owner.get(key)
            if o is None:
                after = i >= first_len[inst]
                owner[key] = (inst, after)
                own_later += after
                own_first += not after
                continue
            o_inst, after = o
            if o_inst == inst:
                continue
            ended = o_inst in end_at and end_at[o_inst] <= t_send
            if after:
                fork_shared += 1
                fork_after_end += ended
                per_inst_fork[inst] += 1
                if inst_play.get(o_inst) == play:
                    fork_same_play += 1
                else:
                    fork_other_play += 1
            else:
                head_shared += 1
                head_after_end += ended
            if k == 0 or first_len[inst] == len(bl):
                n_first_shared_blocks += 1
        if is_end:
            end_at[inst] = t_done
    tot = own_first + own_later
    print(f"sessions {len(first_len)}; blocks first sent by their owner: {tot} ({tot * B / 1e6:.2f} M tokens): "
          f"in the owner's first prompt {own_first} ({own_first * B / 1e6:.2f} M), in its later turns {own_later} "
          f"({own_later * B / 1e6:.2f} M)")
    print(f"blocks a session took from another session's first prompt (shared head): {head_shared} "
          f"({head_shared * B / 1e6:.2f} M tokens); of them after the owner had ended: {head_after_end} "
          f"({head_after_end * B / 1e6:.2f} M)")
    print(f"blocks a session took from another session's later turns (fork): {fork_shared} "
          f"({fork_shared * B / 1e6:.3f} M tokens) by {len(per_inst_fork)} sessions; same play {fork_same_play}, "
          f"other play {fork_other_play}; of them after the owner had ended: {fork_after_end} "
          f"({fork_after_end * B / 1e6:.3f} M)")
    print(f"first prompts: {n_first_blocks * B / 1e6:.2f} M tokens, of which shared with an earlier session "
          f"{n_first_shared_blocks * B / 1e6:.2f} M")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
