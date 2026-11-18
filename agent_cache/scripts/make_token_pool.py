"""Build a token pool for replay_programs.py --token-pool: real text tokenized for the served model, as uint32.

  python3 make_token_pool.py --tokenizer Qwen/Qwen3-30B-A3B-Instruct-2507 --text enwik9 [more files] \
      --tokens 20000000 --output qwen3.pool.u32

Synthetic prompts only need the token shape of text, not its meaning (TraceLab's session_runner tokenizes enwik9 for
the same reason); a pool of real text keeps MoE expert routing closer to real prompts than uniform ids. The replay maps
each block id to a window of the pool, so 20M tokens give ample variety. Needs transformers; the tokenizer is read from
the local Hugging Face cache unless --allow-download.
"""
from __future__ import annotations

import argparse
import sys
from array import array


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tokenizer", required=True, help="model id or local path of the served model's tokenizer")
    ap.add_argument("--text", nargs="+", required=True, help="UTF-8 text files, read in order")
    ap.add_argument("--tokens", type=int, default=20_000_000, help="pool size in tokens")
    ap.add_argument("--output", required=True)
    ap.add_argument("--allow-download", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=not args.allow_download)
    special = set(tok.all_special_ids) | set(tok.get_added_vocab().values())   # no control tokens such as <think>
    pool = array("I")
    batch: list = []

    def flush() -> None:
        for ids in tok(batch, add_special_tokens=False)["input_ids"]:
            pool.extend(i for i in ids if i not in special)
        batch.clear()

    for path in args.text:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    batch.append(line)
                if len(batch) >= 512:
                    flush()
                    if len(pool) >= args.tokens:
                        break
        if len(pool) >= args.tokens:
            break
    if batch:
        flush()
    del pool[args.tokens:]
    if len(pool) < 1 << 16:
        sys.exit(f"STOP: only {len(pool)} tokens; give more text")
    with open(args.output, "wb") as f:
        pool.tofile(f)
    print(f"wrote {args.output}: {len(pool)} tokens, ids {min(pool)}..{max(pool)}, {len(set(pool))} distinct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
