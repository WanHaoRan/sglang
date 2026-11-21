"""Tests of replay_baseline.py against mock_engine.py; no GPU. From agent_cache/baselines/common/:
  python3 -m unittest -q test_replay_baseline
The first test is the one the comparison rests on: a baseline is sent the prompts our own arms are sent.
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "scripts")))

import program_trace as pt  # noqa: E402
import replay_baseline as rb  # noqa: E402
import replay_programs as rp  # noqa: E402
from backends import BACKENDS, Turn  # noqa: E402


def _tree() -> dict:
    """Root: 4 requests with tool keys and waits; a subagent spawned after root[1], joined by root[2]."""
    def r(t, deps, wait, base, keep, new, gap=None, key=None):
        return pt.request(t=t, dur=0.05, deps=deps, wait=wait, out=4, gap=gap, key=key, base=base, keep=keep, new=new,
                          tail=2)
    root = [r(0.0, [], 0.0, None, 0, [1, 2, 3], key="Bash/ls"),
            r(0.15, [[0, 0]], 0.1, [0, 0], 3, [4], gap="tool", key="Task"),
            r(0.45, [[0, 1], [1, 1]], 0.05, [0, 1], 4, [5], gap="tool", key="Bash/git"),
            r(0.60, [[0, 2]], 0.07, [0, 2], 5, [6], gap="user")]
    sub = [r(0.25, [[0, 1]], 0.05, None, 0, [50, 51], gap="spawn", key="Read"),
           r(0.35, [[1, 0]], 0.05, [1, 0], 2, [52], gap="tool")]
    return {"kind": "tree", "tree": "S", "scope": "S", "setup": 0, "system": ["h", "m"],
            "streams": [{"name": "main", "role": "main", "requests": root},
                        {"name": "sa", "role": "subagent", "requests": sub}]}


def _write_trace(tree: dict, *, content: str, block_tokens: int | None) -> str:
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    header = pt.make_header(source="test", content=content, block_tokens=block_tokens, converter={})
    pt.write_trace(path, header=header, trees=[tree])
    return path


class Recorder:
    """Wraps a replayer class's _body so a test sees every body sent, keyed by request id."""

    def __init__(self, cls) -> None:
        self.cls, self.bodies, self._orig = cls, {}, cls._body

    def __enter__(self):
        rec, orig = self.bodies, self._orig

        def body(replayer, *, ids_json, out, rid, hints, session):
            text = orig(replayer, ids_json=ids_json, out=out, rid=rid, hints=hints, session=session)
            rec[rid] = json.loads(text)
            return text
        self.cls._body = body
        return self

    def __exit__(self, *exc):
        self.cls._body = self._orig


class BaselineReplayTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        import mock_engine
        from aiohttp import web
        self.port = random.randint(32000, 39000)
        args = mock_engine.parse_args(["--port", str(self.port), "--time-scale", "0.01", "--page", "4"])
        self.runner = web.AppRunner(mock_engine.build_app(args))
        await self.runner.setup()
        await web.TCPSite(self.runner, "127.0.0.1", self.port).start()
        self.tmp = tempfile.mkdtemp()

    async def asyncTearDown(self):
        await self.runner.cleanup()

    def _common(self, tree: dict, *, content: str, block_tokens: int | None) -> tuple:
        trace = _write_trace(tree, content=content, block_tokens=block_tokens)
        out = os.path.join(self.tmp, f"c{random.randint(0, 1 << 30)}.jsonl")
        return out, ["--trace", trace, "--output", out, "--url", f"http://127.0.0.1:{self.port}", "--sample",
                     "sequential", "--tag", "t"]

    async def run_baseline(self, tree: dict, extra: list, *, backend: str, content="blocks", block_tokens=4) -> tuple:
        out, flags = self._common(tree, content=content, block_tokens=block_tokens)
        args, be = rb.parse_args(["--backend", backend] + flags + extra)
        with Recorder(rb.BaselineReplayer) as rec:
            await rb.replay(args, be)
        with open(out) as f:
            return [json.loads(line) for line in f], rec.bodies

    async def run_pinned(self, tree: dict, extra: list, *, content="blocks", block_tokens=4) -> tuple:
        out, flags = self._common(tree, content=content, block_tokens=block_tokens)
        args = rp.parse_args(flags + extra + ["--api", "completions"])
        with Recorder(rp.Replayer) as rec:
            await rp.replay(args)
        with open(out) as f:
            return [json.loads(line) for line in f], rec.bodies

    async def test_a_baseline_is_sent_the_prompts_our_arms_are_sent(self):
        """Same trace, seed and flags: every request of the wrapper carries the prompt and output length the pinned
        client sends under the same request id, and the records agree on what the workload fixes."""
        extra = ["--lanes", "1", "--plays", "2", "--duration", "30", "--cache-bust", "recycle"]
        ours, ours_bodies = await self.run_pinned(_tree(), extra)
        for backend in sorted(BACKENDS):
            rows, bodies = await self.run_baseline(_tree(), extra, backend=backend)
            self.assertEqual(sorted(bodies), sorted(ours_bodies))
            for rid, body in bodies.items():
                self.assertEqual(body["prompt"], ours_bodies[rid]["prompt"], (backend, rid))
                self.assertEqual(body["max_tokens"], ours_bodies[rid]["max_tokens"])
            fixed = ("rid", "stream", "k", "play", "tree", "wait_s", "key", "out_target", "prompt_sent", "ideal_cached")
            pick = lambda rs: sorted([tuple(r[f] for f in fixed) for r in rs if r["kind"] == "req"])  # noqa: E731
            self.assertEqual(pick(rows), pick(ours))
            self.assertFalse([r for r in rows if r["kind"] == "req" and r.get("error")])

    async def test_arrivals_are_the_pinned_clients(self):
        """Poisson arrivals under a cap: the wrapper admits the same sessions at the same replay times."""
        tree = _tree()
        extra = ["--arrival-rate", "50", "--max-active", "2", "--duration", "0", "--trees", "1", "--seed", "3"]
        ours, _ = await self.run_pinned(tree, extra)
        rows, _ = await self.run_baseline(tree, extra, backend="vllm")
        plays = lambda rs: [(r["tree"], r["arrival_s"]) for r in rs if r["kind"] == "play"]  # noqa: E731
        self.assertEqual(plays(rows), plays(ours))

    async def test_turn_carries_what_the_trace_knows(self):
        seen = {}

        class Spy(BACKENDS["openai"]):
            name = "spy"

            def body(self, *, model, out, turn, want_ids):
                seen[turn.rid] = turn
                return super().body(model=model, out=out, turn=turn, want_ids=want_ids)

        BACKENDS["spy"] = Spy
        try:
            await self.run_baseline(_tree(), ["--lanes", "1", "--plays", "1", "--duration", "30"], backend="spy")
        finally:
            del BACKENDS["spy"]
        root = [seen[f"t-c0-t{k}"] for k in range(4)]
        self.assertEqual([t.last for t in root], [False, False, False, True])
        self.assertEqual([t.reply_key for t in root], ["Bash/ls", "Task", "Bash/git", None])
        self.assertEqual([t.next_gap for t in root], ["tool", "tool", "user", None])
        self.assertEqual([t.next_wait_s for t in root], [0.1, 0.05, 0.07, None])
        self.assertEqual({t.session for t in root}, {"t-c0"})
        self.assertEqual(seen["t-c1-t1"].last, True)
        self.assertEqual(seen["t-c1-t0"].role, "subagent")
        self.assertTrue(all(t.est_wait_s is None for t in seen.values()))   # an estimate needs five waits of a key

    async def test_estimate_is_the_running_median_of_the_keys_waits(self):
        """After five waits behind one key the estimate is their median, as our dist hints' table gives it."""
        def r(k, wait, key):
            return pt.request(t=0.1 * k, dur=0.01, deps=[[0, k - 1]] if k else [], wait=wait, out=2, gap="tool",
                              key=key, base=[0, k - 1] if k else None, keep=k + 1 if k else 0, new=[k + 1], tail=0)
        waits = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.03, 0.03]
        reqs = [r(k, w, "Bash/ls") for k, w in enumerate(waits)]
        tree = {"kind": "tree", "tree": "E", "scope": "E", "setup": 0, "system": ["h", "m"],
                "streams": [{"name": "main", "role": "main", "requests": reqs}]}
        seen = {}

        class Spy(BACKENDS["openai"]):
            name = "spy"

            def body(self, *, model, out, turn, want_ids):
                seen[turn.k] = turn.est_wait_s
                return super().body(model=model, out=out, turn=turn, want_ids=want_ids)

        BACKENDS["spy"] = Spy
        try:
            await self.run_baseline(tree, ["--lanes", "1", "--plays", "1", "--duration", "30"], backend="spy")
        finally:
            del BACKENDS["spy"]
        self.assertEqual([seen[k] for k in range(5)], [None] * 5)      # four waits seen before request 4
        self.assertAlmostEqual(seen[5], 0.06, places=6)                # median of 0.02 0.04 0.06 0.08 0.10
        # 0.03 joined: the table interpolates quantiles of log waits, so six give the geometric mean of the middle two
        self.assertAlmostEqual(seen[6], (0.04 * 0.06) ** 0.5, places=6)

    async def test_vllm_body_and_reply_ids(self):
        rows, bodies = await self.run_baseline(_tree(), ["--lanes", "1", "--plays", "1", "--duration", "30"],
                                               backend="vllm")
        body = bodies["t-c0-t1"]
        self.assertEqual((body["request_id"], body["return_token_ids"], body["ignore_eos"], body["temperature"]),
                         ("t-c0-t1", True, True, 0.0))
        self.assertNotIn("rid", body)
        req = [r for r in rows if r["kind"] == "req"]
        self.assertTrue(all(r["completion_tokens"] == r["out_target"] for r in req))
        self.assertTrue(all(r["ttft"] is not None and r["cached_tokens"] is not None for r in req))

    async def test_lengths_carry_the_servers_reply(self):
        reqs = [pt.request(t=0, dur=0.05, deps=[], wait=0, out=4, prefix=0, append=8),
                pt.request(t=0.1, dur=0.05, deps=[[0, 0]], wait=0.05, out=4, prefix=12, append=4)]
        tree = {"kind": "tree", "tree": "L", "streams": [{"name": "main", "role": "main", "requests": reqs}]}
        rows, _ = await self.run_baseline(tree, ["--arrival-rate", "1", "--duration", "0"], backend="vllm",
                                          content="lengths", block_tokens=None)
        req = [r for r in rows if r["kind"] == "req"]
        self.assertEqual([r["prompt_sent"] for r in req], [8, 16])
        self.assertEqual(req[1]["cached_tokens"], 12)   # prompt 8 + the server's 4 output tokens, page-aligned


class TurnTest(unittest.TestCase):
    def test_backend_names_are_unique_and_registered(self):
        self.assertEqual(sorted(BACKENDS), sorted({cls.name for cls in BACKENDS.values()}))
        self.assertIn("vllm", BACKENDS)
        self.assertEqual(Turn._fields[:4], ("rid", "session", "k", "last"))


if __name__ == "__main__":
    unittest.main()
