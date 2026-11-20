"""Tests of program_trace.py, the three converters, replay_programs.py and mock_engine.py; no GPU, no dataset.

  cd agent_cache/scripts && python3 -m unittest test_program_replay -v

The replay tests run the client against mock_engine.py in the same event loop (aiohttp), on small synthetic traces.
"""
from __future__ import annotations

import json
import os
import random
import tempfile
import unittest

import convert_agentx
import convert_freeinference
import convert_tracelab
import program_trace as pt


def _chain(n: int, *, start: int = 0) -> list:
    return list(range(start, start + n))


class LcpAndDeltaTest(unittest.TestCase):
    def test_lcp_matches_brute_force_on_chained_lists(self):
        rng = random.Random(1)
        for _ in range(200):
            common = rng.randint(0, 30)
            a = _chain(common) + _chain(rng.randint(0, 20), start=1000)
            b = _chain(common) + _chain(rng.randint(0, 20), start=2000)
            brute = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            self.assertEqual(pt.lcp(a, b), brute)

    def test_delta_and_materialize_round_trip(self):
        blocks = {(0, 0): [1, 2, 3], (0, 1): [1, 2, 3, 4, 5], (1, 0): [1, 2, 9]}
        streams = [{"requests": []}, {"requests": []}]
        for (s, k) in sorted(blocks, key=lambda x: (x[1], x[0])):
            cands = [([ds, dk], b) for (ds, dk), b in blocks.items() if (ds, dk) != (s, k) and (ds, dk) < (s, k)]
            base, keep, new = pt.delta(blocks[(s, k)], cands)
            r = pt.request(t=float(k), dur=0.5, deps=[], wait=0.0, out=1, base=base, keep=keep, new=new, tail=0)
            streams[s]["requests"].append(r)
        tree = {"tree": "x", "streams": streams}
        self.assertEqual(pt.materialize_blocks(tree), blocks)
        self.assertEqual(streams[0]["requests"][1]["new"], [4, 5])   # only the new blocks are stored

    def test_check_tree_finds_cycles_and_bad_refs(self):
        reqs = [pt.request(t=0, dur=1, deps=[[0, 1]], wait=0, out=1, base=None, keep=0, new=[1]),
                pt.request(t=1, dur=1, deps=[[0, 0]], wait=0, out=1, base=None, keep=0, new=[2])]
        with self.assertRaises(ValueError):
            pt.check_tree({"tree": "c", "streams": [{"requests": reqs}]}, content="blocks")
        reqs[0]["deps"] = [[3, 0]]
        with self.assertRaises(ValueError):
            pt.check_tree({"tree": "c", "streams": [{"requests": reqs}]}, content="blocks")


def _blocks_tree(gaps: list, *, sizes: list) -> dict:
    """A one-stream blocks tree: request k has sizes[k] blocks extending its predecessor, after waits gaps[k]."""
    reqs, t, prev = [], 0.0, []
    for k, (g, n) in enumerate(zip(gaps, sizes)):
        t = t + g if k else 0.0
        blocks = (prev + _chain(n, start=100 * (k + 1)))[:n] if n >= len(prev) else prev[:n]
        base, keep, new = pt.delta(blocks, [([0, k - 1], prev)] if k else [])
        reqs.append(pt.request(t=t, dur=1.0, deps=[[0, k - 1]] if k else [], wait=g if k else 0.0, out=4,
                               base=base, keep=keep, new=new, tail=0))
        t += 1.0
        prev = blocks
    return {"tree": "T", "scope": "T", "setup": 0, "streams": [{"name": "main", "role": "main", "requests": reqs}]}


class SplitProgramsTest(unittest.TestCase):
    def test_idle_split(self):
        tree = _blocks_tree([0, 5, 4000, 3], sizes=[4, 6, 8, 9])
        progs = pt.split_programs(tree, idle_s=1800.0, compaction=False, content="blocks")
        self.assertEqual([len(p["streams"][0]["requests"]) for p in progs], [2, 2])
        second = progs[1]["streams"][0]["requests"]
        self.assertEqual((second[0]["deps"], second[0]["wait"], second[0]["t"]), ([], 0.0, 0.0))
        self.assertIsNone(second[0]["base"])   # its base was in the first program: blocks spelled out
        self.assertEqual(pt.materialize_blocks(progs[1])[(0, 0)], pt.materialize_blocks(tree)[(0, 2)])

    def test_compaction_split(self):
        tree = _blocks_tree([0, 2, 2], sizes=[10, 12, 3])
        progs = pt.split_programs(tree, idle_s=None, compaction=True, content="blocks")
        self.assertEqual([len(p["streams"][0]["requests"]) for p in progs], [2, 1])

    def test_no_cut_across_a_subagent(self):
        tree = _blocks_tree([0, 3000, 2], sizes=[4, 5, 6])
        sub = [pt.request(t=500.0, dur=1.0, deps=[[0, 0]], wait=499.0, out=1, base=None, keep=0, new=[900]),
               pt.request(t=3500.0, dur=1.0, deps=[[1, 0]], wait=2999.0, out=1, base=None, keep=0, new=[901])]
        tree["streams"].append({"name": "sa", "role": "subagent", "requests": sub})
        progs = pt.split_programs(tree, idle_s=1800.0, compaction=False, content="blocks")
        self.assertEqual(len(progs), 1)


class PromptsTest(unittest.TestCase):
    def test_shared_blocks_give_shared_tokens_within_a_scope(self):
        synth = pt.TokenSynth()
        a = _blocks_tree([0, 1], sizes=[3, 5])
        b = _blocks_tree([0], sizes=[3])
        pa = pt.BlockPrompts(a, block_tokens=4, synth=synth, bust=None, play_tag="0")
        pb = pt.BlockPrompts(b, block_tokens=4, synth=synth, bust=None, play_tag="1")
        ia, na, _, _ = pa.prompt(0, 1)
        ib, nb, _, _ = pb.prompt(0, 0)
        self.assertEqual((na, nb), (20, 12))
        self.assertTrue(ia.startswith(ib))   # same scope, same block ids: identical prefix on the wire
        b["scope"] = "other"
        ic, _, _, _ = pt.BlockPrompts(b, block_tokens=4, synth=synth, bust=None, play_tag="1").prompt(0, 0)
        self.assertNotEqual(ic, ib)

    def test_bust_leads_and_tail_is_unique_per_play(self):
        synth = pt.TokenSynth()
        tree = _blocks_tree([0], sizes=[2])
        tree["streams"][0]["requests"][0]["tail"] = 3
        plain = pt.BlockPrompts(tree, block_tokens=4, synth=synth, bust=None, play_tag="0").prompt(0, 0)
        busted = pt.BlockPrompts(tree, block_tokens=4, synth=synth, bust=[7, 8], play_tag="1").prompt(0, 0)
        self.assertEqual((plain[1], busted[1]), (11, 13))
        self.assertTrue(busted[0].startswith("7,8,"))
        self.assertNotEqual(plain[0].split(",")[-3:], busted[0].split(",")[-3:])

    def test_token_pool_windows(self):
        fd, path = tempfile.mkstemp()
        os.close(fd)
        from array import array
        with open(path, "wb") as f:
            array("I", range(70000)).tofile(f)
        ids = pt.TokenSynth(pool_path=path).ids("b|s|1", 16)
        self.assertEqual(ids, list(range(ids[0], ids[0] + 16)))   # a contiguous window of real text
        os.remove(path)

    def test_length_prompts_follow_tracelab(self):
        reqs = [pt.request(t=0, dur=1, deps=[], wait=0, out=3, prefix=0, append=5),
                pt.request(t=2, dur=1, deps=[[0, 0]], wait=1, out=3, prefix=7, append=2),
                pt.request(t=4, dur=1, deps=[[0, 1]], wait=1, out=3, prefix=12, append=1)]
        lp = pt.LengthPrompts({"tree": "L", "streams": [{"requests": reqs}]}, synth=pt.TokenSynth(), play_tag="0")
        p0 = [int(x) for x in lp.prompt(0, 0)[0].split(",")]
        lp.commit(0, [1, 2, 3])
        self.assertEqual(lp.held(0, reqs[1]), 7)
        p1 = [int(x) for x in lp.prompt(0, 1)[0].split(",")]
        self.assertEqual(p1[:7], p0 + [1, 2])   # the previous prompt and the server's reply, then new tokens
        lp.commit(0, [4, 5, 6])
        self.assertEqual(lp.held(0, reqs[2]), 12)


class ConvertAgentxTest(unittest.TestCase):
    def trace(self) -> dict:
        r = lambda t, model, ids, api: {"t": t, "model": model, "in": 64 * len(ids), "out": 9, "hash_ids": ids,  # noqa
                                        "api_time": api, "type": "s"}
        return {"id": "T1", "block_size": 64, "hash_id_scope": "local", "requests": [
            r(0.0, "opus", [1, 2, 3, 4], 1.0),
            r(0.5, "haiku", [100, 101], 0.3),                       # a side one-shot
            r(2.0, "opus", [1, 2, 3, 4, 5, 6], 1.0),                # extends the main chain
            {"t": 3.5, "type": "subagent", "agent_id": "sa1", "requests": [
                {**r(3.5, "haiku", [200, 201], 1.0), "type": "n"},
                {**r(5.0, "haiku", [200, 201, 202], 1.0), "type": "n"}]},
            r(7.0, "opus", [1, 2, 3, 4, 5, 6, 7, 8], 1.0),          # joins the subagent
            r(9.0, "opus", [1, 2, 9], 1.0),                         # compaction: a seam on the main chain
            r(10.5, "opus", [1, 2, 9, 10], 1.0),
            r(10.6, "opus", [1, 2, 9, 11], 1.0),                    # overlaps the previous one: a fork
        ]}

    def test_chains_roles_deps_and_waits(self):
        tree = convert_agentx.build_tree(self.trace())
        pt.check_tree(tree, content="blocks")
        roles = [(st["role"], len(st["requests"])) for st in tree["streams"]]
        self.assertEqual(roles[0], ("main", 5))
        self.assertIn(("side", 1), roles)
        self.assertIn(("fork", 1), roles)
        sa = next(s for s, st in enumerate(tree["streams"]) if st["role"] == "subagent")
        sub = tree["streams"][sa]["requests"]
        self.assertEqual((sub[0]["deps"], sub[0]["wait"], sub[0]["gap"]), ([[0, 1]], 0.5, "spawn"))
        join = tree["streams"][0]["requests"][2]
        self.assertEqual(join["deps"], [[0, 1], [sa, 1]])
        self.assertAlmostEqual(join["wait"], 1.0)
        self.assertEqual(join["gap"], "join")
        side = next(st for st in tree["streams"] if st["role"] == "side")["requests"][0]
        self.assertEqual((side["deps"], side["wait"]), ([], 0.5))   # nothing had finished: timed from the start
        blocks = pt.materialize_blocks(tree)
        self.assertEqual(blocks[(0, 3)], [1, 2, 9])
        self.assertEqual(tree["streams"][0]["requests"][3]["keep"], 2)

    def test_subagent_side_chain_follows_the_subagent(self):
        trace = self.trace()
        sub = next(e for e in trace["requests"] if e.get("type") == "subagent")
        sub["requests"].append({"t": 4.7, "model": "opus", "in": 128, "out": 9, "hash_ids": [300, 301],
                                "api_time": 0.2, "type": "n"})       # a utility call inside the subagent
        tree = convert_agentx.build_tree(trace)
        pt.check_tree(tree, content="blocks")
        subs = [s for s, st in enumerate(tree["streams"]) if st["role"] == "subagent"]
        self.assertEqual(len(subs), 2)
        side = tree["streams"][subs[1]]["requests"][0]
        self.assertEqual((side["deps"], side["gap"]), ([[subs[0], 0]], None))
        self.assertAlmostEqual(side["wait"], 0.2)
        self.assertEqual(tree["streams"][subs[0]]["requests"][0]["gap"], "spawn")


class ConvertFreeinferenceTest(unittest.TestCase):
    def session(self) -> dict:
        def r(start, end, ids, role_tokens, out, trigger, calls=None):
            return {"start_ms": start, "end_ms": end, "block_ids": ids, "role_tokens": role_tokens,
                    "provider_tokens": {"completion": out}, "step_trigger": trigger, "tool_calls": calls or [],
                    "model": "m", "mutation": {"transition": "append"}}
        sub = {"session_id": "S1/sub",
               "requests": [r(3000, 4000, [50, 51], {"system": 35}, 3, "user"),
                            r(4500, 5000, [50, 51, 52], {"system": 35, "tool": 16}, 3, "tool")]}
        return {"session_id": "S1", "harness": "codex", "user_id": "u", "requests": [
            r(0, 1000, [1, 2], {"system": 30, "user": 10}, 7, "user",
              [{"call_index": 0, "name": "bash", "programs": ["git"], "latency_ms": 500},
               {"call_index": 1, "name": "read", "latency_ms": 100}]),
            r(1500, 2500, [1, 2, 3], {"system": 30, "user": 10, "tool": 13}, 5, "tool",
              [{"call_index": 0, "name": "task", "latency_ms": 3000, "subagent": sub}]),
            r(5500, 6000, [1, 2, 3, 4], {"system": 30, "user": 10, "tool": 26}, 5, "tool")]}

    def test_streams_deps_waits_keys_tails(self):
        tree = convert_freeinference.build_tree("W", self.session())
        pt.check_tree(tree, content="blocks")
        root, sub = (st["requests"] for st in tree["streams"])
        self.assertEqual((sub[0]["deps"], sub[0]["wait"], sub[0]["gap"]), ([[0, 1]], 0.5, "spawn"))
        self.assertEqual((root[2]["deps"], root[2]["wait"], root[2]["gap"]), ([[0, 1], [1, 1]], 0.5, "tool"))
        self.assertEqual(root[0]["key"], "bash/git")      # the call with the longest latency
        self.assertEqual(root[1]["key"], "task")
        self.assertEqual([r["tail"] for r in root], [8, 5, 2])
        self.assertEqual(tree["scope"], "W")


class ConvertTracelabTest(unittest.TestCase):
    def test_waits_prefix_cap_and_key(self):
        rounds = [{"round_pk": 1, "provider": "claude", "session_id": "s", "round_index": 0, "model": "m",
                   "input_tokens_total": 1000, "prefix_tokens": 0, "output_tokens": 50,
                   "first_input_event_type": "user_message"},
                  {"round_pk": 2, "provider": "claude", "session_id": "s", "round_index": 1, "model": "m",
                   "input_tokens_total": 1200, "prefix_tokens": 1100, "output_tokens": 20,
                   "first_input_event_type": "tool_result"},
                  {"round_pk": 3, "provider": "claude", "session_id": "s", "round_index": 2, "model": "m",
                   "input_tokens_total": 1300, "prefix_tokens": 1220, "output_tokens": 10,
                   "first_input_event_type": "user_message"}]
        tools = {1: [(0, "Bash", 2.0, "git"), (1, "Read", 0.5, None)], 2: [(0, "Edit", 0.1, None)]}
        events = {1: [(1, "user_message", 100.0), (2, "text", 101.0), (3, "tool_call", 102.0)],
                  2: [(1, "tool_result", 104.5), (2, "tool_call", 105.0)],
                  3: [(1, "user_message", 165.0), (2, "text", 166.0)]}
        stats = {"prefix_capped": 0, "wall_s": 0.0, "human_s": 0.0, "human_waits": [], "tool_latencies": []}
        tree = convert_tracelab.build_tree(rounds, tools=tools, events=events, stats=stats)
        r = tree["streams"][0]["requests"]
        self.assertEqual((r[1]["wait_tool"], r[1]["wait"], r[1]["gap"]), (2.5, 2.5, "tool"))
        self.assertEqual((r[1]["prefix"], r[1]["append"]), (1050, 150))   # capped at the previous input + output
        self.assertEqual((r[2]["wait_tool"], r[2]["wait_human"], r[2]["gap"]), (0.1, 60.0, "user"))
        self.assertAlmostEqual(r[2]["wait"], 60.1)
        self.assertEqual(r[0]["key"], "Bash/git")
        self.assertEqual(stats["prefix_capped"], 1)
        self.assertEqual([x["dur"] for x in r], [2.0, 0.5, 1.0])


# ===== The replay against the mock engine =====

def _write_trace(tree: dict, *, content: str, block_tokens: int | None) -> str:
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    header = pt.make_header(source="test", content=content, block_tokens=block_tokens, converter={})
    pt.write_trace(path, header=header, trees=[tree])
    return path


def _subagent_tree() -> dict:
    """Root: 3 requests; a subagent spawned after root[1] (0.05 s), joined by root[2] (0.05 s after its end)."""
    def r(t, deps, wait, base, keep, new, gap=None, key=None):
        return pt.request(t=t, dur=0.05, deps=deps, wait=wait, out=4, gap=gap, key=key, base=base, keep=keep, new=new,
                          tail=2)
    root = [r(0.0, [], 0.0, None, 0, [1, 2, 3], key="bash/ls"),
            r(0.15, [[0, 0]], 0.1, [0, 0], 3, [4], gap="tool", key="task"),
            r(0.45, [[0, 1], [1, 1]], 0.05, [0, 1], 4, [5], gap="tool")]
    sub = [r(0.25, [[0, 1]], 0.05, None, 0, [50, 51], gap="spawn", key="read"),
           r(0.35, [[1, 0]], 0.05, [1, 0], 2, [52], gap="tool")]
    return {"kind": "tree", "tree": "S", "scope": "S", "setup": 0, "system": ["h", "m"],
            "streams": [{"name": "main", "role": "main", "requests": root},
                        {"name": "sa", "role": "subagent", "requests": sub}]}


class ReplayTest(unittest.IsolatedAsyncioTestCase):
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

    async def run_client(self, tree: dict, extra: list, *, content="blocks", block_tokens=4) -> list:
        import replay_programs
        trace = _write_trace(tree, content=content, block_tokens=block_tokens)
        out = os.path.join(self.tmp, f"c{random.randint(0, 1 << 30)}.jsonl")
        args = replay_programs.parse_args(["--trace", trace, "--output", out, "--url", f"http://127.0.0.1:{self.port}",
                                           "--sample", "sequential"] + extra)
        await replay_programs.replay(args)
        with open(out) as f:
            return [json.loads(line) for line in f]

    async def test_closed_loop_spawn_join_and_release_hints(self):
        rows = await self.run_client(_subagent_tree(), ["--lanes", "1", "--plays", "1", "--duration", "30",
                                                        "--hints", "release"])
        req = {(r["stream"], r["k"]): r for r in rows if r["kind"] == "req"}
        self.assertEqual(len(req), 5)
        self.assertFalse([r for r in req.values() if r.get("error")])
        end = {key: r["t_send"] + r["latency"] for key, r in req.items()}
        tol = 0.01
        self.assertGreaterEqual(req[(1, 0)]["t_send"] + tol, end[(0, 1)] + 0.05)              # the spawn
        self.assertGreaterEqual(req[(0, 2)]["t_send"] + tol, max(end[(0, 1)], end[(1, 1)]) + 0.05)   # the join
        self.assertGreaterEqual(req[(0, 1)]["t_send"] + tol, end[(0, 0)] + 0.1)
        self.assertEqual([req[(0, k)]["prompt_sent"] for k in range(3)], [14, 18, 22])
        self.assertTrue(all(r["prompt_sent"] == r["prompt_tokens"] for r in req.values()))
        self.assertEqual(req[(0, 2)]["hint"]["reply_key"], "_end")
        self.assertEqual(req[(1, 1)]["hint"]["reply_key"], "_end")
        self.assertIsNone(req[(0, 0)]["hint"]["reply_key"])
        self.assertEqual(req[(0, 1)]["cached_tokens"], 12)   # root[0]'s three blocks, page-aligned

    async def test_dist_hints_carry_keys_and_learn_online(self):
        rows = await self.run_client(_subagent_tree(), ["--lanes", "1", "--plays", "2", "--duration", "30",
                                                        "--hints", "dist"])
        req = [r for r in rows if r["kind"] == "req"]
        self.assertEqual(req[0]["hint"]["reply_key"], "bash/ls")
        self.assertGreater(req[-1]["hint"]["list_len"], 1)   # gaps observed so far joined the table

    async def test_no_call_key_gives_tool_less_replies_their_own_entry(self):
        tree = _blocks_tree([0] + [0.05] * 7, sizes=list(range(2, 10)))   # no request of it names a tool call
        common = ["--lanes", "1", "--plays", "1", "--duration", "30", "--hints", "dist"]
        hints = [r["hint"] for r in await self.run_client(tree, common + ["--hint-no-call"]) if r["kind"] == "req"]
        self.assertEqual([h["reply_key"] for h in hints], ["_no_call"] * 7 + ["_end"])
        self.assertEqual([h["matched"] for h in hints[:5]], [None] * 5)   # an entry needs five gaps
        self.assertEqual([h["matched"] for h in hints[5:7]], ["_no_call"] * 2)
        plain = [r["hint"] for r in await self.run_client(tree, common) if r["kind"] == "req"]
        self.assertEqual([h["reply_key"] for h in plain], [None] * 7 + ["_end"])
        self.assertEqual([h["matched"] for h in plain[5:7]], ["*"] * 2)

    async def test_idle_guard_compresses_whole_system_idle(self):
        tree = _blocks_tree([0, 5.0, 5.0], sizes=[2, 3, 4])
        rows = await self.run_client(tree, ["--lanes", "1", "--plays", "1", "--duration", "60",
                                            "--system-idle-cap", "0.2"])
        summ = next(r for r in rows if r["kind"] == "summary")
        self.assertEqual(summ["profile"], 3)
        self.assertEqual(summ["idle_jumps"], 2)
        self.assertLess(summ["wall_s"], 3.0)   # replay time still counts the 10 s, wall time does not
        self.assertGreater(summ["elapsed_s"], 9.0)

    async def test_trajectory_start_with_primer(self):
        tree = _blocks_tree([0, 0.05, 0.05, 0.05], sizes=[2, 3, 4, 5])
        rows = await self.run_client(tree, ["--lanes", "1", "--plays", "1", "--duration", "30", "--warmup", "primer",
                                            "--start-window", "0.5", "0.5"])
        req = [r for r in rows if r["kind"] == "req"]
        warm = [r for r in req if r["phase"] == "warmup"]
        prof = [r for r in req if r["phase"] == "profile"]
        self.assertEqual([(r["k"], r["out_target"], r["completion_tokens"]) for r in warm], [(1, 1, 1)])
        self.assertEqual([r["k"] for r in prof], [2, 3])
        self.assertEqual(prof[0]["cached_tokens"], 12)   # the primer's prompt is in the cache

    async def test_deadline_stops_sleeping_streams(self):
        tree = _blocks_tree([0, 0.05, 50.0], sizes=[2, 3, 4])
        rows = await self.run_client(tree, ["--lanes", "1", "--duration", "0.5"])
        summ = next(r for r in rows if r["kind"] == "summary")
        self.assertEqual((summ["profile"], summ["skipped"]), (2, 1))
        self.assertLess(summ["wall_s"], 2.0)

    async def test_lengths_carry_the_servers_reply(self):
        reqs = [pt.request(t=0, dur=0.05, deps=[], wait=0, out=4, prefix=0, append=8),
                pt.request(t=0.1, dur=0.05, deps=[[0, 0]], wait=0.05, out=4, prefix=12, append=4)]
        tree = {"kind": "tree", "tree": "L", "streams": [{"name": "main", "role": "main", "requests": reqs}]}
        rows = await self.run_client(tree, ["--arrival-rate", "1", "--duration", "0"], content="lengths",
                                     block_tokens=None)
        req = [r for r in rows if r["kind"] == "req"]
        self.assertEqual([r["prompt_sent"] for r in req], [8, 16])
        self.assertEqual(req[1]["cached_tokens"], 12)   # prompt 8 + the server's 4 output tokens, page-aligned
        self.assertEqual(req[1]["ideal_cached"], 12)

    async def test_completions_api(self):
        rows = await self.run_client(_subagent_tree(), ["--lanes", "1", "--plays", "1", "--duration", "30",
                                                        "--api", "completions", "--release-path", "/programs/release"])
        req = [r for r in rows if r["kind"] == "req"]
        self.assertEqual(len(req), 5)
        self.assertFalse([r for r in req if r.get("error")])
        self.assertTrue(all(r["prompt_tokens"] == r["prompt_sent"] for r in req))
        self.assertEqual(next(r for r in rows if r["kind"] == "summary")["releases"], 2)


class RecipeTest(unittest.TestCase):
    def test_recipes_resolve_and_explicit_flags_win(self):
        import replay_programs as rp
        a = rp.parse_args(["--trace", "x", "--output", "y", "--recipe", "agentx-mvp"])
        self.assertEqual((a.lanes, a.duration, a.start_window, a.warmup, a.system_idle_cap, a.cache_bust),
                         (32, 1800.0, (0.0, 1.0), "primer", 10.0, "recycle"))
        a = rp.parse_args(["--trace", "x", "--output", "y", "--recipe", "mori", "--lanes", "40"])
        self.assertEqual((a.lanes, a.split_idle, a.split_compaction, a.duration), (40, 1800.0, True, 7200.0))
        a = rp.parse_args(["--trace", "x", "--output", "y", "--recipe", "tracelab"])
        self.assertEqual((a.lanes, a.arrival_rate, a.wait_field), (0, 0.05, "wait_tool"))
        with self.assertRaises(SystemExit):
            rp.parse_args(["--trace", "x", "--output", "y", "--lanes", "2", "--program-id-field", "program_id"])


if __name__ == "__main__":
    unittest.main()
