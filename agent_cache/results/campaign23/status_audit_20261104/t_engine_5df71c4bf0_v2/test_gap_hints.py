"""Tests of gap_hints.py and of replay_agentic.py's hint modes; no server, no GPU.

  cd agent_cache/scripts && python3 -m unittest test_gap_hints -v

replay_agentic.py imports aiohttp at load time; the integration tests install a stub module when aiohttp is missing,
and replace the HTTP call with a fake. The engine-reader tests need msgspec and are skipped without it.
"""
from __future__ import annotations

import ast
import asyncio
import contextlib
import importlib.util
import io
import json
import logging
import math
import os
import sys
import tempfile
import types
import unittest
from typing import Annotated, Optional
from unittest import mock

from gap_hints import (ANY_KEY, CALL_CAP, END_ENTRY, END_KEY, MIN_POOLED, GapHistory, GapTable, OnlineGapTable,
                       RecordedSession, SessionHinter, check_knobs, command_word, hint_record, key_levels, load_reply_keys, next_turn_envelope,
                       pick_entry, recorded_sessions, reply_key, reply_keys_path, scaled_gap, task_of, text_call,
                       tool_category, v1_point_s)

try:
    import msgspec
except ImportError:
    msgspec = None

ENGINE = '/home/wanhr/sglang/agent_cache/frozen/engine_5df71c4bf0_v2/python/sglang/srt'


def load_engine_reader():
    """KvHintsEnvelope (managers/kv_hints.py) and read_next_turn_hint (mem_cache/next_turn_prefetch.py) from the
    repo's engine source, so these tests follow it; importing the engine itself would pull in torch."""
    stub = types.ModuleType("sglang.srt.utils.msgspec_utils")
    stub.msgspec_struct_pydantic_core_schema = None   # called only when pydantic builds a schema
    spec = importlib.util.spec_from_file_location("_engine_kv_hints", os.path.join(ENGINE, "managers", "kv_hints.py"))
    kv_hints = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = kv_hints   # msgspec resolves the postponed annotations through it
    with mock.patch.dict(sys.modules, {stub.__name__: stub}):
        spec.loader.exec_module(kv_hints)
    path = os.path.join(ENGINE, "mem_cache", "next_turn_prefetch.py")
    with open(path) as f:
        src = f.read()
    wanted = {"NEXT_TURN_ACTION_TYPE", "NEXT_TURN_ACTION_VERSION", "NextTurnHint", "read_next_turn_hint"}
    pieces, found = [], set()
    for node in ast.parse(src).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            names = {node.name}
        elif isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        else:
            continue
        if names & wanted:
            pieces.append(ast.get_source_segment(src, node))
            found |= names & wanted
    if found != wanted:
        raise RuntimeError(f"{path} no longer defines {sorted(wanted - found)}: update this loader")
    reader = types.ModuleType("_engine_next_turn")
    sys.modules[reader.__name__] = reader
    reader.__dict__.update(msgspec=msgspec, Annotated=Annotated, Optional=Optional,
                           KvHintsEnvelope=kv_hints.KvHintsEnvelope, logger=logging.getLogger(reader.__name__))
    exec(compile("from __future__ import annotations\n" + "\n\n".join(pieces), path, "exec"), reader.__dict__)
    return kv_hints.KvHintsEnvelope, reader.read_next_turn_hint


def turn(text: str, pre_gap: float = 1.0, output_length: int = 4) -> dict:
    return {"messages": [{"role": "user", "content": text}], "pre_gap": pre_gap, "output_length": output_length,
            "source": "swebench", "replay_prompt_tokens": 10}


def bash(pre_gap: float) -> dict:
    return turn("EXECUTION RESULT of [execute_bash]:\nok", pre_gap)


def entry(key: str, p50: float, n: int = 10) -> dict:
    return {"key": key, "p10": p50 / 2, "p50": p50, "p90": p50 * 2, "n": n}


def read_v1(envelope: dict | None) -> dict | None:
    """The payload of the first sglang.next_turn action of version "1", the one today's engine reads."""
    if envelope is None:
        return None
    for action in envelope["actions"]:
        if action["action_type"] == "sglang.next_turn" and action["action_version"] == "1":
            return action["payload"]
    return None


class ToolCategoryTest(unittest.TestCase):
    def test_headers(self):
        cases = {
            "EXECUTION RESULT of [execute_bash]:\n/workspace": "execute_bash",
            "  EXECUTION RESULT of [str_replace_editor]:": "str_replace_editor",
            "[TOOL RESULT web_browser_go] page": "web_browser_go",
            "[TOOL RESULT ] page": "unnamed_tool",
            "\n\nMissing required parameters for function 'execute_bash': {'x'}": "error/execute_bash",
            "Parameter 'cmd' is not allowed for function 'str_replace_editor'": "error/str_replace_editor",
            "Missing required argument: path": "error",
            "Parameter 'command' is expected to be one of": "error",
            "Please continue on whatever approach you think is suitable.": "nudge",
            "Please proceed to the next step.": "nudge",
            "Fix the failing test in foo.py": "other",
        }
        for text, label in cases.items():
            self.assertEqual(tool_category(turn(text)), label, text)
            self.assertRegex(label, r"^[\w./-]+$")   # the engine's HICACHE_EVT value grammar

    def test_non_string_content_and_empty(self):
        self.assertEqual(tool_category({"messages": [{"role": "user", "content": [{"type": "text"}]}]}), "other")
        self.assertEqual(tool_category({"messages": []}), "other")


class ReplyKeyTest(unittest.TestCase):
    def test_command_word(self):
        cases = {
            "git status": "git",
            "cd /workspace/repo && git log": "git",
            "cd /a; cd /b && grep -r x .": "grep",
            "FOO=1 BAR=2 pip install -e .": "pip",
            "/usr/bin/find . -name x": "find",
            "python -m pytest -q": "python-m-pytest",
            "python3 -m pip install x": "python-m-pip",
            "python3 -c 'print(1)'": "python-c",
            "python3 run.py": "python",
            "cat a | grep b": "cat",
            "view": "view",
            "": None,
            "   ": None,
            "$(foo) bar": None,
            "{ echo; }": None,
        }
        for command, word in cases.items():
            self.assertEqual(command_word(command), word, command)

    def test_reply_key(self):
        self.assertEqual(reply_key("execute_bash", {"command": "cd /x && pytest -q"}), "execute_bash/pytest")
        self.assertEqual(reply_key("bash", {"cmd": "ls -l"}), "bash/ls")
        self.assertEqual(reply_key("str_replace_editor", {"command": "view", "path": "/a"}), "str_replace_editor/view")
        self.assertEqual(reply_key("think", {"thought": "hm"}), "think")
        self.assertEqual(reply_key("execute_bash", {"command": 7}), "execute_bash")

    def test_text_call(self):
        reply = "Let me look.\n<function=execute_bash>\n<parameter=command>\ngit status\n</parameter>\n</function>"
        self.assertEqual(text_call(reply), ("execute_bash", {"command": "git status"}))
        self.assertIsNone(text_call("No tool here."))

    def test_text_call_skips_reasoning_and_takes_the_first_call(self):
        reply = ("I could call <function=wrong> here.\n</think>\n\n<function=a><parameter=x>1</parameter></function>"
                 "<function=b></function>")
        self.assertEqual(text_call(reply), ("a", {"x": "1"}))

    def test_key_levels(self):
        self.assertEqual(key_levels("execute_bash/git", "command"), ("execute_bash/git", "execute_bash", ANY_KEY))
        self.assertEqual(key_levels("execute_bash/git", "tool"), ("execute_bash", ANY_KEY))
        self.assertEqual(key_levels("execute_bash/git", "none"), (ANY_KEY,))
        self.assertEqual(key_levels("think", "command"), ("think", ANY_KEY))
        self.assertEqual(key_levels(None, "command"), (ANY_KEY,))

    def test_pick_falls_back_from_key_to_tool_to_default(self):
        by_key = {e["key"]: e for e in (entry("execute_bash/git", 1), entry("execute_bash", 2), entry(ANY_KEY, 3))}
        self.assertEqual(pick_entry(by_key, "execute_bash/git")["p50"], 1)
        self.assertEqual(pick_entry(by_key, "execute_bash/pip")["p50"], 2)
        self.assertEqual(pick_entry(by_key, "think")["p50"], 3)
        self.assertEqual(pick_entry(by_key, None)["p50"], 3)
        self.assertIsNone(pick_entry({}, "think"))

    def test_task_of(self):
        self.assertEqual(task_of("swebench__astropy__astropy-12907__minimax"), "swebench__astropy__astropy-12907")
        self.assertEqual(task_of("wildclaw__t1__claude__chain1"), "wildclaw__t1")

    def test_scaled_gap_caps(self):
        self.assertEqual(scaled_gap({"pre_gap": 2.0}, scale=70.0), 140.0)
        self.assertEqual(scaled_gap({"pre_gap": 100.0}, scale=70.0), 1200.0)
        self.assertEqual(scaled_gap({"pre_gap": None}, scale=70.0), 0.0)


def session(task: str, gaps: list, *, keys: list | None = None, model: str = "m", source: str = "s") -> RecordedSession:
    return RecordedSession(task=task, source=source, model=model, keys=keys or ["tool/a"] * len(gaps), gaps=gaps)


class GapTableTest(unittest.TestCase):
    def test_quantiles_are_empirical_in_ln_space(self):
        gaps = [1.0, 2.0, 4.0, 8.0, 16.0]
        table = GapTable([session("other", gaps)], key_mode="command")
        by_key = {e["key"]: e for e in table.entries(source="s", model="m", task="me")}
        e = by_key["tool/a"]
        self.assertEqual(e["n"], 5)
        self.assertAlmostEqual(e["p50"], 4.0)
        self.assertAlmostEqual(e["p10"], 2 ** 0.4, places=4)   # 0.4 of the way from ln 1 to ln 2
        self.assertAlmostEqual(e["p90"], 2 ** 3.6, places=3)
        self.assertEqual(set(by_key), {"tool/a", "tool", ANY_KEY, END_KEY})
        self.assertEqual(by_key[END_KEY], END_ENTRY)

    def test_leaves_out_the_sessions_own_task(self):
        others = session("other", [10.0] * 5)
        mine = session("me", [1000.0] * 5)
        table = GapTable([others, mine], key_mode="command")
        by_key = {e["key"]: e for e in table.entries(source="s", model="m", task="me")}
        self.assertEqual((by_key["tool/a"]["p50"], by_key["tool/a"]["n"]), (10.0, 5))
        by_key = {e["key"]: e for e in table.entries(source="s", model="m", task="other")}
        self.assertEqual((by_key["tool/a"]["p50"], by_key["tool/a"]["n"]), (1000.0, 5))
        by_key = {e["key"]: e for e in table.entries(source="s", model="m", task="third")}
        self.assertEqual(by_key["tool/a"]["n"], 10)

    def test_too_few_gaps_make_no_entry(self):
        table = GapTable([session("other", [10.0] * 4)], key_mode="command")
        self.assertEqual(table.entries(source="s", model="m", task="me"), [END_ENTRY])

    def test_a_rarer_key_falls_to_its_tool(self):
        s = session("other", [1.0] * 4 + [100.0] * 5, keys=["tool/rare"] * 4 + ["tool/common"] * 5)
        by_key = {e["key"]: e for e in GapTable([s], key_mode="command").entries(source="s", model="m", task="me")}
        self.assertNotIn("tool/rare", by_key)
        self.assertEqual(pick_entry(by_key, "tool/rare")["n"], 9)

    def test_coarser_pools_fill_only_tools_and_the_default(self):
        other_model = session("other", [10.0] * 5, keys=["only_there/x"] * 5, model="n")
        table = GapTable([other_model], key_mode="command")
        by_key = {e["key"]: e for e in table.entries(source="s", model="m", task="me")}
        self.assertEqual(set(by_key), {"only_there", ANY_KEY, END_KEY})   # not the other model's "only_there/x"
        other_source = session("other", [10.0] * 5, source="t")
        by_key = {e["key"]: e for e in GapTable([other_source], key_mode="command").entries(source="s", model="m", task="me")}
        self.assertEqual(set(by_key), {ANY_KEY, END_KEY})

    def test_key_modes(self):
        s = session("other", [10.0] * 5)
        by_mode = {mode: {e["key"] for e in GapTable([s], key_mode=mode).entries(source="s", model="m", task="me")}
                   for mode in ("command", "tool", "none")}
        self.assertEqual(by_mode["command"], {"tool/a", "tool", ANY_KEY, END_KEY})
        self.assertEqual(by_mode["tool"], {"tool", ANY_KEY, END_KEY})
        self.assertEqual(by_mode["none"], {ANY_KEY, END_KEY})

    def test_quantiles_stay_ordered_and_positive(self):
        gaps = [0.0, 0.0, 0.0, 1200.0, 0.001, 3.0, 1199.0]
        for e in GapTable([session("other", gaps)], key_mode="command").entries(source="s", model="m", task="me"):
            if "p50" in e:
                self.assertTrue(0 < e["p10"] <= e["p50"] <= e["p90"], e)

    def test_recorded_sessions_skips_empty_conversations(self):
        data = {"sessions": [{"session_id": "a__t__m", "source": "s", "orig_model": "m"},
                             {"session_id": "b__t__m", "source": "s", "orig_model": "m"}],
                "conversations": [[], [turn("go", 0.0), bash(2.0)]]}
        rec = recorded_sessions(data, keys=[[], ["x", None]], gap_of=lambda t: t["pre_gap"] * 10)
        self.assertIsNone(rec[0])
        self.assertEqual((rec[1].task, rec[1].gaps, rec[1].keys), ("b__t", [20.0], ["x", None]))


class GapHistoryTest(unittest.TestCase):
    def test_entries_appear_with_enough_gaps(self):
        h = GapHistory(min_samples=2, sigma_floor=0.5)
        self.assertEqual(h.entries(), [END_ENTRY])
        h.observe(key="tool/a", gap_s=10.0)
        self.assertEqual(h.entries(), [END_ENTRY])
        h.observe(key="tool/b", gap_s=40.0)
        by_key = {e["key"]: e for e in h.entries()}
        self.assertEqual(set(by_key), {"tool", ANY_KEY, END_KEY})   # two gaps of the tool, one of each word
        h.observe(key="tool/a", gap_s=40.0)
        by_key = {e["key"]: e for e in h.entries()}
        self.assertEqual(set(by_key), {"tool/a", "tool", ANY_KEY, END_KEY})
        self.assertAlmostEqual(by_key["tool/a"]["p50"], 20.0, places=3)   # the geometric mean of 10 and 40
        self.assertEqual(by_key["tool/a"]["n"], 2)

    def test_sigma_floor(self):
        h = GapHistory(sigma_floor=0.5)
        h.observe(key="t", gap_s=5.0)
        h.observe(key="t", gap_s=5.0)
        e = {x["key"]: x for x in h.entries()}["t"]
        self.assertAlmostEqual(e["p90"] / e["p50"], math.exp(0.5 * 1.2815515655446004), places=4)

    def test_zero_gaps_are_floored(self):
        h = GapHistory()
        h.observe(key="t", gap_s=0.0)
        h.observe(key="t", gap_s=0.0)
        self.assertAlmostEqual({x["key"]: x for x in h.entries()}["t"]["p50"], 0.01)

    def test_bad_settings(self):
        for kw in ({"key_mode": "user"}, {"min_samples": 1}, {"sigma_floor": 0}, {"sigma_floor": math.inf},
                   {"sigma_floor": math.nan}, {"sigma_floor": 600.0}):
            with self.assertRaises(ValueError, msg=kw):
                GapHistory(**kw)

    def test_key_mode_none_keeps_only_the_default(self):
        h = GapHistory(key_mode="none")
        for g in (1.0, 2.0, 3.0):
            h.observe(key="tool/a", gap_s=g)
        self.assertEqual({x["key"] for x in h.entries()}, {ANY_KEY, END_KEY})


class OnlineGapTableTest(unittest.TestCase):
    def test_entry_appears_at_the_minimum_with_exact_quantiles(self):
        t = OnlineGapTable()
        self.assertEqual(t.entries(), [END_ENTRY])
        gaps = [16.0, 1.0, 8.0, 2.0, 4.0]   # out of order: the table keeps them sorted
        for g in gaps[:MIN_POOLED - 1]:
            t.observe(key="tool/a", gap_s=g)
        self.assertEqual(t.entries(), [END_ENTRY])
        t.observe(key="tool/a", gap_s=gaps[MIN_POOLED - 1])
        by_key = {e["key"]: e for e in t.entries()}
        self.assertEqual(set(by_key), {"tool/a", "tool", ANY_KEY, END_KEY})
        for key in ("tool/a", "tool", ANY_KEY):
            self.assertEqual(by_key[key]["n"], 5)
            self.assertAlmostEqual(by_key[key]["p50"], 4.0, places=3)
            self.assertAlmostEqual(by_key[key]["p10"], 2.0 ** 0.4, places=3)   # between order statistics, in ln(gap)
            self.assertAlmostEqual(by_key[key]["p90"], 2.0 ** 3.6, places=2)

    def test_entries_grow_with_every_gap(self):
        t = OnlineGapTable()
        for k, g in enumerate((1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)):
            t.observe(key="t", gap_s=g)
            if k + 1 >= MIN_POOLED:
                self.assertEqual({e["key"]: e for e in t.entries()}["t"]["n"], k + 1)

    def test_a_key_falls_back_to_its_tool_and_the_default(self):
        t = OnlineGapTable()
        for g in (1.0, 2.0, 3.0):
            t.observe(key="tool/a", gap_s=g)
            t.observe(key="tool/b", gap_s=g)
        by_key = {e["key"]: e for e in t.entries()}
        self.assertEqual(set(by_key), {"tool", ANY_KEY, END_KEY})   # six gaps of the tool, three of each word
        self.assertEqual(pick_entry(by_key, "tool/a")["key"], "tool")
        self.assertEqual(pick_entry(by_key, "other/x")["key"], ANY_KEY)

    def test_key_modes(self):
        for mode, want in (("tool", {"tool", ANY_KEY, END_KEY}), ("none", {ANY_KEY, END_KEY})):
            t = OnlineGapTable(key_mode=mode)
            for g in (1.0, 2.0, 3.0, 4.0, 5.0):
                t.observe(key="tool/a", gap_s=g)
            self.assertEqual({e["key"] for e in t.entries()}, want, mode)
        with self.assertRaises(ValueError):
            OnlineGapTable(key_mode="user")

    def test_same_quantiles_as_the_pre_calculated_list_on_the_same_gaps(self):
        gaps = [3.0, 0.5, 9.0, 1.5, 27.0, 2.0, 40.0]
        online = OnlineGapTable()
        for g in gaps:
            online.observe(key="tool/a", gap_s=g)
        table = GapTable([RecordedSession(task="other", source="s", model="m", keys=["tool/a"] * len(gaps), gaps=gaps)],
                         key_mode="command")
        self.assertEqual(online.entries(), table.entries(source="s", model="m", task="mine"))

    def test_sessions_share_it_and_only_finished_gaps_count(self):
        turns = [turn("Fix it", 0.0)] + [bash(float(g)) for g in (1, 2, 3, 4, 5, 6, 7)]
        keys = ["t"] * 7 + [None]
        shared = OnlineGapTable()
        a = SessionHinter(turns, keys=keys, gap_of=lambda x: x["pre_gap"], history=shared)
        b = SessionHinter(turns, keys=keys, gap_of=lambda x: x["pre_gap"], history=shared)
        self.assertIsNone(a.step(0).picked)
        self.assertIsNone(b.step(0).picked)
        for t in range(1, 3):
            a.step(t)
            b.step(t)
        # a and b each finished the gaps before turns 1 and 2: 4 in all, and neither has seen a gap still to come
        h = a.step(3)     # a's third gap has just finished: 5 of them, the minimum
        self.assertEqual((h.picked["key"], h.picked["n"]), ("t", 5))
        self.assertEqual(sorted(e["n"] for e in h.entries if "n" in e), [5, 5])
        self.assertEqual(b.step(3).picked["n"], 6)


class SessionHinterTest(unittest.TestCase):
    def setUp(self):
        self.turns = [turn("Fix it", 0.0), bash(2.0), bash(4.0), bash(0.5), bash(8.0), bash(0.7)]
        self.keys = ["execute_bash/git", "execute_bash/ls", "think", "execute_bash/git", "think", None]
        self.list = [entry("execute_bash/git", 5.0), entry("execute_bash", 6.0), entry(ANY_KEY, 7.0), END_ENTRY]

    def hinter(self, turns=None, keys=None, **kw):
        return SessionHinter(turns or self.turns, keys=keys or self.keys, gap_of=lambda t: t["pre_gap"] * 10, **kw)

    def test_fixed_list_and_picks(self):
        h = self.hinter(entries=self.list)
        hints = [h.step(t) for t in range(len(self.turns))]
        self.assertEqual([x.reply_key for x in hints], self.keys[:-1] + [END_KEY])
        self.assertEqual([x.true_gap_s for x in hints[:-1]], [20.0, 40.0, 5.0, 80.0, 7.0])
        self.assertEqual([x.picked["p50"] for x in hints[:-1]], [5.0, 6.0, 7.0, 5.0, 7.0])
        self.assertTrue(all(x.entries is self.list for x in hints))
        self.assertTrue(hints[-1].ended)
        self.assertFalse(hints[0].ended)
        self.assertIsNone(hints[-1].true_gap_s)

    def test_session_list_is_causal(self):
        """The list at request t never sees the gap after reply t or any later one."""
        def lists(turns):
            h = self.hinter(turns, history=GapHistory())
            return [h.step(t).entries for t in range(6)]
        base = lists(self.turns)
        self.assertEqual(base[0], [END_ENTRY])   # nothing observed yet
        self.assertEqual(base[1], [END_ENTRY])   # one gap so far
        self.assertIn(ANY_KEY, {e["key"] for e in base[2]})
        for k in range(1, 6):
            turns = [dict(x) for x in self.turns]
            turns[k]["pre_gap"] = 9999.0
            self.assertEqual(lists(turns)[: k], base[: k], f"gap of turn {k} leaked into an earlier list")

    def test_session_list_keys_a_gap_by_the_reply_before_it(self):
        h = self.hinter(history=GapHistory(key_mode="tool", min_samples=2))
        hints = [h.step(t) for t in range(4)]
        # by turn 3: execute_bash (gaps 20, 40) from replies 0 and 1, think (5) from reply 2
        by_key = {e["key"]: e for e in hints[3].entries}
        self.assertEqual(by_key["execute_bash"]["n"], 2)
        self.assertNotIn("think", by_key)

    def test_order_enforced(self):
        h = self.hinter(entries=self.list)
        h.step(0)
        with self.assertRaises(ValueError):
            h.step(2)

    def test_no_keys_picks_the_default(self):
        h = SessionHinter(self.turns, keys=None, gap_of=lambda t: t["pre_gap"], entries=self.list)
        hints = [h.step(t) for t in range(6)]
        self.assertTrue(all(x.reply_key is None and x.picked["p50"] == 7.0 for x in hints[:-1]))
        self.assertTrue(hints[-1].ended)

    def test_source_none_only_knows_the_end(self):
        h = self.hinter()
        hints = [h.step(t) for t in range(6)]
        self.assertTrue(all(x.picked is None and x.entries == [END_ENTRY] for x in hints[:-1]))
        self.assertTrue(hints[-1].ended)

    def test_replay_cut_is_not_an_end(self):
        """A replay that sends only the first 4 turns: the 4th still names its recorded reply."""
        h = self.hinter(entries=self.list)
        last = [h.step(t) for t in range(4)][-1]
        self.assertFalse(last.ended)
        self.assertEqual((last.reply_key, last.true_gap_s), ("execute_bash/git", 80.0))

    def test_end_modes(self):
        capped = [turn("Fix it", 0.0)] + [bash(1.0 + k % 3) for k in range(CALL_CAP - 1)]
        for turns, calls, end, ended in ((self.turns, None, "explicit", True), (capped, None, "explicit", True),
                                         (self.turns, None, "natural", True), (capped, None, "natural", False),
                                         (self.turns, None, "none", False), (capped, None, "none", False),
                                         (self.turns, 13, "explicit", False), (self.turns, 13, "natural", False),
                                         (self.turns, CALL_CAP, "explicit", False)):
            h = SessionHinter(turns, keys=None, gap_of=lambda t: t["pre_gap"], entries=self.list, end=end, calls=calls)
            last = [h.step(t) for t in range(len(turns))][-1]
            self.assertEqual(last.ended, ended, (len(turns), calls, end))
            if not ended:   # an end the agent does not know: the default entry, nothing to score
                self.assertEqual((last.reply_key, last.true_gap_s, last.picked["key"]), (None, None, ANY_KEY))
        with self.assertRaises(ValueError):
            SessionHinter(self.turns, keys=None, gap_of=lambda t: 0.0, end="sometimes")


class EnvelopeTest(unittest.TestCase):
    def setUp(self):
        self.list = [entry("execute_bash/git", 5.0), entry(ANY_KEY, 7.0), END_ENTRY]
        hinter = SessionHinter([turn("go", 0), bash(2), bash(4)], keys=["execute_bash/git", "think", None],
                               gap_of=lambda t: t["pre_gap"], entries=self.list)
        self.hints = [hinter.step(t) for t in range(3)]

    def envelope(self, hint, v1_point="p50"):
        return next_turn_envelope(message_id="x-c1-t2", session="x-c1", next_rid="x-c1-t3", hint=hint, v1_point=v1_point)

    def test_v2_list_then_v1_median(self):
        env = self.envelope(self.hints[0])
        self.assertEqual([a["action_version"] for a in env["actions"]], ["2", "1"])
        v2 = env["actions"][0]["payload"]
        self.assertEqual(v2, {"session": "x-c1", "next_rid": "x-c1-t3", "gaps": self.list,
                              "replay_reply_key": "execute_bash/git"})
        self.assertEqual(read_v1(env), {"session": "x-c1", "next_rid": "x-c1-t3", "next_gap_s": 5.0})

    def test_record(self):
        rec = hint_record(self.hints[0], next_rid="x-c1-t3", v1_point="p50")
        self.assertEqual(rec, {"next_rid": "x-c1-t3", "reply_key": "execute_bash/git", "list_len": 3,
                               "matched": "execute_bash/git", "point_s": 5.0, "true_gap_s": 2.0, "p10": 2.5, "p50": 5.0,
                               "p90": 10.0, "n": 10})
        fallback = hint_record(self.hints[1], next_rid="r", v1_point="none")
        self.assertEqual((fallback["reply_key"], fallback["matched"], fallback["point_s"]), ("think", ANY_KEY, None))

    def test_end(self):
        env = self.envelope(self.hints[2])
        self.assertEqual(len(env["actions"]), 1)   # no v1: today's engine then registers the session ENDED
        self.assertEqual(env["actions"][0]["payload"]["replay_reply_key"], END_KEY)
        self.assertIsNone(read_v1(env))
        rec = hint_record(self.hints[2], next_rid="r", v1_point="p50")
        self.assertEqual((rec["ended"], rec["matched"], rec["point_s"], rec["true_gap_s"]), (True, END_KEY, None, None))
        self.assertNotIn("p50", rec)

    def test_point_none(self):
        self.assertEqual(len(self.envelope(self.hints[0], v1_point="none")["actions"]), 1)
        self.assertIsNone(v1_point_s(self.hints[0], v1_point="none"))

    def test_no_pick_sends_only_the_link(self):
        hint = SessionHinter([turn("go", 0), bash(2)], keys=None, gap_of=lambda t: 0.0).step(0)
        env = self.envelope(hint)
        self.assertEqual(len(env["actions"]), 1)
        self.assertEqual(env["actions"][0]["payload"]["gaps"], [END_ENTRY])
        rec = hint_record(hint, next_rid="r", v1_point="p50")
        self.assertEqual((rec["matched"], rec["point_s"]), (None, None))
        self.assertNotIn("ended", rec)


class KnobsTest(unittest.TestCase):
    def test_values(self):
        for env in ({}, {"HINT_POINT": "", "HINT_MIN_SAMPLES": ""}, {"HINT_SOURCE": "session", "HINT_END": "natural"},
                    {"HINT_KEY": "none", "HINT_MIN_SAMPLES": "3", "HINT_SIGMA_FLOOR": "0.25"}):
            check_knobs(env)
        for env in ({"HINT_MIN_SAMPLES": "1"}, {"HINT_MIN_SAMPLES": "two"}, {"HINT_SIGMA_FLOOR": "nan"},
                    {"HINT_POINT": "mean"}, {"HINT_END": "sometimes"}, {"HINT_SOURCE": "oracle"}, {"HINT_KEY": "scope"}):
            with self.assertRaises(ValueError, msg=env):
                check_knobs(env)

    def test_reply_keys_needed_unless_source_or_key_is_none(self):
        with tempfile.TemporaryDirectory() as d:
            trace = os.path.join(d, "trace.json")
            with self.assertRaisesRegex(ValueError, "reply_keys"):
                check_knobs({}, trace=trace)
            check_knobs({"HINT_SOURCE": "none"}, trace=trace)
            check_knobs({"HINT_KEY": "none"}, trace=trace)
            with open(reply_keys_path(trace), "w") as f:
                f.write("{}")
            check_knobs({}, trace=trace)

    def test_load_reply_keys_checks_the_shape(self):
        with tempfile.TemporaryDirectory() as d:
            trace = os.path.join(d, "trace.json")
            with open(reply_keys_path(trace), "w") as f:
                json.dump({"conversations": [["a", None]]}, f)
            self.assertEqual(load_reply_keys(trace, [[{}, {}]]), [["a", None]])
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_reply_keys(trace, [[{}, {}, {}]])


@unittest.skipIf(msgspec is None, "msgspec not installed")
class EngineReaderTest(unittest.TestCase):
    """The envelopes as the repo's engine decodes and reads them. They are serialized with allow_nan=False: the
    server's orjson rejects NaN and Infinity, which stdlib json writes."""

    @classmethod
    def setUpClass(cls):
        cls.envelope_type, reader = load_engine_reader()
        cls.read_next_turn_hint = staticmethod(reader)

    def engine_reads(self, env: dict):
        return self.read_next_turn_hint(msgspec.convert(json.loads(json.dumps(env, allow_nan=False)),
                                                        self.envelope_type))

    def envelopes(self, hinter: SessionHinter, n: int):
        for t in range(n):
            h = hinter.step(t)
            yield h, next_turn_envelope(message_id=f"m{t}", session="s", next_rid=f"r{t + 1}", hint=h, v1_point="p50")

    def test_today_reads_the_picked_median_or_nothing(self):
        turns = [turn("go", 0)] + [bash(g) for g in (2.0, 4.0, 6.0, 3.0)]
        hinter = SessionHinter(turns, keys=["t", "t", "t", "t", None], gap_of=lambda t: t["pre_gap"],
                               history=GapHistory())
        for h, env in self.envelopes(hinter, len(turns)):
            hint = self.engine_reads(env)
            if h.ended or h.picked is None:
                self.assertIsNone(hint)   # today's engine registers such a turn ENDED (a known flaw)
            else:
                self.assertEqual((hint.session, hint.next_rid, hint.next_gap_s), ("s", f"r{h.turn + 1}", h.picked["p50"]))

    def test_engine_rejects_a_negative_point(self):
        """The loaded reader carries the engine's own payload rules (next_gap_s >= 0), not a copy."""
        hinter = SessionHinter([turn("go", 0), bash(2)], keys=["t", None], gap_of=lambda t: t["pre_gap"],
                               entries=[entry("t", 5.0), END_ENTRY])
        _, env = next(self.envelopes(hinter, 2))
        self.assertIsNotNone(self.engine_reads(env))
        env["actions"][1]["payload"]["next_gap_s"] = -1.0
        with self.assertLogs(level="WARNING"):
            self.assertIsNone(self.engine_reads(env))

    def test_extreme_gaps_stay_valid(self):
        """Zero, tiny, capped and wildly mixed gaps, both sources, small and large floors: every envelope is strict JSON,
        the engine accepts it, and the quantiles are finite, ordered and positive."""
        gaps = (0.0, 0.0, 1200.0, 0.001, 1200.0, 0.5, 0.0, 1200.0, 3.0, 1199.0)
        turns = [turn("go", 0)] + [bash(g) for g in gaps]
        keys = ["t/a" if k % 3 else "t/b" for k in range(len(gaps))] + [None]
        table = GapTable([RecordedSession(task="other", source="s", model="m", keys=keys, gaps=list(gaps))],
                         key_mode="command")
        hinters = [SessionHinter(turns, keys=keys, gap_of=lambda t: t["pre_gap"],
                                 entries=table.entries(source="s", model="m", task="me"))]
        hinters += [SessionHinter(turns, keys=keys, gap_of=lambda t: t["pre_gap"], history=GapHistory(sigma_floor=floor))
                    for floor in (1e-6, 0.5, 10.0)]
        for hinter in hinters:
            for _, env in self.envelopes(hinter, len(turns)):
                self.engine_reads(env)
                for e in env["actions"][0]["payload"]["gaps"]:
                    if "p50" in e:
                        q = (e["p10"], e["p50"], e["p90"])
                        self.assertTrue(all(math.isfinite(x) and x > 0 for x in q) and q == tuple(sorted(q)), e)


def import_replay():
    """replay_agentic with a stub aiohttp when the real one is missing (only its names are needed here)."""
    if "aiohttp" not in sys.modules:
        try:
            import aiohttp  # noqa: F401
        except ImportError:
            stub = types.ModuleType("aiohttp")
            stub.ClientConnectionError = type("ClientConnectionError", (Exception,), {})
            stub.ClientSession = stub.ClientTimeout = stub.TCPConnector = object
            sys.modules["aiohttp"] = stub
    import replay_agentic
    return replay_agentic


def parse(replay, argv: list):
    old = sys.argv
    sys.argv = ["replay_agentic.py"] + argv
    try:
        return replay.parse_args()
    finally:
        sys.argv = old


class ReplayIntegrationTest(unittest.TestCase):
    TURNS = [turn("Fix it", 0.0), bash(2.0), bash(3.0), bash(0.5), bash(4.0), bash(0.6)]
    KEYS = ["execute_bash/git", "execute_bash/ls", "think", "execute_bash/git", "think", None]
    LIST = [entry("execute_bash/git", 5.0), entry(ANY_KEY, 7.0), END_ENTRY]

    def play(self, argv: list, replayed: int | None = None, gap_list: list | None = None):
        """One user replays TURNS (its first `replayed` turns, as --turns would) against a fake server."""
        replay = import_replay()
        args = parse(replay, ["--trace", "unused.json", "--output", "unused.jsonl", "--gap-scale", "0.001",
                              "--tag", "t"] + argv)
        out = io.StringIO()
        r = replay.Replayer(args, out, "model", 1)
        sent = []

        async def fake_chat_turn(http, messages, max_tokens, rid, kv_hints):
            sent.append((rid, kv_hints))
            return {"text": "ok", "ttft": 0.01, "usage": {"prompt_tokens": 10 * len(messages), "completion_tokens": 1},
                    "cached_details": None, "t_send": r.t0, "latency": 0.02}

        r.chat_turn = fake_chat_turn
        user = replay.User(conv=0, task_conv=0, session_id="sess", turns=self.TURNS[:replayed], arrival_s=0.0,
                           recorded=self.TURNS, calls=len(self.TURNS), reply_keys=self.KEYS, gap_list=gap_list)
        asyncio.run(r.play(None, user))
        recs = [json.loads(line) for line in out.getvalue().splitlines()]
        return args, sent, [x for x in recs if x["kind"] == "turn"]

    def test_dist_mode(self):
        args, sent, recs = self.play(["--hints", "dist", "--hint-source", "precalc"], gap_list=self.LIST)
        self.assertEqual((args.hints, args.oracle_hints), ("dist", False))
        self.assertEqual([rid for rid, _ in sent], [f"t-c0-t{k}" for k in range(6)])
        for k, (rid, env) in enumerate(sent[:-1]):
            v2 = env["actions"][0]
            self.assertEqual((v2["action_version"], v2["payload"]["next_rid"], v2["payload"]["replay_reply_key"]),
                             ("2", f"t-c0-t{k + 1}", self.KEYS[k]))
            self.assertEqual(v2["payload"]["gaps"], self.LIST)
            rec = recs[k]["dist_hint"]
            self.assertEqual(rec["true_gap_s"], round(self.TURNS[k + 1]["pre_gap"] * 0.001, 3))
            self.assertEqual(rec["true_gap_s"], recs[k + 1]["gap_slept"])
            self.assertEqual((rec["next_rid"], rec["reply_key"], rec["point_s"]),
                             (f"t-c0-t{k + 1}", self.KEYS[k], read_v1(env)["next_gap_s"]))
        self.assertEqual(sent[-1][1]["actions"][0]["payload"]["replay_reply_key"], END_KEY)
        self.assertEqual(len(sent[-1][1]["actions"]), 1)
        self.assertTrue(recs[-1]["dist_hint"]["ended"])

    def test_dist_session_source_builds_its_list_as_it_goes(self):
        _, sent, recs = self.play(["--hints", "dist", "--hint-source", "session"])
        self.assertEqual(sent[0][1]["actions"][0]["payload"]["gaps"], [END_ENTRY])
        self.assertIsNone(read_v1(sent[0][1]))
        self.assertIn(ANY_KEY, {e["key"] for e in sent[2][1]["actions"][0]["payload"]["gaps"]})
        self.assertIsNone(recs[0]["dist_hint"]["matched"])
        self.assertEqual(recs[2]["dist_hint"]["matched"], ANY_KEY)   # "think" has no past gap, the default has two

    def test_dist_online_is_the_default_and_shared_per_agent_system(self):
        replay = import_replay()
        args = parse(replay, ["--trace", "unused.json", "--output", "unused.jsonl", "--gap-scale", "0.001", "--tag", "t",
                              "--hints", "dist"])
        self.assertEqual(args.hint_source, "online")
        out = io.StringIO()
        r = replay.Replayer(args, out, "model", 3)
        sent = {}

        async def fake_chat_turn(http, messages, max_tokens, rid, kv_hints):
            sent[rid] = kv_hints
            return {"text": "ok", "ttft": 0.01, "usage": {"prompt_tokens": 10 * len(messages), "completion_tokens": 1},
                    "cached_details": None, "t_send": r.t0, "latency": 0.02}

        r.chat_turn = fake_chat_turn
        for conv, system in enumerate((("h", "m1"), ("h", "m1"), ("h", "m2"))):
            asyncio.run(r.play(None, replay.User(conv=conv, task_conv=conv, session_id="s", turns=self.TURNS, arrival_s=0.0,
                                                  recorded=self.TURNS, calls=len(self.TURNS), reply_keys=self.KEYS,
                                                  system=system)))

        def star_n(rid):
            by_key = {e["key"]: e for e in sent[rid]["actions"][0]["payload"]["gaps"]}
            return by_key[ANY_KEY]["n"] if ANY_KEY in by_key else 0
        self.assertEqual(star_n("t-c0-t0"), 0)
        self.assertEqual(star_n("t-c0-t5"), 5)       # the first user's five finished gaps
        self.assertEqual(star_n("t-c1-t0"), 5)       # the same system: a new session starts from them
        self.assertEqual(star_n("t-c1-t5"), 10)
        self.assertEqual(star_n("t-c2-t0"), 0)       # another system has its own table
        self.assertEqual(set(r.online_tables), {("h", "m1"), ("h", "m2")})

    def test_dist_online_follows_the_key_setting(self):
        _, sent, _ = self.play(["--hints", "dist", "--hint-key", "tool"])
        self.assertEqual({e["key"] for e in sent[5][1]["actions"][0]["payload"]["gaps"]}, {ANY_KEY, END_KEY})
        _, sent, _ = self.play(["--hints", "dist", "--hint-key", "none"])
        self.assertEqual({e["key"] for e in sent[5][1]["actions"][0]["payload"]["gaps"]}, {ANY_KEY, END_KEY})
        replay = import_replay()
        args = parse(replay, ["--trace", "x", "--output", "y", "--hints", "dist", "--hint-key", "tool"])
        r = replay.Replayer(args, io.StringIO(), "model", 1)
        r.new_hinter(replay.User(conv=0, task_conv=0, session_id="s", turns=self.TURNS, arrival_s=0.0, recorded=self.TURNS,
                                 calls=len(self.TURNS), reply_keys=self.KEYS))
        table = r.online_tables[None]
        self.assertEqual(table.key_mode, "tool")

    def test_dist_end_none(self):
        _, sent, recs = self.play(["--hints", "dist", "--hint-source", "precalc", "--hint-end", "none"], gap_list=self.LIST)
        v2 = sent[-1][1]["actions"][0]["payload"]
        self.assertEqual((v2["next_rid"], v2["replay_reply_key"]), ("t-c0-t6", None))
        self.assertEqual(read_v1(sent[-1][1])["next_gap_s"], 7.0)   # the default entry
        self.assertIsNone(recs[-1]["dist_hint"]["true_gap_s"])

    def test_dist_replay_cut(self):
        """--turns 4 on a 6-turn conversation: the 4th turn names its recorded reply and does not end."""
        _, sent, recs = self.play(["--hints", "dist", "--hint-source", "precalc"], replayed=4, gap_list=self.LIST)
        self.assertEqual(len(sent), 4)
        v2 = sent[-1][1]["actions"][0]["payload"]
        self.assertEqual((v2["next_rid"], v2["replay_reply_key"]), ("t-c0-t4", "execute_bash/git"))
        self.assertEqual(recs[-1]["dist_hint"]["true_gap_s"], round(4.0 * 0.001, 3))
        self.assertNotIn("ended", recs[-1]["dist_hint"])

    def test_oracle_mode_unchanged(self):
        for argv in (["--oracle-hints"], ["--hints", "oracle"]):
            args, sent, recs = self.play(argv)
            self.assertEqual((args.hints, args.oracle_hints), ("oracle", True))
            for k, (rid, env) in enumerate(sent[:-1]):
                gap = round(self.TURNS[k + 1]["pre_gap"] * 0.001, 3)
                self.assertEqual(env, {"protocol_version": "1", "message_id": f"t-c0-t{k}",
                                       "actions": [{"action_id": "next_turn", "action_type": "sglang.next_turn",
                                                    "action_version": "1",
                                                    "payload": {"session": "t-c0", "next_rid": f"t-c0-t{k + 1}",
                                                                "next_gap_s": gap}}]})
                self.assertNotIn("dist_hint", recs[k])
            self.assertIsNone(sent[-1][1])

    def test_no_hints(self):
        args, sent, _ = self.play([])
        self.assertEqual(args.hints, "none")
        self.assertTrue(all(env is None for _, env in sent))

    def test_bad_flags(self):
        replay = import_replay()
        base = ["--trace", "x", "--output", "y"]
        for argv in (["--oracle-hints", "--hints", "dist"], ["--hints", "dist", "--hint-sigma-floor", "inf"],
                     ["--hints", "dist", "--hint-sigma-floor", "nan"], ["--hints", "dist", "--hint-min-samples", "1"],
                     ["--hints", "dist", "--hint-point", "mean"], ["--hints", "dist", "--hint-source", "oracle"],
                     ["--hints", "dist", "--hint-key", "scope"]):
            with self.assertRaises(SystemExit, msg=argv), contextlib.redirect_stderr(io.StringIO()):
                parse(replay, base + argv)

    def write_trace(self, directory: str, *, with_keys: bool = True) -> str:
        path = os.path.join(directory, "trace.json")
        convs = [self.TURNS, self.TURNS[:3], self.TURNS]
        sessions = [{"session_id": f"s__task{i}__m", "source": "s", "orig_model": "m", "n_turns_total": n}
                    for i, n in enumerate((6, 9, 6))]
        with open(path, "w") as f:
            json.dump({"conversations": convs, "sessions": sessions, "session_ids": [s["session_id"] for s in sessions]}, f)
        if with_keys:
            with open(reply_keys_path(path), "w") as f:
                json.dump({"conversations": [self.KEYS, self.KEYS[:2] + [None], self.KEYS]}, f)
        return path

    def test_load_users_keeps_the_recording_and_calls(self):
        replay = import_replay()
        with tempfile.TemporaryDirectory() as d:
            path = self.write_trace(d, with_keys=False)
            users = replay.load_users(parse(replay, ["--trace", path, "--output", "y", "--turns", "2"]))
        self.assertEqual(sorted((len(u.turns), len(u.recorded), u.calls) for u in users), [(2, 3, 9), (2, 6, 6), (2, 6, 6)])
        self.assertTrue(all(u.reply_keys is None and u.gap_list is None for u in users))
        self.assertTrue(all(u.system == ("s", "m") for u in users))

    def test_load_users_builds_the_list_from_other_tasks(self):
        replay = import_replay()
        with tempfile.TemporaryDirectory() as d:
            path = self.write_trace(d)
            users = replay.load_users(parse(replay, ["--trace", path, "--output", "y", "--hints", "dist",
                                                     "--hint-source", "precalc", "--gap-scale", "10"]))
        by_conv = {u.task_conv: u for u in users}
        self.assertEqual(by_conv[1].reply_keys, self.KEYS[:2] + [None])
        star = {e["key"]: e for e in by_conv[0].gap_list}[ANY_KEY]
        self.assertEqual(star["n"], 2 + 5)    # conversation 1's two gaps and conversation 2's five, not its own
        self.assertEqual(by_conv[0].gap_list[-1], END_ENTRY)

    def test_missing_reply_keys_stop_before_any_request(self):
        replay = import_replay()
        with tempfile.TemporaryDirectory() as d:
            path = self.write_trace(d, with_keys=False)
            with self.assertRaises(SystemExit) as stop:
                replay.load_users(parse(replay, ["--trace", path, "--output", "y", "--hints", "dist"]))
            self.assertIn("reply_keys", str(stop.exception))
            for argv in (["--hint-source", "none"], ["--hint-key", "none"]):   # these need no keys
                users = replay.load_users(parse(replay, ["--trace", path, "--output", "y", "--hints", "dist"] + argv))
                self.assertTrue(all(u.reply_keys is None for u in users))


if __name__ == "__main__":
    unittest.main()
