"""replay_programs.py against servers that are not our SGLang fork (vLLM, vLLM + LMCache, Continuum, TokenCake).

  python3 replay_baseline.py --backend NAME [--backend-opt KEY=VALUE ...] <every flag of scripts/replay_programs.py>

The workload is not defined here. Trees, arrivals, the closed loop, waits and token ids all come from
scripts/replay_programs.py, imported unchanged, so a baseline is sent the requests our own arms are sent. A backend
only shapes the wire: the body fields a system needs, how the reply's token ids are read back, and any call the system
expects around a tool wait. A request record is the one replay_programs.py writes plus "bl" (what the backend added),
so program_summary.py and compare_arms.py read these runs as they read ours.

Backends (backends.py has one class each): openai, vllm, continuum, tokencake. Each baseline's README names its own.
Stdlib and aiohttp only, like the client it wraps.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

import aiohttp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "scripts")))

import replay_programs as rp  # noqa: E402
from backends import BACKENDS, Turn  # noqa: E402
from gap_hints import OnlineGapTable, pick_entry  # noqa: E402


class BaselineReplayer(rp.Replayer):
    """The pinned replayer with a backend between it and the wire. Only wire-format methods are overridden."""

    def __init__(self, args: argparse.Namespace, *, backend, **kwargs) -> None:
        super().__init__(args, **kwargs)
        self.backend = backend
        self._turns: dict = {}       # request id -> Turn, while the request is in flight
        self._waits: dict = {}       # agent system -> OnlineGapTable of the waits replayed so far

    # ----- what a backend may know about a request -----

    def turn_of(self, play: rp.Play, s: int, k: int) -> Turn:
        reqs = play.streams[s]["requests"]
        r, last = reqs[k], k + 1 >= len(reqs)
        nxt = None if last else reqs[k + 1]
        return Turn(rid=self.rid(play, s, k), session=f"{self.args.tag}-c{play.inst[s]}", k=k, last=last,
                    reply_key=r.get("key"), next_gap=None if last else nxt.get("gap"),
                    next_wait_s=None if last else self.scaled_wait(nxt),
                    est_wait_s=self._estimate(play, reqs, k), role=play.streams[s].get("role"))

    def _estimate(self, play: rp.Play, reqs: list, k: int) -> float | None:
        """Median wait seen so far after replies with this request's key: the estimate our dist hints carry.
        The wait that led to request k has just ended, so it joins the table first."""
        table = self._waits.setdefault(tuple(play.tree.get("system") or ()), OnlineGapTable(key_mode="command"))
        if k > 0:
            table.observe(key=reqs[k - 1].get("key"), gap_s=self.scaled_wait(reqs[k]))
        picked = pick_entry({e["key"]: e for e in table.entries()}, reqs[k].get("key"))
        return None if picked is None or picked.get("ended") else picked["p50"]

    # ----- the wire -----

    async def send(self, play: rp.Play, s: int, k: int, *, phase: str, max_tokens: int | None = None) -> None:
        turn = self.turn_of(play, s, k)
        self._turns[turn.rid] = turn
        try:
            await self.backend.before_request(self, turn)
            await super().send(play, s, k, phase=phase, max_tokens=max_tokens)
        finally:
            del self._turns[turn.rid]
        await self.backend.after_reply(self, turn)

    async def _post(self, *, ids_json: str, out: int, rid: str, hints, session: str, rec: dict) -> dict:
        sent = self.backend.sent(self._turns[rid])
        if sent is not None:
            rec["bl"] = sent
        return await super()._post(ids_json=ids_json, out=out, rid=rid, hints=hints, session=session, rec=rec)

    def _body(self, *, ids_json: str, out: int, rid: str, hints, session: str) -> str:
        want_ids = self.header["content"] == "lengths"   # a lengths trace re-sends the server's own reply
        rest = self.backend.body(model=self.model, out=out, turn=self._turns[rid], want_ids=want_ids)
        return '{"prompt":[' + ids_json + "]," + json.dumps(rest)[1:]

    def _absorb(self, d: dict, res: dict, rec: dict, t_send: float) -> None:
        if d.get("error"):
            raise RuntimeError(f"stream error: {json.dumps(d['error'])[:200]}")
        for ch in d.get("choices") or []:
            ids = self.backend.reply_ids(ch)
            if (ch.get("text") or ids) and rec.get("ttft") is None:
                rec["ttft"] = round(self.clock.now() - t_send, 4)
            res["output_ids"] += ids
        if d.get("usage"):
            res["usage"] = d["usage"]

    async def release(self, play: rp.Play, s: int) -> None:
        await super().release(play, s)
        await self.backend.end_of_stream(self, session=f"{self.args.tag}-c{play.inst[s]}")

    async def notify(self, path: str, body: dict) -> None:
        """A side call a backend makes around a wait; a failure is recorded and the replay goes on."""
        try:
            async with self.http.post(self.args.url + path, json=body) as resp:
                await resp.read()
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status}")
        except Exception as e:  # noqa: BLE001 - as release(): logged, never fatal
            self.write({"kind": "notify_error", "path": path, "error": f"{type(e).__name__}: {str(e)[:160]}"})


def parse_args(argv=None) -> tuple:
    """(replay_programs.py's namespace, the backend). Our flags are taken out first; the rest is the pinned client's."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    ap.add_argument("--backend-opt", action="append", default=[], metavar="KEY=VALUE",
                    help="a backend's own option, repeatable")
    ours, rest = ap.parse_known_args(argv)
    opts = dict(o.split("=", 1) for o in ours.backend_opt)
    args = rp.parse_args(rest + ["--api", "completions"])
    args.backend, args.backend_opts = ours.backend, opts   # kept in the run record
    return args, BACKENDS[ours.backend](opts)


async def replay(args: argparse.Namespace, backend) -> None:
    header, trees, dropped = rp.load_trees(args)
    print(f"{len(trees)} trees from {args.trace} ({header['source']}, {header['content']}); dropped over "
          f"--max-context: {dropped}; backend {backend.name} {args.backend_opts or ''}; "
          + (f"{args.lanes} lanes" if args.lanes else f"arrivals {args.arrival_rate:g}/s, max active "
             f"{args.max_active or 'uncapped'}") + f", duration {args.duration or 'until done'}", flush=True)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=3600)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0),
                                     read_bufsize=1 << 24) as http:
        model = await rp.served_model(http, args.url)
        with open(args.output, "a") as out:
            rep = BaselineReplayer(args, backend=backend, header=header, trees=trees, out=out, http=http, model=model)
            await rep.run()


def main(argv=None) -> int:
    args, backend = parse_args(argv)
    asyncio.run(replay(args, backend))
    return 0


if __name__ == "__main__":
    sys.exit(main())
