"""Wire formats of the baseline servers, used by replay_baseline.py. Stdlib only.

A backend never changes what is sent when or with which tokens; it adds the body fields a system reads, reads the
reply's token ids back, and makes the calls a system expects around a tool wait. Each field a backend adds is filled
from the trace alone (Turn), the same facts our own hint arms use.
"""
from __future__ import annotations

import hashlib
import uuid
from typing import NamedTuple


class Turn(NamedTuple):
    """What a backend may use about one request; every field comes from the trace."""

    rid: str
    session: str               # one stream of one play: <tag>-c<instance>
    k: int                     # the request's index in its stream
    last: bool                 # no request follows in this stream
    reply_key: str | None      # tool key of the recorded reply ("Bash/git"); None without a tool call
    next_gap: str | None       # what the wait after this reply is: tool, user, spawn, join
    next_wait_s: float | None  # that wait as replayed; None after the last request
    est_wait_s: float | None   # median wait seen so far after replies of this key; None before five were seen
    role: str | None           # main, subagent, ...


class OpenAIBackend:
    """Plain /v1/completions, the fields replay_programs.py's own completions mode sends. Option details=on asks our
    SGLang for the cached tokens by tier (return_cached_tokens_details, answered under "sglext")."""

    name = "openai"

    def __init__(self, opts: dict) -> None:
        self.opts = opts

    def body(self, *, model: str, out: int, turn: Turn, want_ids: bool) -> dict:
        """Every body field but the prompt."""
        body = {"model": model, "max_tokens": out, "stream": True, "stream_options": {"include_usage": True},
                "ignore_eos": True, "temperature": 0.0, "rid": turn.rid, "return_token_ids": want_ids}
        if self.opts.get("details") == "on":
            body["return_cached_tokens_details"] = True
        return body

    def reply_ids(self, choice: dict) -> list:
        """The generated token ids one streamed choice carries."""
        return list(choice.get("token_ids") or [])

    def sent(self, turn: Turn) -> dict | None:
        """What this backend added for the request, for the client record; None when nothing."""
        return None

    async def before_request(self, rep, turn: Turn) -> None:
        """Called when the wait before a request has passed, before it is sent."""

    async def after_reply(self, rep, turn: Turn) -> None:
        """Called once a reply is complete, as the wait that follows it starts."""

    async def end_of_stream(self, rep, *, session: str) -> None:
        """Called after a stream's last reply."""


class VllmBackend(OpenAIBackend):
    """Stock vLLM and vLLM + LMCache: the reply's token ids on every request, so TTFT is the first generated token
    whatever its text, and vLLM's own request id field so the server's log carries ours."""

    name = "vllm"

    def body(self, *, model: str, out: int, turn: Turn, want_ids: bool) -> dict:
        body = super().body(model=model, out=out, turn=turn, want_ids=True)
        del body["rid"]
        body["request_id"] = turn.rid
        return body


def continuum_func(reply_key: str | None) -> str | None:
    """Continuum's name for a tool call. Its parser takes the first word of a reply's bash block ("git" for
    `git status`), so a Bash key gives its command word and any other tool its own name."""
    if not reply_key:
        return None
    tool, _, word = reply_key.partition("/")
    return word or tool


class ContinuumBackend(VllmBackend):
    """Continuum (vllm-continuum, a vLLM 0.10.2 fork): the program identity and end it schedules by, as top-level
    fields of the completion request (its protocol.py). job_id is the stream, so a job never has two requests in
    flight; is_last_step marks the stream's last request, which Continuum never pins. this_func_call names the tool
    the recorded reply calls; the release parses it from the reply text, which a replay's synthetic replies lack, so
    it reaches the scheduler only with the replay patch (continuum/patches/0001). Options: func=off sends no tool."""

    name = "continuum"

    def body(self, *, model: str, out: int, turn: Turn, want_ids: bool) -> dict:
        body = super().body(model=model, out=out, turn=turn, want_ids=want_ids)
        body.update(self.fields(turn))
        return body

    def fields(self, turn: Turn) -> dict:
        out = {"job_id": turn.session, "is_last_step": turn.last}
        func = None if turn.last or self.opts.get("func", "on") == "off" else continuum_func(turn.reply_key)
        if func is not None:
            out["this_func_call"] = func
        return out

    def sent(self, turn: Turn) -> dict | None:
        return self.fields(turn)


def lifecycle_id(rid: str) -> str:
    """TokenCake's id for a request: tc- and a canonical UUID4 hex, derived from our request id so it is stable."""
    return "tc-" + uuid.UUID(bytes=hashlib.md5(rid.encode()).digest(), version=4).hex


class TokenCakeBackend(VllmBackend):
    """TokenCake (pku-lemonade/TokenCake, a vLLM 0.22.0 fork), driven as its own client drives it
    (tools/tokencake_experiments/dataset_client.py). A request's request_id is its lifecycle id, and its metadata in
    vllm_xargs.tokencake marks a turn the session will extend as offload_eligible with a reusable_prefix (its
    datasets mark tool nodes so). After a reply, stall_started tells it the session now waits, with an estimate of
    how long; stall_finished, sent when the wait is over, precedes the next turn.
    Options: est=table (the running median wait of the reply's key, our dist hints' table; default) | exact (the
    recorded wait) | none (no estimate: TokenCake learns one per kind); events=off (no stall events, for modes
    without offload); meta=off (no TokenCake metadata, for its native mode)."""

    name = "tokencake"
    EVENTS = "/v1/tokencake/events"

    def __init__(self, opts: dict) -> None:
        super().__init__(opts)
        self._stalled: dict = {}   # session -> lifecycle id of the reply whose wait is running

    def body(self, *, model: str, out: int, turn: Turn, want_ids: bool) -> dict:
        body = super().body(model=model, out=out, turn=turn, want_ids=want_ids)
        body["request_id"] = lifecycle_id(turn.rid)
        if self.opts.get("meta", "on") != "off":
            body["vllm_xargs"] = {"tokencake": self.metadata(turn)}
        return body

    def metadata(self, turn: Turn) -> dict:
        reused = not turn.last
        return {"lifecycle_id": lifecycle_id(turn.rid), "agent_type": turn.role or "agent", "agent_name": turn.session,
                "offload_eligible": reused, "reusable_prefix": reused}

    def estimate(self, turn: Turn) -> float | None:
        mode = self.opts.get("est", "table")
        est = {"table": turn.est_wait_s, "exact": turn.next_wait_s, "none": None}[mode]
        return round(est, 3) if est is not None and est > 0 else None

    def stall_started(self, turn: Turn) -> dict:
        event = {"event": "stall_started", "lifecycle_id": lifecycle_id(turn.rid), "kind": turn.reply_key or "user"}
        est = self.estimate(turn)
        if est is not None:
            event["estimated_duration_s"] = est
        return event

    def sent(self, turn: Turn) -> dict | None:
        out = {"request_id": lifecycle_id(turn.rid)}
        if self.opts.get("events", "on") != "off" and not turn.last:
            out["stall"] = self.stall_started(turn)
        return out

    async def before_request(self, rep, turn: Turn) -> None:
        held = self._stalled.pop(turn.session, None)
        if held is not None:
            await rep.notify(self.EVENTS, {"event": "stall_finished", "lifecycle_id": held})

    async def after_reply(self, rep, turn: Turn) -> None:
        if turn.last or self.opts.get("events", "on") == "off":
            return
        self._stalled[turn.session] = lifecycle_id(turn.rid)
        await rep.notify(self.EVENTS, self.stall_started(turn))

    async def end_of_stream(self, rep, *, session: str) -> None:
        self._stalled.pop(session, None)


BACKENDS = {cls.name: cls for cls in (OpenAIBackend, VllmBackend, ContinuumBackend, TokenCakeBackend)}
