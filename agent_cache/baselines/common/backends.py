"""Wire formats of the baseline servers, used by replay_baseline.py. Stdlib only.

A backend never changes what is sent when or with which tokens; it adds the body fields a system reads, reads the
reply's token ids back, and makes the calls a system expects around a tool wait. Each field a backend adds is filled
from the trace alone (Turn), the same facts our own hint arms use.
"""
from __future__ import annotations

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
    """Plain /v1/completions, the fields replay_programs.py's own completions mode sends."""

    name = "openai"

    def __init__(self, opts: dict) -> None:
        self.opts = opts

    def body(self, *, model: str, out: int, turn: Turn, want_ids: bool) -> dict:
        """Every body field but the prompt."""
        return {"model": model, "max_tokens": out, "stream": True, "stream_options": {"include_usage": True},
                "ignore_eos": True, "temperature": 0.0, "rid": turn.rid, "return_token_ids": want_ids}

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


BACKENDS = {cls.name: cls for cls in (OpenAIBackend, VllmBackend)}
