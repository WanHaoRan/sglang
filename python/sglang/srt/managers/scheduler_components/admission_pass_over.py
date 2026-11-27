from __future__ import annotations

import logging
from typing import Optional

import msgspec

from sglang.srt.disaggregation.utils import DisaggregationMode

logger = logging.getLogger(__name__)

# Most admissions made behind one refused request before admission waits for it again.
# Arbitrary; it bounds what being passed can cost the refused request.
MAX_PASSES = 8
# Pass counts kept; an aborted request leaves its count behind, so the oldest go first.
# Arbitrary; far above the waiting queue's length in any run of this fork.
MAX_TRACKED = 4096


class AdmissionPassOver(msgspec.Struct):
    """Lets later waiting requests take GPU room that an earlier one cannot use.

    Admission walks the waiting queue in order and stops at the first request refused
    for room, although a later, smaller one may fit what is free. With a lookahead the
    walk goes on behind that request for a bounded number of candidates. Once
    MAX_PASSES requests have been admitted behind it, the walk stops there again until
    it is admitted.
    """

    # Most candidates tried behind a refused request in one walk; 0 stops the walk there.
    lookahead: int
    # Requests admitted behind each waiting request, by rid.
    passes: dict[str, int] = {}
    # This walk: the first request refused for room, and the candidates tried and
    # admitted behind it.
    blocked_rid: Optional[str] = None
    tried: int = 0
    admitted_behind: int = 0

    @classmethod
    def create(
        cls,
        *,
        requested_lookahead: int,
        single_rank: bool,
        disaggregation_mode: DisaggregationMode,
        priority_scheduling: bool,
        lora: bool,
        dllm: bool,
        hybrid_ssm: bool,
        radix_cache: bool,
    ) -> AdmissionPassOver:
        unsupported = [
            name
            for name, unmet in (
                ("one rank (TP = PP = 1, no DP attention)", not single_rank),
                (
                    "no disaggregation",
                    disaggregation_mode != DisaggregationMode.NULL,
                ),
                # Lower priorities would take the room a refused higher one waits for.
                ("no priority scheduling", priority_scheduling),
                ("no LoRA", lora),
                ("no dLLM", dllm),
                # The walk would go past the Mamba slot a refused request frees.
                ("no Mamba state", hybrid_ssm),
                # Without it, a refused ignore_eos request stays in the adder's state.
                ("the radix cache", not radix_cache),
            )
            if unmet
        ]
        if requested_lookahead > 0 and unsupported:
            logger.warning(
                "SGLANG_PREFILL_PASS_OVER_LOOKAHEAD is ignored: it needs %s.",
                ", ".join(unsupported),
            )
        return cls(lookahead=0 if unsupported else requested_lookahead)

    def begin(self) -> None:
        self.blocked_rid = None
        self.tried = 0
        self.admitted_behind = 0

    @property
    def behind_refusal(self) -> bool:
        """The walk has passed a request refused for room."""
        return self.blocked_rid is not None

    @property
    def exhausted(self) -> bool:
        """The walk behind a refused request has used its lookahead, or the request has
        been passed MAX_PASSES times."""
        return self.blocked_rid is not None and (
            self.tried >= self.lookahead
            or self.passes.get(self.blocked_rid, 0) >= MAX_PASSES
        )

    def refused(self, rid: str) -> bool:
        """A request was refused for room; whether the walk may go on behind it."""
        if self.lookahead <= 0:
            return False
        if self.blocked_rid is None:
            self.blocked_rid = rid
        else:
            self.tried += 1
        return not self.exhausted

    def admitted(self, rid: str) -> None:
        """A request joined the batch; behind a refused one that is a pass."""
        self.passes.pop(rid, None)
        if self.blocked_rid is None:
            return
        self.tried += 1
        self.admitted_behind += 1
        self.passes[self.blocked_rid] = self.passes.get(self.blocked_rid, 0) + 1
        if len(self.passes) > MAX_TRACKED:
            for stale in list(self.passes)[: MAX_TRACKED // 2]:
                del self.passes[stale]

    def end(self) -> bool:
        """Whether a request was refused for room in this walk, after which the walk may
        have gone on; logs the requests admitted behind it."""
        if self.blocked_rid is None:
            return False
        if self.admitted_behind > 0:
            logger.info(
                "HICACHE_EVT pass_over head=%s admitted=%d tried=%d passes=%d",
                self.blocked_rid,
                self.admitted_behind,
                self.tried,
                self.passes.get(self.blocked_rid, 0),
            )
        return True
