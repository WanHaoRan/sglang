from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import msgspec

from sglang.srt.disaggregation.utils import DisaggregationMode

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

logger = logging.getLogger(__name__)

# Fewest tokens a batch loads from host memory for it to be held. Measured on an H200:
# 2.5 us a token loaded and 24 ms a decode step, so a smaller copy ends within one step.
MIN_LOAD_TOKENS = 8192


class PrefillLoadHold(msgspec.Struct):
    """Keeps a prefill batch back while its prefix is copied from host memory to the GPU.

    The prefill forward waits on that copy layer by layer, and nothing decodes meanwhile.
    A held batch lets the running requests decode beside the copy; it runs once the copy
    has landed, after max_steps more decode steps, or when nothing is left to decode.
    The pass that holds a batch runs one decode step itself, which steps does not count.
    Time stats count the hold as prefill time: forward_entry_time stays the admission
    stamp, which the next-turn prefetcher's admission estimates need.
    """

    # Most decode steps after the holding pass that one batch is held; 0 never holds.
    max_steps: int
    held: Optional[ScheduleBatch] = None
    steps: int = 0

    @classmethod
    def create(
        cls,
        *,
        requested_steps: int,
        hierarchical_cache: bool,
        tp_size: int,
        pp_size: int,
        dp_attention: bool,
        disaggregation_mode: DisaggregationMode,
        speculative: bool,
        dllm: bool,
        hisparse: bool,
    ) -> PrefillLoadHold:
        unsupported = [
            name
            for name, unmet in (
                ("the hierarchical cache", not hierarchical_cache),
                # Whether a copy has landed is local to a rank; ranks must run one batch.
                (
                    "one rank (TP = PP = 1, no DP attention)",
                    tp_size != 1 or pp_size != 1 or dp_attention,
                ),
                # Their batch hand-over between passes is not modelled by the hold.
                (
                    "no disaggregation",
                    disaggregation_mode != DisaggregationMode.NULL,
                ),
                ("no speculative decoding", speculative),
                ("no dLLM", dllm),
                ("no HiSparse", hisparse),
            )
            if unmet
        ]
        if requested_steps > 0 and unsupported:
            logger.warning(
                "SGLANG_HICACHE_PREFILL_LOAD_HOLD_STEPS is ignored: it needs %s.",
                ", ".join(unsupported),
            )
        return cls(max_steps=0 if unsupported else requested_steps)

    def should_hold(self, *, load_tokens: int, decoding: bool, chunked: bool) -> bool:
        # load_tokens: what this batch's own copy moves. chunked: a chunked prefill is in
        # progress, whose chunk the next pass would cache as computed.
        return (
            self.max_steps > 0
            and load_tokens >= MIN_LOAD_TOKENS
            and decoding
            and not chunked
        )

    def hold(self, batch: ScheduleBatch) -> None:
        self.held = batch
        self.steps = 0

    def take(self, *, load_done: bool, decoding: bool) -> Optional[ScheduleBatch]:
        """The held batch when it should run now; None while it stays held."""
        if not load_done and decoding and self.steps < self.max_steps:
            self.steps += 1
            return None
        batch, self.held = self.held, None
        logger.info(
            "HICACHE_EVT load_hold steps=%d landed=%d", self.steps, int(load_done)
        )
        return batch
