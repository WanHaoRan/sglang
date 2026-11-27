"""Fixer's check (notes folder, not an engine test): which regressions do the unit tests of
copy A let through after the fixes?

Each regression is applied in memory only: one function's source is re-compiled with
snippets replaced and bound in place of the original for the run, then restored. No file of
the copy is edited. For every regression the five test modules of the change run in this
process; SURVIVED means no test failed.

Four groups:
  PRE   the behaviour before a fix of this round (the reviewers' findings): must be KILLED.
  ENG   the engineer's 33 mutants (checks/mutation_check.py), ported to the present text.
  OFF   the default-off reviewer's mutants (review_default_off/make_mutants.py).
  GAP   the style reviewer's mutants (review_style_tests/zz_review_style_mutations*.py).
A mutant marked EQUIVALENT changes no behaviour and one marked UNPINNED a corner left
without a test; both are expected to survive, and the reason is beside each.

Run (from the host, CPU only, GPU hidden, niced):
  COPY=/sgl-workspace/sglang/agent_cache/frozen/impl_unified_A
  docker exec -e PYTHONPATH=$COPY/python:$COPY/test/registered/unit/mem_cache \
      -e CUDA_VISIBLE_DEVICES=9 -e PYTHONDONTWRITEBYTECODE=1 sglang_hicache bash -c \
      "cd $COPY && nice -n 10 python3 -m pytest -q -s -p no:cacheprovider --rootdir $COPY \
       /sgl-workspace/sglang/agent_cache/results/design_unified_1126/impl/A/fix/zz_fix_mutations.py"
Set FIX_MUTATIONS=PRE,OFF (group names or label prefixes) to run a part.
"""

import __future__

import contextlib
import inspect
import io
import logging
import os
import sys
import unittest

import test_next_turn_estimators
import test_next_turn_prefetch
import test_next_use_eviction
import test_storage_read_dispatch
import test_storage_read_scheduler

from sglang.srt.mem_cache import (
    next_turn_meter,
    next_turn_prefetch,
    next_turn_prices,
    next_turn_switches,
    next_use_eviction,
    storage_read_scheduler,
    unified_radix_cache,
)
from sglang.srt.mem_cache.unified_cache.components import full

TEST_MODULES = (
    test_next_turn_prefetch,
    test_next_turn_estimators,
    test_next_use_eviction,
    test_storage_read_dispatch,
    test_storage_read_scheduler,
)

P = next_turn_prefetch.NextTurnPrefetcher
M = next_turn_meter.PriceMeter
S = next_turn_switches.Switches
BOARD = next_turn_prices.PriceBoard
FIT = next_turn_prices.PrefillCostModel
TRACKER = next_turn_prices.AdmissionErrorTracker
NU = next_use_eviction.NextUseStrategy
RC = next_use_eviction.ReturnChanceStrategy
SRS = storage_read_scheduler.StorageReadScheduler
CACHE = unified_radix_cache.UnifiedRadixCache
FULL = full.FullComponent

EQUIVALENT = "EQUIVALENT"
UNPINNED = "UNPINNED"
EXPECTED_TO_SURVIVE = (EQUIVALENT, UNPINNED)

# (label, owner, function name, [(snippet, replacement), ...]); a label that starts with
# EQUIVALENT or UNPINNED is expected to survive.
MUTATIONS = [
    # ---- PRE: the behaviour before this round's fixes ----
    (
        "PRE L2 pricing a victim ranks it as eviction does (holding_rate adopts)",
        RC,
        "holding_rate",
        [("self._session_registration(node, adopt=False)", "self._session_registration(node)")],
    ),
    (
        "PRE L2 naming the next victim ranks it as eviction does (get_priority)",
        FULL,
        "_peek_host_priority",
        [("eviction_strategy.peek_priority(node)", "eviction_strategy.get_priority(node)")],
    ),
    (
        "PRE L2 the next victim ignores session refs",
        FULL,
        "_peek_host_priority",
        [("        if self.tree_core.enable_session_radix_cache:", "        if False:")],
    ),
    (
        "PRE L3 the starved clock is not restarted after a drop",
        P,
        "_relieve_starved_gpu",
        [("                    self._starved_since = now\n", "")],
    ),
    (
        "PRE L4 the starved rule runs before this pass's arrivals are registered",
        P,
        "tick",
        [
            (
                "        starved = self._recompute_when_starved and self._gpu_starved(\n"
                "            waiting_queue, now=now\n"
                "        )\n",
                "        starved = False\n"
                "        if self._recompute_when_starved and self._gpu_starved(\n"
                "            waiting_queue, now=now\n"
                "        ):\n"
                "            self._relieve_starved_gpu(waiting_queue, now=now)\n",
            )
        ],
    ),
    (
        "PRE L5 the gate keeps the absent mark when the match grew past it",
        P,
        "prefetch_gate",
        [
            (
                "        if plan.absent_from is not None and matched_tokens > plan.absent_from:\n"
                "            # The match reaches past where the last lookup stopped: that lookup says\n"
                "            # nothing of what storage holds beyond.\n"
                "            plan.absent_from = None\n"
                "        # Storage had nothing from the mark on; a match below it leaves the span up to it.\n"
                "        chain = history if plan.absent_from is None else min(history, plan.absent_from)\n",
                "        chain = history\n"
                "        if plan.absent_from is not None:\n"
                "            chain = min(history, max(plan.absent_from, matched_tokens))\n",
            )
        ],
    ),
    (
        "PRE L5 the queue check keeps the absent mark when memory grew past it",
        P,
        "_probe_missing",
        [
            (
                "        if plan.absent_from is not None and resident > plan.absent_from:\n",
                "        if False:\n",
            )
        ],
    ),
    (
        "PRE L6 turns that found the queue empty feed the admission error",
        M,
        "observe_admission",
        [("        elif predicted > 0.0:\n", "        else:\n")],
    ),
    (
        "PRE L7 the frontier's restore cost holds kappa's pending-write term",
        M,
        "restore_s_per_token",
        [("board.kappa_of_reads(now)", "board.kappa(now)")],
    ),
    (
        "PRE L8 a queued victim sets no price at the frontier",
        M,
        "on_host_evicted",
        [("holding_rate(node, queued=self._queued_rate)", "holding_rate(node)")],
    ),
    (
        "PRE L8 a queued victim is priced at the top rate whatever the order",
        M,
        "follow",
        [("                self._holding_strategy.queued_order,\n", '                "lru",\n')],
    ),
    (
        "PRE L8 measured prices accepted under an eviction policy with no holding rates",
        S,
        "_unmet",
        [("            and not priced_eviction\n", "            and False\n")],
    ),
    (
        "PRE D4 the lateness weight acts in mode observe",
        S,
        "read",
        [("                acting\n                and staging_by_cost\n", "                staging_by_cost\n")],
    ),
    (
        "PRE D4 the read line acts in mode observe",
        S,
        "read",
        [("                acting and parking and envs", "                parking and envs")],
    ),
    (
        "PRE L1 the queue check counts the line for every queued turn on every pass",
        P,
        "_check_queued",
        [
            (
                "        load_s = self._load_seconds(plan.missing_tokens, backlog_pages)\n",
                "        load_s, backlog_pages = self._line_seconds(\n"
                "            plan.missing_tokens,\n"
                "            deadline=None if position <= 1 else admission_at,\n"
                "            backlog_pages=backlog_pages,\n"
                "            now=now,\n"
                "        )\n",
            ),
            ("            if self._read_line:\n", "            if False:\n"),
        ],
    ),
    (
        "PRE L1 a turn's own parked read is counted in its line",
        P,
        "_queued_line_seconds",
        [("        if cache.read_scheduler.is_parked(handle):\n", "        if False:\n")],
    ),
    # ---- OFF: the default-off reviewer's mutants ----
    (
        "OFF control: ntp_admit gains its two fields with every switch off",
        P,
        "_on_departure",
        [
            (
                "        measured = {}\n        if self._prices is not None:",
                '        measured = {"reader": int(reader), "land_ms": "na"}\n'
                "        if self._prices is not None:",
            )
        ],
    ),
    (
        "OFF ec_start gains c= with every switch off",
        P,
        "_log_start_decision",
        [
            (
                'weighed = {"c": f"{stall_weight:.2f}"} if self._lateness_weight else {}',
                'weighed = {"c": f"{stall_weight:.2f}"}',
            )
        ],
    ),
    (
        "OFF ntp_hold gains margin_ms= and q= with every switch off",
        P,
        "read_not_before",
        [
            (
                "margin_s, by_quantile = NTP_HOLD_MARGIN_S, {}",
                "margin_s, by_quantile = NTP_HOLD_MARGIN_S, "
                '{"margin_ms": _ms(NTP_HOLD_MARGIN_S), "q": "0.00"}',
            )
        ],
    ),
    (
        "OFF with only the log on, a staged read's stall is weighed by c",
        P,
        "_staged_start",
        [
            (
                "stall_weight = self._prices.c(now) if self._lateness_weight else 1.0",
                "stall_weight = self._prices.c(now) if self._prices is not None else 1.0",
            )
        ],
    ),
    (
        "OFF with only the log on, a refused read's harm is priced on the meter",
        P,
        "_follow_room",
        [
            (
                "                if self._measured_prices:\n",
                "                if self._prices is not None:\n",
            )
        ],
    ),
    (
        "OFF with only the log on, the hold takes the quantile margin",
        P,
        "read_not_before",
        [("        if self._hold_quantile:\n", "        if self._prices is not None:\n")],
    ),
    (
        "OFF with only the log on, the room price becomes lam",
        P,
        "_memory_price",
        [
            (
                "return self._prices.lam(now) if self._measured_prices else self._price.price",
                "return self._prices.lam(now) if self._prices is not None else self._price.price",
            )
        ],
    ),
    # ---- GAP: the style reviewer's mutants ----
    (
        "GAP G1 a staging ignores the frontier's price lam (victim price only)",
        P,
        "_staging_price",
        [
            (
                "return max(price, self._meter.next_victim_price(load=self._load, now=now))",
                "return self._meter.next_victim_price(load=self._load, now=now)",
            )
        ],
    ),
    (
        "GAP G2 the harm of a refused read keeps the constant per token",
        P,
        "_follow_room",
        [
            (
                "self._meter.prefill_cost.seconds(\n"
                "                        new_tokens=lost_tokens, total_tokens=lost_tokens\n"
                "                    )",
                "(lost_tokens * RECOMPUTE_S_PER_TOKEN)",
            )
        ],
    ),
    (
        "GAP G5a a timed read's estimate never adds what is left of the write hold",
        P,
        "_seconds_in_line",
        [("        if timed:\n", "        if False:\n")],
    ),
    (
        "GAP G5b a read needed now also waits for the write hold in the estimate",
        P,
        "_seconds_in_line",
        [("        if timed:\n", "        if True:\n")],
    ),
    (
        "GAP G5c the starved-GPU rule never adds the write hold to a timed read's landing",
        P,
        "_relieve_starved_gpu",
        [("timed=self.read_deadline(handle) is not None,", "timed=False,")],
    ),
    (
        "GAP G6 the queue check leaves the read line out of ntp_due",
        P,
        "_check_queued",
        [("            if self._read_line:\n", "            if False:\n")],
    ),
    (
        f"{EQUIVALENT} GAP G7a a free room retry keeps an earlier lookup marked as charged: "
        "the retry needs room_refused, set after a found answer (which clears the mark) and "
        "cleared by any later charged call",
        P,
        "prefetch_gate",
        [
            (
                "            self._take_room_retry(plan)\n            plan.lookup_charged = False\n",
                "            self._take_room_retry(plan)\n",
            )
        ],
    ),
    (
        "GAP G7b a room retry re-issued by the queue check is marked as a charged refetch",
        P,
        "on_refetch_result",
        [("plan.refetch_charged = not decision.room_retry", "plan.refetch_charged = True")],
    ),
    (
        "GAP G9 a hit under the prefetch threshold counts as found",
        P,
        "on_lookup_result",
        [
            (
                "found = hit_tokens >= self._tree_cache.prefetch_threshold",
                "found = hit_tokens > 0",
            )
        ],
    ),
    (
        "GAP G10a the prices line carries the hold counters as totals",
        M,
        "_read_path_fields",
        [
            (
                "**{name: count - before for name, count, before in zip(names, counts, logged)},",
                "**{name: count for name, count, before in zip(names, counts, logged)},",
            )
        ],
    ),
    (
        "GAP G10b the prices line drops the hold counters and the measured lags",
        M,
        "_price_fields",
        [("            **self._read_path_fields(),\n", "")],
    ),
    (
        "GAP G12a the starved-GPU rule counts ready turns with the 0.5 s debounce",
        P,
        "_gpu_starved",
        [
            (
                "if not waiting_queue or self._meter.ready_since:",
                "if not waiting_queue or self._prices.ready_waiting > 0:",
            )
        ],
    ),
    (
        "GAP G12b the queue check keeps re-issuing the prefetch of a turn that gave its read up",
        P,
        "_check_queued",
        [("        if plan.recompute:\n            return False\n", "")],
    ),
    (
        "GAP G12c the starved-GPU rule acts on the seeded lags",
        P,
        "_gpu_starved",
        [
            (
                "return now - self._starved_since >= NTP_GPU_BOUND_S and self._line_measured",
                "return now - self._starved_since >= NTP_GPU_BOUND_S",
            )
        ],
    ),
    (
        "GAP G16 a landing adopts into a strategy that is not there",
        P,
        "on_read_landed",
        [("if plan is None or self._return_chance is None:", "if plan is None:")],
    ),
    (
        "GAP G17 landing times are kept for every request",
        P,
        "on_read_landed",
        [
            (
                "if self._prices is not None and (plan is not None or handle.rid in self._queued):",
                "if self._prices is not None:",
            )
        ],
    ),
    (
        "GAP G18 queued turns with no plan get no predicted wait",
        P,
        "tick",
        [
            (
                "if self._prices is not None and req.rid not in self._pred_wait:",
                "if False:",
            )
        ],
    ),
    (
        "GAP G19 a reader whose landing was not seen feeds the error of the GPU turn",
        M,
        "observe_admission",
        [
            (
                "        if reader:\n            if landed_after is not None:\n"
                "                self.admission_error.observe(\n"
                "                    predicted_wait_s=predicted,\n"
                "                    wait_s=wait,\n"
                "                    landed_after_arrival_s=landed_after,\n"
                "                )\n",
                "        if reader and landed_after is not None:\n"
                "            self.admission_error.observe(\n"
                "                predicted_wait_s=predicted,\n"
                "                wait_s=wait,\n"
                "                landed_after_arrival_s=landed_after,\n"
                "            )\n",
            )
        ],
    ),
    (
        "GAP G21 the next victim is priced once and never again",
        M,
        "next_victim_price",
        [
            (
                "if now - self._victim_priced_at >= VICTIM_PRICE_INTERVAL_S:",
                'if self._victim_priced_at == float("-inf"):',
            )
        ],
    ),
    (
        "GAP G22 measured prices accepted with price learning on",
        S,
        "_unmet",
        [
            (
                'if self.measured_prices and price_learning != "off":',
                'if self.measured_prices and price_learning == "harm":',
            )
        ],
    ),
    (
        "GAP G25 the board never gets the admission rate (mu_for loses the arrivals)",
        M,
        "follow",
        [("arrival_rate=0.0 if rate is None else rate.per_s,", "arrival_rate=0.0,")],
    ),
    (
        "GAP G28 a staged read's line counts only the reads needed now",
        P,
        "_staged_start",
        [
            (
                "deadline=self.read_deadline(_handle(plan)) if self._read_line else None,",
                "deadline=None,",
            )
        ],
    ),
    (
        "GAP G39 an aborted request leaving the queue counts as an admission in c",
        P,
        "_on_departure",
        [
            (
                '                    "ntp_expire", req.rid, staged=int(plan.staged), reason="abort"\n'
                "                )\n"
                "            return 0\n",
                '                    "ntp_expire", req.rid, staged=int(plan.staged), reason="abort"\n'
                "                )\n"
                "            return 1\n",
            )
        ],
    ),
    (
        f"{UNPINNED} GAP N2 a refunded lookup leaves the turn capped: differs only when a "
        "refetch's lookup was terminated (no answer, its mark stays), another trigger's "
        "lookup then answers empty, and the match later grows past the absent mark",
        P,
        "on_lookup_result",
        [
            (
                "                plan.free_lookups += 1\n                plan.refetch_capped = False\n",
                "                plan.free_lookups += 1\n",
            )
        ],
    ),
    (
        "GAP N4 the shared length is computed even when the match is past the key",
        P,
        "_history_tokens",
        [
            (
                "            and new_input_tokens is not None\n            and matched_tokens < len(key)\n",
                "            and new_input_tokens is not None\n",
            )
        ],
    ),
    (
        "GAP X3 the ntp_lookup line is not logged",
        P,
        "on_lookup_result",
        [
            (
                '        _log_event(\n            "ntp_lookup",\n            handle.rid,\n'
                "            start=storage_start,\n            asked=requested_tokens,\n"
                "            hit=hit_tokens,\n            refunded=int(refunded),\n"
                "            lookups=plan.lookups,\n        )\n",
                "        pass\n",
            )
        ],
    ),
    (
        "GAP G43 the prices line carries no mu",
        M,
        "_price_fields",
        [('            mu=f"{board.mu(now):.2f}",\n', "")],
    ),
    (
        "GAP G44 the prices line carries no kappa",
        M,
        "_price_fields",
        [('            kappa=f"{board.kappa(now):.3f}",\n', "")],
    ),
    (
        "GAP G45 ntp_due logs the launched pages, not the line, for a pending plan",
        P,
        "_check_pending_deadline",
        [
            (
                "            load_s, backlog_pages = self._line_seconds(",
                "            load_s, _ = self._line_seconds(",
            )
        ],
    ),
    (
        f"{EQUIVALENT} GAP G49 an unpriced victim resets the frontier's highest rate: "
        "with no rate kept yet, assigning None changes nothing",
        M,
        "on_host_evicted",
        [
            (
                "if rate is not None and (self._evicted_rate is None or rate > self._evicted_rate):",
                "if self._evicted_rate is None or (rate is not None and rate > self._evicted_rate):",
            )
        ],
    ),
    (
        "GAP B3a the gate's read time for a recompute quote leaves out the line ahead",
        P,
        "_recompute_quote",
        [
            (
                "read_s, _ = self._line_seconds(missing, deadline=None, backlog_pages=0, now=now)",
                "read_s = self._line_load.seconds(own_pages=math.ceil(missing / self._page_size), backlog_pages=0)",
            )
        ],
    ),
    (
        "GAP B3n the starved clock is not reset when a ready turn appears",
        P,
        "_gpu_starved",
        [
            (
                "        if not waiting_queue or self._meter.ready_since:\n"
                "            self._starved_since = None\n            return False\n",
                "        if not waiting_queue or self._meter.ready_since:\n            return False\n",
            )
        ],
    ),
    (
        "GAP B3v the write hold is doubled in a timed read's estimate",
        P,
        "_seconds_in_line",
        [
            (
                "            seconds += self._tree_cache.read_lags.write_hold_left_s(now)\n",
                "            seconds += 2 * self._tree_cache.read_lags.write_hold_left_s(now)\n",
            )
        ],
    ),
    (
        "GAP Q1 next_use: a registration queued past QUEUED_MAX_AGE_S still gets the queued rate",
        NU,
        "holding_rate",
        [("            return None if stale else queued\n", "            return queued\n")],
    ),
    (
        "GAP Q2 return_chance: a registration queued past QUEUED_MAX_AGE_S still gets the queued rate",
        RC,
        "holding_rate",
        [("            return None if stale else queued\n", "            return queued\n")],
    ),
    (
        "GAP Q3 the prefill fit never forgets",
        FIT,
        "observe",
        [("        fade = self.forget\n", "        fade = 1.0\n")],
    ),
    (
        "GAP Q13 the next victim's price has no ceiling",
        M,
        "next_victim_price",
        [
            (
                "                else min(\n"
                "                    LAM_CEILING, rate * self.restore_s_per_token(load=load, now=now)\n"
                "                )\n",
                "                else rate * self.restore_s_per_token(load=load, now=now)\n",
            )
        ],
    ),
    # ---- ENG: the engineer's 33, on the present text ----
    ("ENG M1 c ignores ready turns left", M, "follow", [("ready_left=ready_waiting > 0", "ready_left=False")]),
    (
        "ENG M2 waiting leaves out unplanned queued turns",
        P,
        "_follow_prices",
        [
            (
                "            sessions_waiting=len(self._plans)\n"
                "            + sum(req.rid not in self._plans for req in waiting_queue),",
                "            sessions_waiting=len(self._plans),",
            )
        ],
    ),
    (
        "ENG M3 late reads by deadline alone",
        M,
        "restore_s_per_token",
        [
            (
                "late += deadline is None or deadline - own_s <= now",
                "late += deadline is None or deadline <= now",
            )
        ],
    ),
    (
        "ENG M4 frontier takes the lowest rate",
        M,
        "on_host_evicted",
        [("rate > self._evicted_rate", "rate < self._evicted_rate")],
    ),
    (
        "ENG M5 prefill fit fed by queued turns",
        M,
        "observe_prefill",
        [("or not 0.0 <= waited < at_once_s", "or False")],
    ),
    (
        "ENG M6 readers feed the admission error",
        M,
        "observe_admission",
        [("        if reader:\n", "        if False:\n")],
    ),
    (
        "ENG M7 staged lookup sets the shared length",
        P,
        "_history_tokens",
        [
            (
                "if not self._free_empty_lookups or plan.staging_call:",
                "if not self._free_empty_lookups:",
            )
        ],
    ),
    (
        "ENG M8 empty answer keeps the refetch charged",
        P,
        "on_lookup_result",
        [
            (
                "            if plan.refetch_charged:\n                plan.refetches -= 1\n"
                "                refunded = True\n",
                "",
            )
        ],
    ),
    (
        "ENG M9 gate gets the prompt cut to the known hit",
        CACHE,
        "prefetch_from_storage",
        [
            (
                "request, matched_tokens=storage_start, prompt_tail=prompt_tail",
                "request, matched_tokens=storage_start, prompt_tail=new_input_tokens",
            )
        ],
    ),
    (
        "ENG M10 terminated lookups are reported",
        CACHE,
        "_report_storage_lookup",
        [
            (
                "if self.prefetch_lookup_fn is None or operation.is_terminated():",
                "if self.prefetch_lookup_fn is None:",
            )
        ],
    ),
    (
        "ENG M11 lateness weight never applied",
        P,
        "_staged_start",
        [
            (
                "stall_weight = self._prices.c(now) if self._lateness_weight else 1.0",
                "stall_weight = 1.0",
            )
        ],
    ),
    (
        "ENG M12 staging priced by the frontier alone",
        P,
        "_staging_price",
        [
            (
                "        return max(price, self._meter.next_victim_price(load=self._load, now=now))",
                "        return price",
            )
        ],
    ),
    (
        "ENG M13 line pages not counted",
        P,
        "_line_seconds",
        [
            (
                "        ahead_pages = self._tree_cache.storage_read_line_pages(deadline)\n",
                "        ahead_pages = backlog_pages\n",
            )
        ],
    ),
    (
        "ENG M14 just-in-time counts the line at saturation",
        P,
        "_check_pending_deadline",
        [("if self._read_line and self._prices.c(now) < READ_LINE_MIN_C:", "if False:")],
    ),
    (
        "ENG M15 hold keeps the seeded lags",
        P,
        "read_not_before",
        [
            (
                "load_s = self._line_load.seconds(own_pages=read.pages, backlog_pages=ahead_pages)",
                "load_s = self._load.seconds(own_pages=read.pages, backlog_pages=ahead_pages)",
            )
        ],
    ),
    (
        "ENG M16 fixed lag charged the pages ahead",
        CACHE,
        "_observe_read_landing",
        [
            (
                "pages = operation.ahead_pages + completed_tokens // self.page_size",
                "pages = completed_tokens // self.page_size",
            )
        ],
    ),
    (
        "ENG M17 hold margin takes the error's sign",
        TRACKER,
        "margin",
        [
            (
                "return fallback_s if error is None else -error",
                "return fallback_s if error is None else error",
            )
        ],
    ),
    (
        "ENG M18 recompute priced behind another turn",
        P,
        "_recompute_quote",
        [("if others > 0 or board.ready_waiting > 0:", "if False:")],
    ),
    (
        "ENG M19 recompute priced on the seeded lags",
        P,
        "_recompute_quote",
        [
            (
                "            # On the load model's seeds a recompute would win at every size.\n"
                "            or not self._line_measured\n",
                "",
            )
        ],
    ),
    (
        "ENG M20 starved rule fires with a ready turn",
        P,
        "_gpu_starved",
        [("if not waiting_queue or self._meter.ready_since:", "if not waiting_queue:")],
    ),
    (
        "ENG M21 starved rule counts reads parked later",
        P,
        "_relieve_starved_gpu",
        [
            (
                "ahead_pages=cache.parked_read_pages_ahead(handle),",
                "ahead_pages=cache.storage_read_line_pages(self.read_deadline(handle)),",
            )
        ],
    ),
    (
        "ENG M22 gate forgets a starved turn recomputes",
        P,
        "prefetch_gate",
        [
            (
                '        if plan.recompute:\n            decision = "recompute"\n'
                "        elif chain - matched_tokens < threshold:",
                "        if chain - matched_tokens < threshold:",
            )
        ],
    ),
    (
        "ENG M23 a dispatched read is dropped",
        CACHE,
        "drop_parked_read",
        [
            (
                "        if self.parked_read_span(request) is None:\n            return False\n"
                "        self.revoke_pending_prefetch(request)",
                "        self.revoke_pending_prefetch(request)",
            )
        ],
    ),
    (
        "ENG M24 mu at saturation is the ready queue",
        BOARD,
        "mu",
        [("        return max(light, float(self.sessions_waiting))", "        return light")],
    ),
    (
        "ENG M25 measured prices decide in observe mode",
        S,
        "read",
        [
            (
                "acting and envs.SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES.get()",
                "envs.SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES.get()",
            )
        ],
    ),
    (
        "ENG M26 quote ignores arrivals during the recompute",
        BOARD,
        "mu_for",
        [
            (
                "return self._mu_light(now) + self.arrival_rate * job_s / 2.0",
                "return self._mu_light(now)",
            )
        ],
    ),
    (
        "ENG M27 observer told after the eviction",
        FULL,
        "drive_host_eviction",
        [
            (
                "            if observer is not None:\n"
                "                observer(x, len(x.component_data[ct].host_value))\n"
                "            self.tree_core._evict_host_leaf(x, tracker, device_frees, host_frees)",
                "            self.tree_core._evict_host_leaf(x, tracker, device_frees, host_frees)\n"
                "            if observer is not None:\n"
                "                observer(x, 0)",
            )
        ],
    ),
    (
        "ENG M28 queued victim priced as unknown",
        NU,
        "holding_rate",
        [("            return None if stale else queued\n", "            return None\n")],
    ),
    (
        "ENG M29 pages due by ignores the deadline",
        SRS,
        "pages_due_by",
        [("if due is None or (deadline is not None and due <= deadline):", "if True:")],
    ),
    (
        "ENG M30 absent mark ignored by the queue check",
        P,
        "_probe_missing",
        [
            (
                "        if plan.absent_from is not None:\n"
                "            chain = min(chain, plan.absent_from)\n",
                "",
            )
        ],
    ),
    (
        "ENG M31 quote prices the key's tail the prompt lacks",
        P,
        "_recompute_quote",
        [("        missing = min(missing, len(new_input_tokens))\n", "")],
    ),
    (
        "ENG M32 read line counts later deadlines for a staged start",
        SRS,
        "pages_due_by",
        [
            (
                "if due is None or (deadline is not None and due <= deadline):",
                "if due is None or deadline is not None:",
            )
        ],
    ),
    (
        "ENG M33 next victim named from the other end of the order",
        FULL,
        "next_host_victim",
        [
            (
                "        _, victim = min((self._peek_host_priority(n), n) for n in leaves)",
                "        _, victim = max((self._peek_host_priority(n), n) for n in leaves)",
            )
        ],
    ),
]


def mutated(owner, name, pairs):
    """The function `name` of `owner` re-compiled with each snippet replaced, in the globals
    of the module that defines it. Snippets carry the file's own indentation."""
    original = owner.__dict__[name]
    function = (
        original.__func__ if isinstance(original, (staticmethod, classmethod)) else original
    )
    function = inspect.unwrap(function)
    source = inspect.getsource(function)
    for old, new in pairs:
        if source.count(old) != 1:
            raise LookupError(f"snippet found {source.count(old)} times in {name}: {old[:50]!r}")
        source = source.replace(old, new)
    if source[0] in " \t":
        source = "if True:\n" + source
    code = compile(
        source,
        f"<mutated {name}>",
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    namespace = {}
    exec(code, sys.modules[owner.__module__].__dict__, namespace)
    made = namespace[name]
    if isinstance(original, staticmethod):
        made = staticmethod(made)
    elif isinstance(original, classmethod):
        made = classmethod(made)
    return original, made


def run_tests():
    """Ids of the tests (and subtests) that fail or error across the five modules."""
    failed = []
    out = io.StringIO()
    previous = logging.root.manager.disable
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        for module in TEST_MODULES:
            suite = unittest.defaultTestLoader.loadTestsFromModule(module)
            result = unittest.TextTestRunner(stream=out, verbosity=0).run(suite)
            failed += [test.id() for test, _ in result.failures + result.errors]
    logging.disable(previous)
    return failed


def test_which_regressions_survive():
    wanted = [part for part in os.environ.get("FIX_MUTATIONS", "").split(",") if part]
    clean = run_tests()
    lines = [f"clean run: {len(clean)} failing {clean[:3]}"]
    unexpected = []
    for label, owner, name, pairs in MUTATIONS:
        survives = label.startswith(EXPECTED_TO_SURVIVE)
        plain = label.split(" ", 1)[1] if survives else label
        if wanted and not any(plain.startswith(part) for part in wanted):
            continue
        try:
            original, made = mutated(owner, name, pairs)
        except LookupError as error:
            lines.append(f"SKIP     {label}: {error}")
            unexpected.append(label)
            continue
        setattr(owner, name, made)
        try:
            failed = run_tests()
        finally:
            setattr(owner, name, original)
        if failed:
            short = sorted({test_id.split(".", 1)[1].split(" ")[0] for test_id in failed})
            lines.append(f"KILLED   {plain}: {len(failed)} failing, e.g. {short[0]}")
            if survives:
                unexpected.append(label)
        else:
            lines.append(f"SURVIVED {label}")
            if not survives:
                unexpected.append(label)
    after = run_tests()
    lines.append(f"clean run after restoring: {len(after)} failing {after[:3]}")
    lines.append(f"unexpected: {len(unexpected)}")
    lines += [f"  UNEXPECTED {label}" for label in unexpected]
    print("\n".join(["", "==== fixer mutation check ====", *lines, "==== end ===="]))
    assert not clean and not after
