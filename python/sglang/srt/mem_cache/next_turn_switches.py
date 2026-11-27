"""The env switches of the next-turn prefetcher's priced decisions ("Four actions, three
prices", agent_cache/results/design_unified_1126/), read once per prefetcher.

Only the prices log works in mode observe: every other switch changes a decision and is off
there. In mode on, a switch with nothing to act on in a setup is off as well, so one block of
switches can serve arms that differ in read order or timing. A switch that would act without
something it needs is refused, since it would run on the constants it was made to replace.
"""

from __future__ import annotations

import msgspec

from sglang.srt.environ import envs


class Switches(msgspec.Struct, frozen=True, kw_only=True):
    """Which switched changes act in one prefetcher."""

    # Seconds between HICACHE_EVT prices lines; 0 logs none.
    prices_log_interval_s: float
    measured_prices: bool
    free_empty_lookups: bool
    lateness_weight: bool
    read_line: bool
    hold_quantile: bool
    recompute_by_price: bool
    recompute_when_starved: bool

    @classmethod
    def read(
        cls,
        *,
        acting: bool,
        gated: bool,
        holding: bool,
        parking: bool,
        staging_by_cost: bool,
        price_learning: str,
        priced_eviction: bool,
    ) -> Switches:
        """acting: mode on. gated, holding: acting with the arrival gate, the arrival hold.
        parking: the deadline read order, under which reads park. staging_by_cost: read
        timing expected_cost, which weighs a staged read's stall against its rent.
        priced_eviction: the eviction policy knows what each holder is worth per second."""
        switches = cls(
            prices_log_interval_s=envs.SGLANG_LOG_HICACHE_PRICES_INTERVAL.get(),
            measured_prices=(
                acting and envs.SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES.get()
            ),
            free_empty_lookups=(
                gated and envs.SGLANG_HICACHE_NEXT_TURN_FREE_EMPTY_LOOKUPS.get()
            ),
            lateness_weight=(
                acting
                and staging_by_cost
                and envs.SGLANG_HICACHE_NEXT_TURN_LATENESS_WEIGHT.get()
            ),
            read_line=(
                acting and parking and envs.SGLANG_HICACHE_NEXT_TURN_READ_LINE.get()
            ),
            hold_quantile=(
                holding and envs.SGLANG_HICACHE_NEXT_TURN_HOLD_QUANTILE.get()
            ),
            # Both answer through the gate: a turn it declines is admitted without a read.
            recompute_by_price=(
                gated and envs.SGLANG_HICACHE_NEXT_TURN_RECOMPUTE_BY_PRICE.get()
            ),
            recompute_when_starved=(
                gated and envs.SGLANG_HICACHE_NEXT_TURN_RECOMPUTE_WHEN_STARVED.get()
            ),
        )
        unmet = switches._unmet(
            staging_by_cost=staging_by_cost,
            price_learning=price_learning,
            priced_eviction=priced_eviction,
        )
        if unmet:
            raise ValueError("; ".join(unmet))
        return switches

    @property
    def measuring(self) -> bool:
        """The prices are measured for the log and for every switch that reads one."""
        return (
            self.prices_log_interval_s > 0
            or self.measured_prices
            or self.lateness_weight
            or self.read_line
        )

    def _unmet(
        self, *, staging_by_cost: bool, price_learning: str, priced_eviction: bool
    ) -> list[str]:
        unmet = []
        if self.measured_prices and price_learning != "off":
            unmet.append(
                "SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES replaces the learned price: "
                "it needs --hicache-storage-next-turn-price-learning off"
            )
        if (
            self.measured_prices
            and not priced_eviction
            and (staging_by_cost or self.hold_quantile)
        ):
            # The room's price is read off host eviction's victims: under a policy with
            # no forecast it is 0 for ever, and a staged or held read would pay no rent.
            unmet.append(
                "SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES prices host room for staged "
                "and held reads off the eviction policy: it needs "
                "--radix-eviction-policy next_use or return_chance"
            )
        if self.hold_quantile and not self.measured_prices:
            # At the knob's 5e-5 a 20K-token read lands at the median, late half the time.
            unmet.append(
                "SGLANG_HICACHE_NEXT_TURN_HOLD_QUANTILE needs "
                "SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES"
            )
        if self.read_line and staging_by_cost and not self.lateness_weight:
            # With the stall at full weight the line turns late-run decisions into starts.
            unmet.append(
                "SGLANG_HICACHE_NEXT_TURN_READ_LINE under read timing expected_cost "
                "needs SGLANG_HICACHE_NEXT_TURN_LATENESS_WEIGHT"
            )
        # On the seeded lags a recompute beats a read at every size, and without the skip
        # a recomputed history is written to the SSD again.
        needs = {
            "SGLANG_HICACHE_NEXT_TURN_MEASURED_PRICES": self.measured_prices,
            # Set but off without parked reads, so the order it acts under is named too.
            "SGLANG_HICACHE_NEXT_TURN_READ_LINE (under the deadline read order)": (
                self.read_line
            ),
            "SGLANG_HICACHE_NIXL_SKIP_STORED_WRITES": (
                envs.SGLANG_HICACHE_NIXL_SKIP_STORED_WRITES.get()
            ),
        }
        missing = ", ".join(need for need, met in needs.items() if not met)
        for name, switched in (
            ("RECOMPUTE_BY_PRICE", self.recompute_by_price),
            ("RECOMPUTE_WHEN_STARVED", self.recompute_when_starved),
        ):
            if switched and missing:
                unmet.append(f"SGLANG_HICACHE_NEXT_TURN_{name} needs {missing}")
        return unmet
