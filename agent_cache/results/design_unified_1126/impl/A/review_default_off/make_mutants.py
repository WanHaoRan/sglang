"""Review check (default-off lens), throwaway: writes one-line mutants of the copy's
next_turn_prefetch.py into ./mutants/ (the copy is not edited). Each is run with
zz_review_default_off_mutant.py (REVIEW_MUTANT=<path>), which runs the copy's own
test_next_turn_prefetch.py against it. Results: mutants.out.

Usage: python3 make_mutants.py, then per mutant copy zz_review_default_off_mutant.py into the
copy's test folder, run it with the test recipe plus -e REVIEW_MUTANT=..., and delete the copy.
"""

import os

SOURCE = (
    "/home/wanhr/sglang/agent_cache/frozen/impl_unified_A/python/sglang/srt/mem_cache/"
    "next_turn_prefetch.py"
)
MUTANTS = {
    # Control: ntp_admit gains its two fields with every switch off (caught).
    "ntp_admit_fields_always": (
        "        measured = {}\n        if self._prices is not None:",
        '        measured = {"reader": int(reader), "land_ms": "na"}\n'
        "        if self._prices is not None:",
    ),
    # ec_start gains c= with every switch off (NOT caught).
    "ec_start_c_always": (
        'weighed = {"c": f"{stall_weight:.2f}"} if self._lateness_weight else {}',
        'weighed = {"c": f"{stall_weight:.2f}"}',
    ),
    # ntp_hold gains margin_ms= and q= with every switch off (NOT caught).
    "ntp_hold_margin_always": (
        "margin_s, by_quantile = NTP_HOLD_MARGIN_S, {}",
        "margin_s, by_quantile = NTP_HOLD_MARGIN_S, "
        '{"margin_ms": _ms(NTP_HOLD_MARGIN_S), "q": "0.00"}',
    ),
    # ntp_gate gains a priced field with every switch off (caught, by TestRecomputeByPrice).
    "ntp_gate_priced_always": (
        "        priced = {}\n        if quote is not None:",
        '        priced = {"rec_ms": "na"}\n        if quote is not None:',
    ),
    # With only the meter built (the prices log alone), the room price becomes lam (caught).
    "log_only_room_price": (
        "return self._prices.lam(now) if self._measured_prices else self._price.price",
        "return self._prices.lam(now) if self._prices is not None else self._price.price",
    ),
    # With only the meter built, a staged read's stall is weighed by c (NOT caught).
    "log_only_stall_weight": (
        "stall_weight = self._prices.c(now) if self._lateness_weight else 1.0",
        "stall_weight = self._prices.c(now) if self._prices is not None else 1.0",
    ),
    # With only the meter built, a refused read's harm is priced on the meter (NOT caught).
    "log_only_harm": (
        "                if self._measured_prices:\n"
        "                    harm_s = self._prices.mu(now)",
        "                if self._prices is not None:\n"
        "                    harm_s = self._prices.mu(now)",
    ),
    # With only the meter built, the hold takes the quantile margin (NOT caught).
    "log_only_hold_margin": (
        "        if self._hold_quantile:\n            margin_s, quantile = self._hold_margin(",
        "        if self._prices is not None:\n"
        "            margin_s, quantile = self._hold_margin(",
    ),
}


def main():
    source = open(SOURCE).read()
    os.makedirs("mutants", exist_ok=True)
    for name, (old, new) in MUTANTS.items():
        assert source.count(old) == 1, (name, source.count(old))
        with open(f"mutants/{name}.py", "w") as out:
            out.write(source.replace(old, new))
        print("wrote", name)


if __name__ == "__main__":
    main()
