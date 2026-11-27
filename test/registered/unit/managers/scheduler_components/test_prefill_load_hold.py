import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.scheduler_components.prefill_load_hold import (  # noqa: E402
    MIN_LOAD_TOKENS,
    PrefillLoadHold,
)

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


class TestPrefillLoadHold(CustomTestCase):
    def test_a_held_batch_runs_after_the_step_budget_even_if_the_load_never_lands(self):
        hold = PrefillLoadHold(max_steps=3)
        batch = object()
        hold.hold(batch)
        taken = [hold.take(load_done=False, decoding=True) for _ in range(4)]
        self.assertEqual(taken, [None, None, None, batch])
        self.assertIsNone(hold.held)

    def test_a_held_batch_runs_at_once_when_its_load_landed_or_nothing_decodes(self):
        for load_done, decoding in ((True, True), (False, False)):
            with self.subTest(load_done=load_done, decoding=decoding):
                hold = PrefillLoadHold(max_steps=8)
                batch = object()
                hold.hold(batch)
                self.assertIs(hold.take(load_done=load_done, decoding=decoding), batch)

    def test_no_hold_without_decode_work_during_a_chunked_prefill_or_for_a_short_copy(
        self,
    ):
        hold = PrefillLoadHold(max_steps=8)
        big = MIN_LOAD_TOKENS
        self.assertTrue(hold.should_hold(load_tokens=big, decoding=True, chunked=False))
        self.assertFalse(
            hold.should_hold(load_tokens=big, decoding=False, chunked=False)
        )
        self.assertFalse(hold.should_hold(load_tokens=big, decoding=True, chunked=True))
        self.assertFalse(
            hold.should_hold(load_tokens=big - 1, decoding=True, chunked=False)
        )

    def test_the_step_count_starts_again_for_each_batch(self):
        hold = PrefillLoadHold(max_steps=2)
        hold.hold(object())
        for _ in range(3):
            hold.take(load_done=False, decoding=True)
        second = object()
        hold.hold(second)
        self.assertIsNone(hold.take(load_done=False, decoding=True))

    def test_create_disables_the_hold_outside_one_rank_plain_serving(self):
        on = dict(
            hierarchical_cache=True,
            tp_size=1,
            pp_size=1,
            dp_attention=False,
            disaggregation_mode=DisaggregationMode.NULL,
            speculative=False,
            dllm=False,
            hisparse=False,
        )
        self.assertEqual(PrefillLoadHold.create(requested_steps=8, **on).max_steps, 8)
        unsupported = dict(
            hierarchical_cache=False,
            tp_size=2,
            pp_size=2,
            dp_attention=True,
            disaggregation_mode=DisaggregationMode.PREFILL,
            speculative=True,
            dllm=True,
            hisparse=True,
        )
        for name, value in unsupported.items():
            with self.subTest(**{name: value}):
                with self.assertLogs(
                    "sglang.srt.managers.scheduler_components.prefill_load_hold",
                    "WARNING",
                ):
                    hold = PrefillLoadHold.create(
                        requested_steps=8, **{**on, name: value}
                    )
                self.assertEqual(hold.max_steps, 0)
                self.assertFalse(
                    hold.should_hold(
                        load_tokens=MIN_LOAD_TOKENS, decoding=True, chunked=False
                    )
                )


if __name__ == "__main__":
    unittest.main()
