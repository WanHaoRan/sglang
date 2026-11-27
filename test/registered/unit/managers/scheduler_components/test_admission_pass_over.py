import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from sglang.srt.managers.scheduler_components.admission_pass_over import (  # noqa: E402
    MAX_PASSES,
    MAX_TRACKED,
    AdmissionPassOver,
)

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


def walk(pass_over, queue, fits):
    """The scheduler's walk over `queue` (rids in order): a rid in `fits` is admitted,
    any other is refused for room. Returns (admitted rids, whether a request was refused
    for room)."""
    admitted = []
    pass_over.begin()
    for rid in queue:
        if pass_over.exhausted:
            break
        if rid in fits:
            admitted.append(rid)
            pass_over.admitted(rid)
            continue
        if not pass_over.refused(rid):
            break
    return admitted, pass_over.end()


class TestAdmissionPassOver(CustomTestCase):
    def test_without_a_lookahead_the_walk_stops_at_the_first_refusal(self):
        pass_over = AdmissionPassOver(lookahead=0)
        admitted, refused = walk(pass_over, ["a", "big", "b"], fits={"a", "b"})
        self.assertEqual((admitted, refused), (["a"], False))
        self.assertEqual(pass_over.passes, {})

    def test_a_later_request_that_fits_is_admitted_behind_a_refused_one(self):
        pass_over = AdmissionPassOver(lookahead=4)
        admitted, refused = walk(
            pass_over, ["a", "big", "b", "big2", "c"], fits={"a", "b", "c"}
        )
        self.assertEqual((admitted, refused), (["a", "b", "c"], True))
        # Only the first refused request is the one passed.
        self.assertEqual(pass_over.passes, {"big": 2})

    def test_the_lookahead_bounds_the_candidates_tried_behind_a_refusal(self):
        pass_over = AdmissionPassOver(lookahead=2)
        admitted, refused = walk(pass_over, ["big", "x", "y", "fits"], fits={"fits"})
        self.assertEqual((admitted, refused), ([], True))
        admitted, _ = walk(
            pass_over, ["big", "x", "fits", "more"], fits={"fits", "more"}
        )
        self.assertEqual(admitted, ["fits"])

    def test_once_max_passes_requests_went_ahead_the_walk_stops_until_it_is_admitted(
        self,
    ):
        """The bound is MAX_PASSES admissions behind one refused request, whatever the
        lookahead, also when they fall in one walk."""
        for lookahead in (1, 3, 5, MAX_PASSES + 4):
            with self.subTest(lookahead=lookahead):
                pass_over = AdmissionPassOver(lookahead=lookahead)
                admitted = []
                for i in range(2 * MAX_PASSES):
                    smalls = [f"s{i}-{j}" for j in range(lookahead)]
                    admitted += walk(pass_over, ["big", *smalls], fits=set(smalls))[0]
                self.assertEqual(len(admitted), MAX_PASSES)
                self.assertEqual(pass_over.passes, {"big": MAX_PASSES})
                admitted, refused = walk(pass_over, ["big", "late"], fits={"late"})
                self.assertEqual((admitted, refused), ([], True))
                # Once admitted its count is gone: a request of the same rid starts afresh.
                admitted, _ = walk(pass_over, ["big", "late"], fits={"big", "late"})
                self.assertEqual(admitted, ["big", "late"])
                self.assertNotIn("big", pass_over.passes)

    def test_a_walk_with_no_refusal_changes_nothing(self):
        pass_over = AdmissionPassOver(lookahead=4)
        admitted, refused = walk(pass_over, ["a", "b"], fits={"a", "b"})
        self.assertEqual((admitted, refused, pass_over.passes), (["a", "b"], False, {}))

    def test_create_turns_the_pass_over_off_where_a_refusal_has_state_to_undo(self):
        on = dict(
            single_rank=True,
            disaggregation_mode=DisaggregationMode.NULL,
            priority_scheduling=False,
            lora=False,
            dllm=False,
            hybrid_ssm=False,
            radix_cache=True,
        )
        self.assertEqual(
            AdmissionPassOver.create(requested_lookahead=8, **on).lookahead, 8
        )
        unsupported = dict(
            single_rank=False,
            disaggregation_mode=DisaggregationMode.DECODE,
            priority_scheduling=True,
            lora=True,
            dllm=True,
            hybrid_ssm=True,
            radix_cache=False,
        )
        for name, value in unsupported.items():
            with self.subTest(**{name: value}):
                with self.assertLogs(
                    "sglang.srt.managers.scheduler_components.admission_pass_over",
                    "WARNING",
                ):
                    pass_over = AdmissionPassOver.create(
                        requested_lookahead=8, **{**on, name: value}
                    )
                self.assertEqual(pass_over.lookahead, 0)

    def test_counts_of_requests_that_never_came_back_are_dropped_oldest_first(self):
        pass_over = AdmissionPassOver(lookahead=1)
        for i in range(MAX_TRACKED + 1):
            walk(pass_over, [f"gone{i}", f"s{i}"], fits={f"s{i}"})
        self.assertLessEqual(len(pass_over.passes), MAX_TRACKED)
        self.assertNotIn("gone0", pass_over.passes)
        self.assertIn(f"gone{MAX_TRACKED}", pass_over.passes)


if __name__ == "__main__":
    unittest.main()
