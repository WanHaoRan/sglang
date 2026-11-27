"""Review check (rank agreement): what a later PP rank's own short write counts for.

Throwaway: copied into the copy's test folder under this name, run, deleted.
"""

import unittest
from unittest import mock

import torch
from test_storage_read_dispatch import (
    PAGE,
    drain,
    finish_writes,
    stored_tracking_cache,
)


class ReviewPPVerdict(unittest.TestCase):
    def test_a_write_cut_short_on_pp_rank_1_only_is_marked_stored(self):
        cache = stored_tracking_cache({1: 4})
        cache.pp_size, cache.pp_rank = 2, 1
        cache.pp_group = object()
        cache.work_list = []
        received = []

        def recv_from_pp0(data, **kwargs):
            # PP rank 0 wrote all its pages and sends its verdict down the pipeline.
            received.append(data.tolist())
            data.fill_(1)

        cache.write_backup_storage(1)
        # This rank (PP 1) wrote one token short of the op's four pages.
        finish_writes(cache, 4 * PAGE - 1)
        with mock.patch.object(torch.distributed, "recv", recv_from_pp0):
            drain(cache, 0, write_acks=1)
        print("\nlocal vote before the PP sync:", received, "-> finished:", cache.tree_core.finished)
        self.assertEqual(received, [[0]])
        # "On every rank" would give False here.
        self.assertFalse(cache.tree_core.finished[-1][2])


if __name__ == "__main__":
    unittest.main()
