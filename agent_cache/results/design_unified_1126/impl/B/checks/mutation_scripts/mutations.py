"""The mutation checks of the tests for steps 1 to 3 of the Retain plan (copy B): the engineer's 64
(M...), re-anchored where the fix moved their text, and those added with the fixes of 2026-11-26
(F..., T..., S...: one per finding, by the reviewer's lens).
name -> (engine file, text to replace, replacement, test file, pytest -k expression).
Each text occurs exactly once in its file; run with mutate.py."""

MUTATIONS = {'M1 short ack counts as stored': ('python/sglang/srt/mem_cache/storage_retention.py',
                                   '        writes_nothing\n'
                                   '        or operation.completed_tokens >= len(operation.hash_value) * page_size\n',
                                   '        True\n',
                                   'test_storage_read_dispatch',
                                   'acked_short'),
 'M2 marks made with the switch off': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                       '            verdicts = itertools.repeat(None)\n',
                                       '            verdicts = itertools.repeat(True)\n',
                                       'test_storage_read_dispatch',
                                       'switch_off_no_node'),
 'M3 no reduce over ranks': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                             '        if not (synced and votes and self.tp_world_size > 1):\n',
                             '        if True:\n',
                             'test_storage_read_dispatch',
                             'short_on_any_rank'),
 'M3b a rank that writes nothing votes no': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                             '            writes_nothing=self.cache_controller.backup_skip,\n',
                                             '            writes_nothing=False,\n',
                                             'test_storage_read_dispatch',
                                             'short_on_any_rank'),
 'M3c detach path reduces': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                             '        if not (synced and votes and self.tp_world_size > 1):\n',
                             '        if not (votes and self.tp_world_size > 1):\n',
                             'test_storage_read_dispatch',
                             'short_on_any_rank'),
 'M4 detach leaves the op mark': ('python/sglang/srt/mem_cache/unified_cache/storage_attachment.py',
                                  '            cache._forget_storage_copies(detached=True)\n',
                                  '            pass\n',
                                  'test_next_use_eviction',
                                  'clearing_or_detaching'),
 'M5 split does not copy the op': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                   '        new_node.storage_write_op = child.storage_write_op\n',
                                   '',
                                   'test_next_use_eviction',
                                   'split_while_the_write'),
 'M6 read-back pages not stored at birth': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                            '        new_node.storage_stored = True\n',
                                            '',
                                            'test_next_use_eviction',
                                            'child_write_first'),
 'M7 no heal': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                '        for ancestor_id in self.tree_core.unstored_ancestors(node_id):\n'
                '            self._submit_storage_write(ancestor_id)\n',
                '        pass\n',
                'test_next_use_eviction',
                'child_write_first'),
 'M7b heal writes a host copy still in flight': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                                 '                node.write_through_pending_id is None\n'
                                                 '                and not node.storage_stored\n',
                                                 '                not node.storage_stored\n',
                                                 'test_next_use_eviction',
                                                 'child_write_first'),
 'M7c heal ignores a queued write': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                     '                and not node.storage_stored\n'
                                     '                and node.storage_write_op is None\n',
                                     '                and not node.storage_stored\n',
                                     'test_next_use_eviction',
                                     'child_write_first'),
 'M8 clear keeps marks': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                          '            if self.retention.track_stored:\n'
                          '                self._forget_storage_copies(detached=False)\n',
                          '',
                          'test_next_use_eviction',
                          'clearing_or_detaching'),
 'M8b split does not copy stored': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                    '        new_node.storage_stored = child.storage_stored\n',
                                    '',
                                    'test_next_use_eviction',
                                    'clearing_or_detaching'),
 'M9 no entry is an end': ('python/sglang/srt/mem_cache/next_turn_prefetch.py',
                           '        ended = entry is not None and entry.ended\n',
                           '        ended = entry is None or entry.ended\n',
                           'test_next_turn_prefetch',
                           'only_an_explicit_end_ends'),
 'M9b unreadable table on a named turn is an end': ('python/sglang/srt/mem_cache/next_turn_prefetch.py',
                                                    '            return TurnOutlook(session=named_session, '
                                                    'hinted=True)\n',
                                                    '            return TurnOutlook(session=named_session, '
                                                    'hinted=True, end_why=None if named_session is None else '
                                                    '"no_next")\n',
                                                    'test_next_turn_prefetch',
                                                    'only_an_explicit_end_ends or exact_hint_session_ends'),
 'M9c reply mode ends before the reply': ('python/sglang/srt/mem_cache/next_turn_prefetch.py',
                                          '        elif reply_known and req.tokenizer is not None:\n',
                                          '        elif req.tokenizer is not None:\n',
                                          'test_next_turn_prefetch',
                                          'only_an_explicit_end_ends'),
 'M10 un-hinted request is an end': ('python/sglang/srt/mem_cache/next_turn_prefetch.py',
                                     '                end_why="no_next" if exact else None,\n',
                                     '                end_why="no_next",\n',
                                     'test_next_turn_prefetch',
                                     'exact_hint_session_ends'),
 'M11 no entry registers ended': ('python/sglang/srt/mem_cache/next_turn_prefetch.py',
                                  '        if not outlook.hinted and outlook.session is None:\n',
                                  '        if True:\n',
                                  'test_next_turn_prefetch',
                                  'no_usable_entry_is_ranked'),
 'M11b expiry and abort register ended': ('python/sglang/srt/mem_cache/next_turn_prefetch.py',
                                          '        if self._explicit_ends:\n'
                                          '            self._forget_session(rid=rid, session=session, why=why)\n'
                                          '            return\n',
                                          '',
                                          'test_next_turn_prefetch',
                                          'no_usable_entry_is_ranked'),
 'M12 factory forgets the switches': ('python/sglang/srt/mem_cache/next_turn_prefetch.py',
                                      '            explicit_ends=tree_cache.retention.explicit_ends,\n'
                                      '            retire_ended=tree_cache.retention.retire_ended_pages,\n',
                                      '',
                                      'test_next_turn_prefetch',
                                      'factory_follows'),
 'M13 switches granted without their needs': ('python/sglang/srt/mem_cache/storage_retention.py',
                                              '        return wanted and not unmet\n',
                                              '        return wanted\n',
                                              'test_storage_read_dispatch',
                                              'missing_what_it_needs'),
 "M13b cache ignores the store's touch interval": ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                                   '            touch_interval_s = backend.touch_interval_s\n',
                                                   '            touch_interval_s = 0.0\n',
                                                   'test_storage_read_dispatch',
                                                   'two_switches_from_the_envs'),
 'M13c cache reads no env': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                             '            track_stored=envs.SGLANG_HICACHE_STORAGE_TRACK_STORED.get(),\n',
                             '            track_stored=False,\n',
                             'test_storage_read_dispatch',
                             'two_switches_from_the_envs'),
 'M13d a test double counts as a store with a cleaner': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                                         '        if isinstance(backend, HiCacheStorage):\n',
                                                         '        if backend is not None:\n',
                                                         'test_storage_read_dispatch',
                                                         'two_switches_from_the_envs'),
 'M14 a written page is stale at birth': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                          '                node.storage_stored = stored_at is not None\n'
                                          '                if stored_at is not None:\n'
                                          '                    node.storage_touched_at = stored_at\n',
                                          '                node.storage_stored = stored_at is not None\n',
                                          'test_next_use_eviction',
                                          'stale_tail_takes'),
 'M14b only the stale nodes are refreshed': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                             '        for node in reversed(stored):\n'
                                             '            node.storage_touched_at = now\n'
                                             '            page_keys.extend(node.hash_value)\n'
                                             '        return page_keys\n',
                                             '        for node in reversed(stored):\n'
                                             '            if now - node.storage_touched_at >= min_interval_s:\n'
                                             '                node.storage_touched_at = now\n'
                                             '                page_keys.extend(node.hash_value)\n'
                                             '        return page_keys\n',
                                             'test_next_use_eviction',
                                             'stale_tail_takes'),
 'M14c tail first': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                     '        for node in reversed(stored):\n            node.storage_touched_at = now\n',
                     '        for node in stored:\n            node.storage_touched_at = now\n',
                     'test_next_use_eviction',
                     'stale_tail_takes'),
 'M14d refreshed nodes are not stamped': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                          '        for node in reversed(stored):\n'
                                          '            node.storage_touched_at = now\n'
                                          '            page_keys.extend(node.hash_value)\n',
                                          '        for node in reversed(stored):\n'
                                          '            page_keys.extend(node.hash_value)\n',
                                          'test_next_use_eviction',
                                          'stale_tail_takes'),
 'M15 unstored nodes are refreshed too': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                          '            if node.storage_stored and node.hash_value:\n'
                                          '                stored.append(node)\n',
                                          '            if node.hash_value:\n                stored.append(node)\n',
                                          'test_next_use_eviction',
                                          'only_stored_nodes'),
 'M16 a probe counts as a serve': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                   '            self.retention.touch_interval_s > 0.0\n'
                                   '            and params.req is not None\n'
                                   '            and self.enable_storage\n',
                                   '            self.retention.touch_interval_s > 0.0\n'
                                   '            and self.enable_storage\n',
                                   'test_next_use_eviction',
                                   'probe_without_a_request'),
 'M16b refresh with storage detached': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                        '            self.retention.touch_interval_s > 0.0\n'
                                        '            and params.req is not None\n'
                                        '            and self.enable_storage\n',
                                        '            self.retention.touch_interval_s > 0.0\n'
                                        '            and params.req is not None\n',
                                        'test_next_use_eviction',
                                        'probe_without_a_request'),
 'M17 tail stamped newest': ('python/sglang/srt/mem_cache/storage/nixl/nixl_utils.py',
                             '            stamp = 0 if newest_ns is None else newest_ns - index * TOUCH_STEP_NS\n',
                             '            stamp = 0 if newest_ns is None else newest_ns + index * TOUCH_STEP_NS\n',
                             'test_hicache_nixl_cleaner',
                             'refreshed_chain'),
 'M17b every page the same stamp': ('python/sglang/srt/mem_cache/storage/nixl/nixl_utils.py',
                                    '            stamp = 0 if newest_ns is None else newest_ns - index * '
                                    'TOUCH_STEP_NS\n',
                                    '            stamp = 0 if newest_ns is None else newest_ns\n',
                                    'test_hicache_nixl_cleaner',
                                    'refreshed_chain'),
 'M18 a touch creates a missing file': ('python/sglang/srt/mem_cache/storage/nixl/nixl_utils.py',
                                        '                try:\n'
                                        '                    os.utime(path, ns=(stamp, stamp))\n',
                                        '                try:\n'
                                        '                    open(path, "ab").close()\n'
                                        '                    os.utime(path, ns=(stamp, stamp))\n',
                                        'test_hicache_nixl_cleaner',
                                        'deleted_file'),
 'M19 the touch key does not win over the env': ('python/sglang/srt/mem_cache/storage/nixl/nixl_utils.py',
                                                 '        if "touch_interval_s" in self.config:\n'
                                                 '            interval = self.config["touch_interval_s"]\n'
                                                 '            if isinstance(interval, bool) or not '
                                                 'isinstance(interval, (int, float)):\n'
                                                 '                raise ValueError("touch_interval_s must be a '
                                                 'number")\n'
                                                 '        else:\n'
                                                 '            interval = '
                                                 'envs.SGLANG_HICACHE_NIXL_TOUCH_INTERVAL_S.get()\n',
                                                 '        interval = '
                                                 'envs.SGLANG_HICACHE_NIXL_TOUCH_INTERVAL_S.get()\n',
                                                 'test_hicache_nixl_cleaner',
                                                 'touch_switches'),
 'M19b the keys reach the NIXL plugin': ('python/sglang/srt/mem_cache/storage/nixl/nixl_utils.py',
                                         '    "touch_interval_s",\n    "retire_ended_pages",\n}\n',
                                         '}\n',
                                         'test_hicache_nixl_cleaner',
                                         'touch_switches'),
 'M20 a touch reaches K only': ('python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py',
                                '        return [self.file_manager.get_file_path(name) for name in names]\n',
                                '        return [self.file_manager.get_file_path(name) for name in names[:1]]\n',
                                'test_hicache_nixl_storage',
                                'TestNixlTouch'),
 'M20b close leaves the touch thread': ('python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py',
                                        '            self._touch_queue.put(None)\n',
                                        '',
                                        'test_hicache_nixl_storage',
                                        'TestNixlTouch and reaches_every_file'),
 'M20c a thread runs with the switches off': ('python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py',
                                              '        if self.touch_interval_s > 0.0 or self.retire_ended_pages:\n',
                                              '        if True:\n',
                                              'test_hicache_nixl_storage',
                                              'TestNixlTouch'),
 'M20d the retire switch alone starts no thread': ('python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py',
                                                   '        if self.touch_interval_s > 0.0 or '
                                                   'self.retire_ended_pages:\n',
                                                   '        if self.touch_interval_s > 0.0:\n',
                                                   'test_hicache_nixl_storage',
                                                   'TestNixlTouch'),
 "M21 an ended turn's node is written": ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                         '        if self.retention.skip_ended_writes and node_id in '
                                         'self._storage_skip_nodes:\n',
                                         '        if False:\n',
                                         'test_next_use_eviction',
                                         'turn_after_a_skipped'),
 'M21b the flag is lost at a new leaf': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                         '                    state.target_node, '
                                         'skip_storage=state.params.skip_storage_backup\n',
                                         '                    state.target_node\n',
                                         'test_next_use_eviction',
                                         'turn_after_a_skipped'),
 'M21c the flag is lost at a node walked over': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                                 '                    node, '
                                                 'skip_storage=state.params.skip_storage_backup\n',
                                                 '                    node\n',
                                                 'test_next_use_eviction',
                                                 'turn_after_a_skipped'),
 'M21d the skip is never spent': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                  '            self._storage_skip_nodes.discard(node_id)\n'
                                  '            self._retire_at_ack.discard(node_id)\n',
                                  '            self._retire_at_ack.discard(node_id)\n',
                                  'test_next_use_eviction',
                                  'turn_after_a_skipped'),
 'M22 fragments do not inherit the decision': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                               '                if old_node_id in decided:\n'
                                               '                    decided.update(new_node_ids)\n',
                                               '                pass\n',
                                               'test_storage_read_dispatch',
                                               'split_of_a_skipped'),
 'M23 no retire at the ack': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                              '            retire=self._retire_at_ack,\n',
                              '            retire=(),\n',
                              'test_next_use_eviction',
                              'retired_while_its_write'),
 'M23b retired at the ack yet marked stored': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                               '                node.storage_stored = False\n'
                                               '                retired.append(node.id)\n',
                                               '                node.storage_stored = stored_at is not None\n'
                                               '                retired.append(node.id)\n',
                                               'test_next_use_eviction',
                                               'retired_while_its_write'),
 'M24 the first prompt is retired': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                     '            if start < keep_tokens:\n'
                                     '                break\n'
                                     '            take(node)\n',
                                     '            take(node)\n',
                                     'test_next_use_eviction',
                                     'retire_'),
 'M24b own replies are left': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                               '            for child in others:\n'
                               '                take(child)\n'
                               '            if start < keep_tokens:\n',
                               '            if start < keep_tokens:\n',
                               'test_next_use_eviction',
                               'retire_'),
 "M24c another session's leaf is taken as own": ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                                 '                child.children or child.get_last_hash_value() not '
                                                 'in own_leaf_keys\n',
                                                 '                child.children\n',
                                                 'test_next_use_eviction',
                                                 'retire_'),
 'M24d a reply built on is taken as own': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                           '                child.children or child.get_last_hash_value() not in '
                                           'own_leaf_keys\n',
                                           '                child.get_last_hash_value() not in own_leaf_keys\n',
                                           'test_next_use_eviction',
                                           'retire_'),
 'M24e retired pages still count as stored': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                              '            elif owned.storage_stored:\n'
                                              '                owned.storage_stored = False\n',
                                              '            elif owned.storage_stored:\n',
                                              'test_next_use_eviction',
                                              'retire_'),
 'M24f a queued write is zeroed at once': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                           '            if (\n'
                                           '                owned.storage_write_op is not None\n'
                                           '                or owned.write_through_pending_id is not None\n'
                                           '            ):\n'
                                           '                pending.append(owned.id)\n'
                                           '            elif owned.storage_stored:\n',
                                           '            if owned.storage_stored:\n',
                                           'test_next_use_eviction',
                                           'retire_'),
 'M24g a skipped write waits for an ack that never comes': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                                            '            if node_id not in '
                                                            'self._storage_skip_nodes\n',
                                                            '            if True\n',
                                                            'test_next_use_eviction',
                                                            'retire_'),
 "M25 a turn with no entry is not the session's": ('python/sglang/srt/mem_cache/next_turn_prefetch.py',
                                                   '        self._note_session_turn(outlook.session, req=req, '
                                                   'tail_node_id=tail_node_id)\n',
                                                   '',
                                                   'test_next_turn_prefetch',
                                                   'only_an_explicit_end_retires'),
 'M25b the first prompt is not page aligned': ('python/sglang/srt/mem_cache/next_turn_prefetch.py',
                                               '            extent = SessionExtent(first_prompt_tokens=first_pages * '
                                               'self._page_size)\n',
                                               '            extent = '
                                               'SessionExtent(first_prompt_tokens=len(req.origin_input_ids))\n',
                                               'test_next_turn_prefetch',
                                               'only_an_explicit_end_retires'),
 'M25c the first prompt is the latest one': ('python/sglang/srt/mem_cache/next_turn_prefetch.py',
                                             '        extent = self._session_extents.get(session)\n'
                                             '        if extent is None:\n',
                                             '        extent = None\n        if extent is None:\n',
                                             'test_next_turn_prefetch',
                                             'only_an_explicit_end_retires'),
 'M26 the commit hook does not flag the ended turn': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                                      '            insert_params.skip_storage_backup = '
                                                      'self._turn_ends_session(\n'
                                                      '                req, reply_known=True\n'
                                                      '            )\n',
                                                      '',
                                                      'test_next_turn_prefetch',
                                                      'explicit_end_skips_the_last_turns_write'),
 'M26b retire off in the cache': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                  '        if not (self.retention.retire_ended_pages and self.enable_storage):\n'
                                  '            return\n',
                                  '        return\n',
                                  'test_next_turn_prefetch',
                                  'explicit_end_skips_the_last_turns_write'),
 'M24h a host copy in flight is not left to its ack': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                                       '            if (\n'
                                                       '                owned.storage_write_op is not None\n'
                                                       '                or owned.write_through_pending_id is not '
                                                       'None\n'
                                                       '            ):\n',
                                                       '            if owned.storage_write_op is not None:\n',
                                                       'test_next_use_eviction',
                                                       'retire_'),
 'M20e the stop wake-up does not flush the counts': ('python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py',
                                                     '            stopping or now - last_log >= '
                                                     '_TOUCH_LOG_INTERVAL_S\n',
                                                     '            now - last_log >= _TOUCH_LOG_INTERVAL_S\n',
                                                     'test_hicache_nixl_storage',
                                                     'TestNixlTouch'),
 'M27 the commit hook forgets which session named the turn': ('python/sglang/srt/mem_cache/next_turn_prefetch.py',
                                                              '        outlook = self._turn_outlook(req, '
                                                              'reply_known=True, named_session=named_session)\n',
                                                              '        outlook = self._turn_outlook(req, '
                                                              'reply_known=True, named_session=None)\n',
                                                              'test_next_turn_prefetch',
                                                              'explicit_end_retires'),
 'M27b ends_session forgets which session named the turn': ('python/sglang/srt/mem_cache/next_turn_prefetch.py',
                                                            '            '
                                                            'named_session=self._rid_session.get(req.rid),\n',
                                                            '            named_session=None,\n',
                                                            'test_next_turn_prefetch',
                                                            'exact_hint_session_ends'),
 'M27c an exact-hint end is not an explicit end at commit': ('python/sglang/srt/mem_cache/next_turn_prefetch.py',
                                                             '        if outlook.end_why is not None:\n'
                                                             '            self._on_explicit_end(\n',
                                                             '        if outlook.end_why == "entry":\n'
                                                             '            self._on_explicit_end(\n',
                                                             'test_next_turn_prefetch',
                                                             'explicit_end_retires'),
 "F1 the parked reply's write heals the chain the retire zeroed": ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                                                   '        if self.retention.track_stored and '
                                                                   'node_id not in self._retire_at_ack:\n',
                                                                   '        if self.retention.track_stored:\n',
                                                                   'test_next_use_eviction',
                                                                   'last_replys_write'),
 'F1b at_ack counts nodes, not pages': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                        '            sum(len(self.tree_core.get_hash_values(node_id)) for node_id in '
                                        'pending),\n',
                                        '            len(pending),\n',
                                        'test_next_use_eviction',
                                        'last_replys_write'),
 'F2 the ack retires the whole operation, not the fragments taken': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                                                     '            if node.id in retire:\n',
                                                                     '            if node_id in retire:\n',
                                                                     'test_next_use_eviction',
                                                                     'retired_while_its_write'),
 'F2b a parked id is left behind after its ack': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                                  '        self._retire_at_ack.difference_update(retired)\n',
                                                  '',
                                                  'test_next_use_eviction',
                                                  'retired_while_its_write'),
 'F3 the heal stops at the first stored ancestor': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                                    '                found.append(node.id)\n'
                                                    '            node = node.parent\n'
                                                    '        found.reverse()\n',
                                                    '                found.append(node.id)\n'
                                                    '            else:\n'
                                                    '                break\n'
                                                    '            node = node.parent\n'
                                                    '        found.reverse()\n',
                                                    'test_next_use_eviction',
                                                    'child_write_first'),
 'F3b a write below does not take queued pages off the retire': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                                                 '        if self._retire_at_ack:\n'
                                                                 '            # Something is built on these pages '
                                                                 "after their session's end: keep them.\n"
                                                                 '            '
                                                                 'self._retire_at_ack.difference_update(self.tree_core.ancestor_ids(node_id))\n',
                                                                 '',
                                                                 'test_next_use_eviction',
                                                                 'write_below_a_retired_chain'),
 'F3c a write below a retired chain rewrites nothing above a queued node': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                                                            '                found.append(node.id)\n'
                                                                            '            node = node.parent\n'
                                                                            '        found.reverse()\n',
                                                                            '                found.append(node.id)\n'
                                                                            '            else:\n'
                                                                            '                break\n'
                                                                            '            node = node.parent\n'
                                                                            '        found.reverse()\n',
                                                                            'test_next_use_eviction',
                                                                            'write_below_a_retired_chain'),
 'F4 retire is granted without the refresh': ('python/sglang/srt/mem_cache/storage_retention.py',
                                              '        (touch, "SGLANG_HICACHE_NIXL_TOUCH_INTERVAL_S above 0"),\n',
                                              '',
                                              'test_storage_read_dispatch',
                                              'missing_what_it_needs'),
 'F5 a turn a table named ends its session by carrying no hint': ('python/sglang/srt/mem_cache/next_turn_prefetch.py',
                                                                  '            exact = isinstance(named_session, '
                                                                  '_ExactHintSession)\n',
                                                                  '            exact = named_session is not None\n',
                                                                  'test_next_turn_prefetch',
                                                                  'exact_hint_session_ends'),
 "F5b an exact hint's next turn is not marked": ('python/sglang/srt/mem_cache/next_turn_prefetch.py',
                                                 '            if self._explicit_ends and hint.entry_key is None\n',
                                                 '            if False\n',
                                                 'test_next_turn_prefetch',
                                                 'exact_hint_session_ends'),
 "F5c a table's next turn is marked as an exact hint's": ('python/sglang/srt/mem_cache/next_turn_prefetch.py',
                                                          '            if self._explicit_ends and hint.entry_key is '
                                                          'None\n',
                                                          '            if self._explicit_ends\n',
                                                          'test_next_turn_prefetch',
                                                          'exact_hint_session_ends'),
 "F5d a known session's turn without a hint registers it as ended": ('python/sglang/srt/mem_cache/next_turn_prefetch.py',
                                                                     '        if not outlook.hinted and '
                                                                     'outlook.session is None:\n',
                                                                     '        if not outlook.hinted:\n',
                                                                     'test_next_turn_prefetch',
                                                                     'no_usable_entry_is_ranked'),
 'F5e no_hint is logged as no_entry': ('python/sglang/srt/mem_cache/next_turn_prefetch.py',
                                       '            why="no_entry" if outlook.hinted else "no_hint",\n',
                                       '            why="no_entry",\n',
                                       'test_next_turn_prefetch',
                                       'no_usable_entry_is_ranked'),
 'F6 a clear leaves the queued operation on its node': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                                        '            node.storage_stored = False\n'
                                                        '            node.storage_write_op = None\n',
                                                        '            node.storage_stored = False\n',
                                                        'test_next_use_eviction',
                                                        'clearing_or_detaching'),
 'F6b a clear keeps the pages parked for an ack': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                                   '        self.tree_core.clear_storage_marks()\n'
                                                   '        self._retire_at_ack.clear()\n',
                                                   '        self.tree_core.clear_storage_marks()\n',
                                                   'test_next_use_eviction',
                                                   'clearing_or_detaching'),
 'F8 a detach keeps the skips decided': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                         '        if detached:\n'
                                         '            # A host copy still in flight is not written on to a detached '
                                         'store.\n'
                                         '            self._storage_skip_nodes.clear()\n',
                                         '',
                                         'test_next_use_eviction',
                                         'clearing_or_detaching'),
 'T3 a failing touch ends the thread': ('python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py',
                                        '            except Exception:\n'
                                        '                # One failing job must not end the thread',
                                        '            except KeyboardInterrupt:\n'
                                        '                # One failing job must not end the thread',
                                        'test_hicache_nixl_storage',
                                        'touch_that_fails'),
 'T4 the marks are granted with PP above 1': ('python/sglang/srt/mem_cache/storage_retention.py',
                                              '        (single_pp_rank, "PP = 1"),\n',
                                              '',
                                              'test_storage_read_dispatch',
                                              'missing_what_it_needs'),
 'T5 a sidecar pool registered after init keeps the retire': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                                              '        if self.retention.retire_ended_pages:\n'
                                                              "            # A draft model's pools",
                                                              '        if False:\n'
                                                              "            # A draft model's pools",
                                                              'test_storage_read_dispatch',
                                                              'what_the_server_really_has'),
 'T7 the touch thread holds its backend': ('python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py',
                                           '                    backend_ref=weakref.ref(self),\n',
                                           '                    backend_ref=(lambda backend: lambda: '
                                           'backend)(self),\n',
                                           'test_hicache_nixl_storage',
                                           'dropped_without_close'),
 'S1 the prompt insert never flags': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                      '        insert_params.skip_storage_backup = self._turn_ends_session(\n'
                                      '            req, reply_known=False\n'
                                      '        )\n',
                                      '',
                                      'test_next_turn_prefetch',
                                      'prompt_insert_of_an_ending_turn'),
 'S1b the prompt insert reads a reply not generated yet': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                                           '            req, reply_known=False\n',
                                                           '            req, reply_known=True\n',
                                                           'test_next_turn_prefetch',
                                                           'prompt_insert_of_an_ending_turn'),
 'S4 the cache passes host_cache_mode as met': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                                '            host_cache_mode=self.host_memory_mode == "cache",\n',
                                                '            host_cache_mode=True,\n',
                                                'test_storage_read_dispatch',
                                                'what_the_server_really_has'),
 'S4 the cache passes python_tree_core as met': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                                 '            python_tree_core=isinstance(self.tree_core, '
                                                 'UnifiedTreeCore),\n',
                                                 '            python_tree_core=True,\n',
                                                 'test_storage_read_dispatch',
                                                 'what_the_server_really_has'),
 'S4 the cache passes single_pp_rank as met': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                               '            single_pp_rank=self.pp_size == 1,\n',
                                               '            single_pp_rank=True,\n',
                                               'test_storage_read_dispatch',
                                               'what_the_server_really_has'),
 'S4 the cache passes next_turn_prefetch as met': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                                   '            '
                                                   'next_turn_prefetch=get_memory().hicache_storage_next_turn_prefetch '
                                                   '!= "off",\n',
                                                   '            next_turn_prefetch=True,\n',
                                                   'test_storage_read_dispatch',
                                                   'what_the_server_really_has'),
 'S4 the cache passes full_kv_only as met': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                             '            full_kv_only=self.tree_components == '
                                             '(ComponentType.FULL,)\n'
                                             '            and not self.sidecar_pool_specs,\n',
                                             '            full_kv_only=True,\n',
                                             'test_storage_read_dispatch',
                                             'what_the_server_really_has'),
 'S4b a runtime attach keeps the switches of the store before': ('python/sglang/srt/mem_cache/unified_cache/storage_attachment.py',
                                                                 '        cache._init_storage_retention()\n'
                                                                 '        return True, "Attached HiCache storage '
                                                                 'backend successfully."\n',
                                                                 '        return True, "Attached HiCache storage '
                                                                 'backend successfully."\n',
                                                                 'test_storage_read_dispatch',
                                                                 'runtime_attach_resolves'),
 'S4c a rank that writes no files takes the switches': ('python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py',
                                                        '        touches = self.file_manager is not None and not '
                                                        'self.backup_skip\n',
                                                        '        touches = self.file_manager is not None\n',
                                                        'test_hicache_nixl_storage',
                                                        'writes_no_files'),
 'S5 retire with the store detached': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                       '        if not (self.retention.retire_ended_pages and '
                                       'self.enable_storage):\n',
                                       '        if not self.retention.retire_ended_pages:\n',
                                       'test_next_use_eviction',
                                       'takes_the_replies'),
 'S5b the match hook runs with the touch interval at zero': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                                             '            self.retention.touch_interval_s > 0.0\n'
                                                             '            and params.req is not None\n'
                                                             '            and self.enable_storage\n',
                                                             '            params.req is not None\n'
                                                             '            and self.enable_storage\n',
                                                             'test_next_use_eviction',
                                                             'probe_without_a_request'),
 'S6 a split does not copy the refresh stamp': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                                '        new_node.storage_touched_at = child.storage_touched_at\n',
                                                '',
                                                'test_next_use_eviction',
                                                'stale_tail_takes'),
 "S7 every other child is a stranger's leaf": ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                               '                child.children or child.get_last_hash_value() not in '
                                               'own_leaf_keys\n',
                                               '                True\n',
                                               'test_next_use_eviction',
                                               'read_back_as_a_new_node'),
 "S7b the turn's last page key is not kept": ('python/sglang/srt/mem_cache/next_turn_prefetch.py',
                                              '        if leaf_key is not None:\n'
                                              '            extent.leaf_keys.append(leaf_key)\n',
                                              '',
                                              'test_next_turn_prefetch',
                                              'only_an_explicit_end_retires'),
 'F9 a write skip is decided with the store detached': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                                        '            # Detached, no write comes to spend a skip: its '
                                                        'node id would stay for good.\n'
                                                        '            and self.enable_storage\n',
                                                        '',
                                                        'test_next_turn_prefetch',
                                                        'prompt_insert_of_an_ending_turn'),
 'S11 what is kept per session is never dropped': ('python/sglang/srt/mem_cache/next_turn_prefetch.py',
                                                   '            while len(self._session_extents) > '
                                                   'REGISTRY_MAX_ENTRIES:\n'
                                                   '                del '
                                                   'self._session_extents[next(iter(self._session_extents))]\n',
                                                   '',
                                                   'test_next_turn_prefetch',
                                                   'bounded_oldest_first')}
