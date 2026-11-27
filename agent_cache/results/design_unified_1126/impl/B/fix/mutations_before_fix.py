"""The mutation checks of the tests added for steps 1 to 3 of the Retain plan (copy B, 2026-11-26).
name -> (engine file, text to replace, replacement, test file, pytest -k expression).
Each text occurs exactly once in its file; run with mutate.py."""

MUTATIONS = {'M1 short ack counts as stored': ('python/sglang/srt/mem_cache/storage_retention.py',
                                   '        writes_nothing\n'
                                   '        or operation.completed_tokens >= len(operation.hash_value) * '
                                   'page_size\n',
                                   '        True\n',
                                   'test_storage_read_dispatch',
                                   'acked_short'),
 'M2 marks made with the switch off': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                       '            verdicts = itertools.repeat(None)\n',
                                       '            verdicts = itertools.repeat(True)\n',
                                       'test_storage_read_dispatch',
                                       'switch_off_no_node'),
 'M3 no reduce over ranks': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                             '        if not (synced and votes and (self.tp_world_size > 1 or self.pp_size > '
                             '1)):\n',
                             '        if True:\n',
                             'test_storage_read_dispatch',
                             'short_on_any_rank'),
 'M3b a rank that writes nothing votes no': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                             '            '
                                             'writes_nothing=self.cache_controller.backup_skip,\n',
                                             '            writes_nothing=False,\n',
                                             'test_storage_read_dispatch',
                                             'short_on_any_rank'),
 'M3c detach path reduces': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                             '        if not (synced and votes and (self.tp_world_size > 1 or self.pp_size > '
                             '1)):\n',
                             '        if not (votes and (self.tp_world_size > 1 or self.pp_size > 1)):\n',
                             'test_storage_read_dispatch',
                             'short_on_any_rank'),
 'M4 detach leaves the op mark': ('python/sglang/srt/mem_cache/unified_cache/storage_attachment.py',
                                  '                if cache.retention.track_stored:\n',
                                  '                if False:\n',
                                  'test_storage_read_dispatch',
                                  'dropped_at_detach'),
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
                '            for ancestor_id in self.tree_core.unstored_ancestors(node_id):\n'
                '                self._submit_storage_write(ancestor_id)\n',
                '            pass\n',
                'test_next_use_eviction',
                'child_write_first'),
 'M7b heal writes a host copy still in flight': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                                 '            and node.write_through_pending_id is None\n'
                                                 '            and not node.storage_stored\n',
                                                 '            and not node.storage_stored\n',
                                                 'test_next_use_eviction',
                                                 'child_write_first'),
 'M7c heal ignores a queued write': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                     '            and not node.storage_stored\n'
                                     '            and node.storage_write_op is None\n',
                                     '            and not node.storage_stored\n',
                                     'test_next_use_eviction',
                                     'child_write_first'),
 'M8 clear keeps marks': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                          '            if self.retention.track_stored:\n'
                          '                self.tree_core.clear_storage_marks()\n',
                          '',
                          'test_next_use_eviction',
                          'clearing_the_store'),
 'M8b split does not copy stored': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                    '        new_node.storage_stored = child.storage_stored\n',
                                    '',
                                    'test_next_use_eviction',
                                    'clearing_the_store'),
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
                                     '                end_why=None if named_session is None else '
                                     '"no_next",\n',
                                     '                end_why="no_next",\n',
                                     'test_next_turn_prefetch',
                                     'exact_hint_session_ends'),
 'M11 no entry registers ended': ('python/sglang/srt/mem_cache/next_turn_prefetch.py',
                                  '            if outlook.end_why is None and outlook.hinted:\n',
                                  '            if False:\n',
                                  'test_next_turn_prefetch',
                                  'no_usable_entry_is_ranked'),
 'M11b expiry and abort register ended': ('python/sglang/srt/mem_cache/next_turn_prefetch.py',
                                          '        if self._explicit_ends:\n'
                                          '            self._forget_session(rid=rid, session=session, '
                                          'why=why)\n'
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
                                                   '            touch_interval_s = '
                                                   'backend.touch_interval_s\n',
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
                                          '            if stored_at is not None:\n'
                                          '                node.storage_touched_at = stored_at\n',
                                          '',
                                          'test_next_use_eviction',
                                          'stale_tail_takes'),
 'M14b only the stale nodes are refreshed': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                             '        for node in reversed(stored):\n'
                                             '            node.storage_touched_at = now\n'
                                             '            page_keys.extend(node.hash_value)\n'
                                             '        return page_keys\n',
                                             '        for node in reversed(stored):\n'
                                             '            if now - node.storage_touched_at >= '
                                             'min_interval_s:\n'
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
                                          '            if node.hash_value:\n'
                                          '                stored.append(node)\n',
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
                             '            stamp = 0 if newest_ns is None else newest_ns - index * '
                             'TOUCH_STEP_NS\n',
                             '            stamp = 0 if newest_ns is None else newest_ns + index * '
                             'TOUCH_STEP_NS\n',
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
                                                 '                raise ValueError("touch_interval_s must be '
                                                 'a number")\n'
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
                                '        return [self.file_manager.get_file_path(name) for name in '
                                'names[:1]]\n',
                                'test_hicache_nixl_storage',
                                'TestNixlTouch'),
 'M20b close leaves the touch thread': ('python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py',
                                        '            self._touch_queue.put(None)\n'
                                        '            self._touch_thread.join(timeout=5.0)\n',
                                        '',
                                        'test_hicache_nixl_storage',
                                        'TestNixlTouch'),
 'M20c a thread runs with the switches off': ('python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py',
                                              '        if self.touch_interval_s > 0.0 or '
                                              'self.retire_ended_pages:\n',
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
                              '        retire = (\n'
                              '            self.retention.retire_ended_pages and entry.node_id in '
                              'self._retire_at_ack\n'
                              '        )\n',
                              '        retire = False\n',
                              'test_storage_read_dispatch',
                              'retired_while_its_write'),
 'M23b retired at the ack yet marked stored': ('python/sglang/srt/mem_cache/unified_radix_cache.py',
                                               '            stored_at=time.monotonic() if stored and not '
                                               'retire else None,\n',
                                               '            stored_at=time.monotonic() if stored else '
                                               'None,\n',
                                               'test_storage_read_dispatch',
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
                                                 '            if any(child.children or child.id not in '
                                                 'own_leaves for child in others):\n',
                                                 '            if any(child.children for child in others):\n',
                                                 'test_next_use_eviction',
                                                 'retire_'),
 'M24d a reply built on is taken as own': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                           '            if any(child.children or child.id not in own_leaves '
                                           'for child in others):\n',
                                           '            if any(child.id not in own_leaves for child in '
                                           'others):\n',
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
                                                   '                self._note_session_turn(\n'
                                                   '                    outlook.session, req=req, '
                                                   'tail_node_id=tail_node_id\n'
                                                   '                )\n',
                                                   '',
                                                   'test_next_turn_prefetch',
                                                   'only_an_explicit_end_retires'),
 'M25b the first prompt is not page aligned': ('python/sglang/srt/mem_cache/next_turn_prefetch.py',
                                               '            extent = '
                                               'SessionExtent(first_prompt_tokens=first_pages * '
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
                                  '        if not (self.retention.retire_ended_pages and '
                                  'self.enable_storage):\n'
                                  '            return\n',
                                  '        return\n',
                                  'test_next_turn_prefetch',
                                  'explicit_end_skips_the_last_turns_write'),
 'M24h a host copy in flight is not left to its ack': ('python/sglang/srt/mem_cache/unified_cache/unified_tree_core.py',
                                                       '            if (\n'
                                                       '                owned.storage_write_op is not None\n'
                                                       '                or owned.write_through_pending_id is '
                                                       'not None\n'
                                                       '            ):\n',
                                                       '            if owned.storage_write_op is not None:\n',
                                                       'test_next_use_eviction',
                                                       'retire_'),
 'M20e the stop wake-up does not flush the counts': ('python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py',
                                                     '                stopping or now - last_log >= '
                                                     '_TOUCH_LOG_INTERVAL_S\n',
                                                     '                now - last_log >= '
                                                     '_TOUCH_LOG_INTERVAL_S\n',
                                                     'test_hicache_nixl_storage',
                                                     'TestNixlTouch'),
 'M27 the commit hook forgets which session named the turn': ('python/sglang/srt/mem_cache/next_turn_prefetch.py',
                                                              '            outlook = self._turn_outlook(req, '
                                                              'reply_known=True, named_session=session)\n',
                                                              '            outlook = self._turn_outlook(req, '
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
                                                             '            if outlook.end_why is not None:\n'
                                                             '                self._on_explicit_end(req.rid, '
                                                             'outlook, tail_node_id=tail_node_id)\n',
                                                             '            if outlook.end_why == "entry":\n'
                                                             '                self._on_explicit_end(req.rid, '
                                                             'outlook, tail_node_id=tail_node_id)\n',
                                                             'test_next_turn_prefetch',
                                                             'explicit_end_retires')}
