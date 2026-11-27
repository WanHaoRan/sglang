# Review of the 2026-11-23 design note (workflow wf_c267e9d1-02b)

CPU-only review of `agent_cache/agent_cache_design_discussion_2026-11-23.md` and the paper draft's design section,
run 2026-11-23 21:30-22:50Z while campaign 34's hicache rerun used the GPU. Nothing in the engine was changed.

- `critique.md`: the adversarial reviewer's verdicts, recommended synthesis, simulator targets and ranked experiments.
- `facts_{fetch,evict,evidence,calib}.md`: what four fact-finders found in the code and records (path:line, record paths).
- `design_{scheduling,economics,storage}.md`: three independent design proposals.
- `all.json`: everything above in one file.
- `agents/`: the agents' scratch scripts and outputs (scripts take the run folders as input; the `*.evt` extracts
  of the campaign 34 server logs were left out and are rebuilt by `agents/facts_evidence/` scripts).
  Key scorers: `agents/reviewer/admit_err.py`, `agents/reviewer/max_structure.py` (two-queue admission model),
  `agents/designer-storage/scripts/host_room_replay2.py` (open-loop host-room replay), `agents/reviewer/head_control*`
  (cleaner shared-head check), `agents/designer-scheduling/toy_two_queues.py` (toy, not calibrated).

Paths inside these files point at the original scratch folder
`/tmp/claude-1001/-home-wanhr-sglang/cf65791c-c7d4-4411-be47-21b43c92b26f/scratchpad/wf_design/`, which maps to `agents/` here.
