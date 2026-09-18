---
name: runbook-authoring
description: 'Format and clarity rules for evaluation runbooks, plans and HANDOFF files (e.g. agent_cache/RUNBOOK.md, hicache_eval/HANDOFF.md): every step states Goal / Expected result / Expected lessons; every command says where it runs, what it touches, and what it should print. Use when writing, extending or restructuring any runbook, evaluation plan, step-by-step guide or handoff document.'
---

# Runbook Authoring

A runbook is read by someone who was not in the conversation that produced it,
often days later, on a box whose state has changed. Every step must therefore say
what it is for, what a good outcome looks like, and what the reader should take
away from it. Every command must be runnable by pasting, with nothing to guess.

The user asked for this explicitly on 2026-09-18: "for each step, write down the
goal, the expected results and the expected lessons learned ... for each command,
make it easy to use and understand." Apply it to every step, not only the
interesting ones.

## The step template

Every step (every `###` heading, and every `##` section that has no `###`
children) uses this block. Field order is fixed; a field that does not apply says
`none` rather than being omitted, so the reader never wonders whether it was
forgotten.

```markdown
### 2.2 Local NVMe for L3

**Status:** done 2026-09-17; redo every morning (the disk is wiped on VM stop)
**Goal:** give HiCache an L3 directory on the local SSD, not on the root disk.
**Runs on:** host shell, as `wanhr` (needs sudo)
**Touches:** `/dev/nvme0n1` (formats it: destroys its contents), `/mnt/nvme`, `/etc/fstab` (read only)
**Takes:** ~2 min (estimate: the format was never timed); GPU idle

<command blocks, see "Command rules">

**Expected result:** `findmnt /mnt/nvme` prints one line with `SOURCE=/dev/nvme0n1 FSTYPE=ext4`;
`df -h /mnt/nvme` shows ~345 GB free; `/mnt/nvme/hicache_l3` exists and is empty.
**If it differs:** `findmnt` prints nothing: the mount step was skipped, rerun from "Mount".
`df` shows the root disk (1 TB, /dev/sda1): `/mnt/nvme` is a plain directory, the mount failed.
**Expected lessons:** the local SSD does not survive a VM stop/start, so any L3 result
taken after a restart without this step measured the root disk, not the NVMe.
```

Field meanings:

- **Status:** done (with date), not started, provisional, redo-every-time. Never
  put status in the heading; headings are stable anchors for `§` cross references.
- **Goal:** one sentence: the decision this step enables or the artifact it
  produces. "Install X" is not a goal; "make the container able to see the GPU so
  §2.4's check can pass" is.
- **Runs on:** `host` or `container: <name>` (with the `docker exec` form shown
  once), plus the user and working directory when they matter. Paths that differ
  between host and container are spelled out on both sides
  (`/home/wanhr/sglang/agent_cache` = `/sgl-workspace/sglang/agent_cache`).
- **Touches:** every file, directory, disk, container, or process the step
  changes; say `read-only` when nothing. Destructive steps say so in this line,
  and name what is destroyed.
- **Takes:** wall time, and whether the GPU is busy (so the reader knows what can
  run concurrently).
- **Expected result:** concrete and checkable: file names with sizes, counts,
  numbers with a tolerance, the exact log line or the exact printed value. "It
  works" is not an expected result. Tag every number `measured` (with date and
  source), `derived` (with the formula), or `estimate`.
- **If it differs:** the one or two likely causes and the fix. Optional but
  strongly preferred for anything that has actually failed once.
- **Expected lessons:** what the outcome teaches about the system, or the trap
  the step prevents and why. For a measurement step, say what each plausible
  outcome would mean (e.g. "if L3 TTFT is above recompute at every length, the
  disk cannot pay for this model and §7 rows 4-6 are skipped").

## Command rules

1. **One purpose per block, one comment line above it** that says in plain words
   what it does and, when destructive, what it destroys:
   `# Format the local NVMe as ext4 (destroys everything on /dev/nvme0n1)`.
2. **Say where it runs** in the first comment line of the block: `# host` or
   `# container: sglang_hicache`. When a block runs inside a container from the
   host, show the `docker exec -i <name> bash -c '...'` form rather than asking
   the reader to "run this in the container".
3. **Self-contained.** Every variable a block uses is defined in that block, or
   in one named variables block the step tells the reader to paste first.
   Never `$COMMON="--flag a --flag b"`: zsh does not word-split it. Use bash
   arrays under an explicit `bash` heading, or spell the flags out.
4. **Label every printed value.** A block that prints more than one value prefixes
   each: `echo "memlock: $(ulimit -l)"`, `echo "nvme free: $(df -h /mnt/nvme | ...)"`.
   Never `ulimit -l; findmnt ...; df ...; free ...` in one block: the reader
   cannot tell which line is which (this misread an 84 GB /dev/shm as the NVMe).
5. **Show the expected output** right after the block, as `-> prints:` with the
   pass condition, e.g. `-> prints "cuda: True"; anything else means the
   toolkit is not wired into docker (§2.1)`.
6. **Guard destructive commands** with a check that stops the block before the
   destructive line, and never use `exit` inside a pasteable block (it closes the
   reader's shell). Use `|| { echo "STOP: <why>"; false; }` and put the
   destructive line under `&&`, or split into a check block and an act block.
7. **Long commands** are `\`-continued, one flag per line, grouped by purpose
   with a short comment per group (`# tiering`, `# pressure`), so a reader can
   see which flags make the arm differ from the baseline.
8. **Anything longer than a minute** states its duration and gives one way to
   watch it (`tail -f <log>` or a marker file), and one way to stop it cleanly.
9. **Copy-paste over adapt.** Fill in the real values for this box. A placeholder
   is `<ANGLE_BRACKETS>` and every placeholder is defined once, next to its
   first use.
10. **Ordered sequences** are numbered one block per item, or moved into a script
    under `scripts/` that the runbook invokes with one line; the runbook then
    documents the script's stages, not its internals.
11. **Interactive or host-only actions** the reader must do (VM start, `gcloud
    auth`, editing a config) are their own step with the same template.

## Numbers and evidence

- Every number is tagged: `measured` (date, results path), `derived` (formula and
  inputs), or `estimate` (and what would make it wrong).
- Code facts carry an anchor `path:line` in this checkout; the anchor is checked
  when the runbook is edited, not assumed.
- Constants that later steps depend on live in one table (`§5.4` style) and are
  referenced from there, never re-typed with a different value elsewhere.

## Document structure

1. **§0 The short version:** the ordered list of what actually gets run, one line
   per step, with the § reference and the time it takes. A reader who knows the
   box uses only this section.
2. **Box state:** what is installed, mounted, running, with the date it was
   checked and the command that checks it again.
3. **Steps in execution order**, each with the template above.
4. **Traps:** one line each, each pointing at the step it protects.
5. **Layout and hygiene:** where results go, what is tracked, what must be
   reverted before a commit.

Each `##` section opens with two or three sentences on why the section exists and
what the reader should have at its end. Cross references use `§N.M`; never "see
above".

## Checklist before delivering

Run these against the file; fix every hit.

- Every `###` step has Status, Goal, Runs on, Touches, Takes, Expected result,
  Expected lessons (grep for the bold labels and count them per step).
- Every fenced command block starts with a `# host` / `# container:` line.
- No `exit` inside a fenced block that a reader would paste into a shell.
- No `$VARIABLE` used in a block that neither defines it nor names the variables
  block that does.
- No block prints two or more unlabelled values.
- Every `->` expected output states a pass condition.
- Every number has a `measured` / `derived` / `estimate` tag, and every
  `path:line` anchor resolves in this checkout.
- Headings contain no status words (DONE, TODO, NOT started); those live in
  **Status:**.
- §0 lists every GPU-busy step with its duration.
