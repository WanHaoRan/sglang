#!/usr/bin/env bash
# The files listed in agent_cache/large_files.sha256 (over 25 MB when added) are stored
# xz-compressed in git: the .xz copy is tracked and the original is ignored
# (agent_cache/.gitignore). This restores each missing original next to its copy and
# checks every original against the list. Originals already present are kept.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
list=agent_cache/large_files.sha256
command -v xz > /dev/null || { echo "restore_large_files.sh: needs xz (XZ Utils)" >&2; exit 1; }
part=
cleanup() { if [[ -n "$part" ]]; then rm -f -- "$part"; fi; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
restored=0
while read -r _ f; do
  if [[ ! -e "$f" ]]; then
    part="$f.part"
    xz -dc -- "$f.xz" > "$part"
    mv -- "$part" "$f"
    part=
    restored=$((restored + 1))
  fi
done < "$list"
if ! sha256sum --quiet -c "$list"; then
  echo "restore_large_files.sh: delete the files marked FAILED and run this again" >&2
  exit 1
fi
echo "restored $restored and checked $(wc -l < "$list") files"
