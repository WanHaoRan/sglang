#!/usr/bin/env bash
# Patch the installed `dflash` client so it reads acceptance length from where this
# SGLang checkout puts it (choices[0].meta_info) instead of a top-level meta_info key
# that the server never emits. See RUNBOOK.md §2.4. Idempotent.
set -euo pipefail

PY=${PYTHON:-python3}
TARGET=$($PY -c 'import dflash.benchmark as b; print(b.__file__)')
OLD='meta = out.get("meta_info", {}) or {}'
NEW='meta = out.get("meta_info") or ((out.get("choices") or [{}])[0].get("meta_info")) or {}'

if grep -qF "$NEW" "$TARGET"; then
  echo "already patched: $TARGET"
  exit 0
fi
if ! grep -qF "$OLD" "$TARGET"; then
  echo "ERROR: expected line not found in $TARGET; the dflash release changed. Patch by hand:" >&2
  echo "  $NEW" >&2
  exit 1
fi
cp "$TARGET" "$TARGET.orig"
$PY - "$TARGET" "$OLD" "$NEW" <<'EOF'
import sys
path, old, new = sys.argv[1:4]
src = open(path).read()
assert src.count(old) == 1, "anchor line is not unique"
open(path, "w").write(src.replace(old, new))
EOF
echo "patched: $TARGET (backup at $TARGET.orig)"
grep -n 'meta = out.get' "$TARGET"
