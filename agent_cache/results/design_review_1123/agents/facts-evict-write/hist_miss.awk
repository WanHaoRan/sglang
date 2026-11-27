# Join each next-turn gate decision that issues a lookup (read/reread/retry) with the
# next s2h_query of the same rid; classify the history part (chain - matched) as fully
# found, partly found (cut), or not found, per minute.
function val(line, key,   a, b) { split(line, a, key "="); split(a[2], b, " "); return b[1] }
/HICACHE_EVT ntp_gate/ {
  d = val($0, "decision");
  if (d == "read" || d == "reread" || d == "retry") {
    rid = val($0, "rid"); miss[rid] = val($0, "chain") - val($0, "matched");
  }
  next
}
/HICACHE_EVT s2h_query/ {
  rid = val($0, "rid");
  if (!(rid in miss)) next;
  m = miss[rid]; delete miss[rid];
  if (m < 4096) next;
  t = substr($2, 1, 5); hit = val($0, "storage_hit") + 0;
  n[t]++; need[t] += m;
  if (hit == 0) { z[t]++; zt[t] += m }
  else if (hit < m - 64) { p[t]++; pt[t] += m - hit }
  else f[t]++;
}
END { for (k in n) printf "%s lookups=%d need_tok=%d none=%d none_tok=%d partial=%d partial_lost_tok=%d full=%d\n", k, n[k], need[k], z[k], zt[k], p[k], pt[k], f[k] }
