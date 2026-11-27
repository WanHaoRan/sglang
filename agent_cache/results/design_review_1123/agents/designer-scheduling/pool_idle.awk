# For each Decode/Prefill batch log line: token usage and #queue-req. Reports, per 30-min window since the
# first prefill, the share of lines with a non-empty queue whose KV pool use is below 0.90 / 0.95,
# and the mean unused pool (1 - usage) on lines with a non-empty queue.
function ts(s,   a, b) { split(s, a, /[ :]/); return a[2]*3600 + a[3]*60 + a[4] }
/Decode batch|Prefill batch/ {
  t = ts(substr($0, 2, 23));
  if (t0 == "" && /Prefill batch/) t0 = t;
  if (t0 == "") next;
  if (t < t0) t += 86400;
  m = int((t - t0) / 60); w = (m < 30) ? 0 : (m < 60) ? 1 : (m < 90) ? 2 : (m < 110) ? 3 : 4;
  match($0, /token usage: [0-9.]+/); u = substr($0, RSTART + 13, RLENGTH - 13) + 0;
  match($0, /#queue-req: [0-9]+/); q = substr($0, RSTART + 12, RLENGTH - 12) + 0;
  n[w]++;
  if (q > 0) { nq[w]++; un[w] += 1 - u; if (u < 0.90) lo90[w]++; if (u < 0.95) lo95[w]++; qs[w] += q }
}
END {
  split("0-30 30-60 60-90 90-110 110+", lab, " ");
  for (w = 0; w <= 4; w++) if (n[w]) printf "%s min: lines %d, queue>0 %d (%.0f%%), of those use<0.90 %.1f%% use<0.95 %.1f%%, mean unused %.3f, mean queue %.1f\n", lab[w+1], n[w], nq[w], 100*nq[w]/n[w], nq[w]?100*lo90[w]/nq[w]:0, nq[w]?100*lo95[w]/nq[w]:0, nq[w]?un[w]/nq[w]:0, nq[w]?qs[w]/nq[w]:0
}
