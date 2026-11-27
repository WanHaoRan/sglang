# Split the per-minute tallies into the 30 minutes before the first cleaner pass and
# the time from that pass to the end of the run.
{
  for (i = 2; i <= NF; i++) { split($i, a, "="); v[a[1]] = a[2] }
  m = $1
  if (m < W0) next
  w = (m < P) ? "1 30 min before first pass" : "2 first pass to end"
  L[w] += v["lookups"]; N[w] += v["need_tok"]; Z[w] += v["none"]; ZT[w] += v["none_tok"]
  Q[w] += v["partial"]; QT[w] += v["partial_lost_tok"]; F[w] += v["full"]
}
END {
  for (k in L) printf "%s | lookups %d, history %.2fM | none %d (%.2fM, %.1f%%) | partial %d (lost %.2fM, %.1f%%) | full %d\n", k, L[k], N[k]/1e6, Z[k], ZT[k]/1e6, 100*ZT[k]/N[k], Q[k], QT[k]/1e6, 100*QT[k]/N[k], F[k]
}
