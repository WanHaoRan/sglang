Campaign 32: campaign 28 (FreeInference) on FlashAttention 3. The same settings (settings.env is campaign 28's), users
and arms, plus campaign 28's oracle arm ec_exact, each for the full 2 h; only the attention backend and the engine copy
change (queue.txt says why). Queued 2026-11-22 at the author's request after the 20-minute FA3 check
(baselines/README.md, "FA3"); run with campaign 33 by frozen/oracle_scripts_staging/c28_users/run_campaigns.sh, started
by chain_fa3.sh once the oracle arms of campaigns 28 and 29 and the fix check (results/qfix_check/) have ended.
Requests pair with campaign 28's arms and with the baselines of campaign 30 by (play, stream, k).
