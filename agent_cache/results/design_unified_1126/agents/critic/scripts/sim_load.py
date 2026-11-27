"""What-if on the workflow's calibrated closed-loop simulator (../../sim/kvsim.py, read-only): the GPU time the
simulator charges to the host-to-GPU load, removed. Variants of ec_full at campaign 34's setting:
  base        the calibrated cost set (h2d 0.85 us per loaded token, 15 ms per prefill batch)
  h2d0        h2d = 0 (what the simulator itself attributes to the copy)
  h2d0_b0     h2d = 0 and no per-batch term (37 ms a load-carrying batch: near the 40-53 ms a request measured
              on the logs as the time a prefill pass waits on the copy)
  h2d2x       h2d = 1.7 us (the per-pass measurement), per-batch term left: a check of the sensitivity upward
  pool102     the calibrated costs with a GPU pool 2 % larger (681,472 tokens); pool101: 1 % larger
usage: sim_load.py <seed> <variant> [arm]   -> one JSON line on stdout
"""
import json
import os
import sys

SIM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "sim")
sys.path.insert(0, os.path.abspath(SIM))
import kvsim  # noqa: E402

VARIANTS = {
    "base": {},
    "h2d0": {"h2d": 0.0},
    "h2d0_b0": {"h2d": 0.0, "batch_s": 0.0},
    "h2d2x": {"h2d": 1.7e-6},
    # 2 % more usable GPU pool (what first-fit admission among ready turns would reclaim, hol.py): a config change
    "pool102": {"cfg": {"gpu_tokens": 681472}},
    "pool101": {"cfg": {"gpu_tokens": 674752}},
}


def main():
    seed, variant = int(sys.argv[1]), sys.argv[2]
    arm = sys.argv[3] if len(sys.argv) > 3 else "ec_full"
    spec = dict(VARIANTS[variant])
    cfg = kvsim.Config(arm=arm, seed=seed, **spec.pop("cfg", {}))
    sim = kvsim.Sim(cfg)
    for k, v in spec.items():
        sim.cost[k] = v
    sim.run()
    s = kvsim.summary(sim)
    out = {"seed": seed, "variant": variant, "arm": arm, "requests": s["requests"], "req_per_min": s["req_per_min"],
           "ttft_mean": s["ttft_mean"], "ttft_p50": s["ttft_p50"], "ttft_p90": s["ttft_p90"],
           "queue_wait_mean": s["queue_wait_mean"], "slot_wait_mean": s["slot_wait_mean"],
           "cap_binds_at_min": s["cap_binds_at_min"],
           "windows": [{k: w[k] for k in ("window", "requests", "req_per_min", "ttft_mean")} for w in s["windows"]],
           "census_90_110": s["census_90_110"]}
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
