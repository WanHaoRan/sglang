#!/usr/bin/env python3
"""A small closed (or open) network with a pool-bound batch GPU and pass-over admission.  Stdlib only.

Model
- N users.  A user alternates a pause Z and a request.  A request arrives, waits in one arrival-ordered
  queue, is admitted to one of B slots (the KV pool), is prefilled (serial GPU work of p seconds, during
  which decoding of every running request is paused: prefill goes before decode steps, as in the engine's
  scheduler loop), emits its first token at the end of its prefill, then decodes n_out tokens together with
  the other running requests.  A decode step with n running requests takes tau(n) = a + b*n seconds and gives
  each of them one token (decode is treated as a fluid: token rate 1/tau(n) per request).
- B = 1 with n_out = 0 is the single non-preemptive server (service = p).
- A request may need a read: it is "ready" at arrival + r (or at an absolute time given by the caller).
  Admission passes over a turn that is not ready: a free slot goes to the earliest-arrived ready turn.
- Common random numbers: the k-th request of user u draws (Z, p, n_out, r, tag) from the user's own stream,
  so two runs with the same seed differ only by the perturbations passed in.

Returned records: rec[(u, k)] = [arrival, ready, admit, first_token, done, p, n_out, tag, due].
`due` is the first time the turn would have been admitted had it been ready (a slot was free for it and it
was passed over, or the slot stayed empty); -1 if it never waited on its read.
"""
import heapq
import math
import random

ARR, LAND = 0, 1


def simulate(n_users, n_slots, tau_a, tau_b, draw, seed, t_end,
             extra_prefill=None, ready_abs=None, need=None, t_stat0=0.0, hard_stop=None):
    """Run until t_end, and beyond it until every request id in `need` is done (if given).

    draw(u, k, rng) -> (Z, p, n_out, r, tag) or None when user u has no k-th request.
    extra_prefill: {(u, k): seconds added to that request's prefill}.
    ready_abs: {(u, k): absolute time at which that request becomes ready} (overrides arrival + r).
    Returns (rec, stats).
    """
    extra_prefill = extra_prefill or {}
    ready_abs = ready_abs or {}
    need_left = set(need) if need else None
    rngs = [random.Random(seed * 1000003 + u) for u in range(n_users)]
    rec = {}
    heap = []
    seq = 0
    for u in range(n_users):
        d = draw(u, 0, rngs[u])
        if d is None:
            continue
        heapq.heappush(heap, (d[0], seq, ARR, (u, 0, d)))
        seq += 1
    waiting = []          # arrival-ordered list of [key, ready_time]
    slots_used = 0
    prefill_q = []        # admitted, waiting for their (serial) prefill
    pf_key = None
    pf_end = math.inf
    dec = []              # heap of (v_finish, seq, key)
    n_dec = 0
    v = 0.0               # per-request decoded-token clock
    t = 0.0
    kcount = [0] * n_users
    st = dict(busy=0.0, idle=0.0, empty_slot_wait=0.0, empty_slot_nowait=0.0, slot_time=0.0, dec_time=0.0,
              pf_time=0.0, n_dec_int=0.0, T=0.0, passed_over=0, admits=0, q_int=0.0, run_int=0.0,
              starve_time=0.0, admit_alone=0)

    def tau(n):
        return tau_a + tau_b * n

    while True:
        t_heap = heap[0][0] if heap else math.inf
        if pf_key is not None:
            t_gpu = pf_end
        elif n_dec > 0:
            t_gpu = t + max(dec[0][0] - v, 0.0) * tau(n_dec)
        else:
            t_gpu = math.inf
        t_next = t_heap if t_heap < t_gpu else t_gpu
        if t_next == math.inf:
            break
        if hard_stop is not None and t_next > hard_stop:
            break
        if t_next > t_end and need_left is not None and not need_left:
            break
        if t_next > t_end and need_left is None:
            break
        dt = t_next - t
        if dt > 0.0:
            if t >= t_stat0:
                st["T"] += dt
                if pf_key is not None:
                    st["pf_time"] += dt
                    st["busy"] += dt
                elif n_dec > 0:
                    st["dec_time"] += dt
                    st["busy"] += dt
                    st["n_dec_int"] += dt * n_dec
                else:
                    st["idle"] += dt
                free = n_slots - slots_used
                if free > 0:
                    if waiting:
                        st["empty_slot_wait"] += dt * free
                        st["starve_time"] += dt
                    else:
                        st["empty_slot_nowait"] += dt * free
                st["q_int"] += dt * len(waiting)
                st["run_int"] += dt * slots_used
            if pf_key is None and n_dec > 0:
                v += dt / tau(n_dec)
        t = t_next
        if t_gpu <= t_heap:
            if pf_key is not None:
                key = pf_key
                r = rec[key]
                r[3] = t
                pf_key = None
                pf_end = math.inf
                heapq.heappush(dec, (v + r[6], seq, key))
                seq += 1
                n_dec += 1
            else:
                vf, _, key = heapq.heappop(dec)
                v = vf if vf > v else v
                n_dec -= 1
                slots_used -= 1
                rec[key][4] = t
                if need_left is not None:
                    need_left.discard(key)
                u = key[0]
                kcount[u] += 1
                d = draw(u, kcount[u], rngs[u])
                if d is not None:
                    heapq.heappush(heap, (t + d[0], seq, ARR, (u, kcount[u], d)))
                    seq += 1
        else:
            _, _, typ, payload = heapq.heappop(heap)
            if typ == ARR:
                u, k, d = payload
                key = (u, k)
                ready = ready_abs.get(key, t + d[3])
                if ready < t:
                    ready = t
                rec[key] = [t, ready, -1.0, -1.0, -1.0, d[1] + extra_prefill.get(key, 0.0), d[2], d[4], -1.0]
                waiting.append([key, ready])
                if ready > t:
                    heapq.heappush(heap, (ready, seq, LAND, key))
                    seq += 1
            # a LAND event only wakes the admission check below
        # admission: a free slot goes to the earliest-arrived ready turn
        while slots_used < n_slots and waiting:
            pick = -1
            for i, w in enumerate(waiting):
                if w[1] <= t:
                    pick = i
                    break
            if pick < 0:
                break
            for w in waiting[:pick]:
                if rec[w[0]][8] < 0:
                    rec[w[0]][8] = t
            key = waiting.pop(pick)[0]
            rec[key][2] = t
            slots_used += 1
            prefill_q.append(key)
            if t >= t_stat0:
                st["admits"] += 1
                if pick > 0:
                    st["passed_over"] += 1
                other_ready = False
                for w in waiting:
                    if w[1] <= t:
                        other_ready = True
                        break
                if not other_ready:
                    st["admit_alone"] += 1
        if slots_used < n_slots and waiting:
            # a slot is free and nothing is ready: each waiting turn would be admitted now if it were ready
            for w in waiting:
                if rec[w[0]][8] < 0:
                    rec[w[0]][8] = t
        if pf_key is None and prefill_q:
            pf_key = prefill_q.pop(0)
            pf_end = t + rec[pf_key][5]
    st["t_final"] = t
    return rec, st


def summarize(rec, t0, t1, by_tag=False):
    """Means over requests that arrived in [t0, t1) and completed."""
    n = 0
    s_ttft = s_resp = s_wait = s_p = s_pre = 0.0
    tags = {}
    for key, r in rec.items():
        if r[4] < 0 or r[0] < t0 or r[0] >= t1:
            continue
        n += 1
        s_ttft += r[3] - r[0]
        s_resp += r[4] - r[0]
        s_wait += r[2] - r[0]
        s_pre += r[3] - r[2]
        s_p += r[5]
        if by_tag:
            a = tags.setdefault(r[7], [0, 0.0, 0.0])
            a[0] += 1
            a[1] += r[3] - r[0]
            a[2] += r[5]
    res = dict(n=n, ttft=s_ttft / max(n, 1), resp=s_resp / max(n, 1), wait=s_wait / max(n, 1),
               pre=s_pre / max(n, 1), p=s_p / max(n, 1))
    if by_tag:
        res["tags"] = {k: (a[0], a[1] / a[0], a[2] / a[0]) for k, a in tags.items()}
    return res
