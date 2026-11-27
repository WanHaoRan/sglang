"""Keeps a few long generations decoding until the stop file appears, so that a checked request's prefill batch
meets a running batch (the load hold only acts while something decodes).

Usage (in the container): python3 bg_decode.py <stop file> [streams] [max_new_tokens]
"""
import json, os, sys, threading, urllib.request

STOP, N, OUT = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6, int(sys.argv[3]) if len(sys.argv) > 3 else 1500
URL = "http://127.0.0.1:30000/generate"
done = [0] * N


def stream(i: int) -> None:
    while not os.path.exists(STOP):
        body = json.dumps({"text": f"Stream {i}, round {done[i]}. Write the numbers from one to ten thousand in words:",
                           "sampling_params": {"max_new_tokens": OUT, "temperature": 0, "ignore_eos": True}}).encode()
        req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            resp.read()
        done[i] += 1


threads = [threading.Thread(target=stream, args=(i,), daemon=True) for i in range(N)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f"background generations finished: {sum(done)}", flush=True)
