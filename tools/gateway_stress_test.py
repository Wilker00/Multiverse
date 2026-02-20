"""
Gateway stress smoke test for the universal-model API.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from typing import Dict, List


def _default_payload() -> Dict[str, object]:
    sample_obs = {
        "x": 0,
        "y": 0,
        "goal_x": 7,
        "goal_y": 7,
        "battery": 28,
        "nearby_obstacles": 0,
        "nearest_charger_dist": 4,
        "t": 0,
        "on_conveyor": 0,
        "patrol_dist": 5,
        "lidar": [1, 1, 2, 4, 3, 1, 1, 1],
        "flat": [0.0, 0.0, 7.0, 7.0, 28.0, 0.0, 0.0, 0.0, 5.0, 1.0, 1.0, 2.0, 4.0, 3.0, 1.0, 1.0, 1.0],
    }
    return {"obs": sample_obs, "verse": "warehouse_world", "top_k": 5}


def _is_api_healthy(health_url: str, timeout_sec: float) -> bool:
    try:
        with urllib.request.urlopen(health_url, timeout=timeout_sec) as response:
            return int(getattr(response, "status", 0) or 0) == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def run_stress_test(endpoint: str, payload: Dict[str, object], *, num_requests: int, timeout_sec: float) -> int:
    latencies_ms: List[float] = []
    request_bytes = json.dumps(payload).encode("utf-8")
    print(f"Starting stress test: {num_requests} requests to {endpoint}")

    for i in range(int(num_requests)):
        start = time.perf_counter()
        req = urllib.request.Request(endpoint, data=request_bytes, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout_sec) as response:
                body = response.read().decode("utf-8")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"Request {i + 1} failed: {exc}")
            continue

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            print(f"Request {i + 1} failed: invalid JSON response")
            continue

        if not bool(data.get("ok", False)):
            print(f"Request {i + 1} failed: {data.get('error', 'unknown error')}")
            continue

        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_requests} requests...")

    if not latencies_ms:
        print("No successful requests.")
        return 1

    avg_latency = statistics.mean(latencies_ms)
    p95 = statistics.quantiles(latencies_ms, n=20)[18] if len(latencies_ms) >= 20 else max(latencies_ms)
    p99 = statistics.quantiles(latencies_ms, n=100)[98] if len(latencies_ms) >= 100 else max(latencies_ms)

    print("")
    print("--- Gateway Stress Test Results ---")
    print(f"Successful Requests: {len(latencies_ms)}")
    print(f"Average Latency:     {avg_latency:.2f} ms")
    print(f"P95 Latency:         {p95:.2f} ms")
    print(f"P99 Latency:         {p99:.2f} ms")
    print("Target Threshold:    100 ms (P95)")
    print(f"VERDICT: {'PASS' if p95 < 100.0 else 'FAIL'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--endpoint", type=str, default=None)
    ap.add_argument("--num_requests", type=int, default=100)
    ap.add_argument("--timeout_sec", type=float, default=5.0)
    ap.add_argument("--skip_health_check", action="store_true")
    args = ap.parse_args()

    endpoint = str(args.endpoint).strip() if args.endpoint else f"http://127.0.0.1:{int(args.port)}/predict"
    health_url = endpoint.rsplit("/", 1)[0] + "/health"

    if not bool(args.skip_health_check):
        print("Checking if API is running...")
        if not _is_api_healthy(health_url=health_url, timeout_sec=float(args.timeout_sec)):
            print("API is not running. Start it with: python tools/universal_model_api.py --port 8089")
            return 2
        print("API is running.")

    return run_stress_test(
        endpoint=endpoint,
        payload=_default_payload(),
        num_requests=max(1, int(args.num_requests)),
        timeout_sec=max(0.1, float(args.timeout_sec)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
