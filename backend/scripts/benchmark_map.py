"""Measure complete HTTP map latency against a running backend."""

import argparse
import json
import statistics
import time
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    cold, warm = [], []
    for pressure in (101.31, 101.32, 101.33):
        body = json.dumps({"material": "cotton", "pressure_kpa": pressure, "resolution": 50}).encode()
        for timings in (cold, warm):
            start = time.perf_counter()
            request = Request(args.base_url.rstrip("/") + "/map", data=body, headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=10) as response:
                result = json.loads(response.read())
            timings.append(time.perf_counter() - start)
            assert len(result["z"]) == 50 and all(len(row) == 50 for row in result["z"])
            assert result["z"] == result["extinction_probability"]
    print(f"50x50 complete HTTP request: cold median={statistics.median(cold):.4f}s max={max(cold):.4f}s; cached median={statistics.median(warm):.4f}s")
    if max(cold) >= 1:
        raise SystemExit("Map latency exceeded 1 second on this machine.")


if __name__ == "__main__":
    main()
