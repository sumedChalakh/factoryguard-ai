import argparse
import json
import os
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def percentile(values, p):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return float(sorted_values[f])
    return float(sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f))


def make_request(url, payload_bytes, timeout, api_key=None, api_key_header="X-API-Key"):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers[api_key_header] = api_key

    req = Request(url=url, data=payload_bytes, headers=headers, method="POST")

    start = time.perf_counter()
    status = None
    error = None
    try:
        with urlopen(req, timeout=timeout) as response:
            status = response.status
            response.read()
    except HTTPError as exc:
        status = exc.code
        error = str(exc)
    except URLError as exc:
        status = 0
        error = str(exc)
    except Exception as exc:
        status = 0
        error = str(exc)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return elapsed_ms, status, error


def main():
    parser = argparse.ArgumentParser(description="Latency benchmark for FactoryGuard API endpoints")
    parser.add_argument("--url", default="http://127.0.0.1:5000/predict", help="Endpoint URL to benchmark")
    parser.add_argument("--requests", type=int, default=200, help="Total requests")
    parser.add_argument("--concurrency", type=int, default=20, help="Concurrent workers")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup request count")
    parser.add_argument("--timeout", type=float, default=10.0, help="Request timeout in seconds")
    parser.add_argument("--payload", default='{"temperature":60,"vibration":29,"pressure":102}', help="JSON payload")
    parser.add_argument("--api-key", default=os.getenv("API_KEY", ""), help="API key for protected endpoints")
    parser.add_argument("--api-key-header", default=os.getenv("API_KEY_HEADER", "X-API-Key"), help="API key header name")
    args = parser.parse_args()

    try:
        payload_obj = json.loads(args.payload)
        payload_bytes = json.dumps(payload_obj).encode("utf-8")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON payload: {exc}")

    print("=== FactoryGuard Latency Benchmark ===")
    print(f"URL         : {args.url}")
    print(f"Requests    : {args.requests}")
    print(f"Concurrency : {args.concurrency}")
    print(f"Warmup      : {args.warmup}")
    print(f"Timeout(s)  : {args.timeout}")
    print()

    for _ in range(args.warmup):
        make_request(args.url, payload_bytes, args.timeout, args.api_key, args.api_key_header)

    latencies = []
    status_counts = {}
    errors = []

    bench_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                make_request,
                args.url,
                payload_bytes,
                args.timeout,
                args.api_key,
                args.api_key_header,
            )
            for _ in range(args.requests)
        ]

        for future in as_completed(futures):
            latency_ms, status, error = future.result()
            latencies.append(latency_ms)
            status_counts[status] = status_counts.get(status, 0) + 1
            if error:
                errors.append(error)

    bench_elapsed = time.perf_counter() - bench_start

    success_count = sum(count for status, count in status_counts.items() if 200 <= status < 300)
    error_count = args.requests - success_count
    error_rate = (error_count / args.requests) * 100 if args.requests else 0.0
    throughput = args.requests / bench_elapsed if bench_elapsed > 0 else 0.0

    print("--- Results ---")
    print(f"Total time        : {bench_elapsed:.3f} s")
    print(f"Throughput        : {throughput:.2f} req/s")
    print(f"Success           : {success_count}/{args.requests}")
    print(f"Error rate        : {error_rate:.2f}%")
    print(f"Min latency       : {min(latencies):.2f} ms")
    print(f"Avg latency       : {statistics.mean(latencies):.2f} ms")
    print(f"Max latency       : {max(latencies):.2f} ms")
    print(f"p50 latency       : {percentile(latencies, 50):.2f} ms")
    print(f"p95 latency       : {percentile(latencies, 95):.2f} ms")
    print(f"p99 latency       : {percentile(latencies, 99):.2f} ms")
    print(f"Status breakdown  : {status_counts}")

    if errors:
        print("\nSample errors:")
        for message in errors[:5]:
            print(f"- {message}")


if __name__ == "__main__":
    main()
