"""Run a complete research demo against the live backend and save its evidence."""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ARTIFACT_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR / "demo_report.json")
    args = parser.parse_args()
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "base_url": args.base_url, "calls": {}}

    def call(name, path, body=None):
        request = Request(
            args.base_url.rstrip("/") + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "X-Request-ID": f"pitch-{name}"},
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=30) as response:
                result = json.loads(response.read())
                request_id = response.headers.get("X-Request-ID")
        except HTTPError as exc:
            raise SystemExit(f"{path}: HTTP {exc.code}: {exc.read().decode()}") from exc
        elapsed_ms = (time.perf_counter() - started) * 1000
        report["calls"][name] = {"path": path, "request": body, "elapsed_ms": round(elapsed_ms, 3), "request_id": request_id, "response": result}
        print(f"{name:20s} {elapsed_ms:8.1f} ms  source={result['data_source']}")
        return result

    scenario = {"material": "cotton", "o2": 21, "pressure_kpa": 101.3, "flow_cm_s": 10}
    health = call("health", "/health")
    if not health["model_loaded"]:
        raise SystemExit("Generate data, train the model, and restart the backend first.")
    card = call("model_card", "/model-card")
    datasets = call("nasa_datasets", "/datasets")
    call("batch_prediction", "/predict/batch", {"points": [scenario, scenario | {"o2": 34, "pressure_kpa": 56.5}]})
    analysis = call("scenario_analysis", "/analyze", {"scenario": scenario, "samples": 128})
    optimization = call("constrained_search", "/optimize", {
        "scenario": scenario,
        "bounds": {"o2_range": [12, 35], "pressure_range": [50, 115], "flow_range": [.1, 100]},
        "target_extinction_probability": .9, "resolution": 10, "k": 5,
    })
    design = call("experiment_design", "/experiments/design", {"material": "cotton", "pressure_kpa": 56.5, "k": 5})
    print(f"\nModel: {card['model_version']}  identity={card['model_id']}")
    evidence = card.get("training_evidence") or {}
    print(f"NASA training rows: {evidence.get('nasa_row_count', 0)}; verified curated rows: {evidence.get('verified_curated_row_count', 0)}")
    print(f"Registered NASA evidence: {sum(row['record_count'] for row in datasets['datasets'])} records across {len(datasets['datasets'])} datasets")
    print(f"Scenario: P(no sustained flame)={analysis['current']['extinction_probability']:.3f}; assessment={analysis['assessment']}")
    print(f"Input-tolerance support fraction: {analysis['robustness']['support_fraction']:.1%}")
    print(f"Search: {optimization['status']}; {optimization['eligible_count']}/{optimization['evaluated_count']} eligible centres; {len(optimization['candidates'])} target-matching candidates")
    if optimization["candidates"]:
        candidate = optimization["candidates"][0]
        print(f"Nearest research candidate: {json.dumps(candidate['prediction']['inputs'])}")
        print(f"Minimum centre/corner member lower quantile: {candidate['robustness']['minimum_sampled_extinction_lower']:.3f}")
    if design["points"]:
        point = design["points"][0]
        print(f"Next experiment: O2={point['o2']:.2f}%, flow={point['flow_cm_s']:.2f} cm/s, information proxy={point['ensemble_information_gain_proxy_bits']:.3f} bits")
    print(f"Training source={health['data_source']}; approximate physics and limited NASA observation windows make these research demonstrations, not safety certification.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Evidence saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
