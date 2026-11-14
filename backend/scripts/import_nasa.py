"""Merge verified NASA observations with a synthetic CSV, offline by default.

python scripts/import_nasa.py [--input PATH] [--output PATH]
Optional --download-verify fetches original official sources and checks SHA256.
The two large source reports are never needed for an offline import.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from app.config import ARTIFACT_DIR, DATA_PATH, SYNTHETIC_DATA_PATH
from data.loader import load_csv, validate_dataframe
from data.nasa import CURATED_DIR, nasa_training_rows, verify_sha256


def download_verify_sources(directory: Path) -> list[Path]:
    """Fetch through stable NASA endpoints; never log/store signed URL tokens."""
    registry = json.loads((CURATED_DIR / "registry.json").read_text(encoding="utf-8"))
    directory.mkdir(parents=True, exist_ok=True)
    verified = []
    for dataset in registry["datasets"]:
        for source in dataset["source_files"]:
            target = directory / source["name"]
            if target.is_file():
                verify_sha256(target, source["sha256"])
                verified.append(target)
                continue
            download_url = source["url"]
            if source["download_method"] == "psi_signed_redirect":
                request = urllib.request.Request(download_url, headers={"User-Agent": "FlameInFreefall-source-verifier/1"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = response.read().decode("utf-8").strip()
                try:
                    decoded = json.loads(payload)
                except json.JSONDecodeError:
                    decoded = payload
                if not isinstance(decoded, str) or urllib.parse.urlsplit(decoded).scheme != "https":
                    raise ValueError(f"NASA PSI returned an invalid download response for {source['name']}.")
                download_url = decoded
            temporary = target.with_suffix(target.suffix + ".part")
            try:
                request = urllib.request.Request(download_url, headers={"User-Agent": "FlameInFreefall-source-verifier/1"})
                with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                    shutil.copyfileobj(response, output)
                verify_sha256(temporary, source["sha256"])
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            verified.append(target)
            print(f"Verified {source['name']} SHA256 {source['sha256']}")
    return verified


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SYNTHETIC_DATA_PATH, help="Existing synthetic-only CSV.")
    parser.add_argument("--output", type=Path, default=DATA_PATH, help="Separate mixed-data CSV.")
    parser.add_argument("--download-verify", action="store_true", help="Download official originals and verify pinned SHA256 hashes.")
    parser.add_argument("--source-dir", type=Path, default=ARTIFACT_DIR / "nasa_sources")
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("--output must differ from --input; preserve the synthetic source CSV.")
    synthetic = load_csv(args.input)
    if not synthetic["source"].eq("synthetic").all():
        parser.error("--input must be synthetic-only; this prevents duplicate NASA imports.")
    if args.download_verify:
        download_verify_sources(args.source_dir)
    nasa = nasa_training_rows()
    mixed = validate_dataframe(pd.concat([synthetic, nasa], ignore_index=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mixed.to_csv(args.output, index=False)
    print(
        f"Wrote {len(mixed)} observations: {len(synthetic)} synthetic + {len(nasa)} NASA; "
        "2 left-censored NASA rates excluded; NASA rows use one report-level group."
    )
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
