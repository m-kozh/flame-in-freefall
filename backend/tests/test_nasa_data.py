"""Scientific conversions, missingness, provenance and offline NASA import."""

import hashlib
import json
import shutil

import pandas as pd
import pytest

from data import nasa
from data.loader import COLUMNS, load_csv
from data.synthetic import generate_synthetic
from scripts.import_nasa import main as import_main


def test_cr_complete_table_preserves_censoring_and_observation_window():
    observations = nasa.dataset_observations("nasa-cr-182185")
    assert len(observations) == 13
    assert len({row["record_id"] for row in observations}) == 13
    censored = [row for row in observations if not row["training_eligible"]]
    assert [row["table_row"] for row in censored] == [10, 11]
    for row in censored:
        assert row["spread_rate_mm_s"] is None
        assert row["spread_rate_upper_bound_mm_s"] == 0.5
        assert row["rate_observability"] == "left_censored_below_0.5_mm_s"
        assert row["extinguished"] is False
        assert "left-censored" in row["exclusion_reason"]
    for row in observations:
        assert row["observation_window_seconds"] == 5.18
        assert row["source_reference"]["pdf_page"] == 7
        assert row["source_reference"]["table"] == "I"
        if row["training_eligible"]:
            assert row["exclusion_reason"] is None
    extinctions = [row for row in observations if row["extinguished"]]
    assert len(extinctions) == 3
    assert all(row["spread_rate_mm_s"] is None for row in extinctions)


def test_training_exact_source_units_nominal_conditions_and_one_group():
    frame = nasa.nasa_training_rows()
    assert frame.columns.tolist() == COLUMNS
    assert len(frame) == 11
    assert frame.experiment_id.unique().tolist() == ["NASA-CR-182185"]
    assert frame.campaign.unique().tolist() == ["NASA-CR-182185"]
    assert frame.material.unique().tolist() == ["cellulose_thin"]
    assert frame.source.unique().tolist() == ["nasa"]
    assert frame.thickness_mm.eq(0.076).all()
    assert frame.pressure_kpa.eq(101.325).all()
    assert frame.gravity_g.eq(0.0).all()
    assert frame.ignited.all()
    assert int(frame.extinguished.sum()) == 3
    assert frame.loc[~frame.extinguished, "spread_rate_mm_s"].tolist() == [
        8.7, 6.4, 4.0, 3.5, 3.6, 2.7, 2.1, 1.9,
    ]
    # Fuel speed is used directly, without adding the target spread velocity.
    assert frame.flow_cm_s.tolist() == [
        6.06, 6.75, 3.92, 1.70, 6.36, 3.30, 1.31, 0.81, 5.96, 6.08, 4.19,
    ]
    assert frame.loc[frame.extinguished, "spread_rate_mm_s"].isna().all()


def test_catalogue_counts_source_hashes_and_rights_are_distinct():
    cards = {card["id"]: card for card in nasa.dataset_catalog()}
    assert set(cards) == {"nasa-cr-182185", "psi-25-bass-ii", "bass-ii-tm-20210011385"}
    assert [cards[key]["record_count"] for key in cards] == [13, 129, 41]
    assert [cards[key]["training_row_count"] for key in cards] == [11, 0, 0]
    assert cards["psi-25-bass-ii"]["doi"] == "10.60555/4qc4-de67"
    assert cards["psi-25-bass-ii"]["license"] == "CC0-1.0"
    assert cards["nasa-cr-182185"]["license"] != "CC0-1.0"
    assert cards["bass-ii-tm-20210011385"]["license"] != "CC0-1.0"
    for card in cards.values():
        for source in card["source_files"]:
            assert set(source) == {"name", "url", "sha256"}
            assert len(source["sha256"]) == 64
            assert source["url"].startswith("https://")
            assert "X-Amz-" not in source["url"]
    official_csv = next(
        source for source in cards["psi-25-bass-ii"]["source_files"]
        if source["name"].endswith(".csv")
    )
    assert hashlib.sha256((nasa.CURATED_DIR / "psi25_bass_ii.csv").read_bytes()).hexdigest() == official_csv["sha256"]


def test_psi_raw_official_strings_are_not_inferred_targets():
    observations = nasa.dataset_observations("psi-25-bass-ii")
    assert len(observations) == 129
    assert all(row["training_eligible"] is False for row in observations)
    assert all(row["exclusion_reason"] for row in observations)
    assert observations[0]["source_fields"]["Test #"] == "B1"
    assert observations[0]["source_fields"]["Fan display"] == "62"
    assert observations[0]["source_fields"]["Air display"] == "5"
    multistage = next(row for row in observations if row["source_fields"]["Test #"] == "O4")
    assert multistage["source_fields"]["Air display"] == "5   down to ext"
    assert multistage["source_fields"]["Fan display"] == "5   0.7   0.5   0.4   0.36 ext"
    assert "flow_cm_s" not in multistage
    assert "extinguished" not in multistage
    assert "spread_rate_mm_s" not in multistage


def test_report_stages_do_not_impute_oxygen_pressure_or_binary_targets():
    observations = nasa.dataset_observations("bass-ii-tm-20210011385")
    stages = [row for row in observations if row["observation_type"] == "reported_spread_rate_stage"]
    boundaries = [row for row in observations if row["observation_type"] == "aggregate_critical_flow_boundary"]
    assert len(stages) == 39
    assert len(boundaries) == 2
    assert all(not row["training_eligible"] and row["exclusion_reason"] for row in observations)
    assert all(row["o2_percent"] is None and row["pressure_kpa"] is None for row in stages)
    assert all(row["ignited"] is None and row["extinguished"] is None for row in observations)
    assert stages[0]["spread_rate_mm_s"] == 0.071
    assert stages[0]["o2_interval_percent"] == [21.9, 22.2]
    assert [(row["o2_percent"], row["critical_flow_cm_s"]) for row in boundaries] == [(21.0, 1.0), (16.0, 3.0)]


@pytest.mark.parametrize("dataset_id", ["unknown", "", "../registry.json"])
def test_unknown_dataset_raises_key_error(dataset_id):
    with pytest.raises(KeyError):
        nasa.dataset_observations(dataset_id)


def test_snapshot_corruption_is_detected_offline(tmp_path, monkeypatch):
    copied = tmp_path / "curated"
    shutil.copytree(nasa.CURATED_DIR, copied)
    monkeypatch.setattr(nasa, "CURATED_DIR", copied)
    with (copied / "psi25_bass_ii.csv").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(nasa.SourceIntegrityError, match="SHA256 mismatch"):
        nasa.dataset_observations("psi-25-bass-ii")
    with pytest.raises(nasa.SourceIntegrityError):
        nasa.dataset_catalog()


def test_return_values_cannot_mutate_cached_or_bundled_evidence():
    card = nasa.dataset_catalog()[0]
    card["source_files"][0]["sha256"] = "modified"
    row = nasa.dataset_observations("nasa-cr-182185")[0]
    row["source_values"]["fuel_speed_cm_s"] = "9999"
    assert nasa.dataset_catalog()[0]["source_files"][0]["sha256"] != "modified"
    assert nasa.nasa_training_rows().iloc[0].flow_cm_s == 6.06


def test_training_evidence_is_deterministic_and_recognizes_only_exact_rows():
    frame = nasa.nasa_training_rows()
    summary = nasa.training_evidence_summary(frame)
    assert summary == nasa.training_evidence_summary(frame.copy())
    json.dumps(summary, allow_nan=False)
    assert summary["nasa_row_count"] == 11
    assert summary["verified_curated_row_count"] == 11
    assert summary["unverified_nasa_row_count"] == 0
    assert summary["dataset_ids"] == ["nasa-cr-182185"]
    assert summary["observation_window_s"] == 5.18
    assert summary["material_counts"] == {"cellulose_thin": 11}
    assert summary["campaign_counts"] == {"NASA-CR-182185": 11}
    assert "indefinite" in summary["target_definitions"]["extinguished_false"] or any(
        "indefinite" in limitation for limitation in summary["limitations"]
    )
    tampered = frame.copy()
    tampered.loc[0, "flow_cm_s"] = 6.07
    summary = nasa.training_evidence_summary(tampered)
    assert summary["verified_curated_row_count"] == 10
    assert summary["unverified_nasa_row_count"] == 1
    assert summary["observation_window_s"] is None
    assert any("not been independently verified" in limitation for limitation in summary["limitations"])


def test_future_nasa_and_incomplete_frames_do_not_inherit_curated_labels():
    future = nasa.nasa_training_rows().iloc[:1].copy()
    future["campaign"] = "future-nasa-study"
    future["experiment_id"] = "future-nasa-study-1"
    for frame in [future, pd.DataFrame({"source": ["nasa"], "campaign": ["NASA-CR-182185"]})]:
        summary = nasa.training_evidence_summary(frame)
        assert summary["nasa_row_count"] == 1
        assert summary["verified_curated_row_count"] == 0
        assert summary["unverified_nasa_row_count"] == 1
        assert summary["dataset_ids"] == []
        assert summary["observation_window_s"] is None
    assert nasa.training_evidence_summary(pd.DataFrame())["nasa_row_count"] == 0
    assert nasa.training_evidence_summary(pd.DataFrame({"material": ["cellulose_thin"]}))["dataset_ids"] == []


def test_synthetic_row_with_nasa_campaign_is_not_counted_as_nasa():
    frame = nasa.nasa_training_rows().iloc[:1].copy()
    frame["source"] = "synthetic"
    summary = nasa.training_evidence_summary(frame)
    assert summary["nasa_row_count"] == 0
    assert summary["verified_curated_row_count"] == 0
    assert summary["dataset_ids"] == []
    assert summary["campaign_source_conflict_row_count"] == 1


def test_offline_import_preserves_synthetic_file_and_adds_only_eleven(tmp_path, monkeypatch):
    synthetic_path = tmp_path / "synthetic.csv"
    output_path = tmp_path / "mixed.csv"
    generate_synthetic(80, 5).to_csv(synthetic_path, index=False)
    before = synthetic_path.read_bytes()
    def forbid_network(*args, **kwargs):
        pytest.fail("Default import must not access the network")
    monkeypatch.setattr("urllib.request.urlopen", forbid_network)
    import_main(["--input", str(synthetic_path), "--output", str(output_path)])
    mixed = load_csv(output_path)
    assert synthetic_path.read_bytes() == before
    assert mixed.source.value_counts().to_dict() == {"synthetic": 80, "nasa": 11}
    assert mixed.loc[mixed.source.eq("nasa"), "experiment_id"].nunique() == 1
    with pytest.raises(SystemExit):
        import_main(["--input", str(synthetic_path), "--output", str(synthetic_path)])
    with pytest.raises(SystemExit):
        import_main(["--input", str(output_path), "--output", str(tmp_path / "duplicate.csv")])
