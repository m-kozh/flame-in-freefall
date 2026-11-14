# NASA observations, provenance and training scope

The project now ingests verified NASA source observations and an official PSI
experiment catalogue. Source data and training eligibility are separate: an entry
in a NASA repository does not automatically contain all model targets.

## Verified sources

1. NASA-CR-182185, NTRS19890014267, TableI (PDF page7, printed page6):13 reported
   conditions for a0.076mm cellulose/polyamide thin sheet, nominal1atm=101.325kPa,
   opposed relative flow and a5.18s microgravity observation window. Eight rows
   contain measured positive spread rates, three observed extinction, and two
   left-censored rates below0.05cm/s (=0.5mm/s). Keep all13 as evidence; exclude the
   two censored rows from the current complete-target training CSV. Use fuel speed,
   never characteristic relative velocity (which includes the target flame speed).
   Source: https://ntrs.nasa.gov/citations/19890014267 . NASA record rights are
   "Work of the US Gov. Public Use Permitted"; this is distinct from PSI's CC0.
2. PSI-25/BASS-II, DOI10.60555/4qc4-de67, version4, CC0-1.0:129 official catalogue
   rows and the original Test.Matrices.xlsx/ReadMe.docx. Preserve raw field meaning;
   fan display is voltage×10, while air display is a slow-response cm/s indicator.
   Multiple flow values and directions/geometries cannot be collapsed into one
   invented training point. Source: https://psi.nasa.gov/physci/repo/data/investigations/PSI-25 .
3. NASA/TM-20210011385 BASS-II Summary Report:two thin-PMMA aggregated quenching
   boundaries and39 reported speed stages across20 thick-PMMA tests. Stage-specific
   oxygen values are unavailable in the reported table; keep them null. These are
   evidence, not39 complete independent training rows. Source:
   https://ntrs.nasa.gov/citations/20210011385 .

Original downloads, SHA256 hashes, per-record page/table references, unit conversions,
and exclusions are kept with the curated registry. Download scripts verify hashes
and do not save temporary signed-download credentials in the repository.

## Model integration

The default mixed research training set contains2000 synthetic rows and11 complete
NASA-CR-182185 rows. The entire NASA report uses one conservative experiment group,
so all11 are held out together during GroupKFold. They are not represented as11
independent experimental batches. `data_source="mixed"` means both sources were
used in training; it does not mean every material or pressure has NASA observations.

The NASA label `extinguished=false` means **no extinction observed within5.18s**,
not proof of indefinite sustained burning. This short-window label must remain
visible in dataset/model metadata. Current NASA coverage is one thin cellulose
sample and one nominal pressure. Material library thermal properties remain
explicit approximations. Neither13 conditions nor one research group supports a
standalone, independently validated NASA-only model.

Report metrics separately by `source`, in addition to aggregate scores. The NASA
OOF slice is a whole-report transfer test, made by a model trained without that
NASA group; final-model training predictions are not an external validation score.
Retain the synthetic-only model and external NASA baseline before retraining.
The external-evaluation script checks report identifiers and numerical observation
overlap, so renamed campaigns or changed source labels cannot disguise fitted rows
as an unseen NASA validation set. The numerical check is conservative and does not
prove scientific independence of otherwise distinct experiments.

## Additive evidence API

Existing prediction and map endpoints retain their response types. Data endpoints
are usable without a trained model and always carry `data_source="nasa"`:

- `GET /datasets` → `{data_source, datasets:[DatasetCard...]}`.
- `GET /datasets/{dataset_id}` → `{data_source, dataset:DatasetCard}`.
- `GET /datasets/{dataset_id}/observations` →
  `{data_source,dataset_id,count,observations:[object...]}`.

Stable IDs: `nasa-cr-182185`, `psi-25-bass-ii`, `bass-ii-tm-20210011385`.
DatasetCard fields: `id`, `title`, `source_url`, `doi`(nullable), `license`,
`record_count`, `training_row_count`, `observation_window_s`(nullable),
`scope`(object), `source_files`(list of objects with name/url/sha256),
`limitations`(string list), and `included_in_loaded_model`(boolean).
Unknown IDs return404 with the same `detail,data_source` error envelope.

Observation objects preserve source-specific fields. Every record includes a stable
record_id, source reference, eligibility and exclusion reason; incomplete/censored
values remain null and are never converted to invented zeros or burning labels.

`GET /model-card` gains additive NASA training-evidence metadata and per-material
source counts. `/metrics.validation.models[*].by_source` explicitly separates NASA
and synthetic outcomes. The existing `data_source` value changes to mixed only
after the mixed model is actually trained and loaded.
