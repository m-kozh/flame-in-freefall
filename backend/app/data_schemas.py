"""NASA source evidence is distinct from provenance of the currently loaded model."""

from typing import Annotated, Any, Literal

from pydantic import Field

from app.schemas import Nonnegative, ResponseModel

Count = Annotated[int, Field(ge=0)]


class DatasetSourceFile(ResponseModel):
    name: str
    url: str
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class DatasetCard(ResponseModel):
    id: str
    title: str
    source_url: str
    doi: str | None
    license: str
    record_count: Count
    training_row_count: Count
    observation_window_s: Nonnegative | None
    scope: dict[str, Any]
    source_files: list[DatasetSourceFile]
    limitations: list[str]
    included_in_loaded_model: bool


class NasaSourceResponse(ResponseModel):
    # This identifies the observations' source, not the inference model's data.
    data_source: Literal["nasa"] = "nasa"


class DatasetListResponse(NasaSourceResponse):
    datasets: list[DatasetCard]


class DatasetDetailResponse(NasaSourceResponse):
    dataset: DatasetCard


class DatasetObservationsResponse(NasaSourceResponse):
    dataset_id: str
    count: Count
    # Different investigations measure different quantities. Preserve original
    # field meaning and missingness instead of pretending all records are rows
    # of the complete-target model-training CSV.
    observations: list[dict[str, Any]]
