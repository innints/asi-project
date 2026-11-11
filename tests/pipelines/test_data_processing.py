"""Test przepływu danych przez przygotowanie do modelowania."""

from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from asi_project.pipelines.data_processing import create_pipeline


def test_data_processing_validates_normalizes_and_filters(project):
    target = project.contract.target
    category = project.config.categorical_features[0]
    raw = project.sample.groupby(target, sort=False).head(2).reset_index(drop=True)
    raw[target] = raw[target].astype(object)
    raw.at[0, target] = " \t"
    raw[category] = raw[category].astype(object)
    raw.at[1, category] = "  example  "
    raw = raw.rename(columns={target: f" {target} "})

    catalog = DataCatalog({"raw_data": MemoryDataset(raw), "prepared_data": MemoryDataset()})
    SequentialRunner().run(create_pipeline(), catalog)
    prepared = catalog.load("prepared_data")

    assert prepared.columns.tolist() == project.sample.columns.tolist()
    assert prepared.index.tolist() == [1, 2, 3]
    assert prepared[target].tolist() == raw[f" {target} "].iloc[1:].tolist()
    assert prepared.at[1, category] == "example"
