from typing import Dict
import tempfile

import pandas as pd
import pytest

from polypesto.core.experiment import Dataset


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Reusable sample data"""
    return pd.DataFrame(
        {
            "Time (min)": [0.0, 1.0, 2.0],
            "Conversion A": [0.0, 0.5, 0.8],
            "Conversion B": [0.0, 0.3, 0.6],
        }
    )


@pytest.fixture
def sample_obs_map() -> Dict[str, str]:
    """Reusable observable mapping"""
    return {"xA": "Conversion A", "xB": "Conversion B"}


def test_dataset_basic_loading(
    sample_dataframe: pd.DataFrame, sample_obs_map: Dict[str, str]
):
    """Test that Dataset.load() works with basic inputs."""
    dataset = Dataset.load(sample_dataframe, tkey="Time (min)", obs_map=sample_obs_map)

    # Assert: Check that everything worked correctly
    assert dataset.tkey == "Time (min)"
    assert dataset.obs_map == sample_obs_map
    assert len(dataset.data) == 3
    assert "Conversion A" in dataset.data.columns
    assert "Conversion B" in dataset.data.columns
    assert "Time (min)" in dataset.data.columns
    assert dataset.data["Time (min)"].tolist() == [0.0, 1.0, 2.0]
    assert dataset.data["Conversion A"].tolist() == [0.0, 0.5, 0.8]
    assert dataset.data["Conversion B"].tolist() == [0.0, 0.3, 0.6]
    assert isinstance(dataset.id, str)
    assert len(dataset.id) > 0


def test_dataset_load_from_file(
    sample_dataframe: pd.DataFrame, sample_obs_map: Dict[str, str]
):
    """Test Dataset.load() can read from a file."""
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        sample_dataframe.to_csv(tmp_file.name, index=False)

        # Act
        dataset = Dataset.load(tmp_file.name, tkey="Time (min)", obs_map=sample_obs_map)

        # Assert
        assert dataset.tkey == "Time (min)"
        assert dataset.obs_map == sample_obs_map
        assert len(dataset.data) == 3
        assert "Conversion A" in dataset.data.columns
        assert "Conversion B" in dataset.data.columns
        assert "Time (min)" in dataset.data.columns


# Error handling tests
def test_dataset_load_missing_columns(sample_obs_map):
    """Test Dataset.load() doesn't validate tkey column exists at load time."""

    # Missing Time (min) column
    bad_data = pd.DataFrame({"Conversion A": [0.1, 0.2], "Conversion B": [0.1, 0.2]})
    with pytest.raises(KeyError):
        dataset = Dataset.load(bad_data, tkey="Time (min)", obs_map=sample_obs_map)

    # Missing Conversion B column
    bad_data = pd.DataFrame({"Time (min)": [0.0, 1.0], "Conversion A": [0.1, 0.2]})
    with pytest.raises(KeyError):
        dataset = Dataset.load(bad_data, tkey="Time (min)", obs_map=sample_obs_map)


# Edge cases
def test_dataset_load_empty_dataframe():
    """Test Dataset.load() with empty DataFrame."""
    empty_data = pd.DataFrame(columns=["time", "value"])
    obs_map = {"x": "value"}

    with pytest.raises(ValueError):
        dataset = Dataset.load(empty_data, tkey="time", obs_map=obs_map)


def test_dataset_load_with_nan_values():
    """Test Dataset.load() handles NaN values correctly."""
    data_with_nans = pd.DataFrame(
        {"time": [0.0, 1.0, 2.0], "signal": [1.0, float("nan"), 3.0]}
    )

    dataset = Dataset.load(data_with_nans, tkey="time", obs_map={"x": "signal"})

    # NaN should be preserved in the data
    assert len(dataset.data) == 3
    assert pd.isna(dataset.data["signal"].iloc[1])


def test_dataset_load_from_nonexistent_file():
    """Test Dataset.load() fails gracefully with bad file path."""
    with pytest.raises(FileNotFoundError):
        Dataset.load("nonexistent_file.csv", tkey="time", obs_map={"x": "y"})
