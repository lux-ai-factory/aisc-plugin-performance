"""Tests for data input provider."""

import io
import pytest
import pandas as pd
import numpy as np

from predictive_insights.data_input_provider import (
    DataFrameProvider,
    dataframe_iter,
)


class TestDataFrameProvider:
    """Tests for DataFrameProvider class."""

    @pytest.fixture
    def sample_csv_bytes(self):
        """Create sample CSV data as bytes."""
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [10, 20, 30, 40, 50],
                "target": [0, 1, 0, 1, 0],
            }
        )
        buffer = io.BytesIO()
        df.to_csv(buffer, index=False)
        return buffer.getvalue()

    @pytest.fixture
    def sample_parquet_bytes(self):
        """Create sample Parquet data as bytes."""
        pytest.importorskip("pyarrow")
        df = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [10, 20, 30, 40, 50],
                "target": [0, 1, 0, 1, 0],
            }
        )
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False)
        return buffer.getvalue()

    def test_read_csv_single_file(self, sample_csv_bytes):
        provider = DataFrameProvider(sample_csv_bytes)
        data = provider.get_data()
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 5

    @pytest.mark.skipif(
        not pytest.importorskip("pyarrow", reason="pyarrow not installed"),
        reason="pyarrow required",
    )
    def test_read_parquet_single_file(self, sample_parquet_bytes):
        provider = DataFrameProvider(sample_parquet_bytes)
        data = provider.get_data()
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 5


class TestDataFrameProviderIterator:
    """Tests for dataframe_iter method."""

    @pytest.fixture
    def dataframe_with_dates(self):
        """Create provider with date column."""
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        return pd.DataFrame(
            {
                "date": dates,
                "feature1": np.random.rand(30),
                "target": np.random.randint(0, 2, 30),
            }
        )

    def test_iter_without_date_feature(self, dataframe_with_dates):
        """Without date feature, should yield single batch."""
        batches = list(dataframe_iter(dataframe_with_dates, None, None, None))
        assert len(batches) == 1
        date, mask = batches[0]
        assert date is None

    def test_iter_with_date_feature(self, dataframe_with_dates):
        """With date feature, should yield multiple batches."""
        batches = list(dataframe_iter(dataframe_with_dates, "date", "7D", "7D"))
        assert len(batches) > 1
        for date, mask in batches:
            assert date is not None
            assert isinstance(mask, pd.Series)
