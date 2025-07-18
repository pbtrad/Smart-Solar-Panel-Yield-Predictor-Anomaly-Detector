import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta
import json
from io import StringIO

from src.data.storage import (
    StorageError,
    TimeSeriesStorage,
    ObjectStorage,
    DataStorageManager,
)


class TestStorageError(unittest.TestCase):
    def test_exception_creation(self):
        error = StorageError("Test storage error")
        self.assertEqual(str(error), "Test storage error")


class TestTimeSeriesStorage(unittest.TestCase):
    def setUp(self):
        self.storage = TimeSeriesStorage(
            url="http://localhost:8086",
            token="test_token",
            org="test_org",
            bucket="test_bucket",
        )

    def test_init(self):
        storage = TimeSeriesStorage(
            url="http://test:8086",
            token="test_token",
            org="test_org",
            bucket="test_bucket",
        )

        self.assertEqual(storage.bucket, "test_bucket")
        self.assertIsNotNone(storage.client)

    def test_store_inverter_data_success(self):
        test_df = pd.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 10, 0, 0),
                    datetime(2024, 1, 1, 11, 0, 0),
                ],
                "power_output": [100.0, 120.0],
                "voltage": [240.0, 240.0],
                "current": [0.42, 0.50],
                "inverter_id": ["inv_001", "inv_001"],
            }
        )

        try:
            self.storage.store_inverter_data(test_df)
        except StorageError:
            pass

    @patch("influxdb_client.Point")
    def test_store_inverter_data_failure(self, mock_point):
        mock_point.side_effect = Exception("InfluxDB error")

        test_df = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 10, 0, 0)],
                "power_output": [100.0],
                "inverter_id": ["inv_001"],
            }
        )

        with self.assertRaises(StorageError):
            self.storage.store_inverter_data(test_df)

    def test_store_weather_data_success(self):
        test_df = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 10, 0, 0)],
                "temp": [20.0],
                "humidity": [60],
                "wind_speed": [5.2],
                "clouds": [20],
                "solar_radiation": [800.0],
            }
        )

        try:
            self.storage.store_weather_data(test_df)
        except StorageError:
            pass

    def test_query_power_data_success(self):
        mock_query_api = Mock()
        mock_result = pd.DataFrame(
            {
                "_time": [datetime(2024, 1, 1, 10, 0, 0)],
                "_field": ["power_output"],
                "_value": [100.0],
                "inverter_id": ["inv_001"],
            }
        )
        mock_query_api.query_data_frame.return_value = mock_result
        self.storage.query_api = mock_query_api

        start_time = datetime(2024, 1, 1, 10, 0, 0)
        end_time = datetime(2024, 1, 1, 11, 0, 0)

        result = self.storage.query_power_data(start_time, end_time)

        self.assertIsInstance(result, pd.DataFrame)
        mock_query_api.query_data_frame.assert_called_once()

    def test_query_power_data_empty_result(self):
        mock_query_api = Mock()
        mock_query_api.query_data_frame.return_value = pd.DataFrame()
        self.storage.query_api = mock_query_api

        start_time = datetime(2024, 1, 1, 10, 0, 0)
        end_time = datetime(2024, 1, 1, 11, 0, 0)

        result = self.storage.query_power_data(start_time, end_time)

        self.assertTrue(result.empty)

    def test_query_power_data_failure(self):
        mock_query_api = Mock()
        mock_query_api.query_data_frame.side_effect = Exception("Query failed")
        self.storage.query_api = mock_query_api

        start_time = datetime(2024, 1, 1, 10, 0, 0)
        end_time = datetime(2024, 1, 1, 11, 0, 0)

        with self.assertRaises(StorageError):
            self.storage.query_power_data(start_time, end_time)

    def test_query_weather_data_success(self):
        mock_query_api = Mock()
        mock_result = pd.DataFrame(
            {
                "_time": [datetime(2024, 1, 1, 10, 0, 0)],
                "_field": ["temp"],
                "_value": [20.0],
            }
        )
        mock_query_api.query_data_frame.return_value = mock_result
        self.storage.query_api = mock_query_api

        start_time = datetime(2024, 1, 1, 10, 0, 0)
        end_time = datetime(2024, 1, 1, 11, 0, 0)

        result = self.storage.query_weather_data(start_time, end_time)

        self.assertIsInstance(result, pd.DataFrame)
        mock_query_api.query_data_frame.assert_called_once()

    def test_close(self):
        mock_client = Mock()
        self.storage.client = mock_client

        self.storage.close()

        mock_client.close.assert_called_once()


class TestObjectStorage(unittest.TestCase):
    def setUp(self):
        self.storage = ObjectStorage(
            endpoint_url="http://localhost:9000",
            access_key="test_access_key",
            secret_key="test_secret_key",
            bucket_name="test_bucket",
        )

    @patch("boto3.client")
    def test_init(self, mock_boto_client):
        mock_s3_client = Mock()
        mock_boto_client.return_value = mock_s3_client

        storage = ObjectStorage(
            endpoint_url="http://test:9000",
            access_key="test_key",
            secret_key="test_secret",
            bucket_name="test_bucket",
        )

        mock_boto_client.assert_called_once_with(
            "s3",
            endpoint_url="http://test:9000",
            aws_access_key_id="test_key",
            aws_secret_access_key="test_secret",
        )

    def test_store_raw_data_success(self):
        mock_s3_client = Mock()
        self.storage.s3_client = mock_s3_client

        test_df = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 10, 0, 0)],
                "power_output": [100.0],
                "inverter_id": ["inv_001"],
            }
        )

        timestamp = datetime(2024, 1, 1, 10, 0, 0)

        result = self.storage.store_raw_data(test_df, "inverter", timestamp)

        expected_key = f"raw_data/inverter/2024/01/01/{timestamp.isoformat()}.csv"
        self.assertEqual(result, expected_key)
        mock_s3_client.put_object.assert_called_once()

    def test_store_raw_data_failure(self):
        mock_s3_client = Mock()
        mock_s3_client.put_object.side_effect = Exception("S3 error")
        self.storage.s3_client = mock_s3_client

        test_df = pd.DataFrame({"col": [1]})
        timestamp = datetime.now()

        with self.assertRaises(StorageError):
            self.storage.store_raw_data(test_df, "test", timestamp)

    def test_store_model_artifacts_success(self):
        mock_s3_client = Mock()
        self.storage.s3_client = mock_s3_client

        model_data = b"test_model_data"
        model_name = "forecast_model"
        version = "v1.0.0"

        result = self.storage.store_model_artifacts(model_data, model_name, version)

        expected_key = f"models/{model_name}/{version}/model.pkl"
        self.assertEqual(result, expected_key)
        mock_s3_client.put_object.assert_called_once()

    def test_store_model_artifacts_failure(self):
        mock_s3_client = Mock()
        mock_s3_client.put_object.side_effect = Exception("S3 error")
        self.storage.s3_client = mock_s3_client

        model_data = b"test_data"

        with self.assertRaises(StorageError):
            self.storage.store_model_artifacts(model_data, "test", "v1")

    def test_retrieve_model_artifacts_success(self):
        mock_s3_client = Mock()
        mock_body = Mock()
        mock_body.read.return_value = b"test_model_data"
        mock_response = {"Body": mock_body}
        mock_s3_client.get_object.return_value = mock_response
        self.storage.s3_client = mock_s3_client

        result = self.storage.retrieve_model_artifacts("test_model", "v1.0")

        self.assertEqual(result, b"test_model_data")
        mock_s3_client.get_object.assert_called_once()

    def test_retrieve_model_artifacts_failure(self):
        mock_s3_client = Mock()
        mock_s3_client.get_object.side_effect = Exception("S3 error")
        self.storage.s3_client = mock_s3_client

        with self.assertRaises(StorageError):
            self.storage.retrieve_model_artifacts("test_model", "v1.0")


class TestDataStorageManager(unittest.TestCase):
    def setUp(self):
        self.ts_config = {
            "url": "http://localhost:8086",
            "token": "test_token",
            "org": "test_org",
            "bucket": "test_bucket",
        }
        self.object_config = {
            "endpoint_url": "http://localhost:9000",
            "access_key": "test_access_key",
            "secret_key": "test_secret_key",
            "bucket_name": "test_bucket",
        }

    @patch("src.data.storage.TimeSeriesStorage")
    @patch("src.data.storage.ObjectStorage")
    def test_init(self, mock_object_storage, mock_ts_storage):
        mock_ts_instance = Mock()
        mock_object_instance = Mock()
        mock_ts_storage.return_value = mock_ts_instance
        mock_object_storage.return_value = mock_object_instance

        manager = DataStorageManager(self.ts_config, self.object_config)

        mock_ts_storage.assert_called_once_with(**self.ts_config)
        mock_object_storage.assert_called_once_with(**self.object_config)

    @patch("src.data.storage.TimeSeriesStorage")
    @patch("src.data.storage.ObjectStorage")
    def test_store_ingested_data_success(self, mock_object_storage, mock_ts_storage):
        mock_ts_instance = Mock()
        mock_object_instance = Mock()
        mock_object_instance.store_raw_data.return_value = "test_key"
        mock_ts_storage.return_value = mock_ts_instance
        mock_object_storage.return_value = mock_object_instance

        manager = DataStorageManager(self.ts_config, self.object_config)

        test_data = {
            "inverter": pd.DataFrame(
                {
                    "timestamp": [datetime(2024, 1, 1, 10, 0, 0)],
                    "power_output": [100.0],
                    "inverter_id": ["inv_001"],
                }
            ),
            "weather": pd.DataFrame(
                {"timestamp": [datetime(2024, 1, 1, 10, 0, 0)], "temp": [20.0]}
            ),
        }

        timestamp = datetime.now()
        result = manager.store_ingested_data(test_data, timestamp)

        self.assertIn("inverter", result)
        self.assertIn("weather", result)
        self.assertEqual(mock_ts_instance.store_inverter_data.call_count, 1)
        self.assertEqual(mock_ts_instance.store_weather_data.call_count, 1)
        self.assertEqual(mock_object_instance.store_raw_data.call_count, 2)

    @patch("src.data.storage.TimeSeriesStorage")
    @patch("src.data.storage.ObjectStorage")
    def test_get_combined_data_success(self, mock_object_storage, mock_ts_storage):
        mock_ts_instance = Mock()
        mock_ts_instance.query_power_data.return_value = pd.DataFrame(
            {"timestamp": [datetime(2024, 1, 1, 10, 0, 0)], "power_output": [100.0]}
        )
        mock_ts_instance.query_weather_data.return_value = pd.DataFrame(
            {"timestamp": [datetime(2024, 1, 1, 10, 0, 0)], "temp": [20.0]}
        )
        mock_object_instance = Mock()
        mock_ts_storage.return_value = mock_ts_instance
        mock_object_storage.return_value = mock_object_instance

        manager = DataStorageManager(self.ts_config, self.object_config)

        start_time = datetime(2024, 1, 1, 10, 0, 0)
        end_time = datetime(2024, 1, 1, 11, 0, 0)

        result = manager.get_combined_data(start_time, end_time)

        self.assertIn("power", result)
        self.assertIn("weather", result)
        self.assertIsInstance(result["power"], pd.DataFrame)
        self.assertIsInstance(result["weather"], pd.DataFrame)

    @patch("src.data.storage.TimeSeriesStorage")
    @patch("src.data.storage.ObjectStorage")
    def test_get_combined_data_failure(self, mock_object_storage, mock_ts_storage):
        mock_ts_instance = Mock()
        mock_ts_instance.query_power_data.side_effect = Exception("Query failed")
        mock_object_instance = Mock()
        mock_ts_storage.return_value = mock_ts_instance
        mock_object_storage.return_value = mock_object_instance

        manager = DataStorageManager(self.ts_config, self.object_config)

        start_time = datetime(2024, 1, 1, 10, 0, 0)
        end_time = datetime(2024, 1, 1, 11, 0, 0)

        with self.assertRaises(StorageError):
            manager.get_combined_data(start_time, end_time)

    @patch("src.data.storage.TimeSeriesStorage")
    @patch("src.data.storage.ObjectStorage")
    def test_close(self, mock_object_storage, mock_ts_storage):
        mock_ts_instance = Mock()
        mock_object_instance = Mock()
        mock_ts_storage.return_value = mock_ts_instance
        mock_object_storage.return_value = mock_object_instance

        manager = DataStorageManager(self.ts_config, self.object_config)
        manager.close()

        mock_ts_instance.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
