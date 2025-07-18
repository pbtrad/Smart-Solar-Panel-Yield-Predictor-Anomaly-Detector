import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta, timezone
import requests
from io import StringIO

from src.data.ingestion import (
    DataIngestionError,
    BaseIngestionService,
    InverterIngestionService,
    WeatherIngestionService,
    MaintenanceIngestionService,
    DataIngestionOrchestrator,
)


class TestDataIngestionError(unittest.TestCase):
    def test_exception_creation(self):
        error = DataIngestionError("Test error message")
        self.assertEqual(str(error), "Test error message")


class TestBaseIngestionService(unittest.TestCase):
    def setUp(self):
        self.service = InverterIngestionService(
            api_url="https://test.api.com", api_key="test_key", inverter_ids=["test"]
        )

    def test_validate_response_success(self):
        response = Mock()
        response.status_code = 200
        response.text = "Success"

        self.service.validate_response(response)

    def test_validate_response_failure(self):
        response = Mock()
        response.status_code = 404
        response.text = "Not Found"

        with self.assertRaises(DataIngestionError):
            self.service.validate_response(response)


class TestInverterIngestionService(unittest.TestCase):
    def setUp(self):
        self.service = InverterIngestionService(
            api_url="https://test.api.com",
            api_key="test_key",
            inverter_ids=["inv_001", "inv_002"],
        )

    def test_fetch_data_success(self):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "measurements": [
                {
                    "timestamp": "2024-01-01T10:00:00Z",
                    "power_output": 100.5,
                    "voltage": 240.0,
                    "current": 0.42,
                },
                {
                    "timestamp": "2024-01-01T11:00:00Z",
                    "power_output": 120.3,
                    "voltage": 240.0,
                    "current": 0.50,
                },
            ]
        }

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        self.service.session = mock_session

        start_date = datetime(2024, 1, 1, 10, 0, 0)
        end_date = datetime(2024, 1, 1, 11, 0, 0)

        result = self.service.fetch_data(start_date, end_date)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 4)
        self.assertIn("inverter_id", result.columns)
        self.assertIn("power_output", result.columns)

    def test_fetch_data_api_failure(self):
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        self.service.session = mock_session

        start_date = datetime(2024, 1, 1, 10, 0, 0)
        end_date = datetime(2024, 1, 1, 11, 0, 0)

        with self.assertRaises(DataIngestionError):
            self.service.fetch_data(start_date, end_date)

    def test_fetch_data_no_inverters(self):
        service = InverterIngestionService(
            api_url="https://test.api.com", api_key="test_key", inverter_ids=[]
        )

        start_date = datetime(2024, 1, 1, 10, 0, 0)
        end_date = datetime(2024, 1, 1, 11, 0, 0)

        with self.assertRaises(DataIngestionError):
            service.fetch_data(start_date, end_date)


class TestWeatherIngestionService(unittest.TestCase):
    def setUp(self):
        self.service = WeatherIngestionService(
            api_url="https://test.weather.api.com",
            api_key="test_key",
            location={"lat": 40.7128, "lon": -74.0060},
        )

    def test_fetch_data_success(self):
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "hourly": [
                {
                    "dt": 1704110400,
                    "temp": 15.5,
                    "humidity": 65,
                    "wind_speed": 5.2,
                    "clouds": 20,
                    "solar_radiation": 800.0,
                },
                {
                    "dt": 1704114000,
                    "temp": 16.2,
                    "humidity": 60,
                    "wind_speed": 4.8,
                    "clouds": 15,
                    "solar_radiation": 850.0,
                },
            ]
        }

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        self.service.session = mock_session

        start_date = datetime(2024, 1, 1, 10, 0, 0)
        end_date = datetime(2024, 1, 1, 11, 0, 0)

        result = self.service.fetch_data(start_date, end_date)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)
        self.assertIn("temp", result.columns)
        self.assertIn("timestamp", result.columns)

    def test_fetch_data_api_failure(self):
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        self.service.session = mock_session

        start_date = datetime(2024, 1, 1, 10, 0, 0)
        end_date = datetime(2024, 1, 1, 11, 0, 0)

        with self.assertRaises(DataIngestionError):
            self.service.fetch_data(start_date, end_date)


class TestMaintenanceIngestionService(unittest.TestCase):
    def setUp(self):
        self.test_data = StringIO(
            "timestamp,maintenance_type,description,inverter_id\n"
            "2024-01-01T10:00:00,cleaning,Panel cleaning performed,inv_001\n"
            "2024-01-02T14:00:00,inspection,Regular inspection,inv_002\n"
        )

    def test_fetch_data_success(self):
        test_df = pd.read_csv(self.test_data)

        with patch.object(pd, "read_csv", return_value=test_df):
            service = MaintenanceIngestionService("test_maintenance.csv")

            start_date = datetime(2024, 1, 1, 9, 0, 0)
            end_date = datetime(2024, 1, 2, 15, 0, 0)

            result = service.fetch_data(start_date, end_date)

            self.assertIsInstance(result, pd.DataFrame)
            self.assertEqual(len(result), 2)
            self.assertIn("maintenance_type", result.columns)

    @patch("builtins.open", create=True)
    def test_fetch_data_file_not_found(self, mock_open):
        mock_open.side_effect = FileNotFoundError("File not found")

        service = MaintenanceIngestionService("nonexistent.csv")

        start_date = datetime(2024, 1, 1, 10, 0, 0)
        end_date = datetime(2024, 1, 1, 11, 0, 0)

        with self.assertRaises(DataIngestionError):
            service.fetch_data(start_date, end_date)


class TestDataIngestionOrchestrator(unittest.TestCase):
    def setUp(self):
        self.config = {
            "inverter": {
                "enabled": True,
                "api_url": "https://test.api.com",
                "api_key": "test_key",
                "inverter_ids": ["inv_001"],
            },
            "weather": {
                "enabled": True,
                "api_url": "https://test.weather.api.com",
                "api_key": "test_key",
                "location": {"lat": 40.7128, "lon": -74.0060},
            },
            "maintenance": {"enabled": False, "file_path": ""},
        }

    @patch("src.data.ingestion.InverterIngestionService")
    @patch("src.data.ingestion.WeatherIngestionService")
    def test_initialize_services(self, mock_weather_service, mock_inverter_service):
        orchestrator = DataIngestionOrchestrator(self.config)

        self.assertIn("inverter", orchestrator.services)
        self.assertIn("weather", orchestrator.services)
        self.assertNotIn("maintenance", orchestrator.services)

    @patch("src.data.ingestion.InverterIngestionService")
    @patch("src.data.ingestion.WeatherIngestionService")
    def test_ingest_all_data_success(self, mock_weather_service, mock_inverter_service):
        mock_inverter_instance = Mock()
        mock_inverter_instance.fetch_data.return_value = pd.DataFrame(
            {
                "timestamp": [datetime.now()],
                "power_output": [100.0],
                "inverter_id": ["inv_001"],
            }
        )

        mock_weather_instance = Mock()
        mock_weather_instance.fetch_data.return_value = pd.DataFrame(
            {"timestamp": [datetime.now()], "temp": [20.0], "humidity": [60]}
        )

        mock_inverter_service.return_value = mock_inverter_instance
        mock_weather_service.return_value = mock_weather_instance

        orchestrator = DataIngestionOrchestrator(self.config)

        start_date = datetime(2024, 1, 1, 10, 0, 0)
        end_date = datetime(2024, 1, 1, 11, 0, 0)

        result = orchestrator.ingest_all_data(start_date, end_date)

        self.assertIn("inverter", result)
        self.assertIn("weather", result)
        self.assertIsInstance(result["inverter"], pd.DataFrame)
        self.assertIsInstance(result["weather"], pd.DataFrame)

    def test_get_data_summary(self):
        orchestrator = DataIngestionOrchestrator(self.config)

        test_data = {
            "inverter": pd.DataFrame(
                {
                    "timestamp": [
                        datetime(2024, 1, 1, 10, 0, 0),
                        datetime(2024, 1, 1, 11, 0, 0),
                    ],
                    "power_output": [100.0, 120.0],
                    "inverter_id": ["inv_001", "inv_001"],
                }
            ),
            "weather": pd.DataFrame(
                {
                    "timestamp": [datetime(2024, 1, 1, 10, 0, 0)],
                    "temp": [20.0],
                    "humidity": [60],
                }
            ),
        }

        summary = orchestrator.get_data_summary(test_data)

        self.assertIn("inverter", summary)
        self.assertIn("weather", summary)
        self.assertEqual(summary["inverter"]["record_count"], 2)
        self.assertEqual(summary["weather"]["record_count"], 1)


if __name__ == "__main__":
    unittest.main()
