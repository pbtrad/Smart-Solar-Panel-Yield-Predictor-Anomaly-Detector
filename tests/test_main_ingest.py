import unittest
from unittest.mock import Mock, patch
import sys
from datetime import datetime

from src.ingest import parse_date_range, main


class TestParseDateRange(unittest.TestCase):
    def test_parse_date_range_valid(self):
        start_date = "2024-01-01T10:00:00"
        end_date = "2024-01-01T11:00:00"

        start, end = parse_date_range(start_date, end_date)

        self.assertIsInstance(start, datetime)
        self.assertIsInstance(end, datetime)
        self.assertEqual(start.year, 2024)
        self.assertEqual(start.month, 1)
        self.assertEqual(start.day, 1)
        self.assertEqual(start.hour, 10)
        self.assertEqual(end.hour, 11)

    def test_parse_date_range_invalid_format(self):
        start_date = "invalid-date"
        end_date = "2024-01-01T11:00:00"

        with self.assertRaises(ValueError):
            parse_date_range(start_date, end_date)

    def test_parse_date_range_invalid_end_date(self):
        start_date = "2024-01-01T10:00:00"
        end_date = "invalid-date"

        with self.assertRaises(ValueError):
            parse_date_range(start_date, end_date)


class TestMainFunction(unittest.TestCase):
    def setUp(self):
        self.original_argv = sys.argv
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def tearDown(self):
        sys.argv = self.original_argv
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    @patch("src.ingest.ConfigManager")
    @patch("src.ingest.DataIngestionOrchestrator")
    @patch("src.ingest.DataStorageManager")
    def test_main_success(
        self, mock_storage_manager, mock_orchestrator, mock_config_manager
    ):
        sys.argv = [
            "ingest.py",
            "--start-date",
            "2024-01-01T10:00:00",
            "--end-date",
            "2024-01-01T11:00:00",
            "--source",
            "all",
        ]

        mock_config_instance = Mock()
        mock_config_instance.validate_config.return_value = True
        mock_config_instance.get_ingestion_config.return_value = {
            "inverter": {"enabled": False}
        }
        mock_config_instance.get_storage_config.return_value = {
            "timeseries": {"url": "", "token": "", "org": "", "bucket": ""},
            "object": {
                "endpoint_url": "",
                "access_key": "",
                "secret_key": "",
                "bucket_name": "",
            },
        }
        mock_config_manager.return_value = mock_config_instance

        mock_orchestrator_instance = Mock()
        mock_orchestrator_instance.services = {"inverter": Mock()}
        mock_orchestrator_instance.ingest_all_data.return_value = {"inverter": Mock()}
        mock_orchestrator_instance.get_data_summary.return_value = {
            "inverter": {"record_count": 10}
        }
        mock_orchestrator.return_value = mock_orchestrator_instance

        mock_storage_instance = Mock()
        mock_storage_instance.store_ingested_data.return_value = {
            "inverter": "test_key"
        }
        mock_storage_manager.return_value = mock_storage_instance

        result = main()

        self.assertEqual(result, 0)
        mock_config_instance.validate_config.assert_called_once()
        mock_orchestrator_instance.ingest_all_data.assert_called_once()
        mock_storage_instance.store_ingested_data.assert_called_once()
        mock_storage_instance.close.assert_called_once()

    @patch("src.ingest.ConfigManager")
    def test_main_config_validation_failure(self, mock_config_manager):
        sys.argv = [
            "ingest.py",
            "--start-date",
            "2024-01-01T10:00:00",
            "--end-date",
            "2024-01-01T11:00:00",
        ]

        mock_config_instance = Mock()
        mock_config_instance.validate_config.return_value = False
        mock_config_manager.return_value = mock_config_instance

        result = main()

        self.assertEqual(result, 1)

    @patch("src.ingest.ConfigManager")
    @patch("src.ingest.DataIngestionOrchestrator")
    @patch("src.ingest.DataStorageManager")
    def test_main_single_source(
        self, mock_storage_manager, mock_orchestrator, mock_config_manager
    ):
        sys.argv = [
            "ingest.py",
            "--start-date",
            "2024-01-01T10:00:00",
            "--end-date",
            "2024-01-01T11:00:00",
            "--source",
            "inverter",
        ]

        mock_config_instance = Mock()
        mock_config_instance.validate_config.return_value = True
        mock_config_instance.get_ingestion_config.return_value = {
            "inverter": {"enabled": True}
        }
        mock_config_instance.get_storage_config.return_value = {
            "timeseries": {"url": "", "token": "", "org": "", "bucket": ""},
            "object": {
                "endpoint_url": "",
                "access_key": "",
                "secret_key": "",
                "bucket_name": "",
            },
        }
        mock_config_manager.return_value = mock_config_instance

        mock_orchestrator_instance = Mock()
        mock_orchestrator_instance.services = {"inverter": Mock()}
        mock_orchestrator_instance.ingest_all_data.return_value = {"inverter": Mock()}
        mock_orchestrator_instance.get_data_summary.return_value = {
            "inverter": {"record_count": 5}
        }
        mock_orchestrator.return_value = mock_orchestrator_instance

        mock_storage_instance = Mock()
        mock_storage_instance.store_ingested_data.return_value = {
            "inverter": "test_key"
        }
        mock_storage_manager.return_value = mock_storage_instance

        result = main()

        self.assertEqual(result, 0)

    @patch("src.ingest.ConfigManager")
    @patch("src.ingest.DataIngestionOrchestrator")
    @patch("src.ingest.DataStorageManager")
    def test_main_source_not_configured(
        self, mock_storage_manager, mock_orchestrator, mock_config_manager
    ):
        sys.argv = [
            "ingest.py",
            "--start-date",
            "2024-01-01T10:00:00",
            "--end-date",
            "2024-01-01T11:00:00",
            "--source",
            "inverter",
        ]

        mock_config_instance = Mock()
        mock_config_instance.validate_config.return_value = True
        mock_config_instance.get_ingestion_config.return_value = {}
        mock_config_instance.get_storage_config.return_value = {
            "timeseries": {"url": "", "token": "", "org": "", "bucket": ""},
            "object": {
                "endpoint_url": "",
                "access_key": "",
                "secret_key": "",
                "bucket_name": "",
            },
        }
        mock_config_manager.return_value = mock_config_instance

        mock_orchestrator_instance = Mock()
        mock_orchestrator_instance.services = {}
        mock_orchestrator.return_value = mock_orchestrator_instance

        result = main()

        self.assertEqual(result, 1)

    @patch("src.ingest.ConfigManager")
    @patch("src.ingest.DataIngestionOrchestrator")
    @patch("src.ingest.DataStorageManager")
    def test_main_dry_run(
        self, mock_storage_manager, mock_orchestrator, mock_config_manager
    ):
        sys.argv = [
            "ingest.py",
            "--start-date",
            "2024-01-01T10:00:00",
            "--end-date",
            "2024-01-01T11:00:00",
            "--dry-run",
        ]

        mock_config_instance = Mock()
        mock_config_instance.validate_config.return_value = True
        mock_config_instance.get_ingestion_config.return_value = {
            "inverter": {"enabled": False}
        }
        mock_config_instance.get_storage_config.return_value = {
            "timeseries": {"url": "", "token": "", "org": "", "bucket": ""},
            "object": {
                "endpoint_url": "",
                "access_key": "",
                "secret_key": "",
                "bucket_name": "",
            },
        }
        mock_config_manager.return_value = mock_config_instance

        mock_orchestrator_instance = Mock()
        mock_orchestrator_instance.services = {"inverter": Mock()}
        mock_orchestrator_instance.ingest_all_data.return_value = {"inverter": Mock()}
        mock_orchestrator_instance.get_data_summary.return_value = {
            "inverter": {"record_count": 10}
        }
        mock_orchestrator.return_value = mock_orchestrator_instance

        mock_storage_instance = Mock()
        mock_storage_manager.return_value = mock_storage_instance

        result = main()

        self.assertEqual(result, 0)
        mock_storage_instance.store_ingested_data.assert_not_called()

    @patch("src.ingest.ConfigManager")
    @patch("src.ingest.DataIngestionOrchestrator")
    @patch("src.ingest.DataStorageManager")
    def test_main_no_data_ingested(
        self, mock_storage_manager, mock_orchestrator, mock_config_manager
    ):
        sys.argv = [
            "ingest.py",
            "--start-date",
            "2024-01-01T10:00:00",
            "--end-date",
            "2024-01-01T11:00:00",
        ]

        mock_config_instance = Mock()
        mock_config_instance.validate_config.return_value = True
        mock_config_instance.get_ingestion_config.return_value = {
            "inverter": {"enabled": False}
        }
        mock_config_instance.get_storage_config.return_value = {
            "timeseries": {"url": "", "token": "", "org": "", "bucket": ""},
            "object": {
                "endpoint_url": "",
                "access_key": "",
                "secret_key": "",
                "bucket_name": "",
            },
        }
        mock_config_manager.return_value = mock_config_instance

        mock_orchestrator_instance = Mock()
        mock_orchestrator_instance.services = {}
        mock_orchestrator_instance.ingest_all_data.return_value = {}
        mock_orchestrator_instance.get_data_summary.return_value = {}
        mock_orchestrator.return_value = mock_orchestrator_instance

        mock_storage_instance = Mock()
        mock_storage_manager.return_value = mock_storage_instance

        result = main()

        self.assertEqual(result, 0)
        mock_storage_instance.store_ingested_data.assert_not_called()

    @patch("src.ingest.ConfigManager")
    def test_main_exception_handling(self, mock_config_manager):
        sys.argv = [
            "ingest.py",
            "--start-date",
            "2024-01-01T10:00:00",
            "--end-date",
            "2024-01-01T11:00:00",
        ]

        mock_config_instance = Mock()
        mock_config_instance.validate_config.side_effect = Exception("Config error")
        mock_config_manager.return_value = mock_config_instance

        result = main()

        self.assertEqual(result, 1)


if __name__ == "__main__":
    unittest.main()
