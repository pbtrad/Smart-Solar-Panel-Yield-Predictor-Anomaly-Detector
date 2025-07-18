import unittest
from unittest.mock import Mock, patch, mock_open
import os
import yaml

from src.utils.config import ConfigManager


class TestConfigManager(unittest.TestCase):
    def setUp(self):
        self.test_config = {
            "inverter": {
                "enabled": False,
                "api_url": "",
                "api_key": "",
                "inverter_ids": [],
            },
            "weather": {
                "enabled": False,
                "api_url": "",
                "api_key": "",
                "location": {"lat": 0.0, "lon": 0.0},
            },
            "maintenance": {"enabled": False, "file_path": ""},
            "storage": {
                "timeseries": {
                    "url": "http://localhost:8086",
                    "token": "",
                    "org": "",
                    "bucket": "solar_data",
                },
                "object": {
                    "endpoint_url": "http://localhost:9000",
                    "access_key": "",
                    "secret_key": "",
                    "bucket_name": "solar-data",
                },
            },
            "models": {
                "forecast": {
                    "algorithm": "prophet",
                    "horizon_hours": 24,
                    "retrain_frequency_hours": 168,
                },
                "anomaly": {"algorithm": "isolation_forest", "threshold": 0.95},
            },
            "api": {"host": "0.0.0.0", "port": 8000, "debug": False},
            "notifications": {
                "enabled": False,
                "email": {
                    "smtp_server": "",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "from_email": "",
                    "to_emails": [],
                },
                "sms": {
                    "twilio_account_sid": "",
                    "twilio_auth_token": "",
                    "twilio_phone_number": "",
                    "to_phone_numbers": [],
                },
            },
        }

    def test_init_with_default_config(self):
        with patch("builtins.open", side_effect=FileNotFoundError):
            config_manager = ConfigManager()

            self.assertIsInstance(config_manager.config, dict)
            self.assertIn("inverter", config_manager.config)
            self.assertIn("weather", config_manager.config)
            self.assertIn("storage", config_manager.config)

    def test_init_with_custom_config_path(self):
        test_config_yaml = yaml.dump(self.test_config)

        with patch("builtins.open", mock_open(read_data=test_config_yaml)):
            config_manager = ConfigManager("custom_config.yaml")

            self.assertEqual(config_manager.config_path, "custom_config.yaml")
            self.assertEqual(config_manager.config["inverter"]["enabled"], False)

    def test_load_env_vars(self):
        env_vars = {
            "INVERTER_ENABLED": "true",
            "INVERTER_API_URL": "https://test.api.com",
            "WEATHER_LOCATION_LAT": "40.7128",
            "WEATHER_LOCATION_LON": "-74.0060",
            "API_PORT": "9000",
            "MODELS_FORECAST_ALGORITHM": "xgboost",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            config_manager = ConfigManager()

            self.assertTrue(config_manager.config["inverter"]["enabled"])
            self.assertEqual(
                config_manager.config["inverter"]["api_url"], "https://test.api.com"
            )
            self.assertEqual(
                config_manager.config["weather"]["location"]["lat"], 40.7128
            )
            self.assertEqual(
                config_manager.config["weather"]["location"]["lon"], -74.0060
            )
            self.assertEqual(config_manager.config["api"]["port"], 9000)
            self.assertEqual(
                config_manager.config["models"]["forecast"]["algorithm"], "xgboost"
            )

    def test_get_simple_key(self):
        config_manager = ConfigManager()
        config_manager.config = self.test_config

        result = config_manager.get("inverter.enabled")
        self.assertEqual(result, False)

        result = config_manager.get("api.port")
        self.assertEqual(result, 8000)

    def test_get_nested_key(self):
        config_manager = ConfigManager()
        config_manager.config = self.test_config

        result = config_manager.get("weather.location.lat")
        self.assertEqual(result, 0.0)

        result = config_manager.get("storage.timeseries.url")
        self.assertEqual(result, "http://localhost:8086")

    def test_get_nonexistent_key(self):
        config_manager = ConfigManager()
        config_manager.config = self.test_config

        result = config_manager.get("nonexistent.key")
        self.assertIsNone(result)

        result = config_manager.get("nonexistent.key", default="default_value")
        self.assertEqual(result, "default_value")

    def test_get_ingestion_config(self):
        config_manager = ConfigManager()
        config_manager.config = self.test_config

        result = config_manager.get_ingestion_config()

        self.assertIn("inverter", result)
        self.assertIn("weather", result)
        self.assertIn("maintenance", result)
        self.assertEqual(result["inverter"]["enabled"], False)
        self.assertEqual(result["weather"]["enabled"], False)

    def test_get_storage_config(self):
        config_manager = ConfigManager()
        config_manager.config = self.test_config

        result = config_manager.get_storage_config()

        self.assertIn("timeseries", result)
        self.assertIn("object", result)
        self.assertEqual(result["timeseries"]["url"], "http://localhost:8086")
        self.assertEqual(result["object"]["endpoint_url"], "http://localhost:9000")

    def test_get_model_config(self):
        config_manager = ConfigManager()
        config_manager.config = self.test_config

        forecast_config = config_manager.get_model_config("forecast")
        anomaly_config = config_manager.get_model_config("anomaly")

        self.assertEqual(forecast_config["algorithm"], "prophet")
        self.assertEqual(forecast_config["horizon_hours"], 24)
        self.assertEqual(anomaly_config["algorithm"], "isolation_forest")
        self.assertEqual(anomaly_config["threshold"], 0.95)

    def test_get_model_config_nonexistent(self):
        config_manager = ConfigManager()
        config_manager.config = self.test_config

        result = config_manager.get_model_config("nonexistent")
        self.assertEqual(result, {})

    def test_get_api_config(self):
        config_manager = ConfigManager()
        config_manager.config = self.test_config

        result = config_manager.get_api_config()

        self.assertEqual(result["host"], "0.0.0.0")
        self.assertEqual(result["port"], 8000)
        self.assertEqual(result["debug"], False)

    def test_get_notification_config(self):
        config_manager = ConfigManager()
        config_manager.config = self.test_config

        result = config_manager.get_notification_config()

        self.assertEqual(result["enabled"], False)
        self.assertIn("email", result)
        self.assertIn("sms", result)
        self.assertEqual(result["email"]["smtp_port"], 587)

    def test_validate_config_success(self):
        config_manager = ConfigManager()
        config_manager.config = self.test_config

        result = config_manager.validate_config()
        self.assertTrue(result)

    def test_validate_config_missing_section(self):
        config_manager = ConfigManager()
        config_manager.config = {
            "inverter": {"enabled": False},
            "weather": {"enabled": False},
        }

        result = config_manager.validate_config()
        self.assertFalse(result)

    def test_validate_config_incomplete_inverter(self):
        config_manager = ConfigManager()
        config_manager.config = {
            "inverter": {
                "enabled": True,
                "api_url": "",
                "api_key": "",
                "inverter_ids": [],
            },
            "weather": {"enabled": False},
            "maintenance": {"enabled": False},
            "storage": {
                "timeseries": {"url": "", "token": "", "org": "", "bucket": ""},
                "object": {
                    "endpoint_url": "",
                    "access_key": "",
                    "secret_key": "",
                    "bucket_name": "",
                },
            },
            "models": {
                "forecast": {
                    "algorithm": "",
                    "horizon_hours": 0,
                    "retrain_frequency_hours": 0,
                },
                "anomaly": {"algorithm": "", "threshold": 0.0},
            },
            "api": {"host": "", "port": 0, "debug": False},
            "notifications": {"enabled": False},
        }

        result = config_manager.validate_config()
        self.assertFalse(result)

    def test_validate_config_incomplete_weather(self):
        config_manager = ConfigManager()
        config_manager.config = {
            "inverter": {"enabled": False},
            "weather": {
                "enabled": True,
                "api_url": "",
                "api_key": "",
                "location": {"lat": 0.0, "lon": 0.0},
            },
            "maintenance": {"enabled": False},
            "storage": {
                "timeseries": {"url": "", "token": "", "org": "", "bucket": ""},
                "object": {
                    "endpoint_url": "",
                    "access_key": "",
                    "secret_key": "",
                    "bucket_name": "",
                },
            },
            "models": {
                "forecast": {
                    "algorithm": "",
                    "horizon_hours": 0,
                    "retrain_frequency_hours": 0,
                },
                "anomaly": {"algorithm": "", "threshold": 0.0},
            },
            "api": {"host": "", "port": 0, "debug": False},
            "notifications": {"enabled": False},
        }

        result = config_manager.validate_config()
        self.assertFalse(result)

    @patch("src.utils.config.Path")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_config(self, mock_file, mock_path):
        mock_path_instance = Mock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.parent.mkdir.return_value = None

        mock_file.return_value.read.return_value = ""

        config_manager = ConfigManager()
        config_manager.config = self.test_config

        config_manager.save_config("test_config.yaml")

        mock_path_instance.parent.mkdir.assert_called_once_with(
            parents=True, exist_ok=True
        )
        self.assertEqual(mock_file.call_count, 2)
        mock_file.assert_any_call("configs/app.yaml", "r")
        mock_file.assert_any_call("test_config.yaml", "w")

    @patch("src.utils.config.Path")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_config_default_path(self, mock_file, mock_path):
        mock_path_instance = Mock()
        mock_path.return_value = mock_path_instance
        mock_path_instance.parent.mkdir.return_value = None

        mock_file.return_value.read.return_value = ""

        config_manager = ConfigManager()
        config_manager.config = self.test_config

        config_manager.save_config()

        self.assertEqual(mock_file.call_count, 2)
        mock_file.assert_any_call("configs/app.yaml", "r")
        mock_file.assert_any_call("configs/app.yaml", "w")

    def test_get_default_config(self):
        config_manager = ConfigManager()
        default_config = config_manager._get_default_config()

        self.assertIn("inverter", default_config)
        self.assertIn("weather", default_config)
        self.assertIn("storage", default_config)
        self.assertIn("models", default_config)
        self.assertIn("api", default_config)
        self.assertIn("notifications", default_config)

        self.assertFalse(default_config["inverter"]["enabled"])
        self.assertFalse(default_config["weather"]["enabled"])
        self.assertEqual(default_config["api"]["port"], 8000)
        self.assertEqual(default_config["models"]["forecast"]["algorithm"], "prophet")


if __name__ == "__main__":
    unittest.main()
