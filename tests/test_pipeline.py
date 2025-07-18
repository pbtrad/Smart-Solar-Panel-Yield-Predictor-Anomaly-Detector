import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os
import json

from src.models.pipeline import (
    SmartSolarMLPipeline,
    MLPipelineError,
)
from src.data.features import ProphetFeatureEngineer


class TestSmartSolarMLPipeline(unittest.TestCase):
    def setUp(self):
        self.config = {
            "forecast": {
                "holidays": True,
                "seasonality_mode": "multiplicative",
            },
            "anomaly": {
                "contamination": 0.1,
                "threshold": 0.95,
            },
            "model_paths": {
                "forecast": "models/forecast_model.pkl",
                "anomaly": "models/anomaly_model.pkl",
            },
        }
        self.feature_engineer = ProphetFeatureEngineer()
        self.pipeline = SmartSolarMLPipeline(self.config)
        self.pipeline.set_feature_engineer(self.feature_engineer)

        self.sample_inverter_data = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 10 + i, 0, 0) for i in range(5)],
                "power_output": [100.5, 120.3, 150.7, 140.2, 130.8],
                "voltage": [240.0] * 5,
                "current": [0.42, 0.50, 0.63, 0.58, 0.54],
                "inverter_id": ["inv_001"] * 5,
            }
        )

        self.sample_weather_data = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 10 + i, 0, 0) for i in range(5)],
                "temp": [15.5, 16.2, 18.0, 17.5, 16.8],
                "humidity": [65, 60, 55, 58, 62],
                "wind_speed": [5.2, 4.8, 6.1, 5.5, 5.0],
                "clouds": [20, 15, 10, 12, 18],
                "solar_radiation": [800.0, 850.0, 900.0, 880.0, 860.0],
            }
        )

        self.sample_maintenance_data = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 9, 0, 0)],
                "maintenance_type": ["cleaning"],
                "description": ["Panel cleaning performed"],
                "inverter_id": ["inv_001"],
            }
        )

        self.training_data = {
            "inverter": self.sample_inverter_data,
            "weather": self.sample_weather_data,
            "maintenance": self.sample_maintenance_data,
        }

    def test_init(self):
        self.assertIsNotNone(self.pipeline)
        self.assertIsNotNone(self.pipeline.forecasting_pipeline)
        self.assertIsNotNone(self.pipeline.anomaly_pipeline)
        self.assertEqual(self.pipeline.feature_engineer, self.feature_engineer)
        self.assertEqual(self.pipeline.models_trained["forecast"], False)
        self.assertEqual(self.pipeline.models_trained["anomaly"], False)

    def test_set_feature_engineer(self):
        new_engineer = ProphetFeatureEngineer()
        self.pipeline.set_feature_engineer(new_engineer)

        self.assertEqual(self.pipeline.feature_engineer, new_engineer)
        self.assertEqual(
            self.pipeline.forecasting_pipeline.feature_engineer, new_engineer
        )
        self.assertEqual(self.pipeline.anomaly_pipeline.feature_engineer, new_engineer)

    @patch("src.models.forecasting.Prophet")
    @patch("src.models.anomaly.IsolationForest")
    @patch("src.models.anomaly.StandardScaler")
    @patch("builtins.open", create=True)
    def test_train_all_models_success(
        self, mock_open, mock_scaler, mock_isolation_forest, mock_prophet
    ):
        mock_forecast_model = Mock()
        mock_anomaly_model = Mock()
        mock_scaler_instance = Mock()

        mock_prophet.return_value = mock_forecast_model
        mock_isolation_forest.return_value = mock_anomaly_model
        mock_scaler.return_value = mock_scaler_instance

        mock_anomaly_model.predict.return_value = np.array([1, 1, -1, 1, 1])
        mock_anomaly_model.decision_function.return_value = np.array(
            [-0.1, -0.2, -0.8, -0.3, -0.4]
        )
        mock_scaler_instance.transform.return_value = np.random.rand(5, 10)

        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file

        with patch.object(
            self.pipeline.forecasting_pipeline.forecaster, "save_model"
        ) as mock_save_forecast:
            with patch.object(
                self.pipeline.anomaly_pipeline.detector, "save_model"
            ) as mock_save_anomaly:
                results = self.pipeline.train_all_models(self.training_data)

                self.assertIn("forecast", results)
                self.assertIn("anomaly", results)
                self.assertEqual(results["forecast"]["status"], "success")
                self.assertEqual(results["anomaly"]["status"], "success")
                self.assertTrue(self.pipeline.models_trained["forecast"])
                self.assertTrue(self.pipeline.models_trained["anomaly"])

                # Verify save methods were called
                mock_save_forecast.assert_called()
                mock_save_anomaly.assert_called()

    def test_train_all_models_missing_data(self):
        incomplete_data = {
            "inverter": pd.DataFrame(),
            "weather": self.sample_weather_data,
        }

        with self.assertRaises(MLPipelineError):
            self.pipeline.train_all_models(incomplete_data)

    @patch("src.models.forecasting.Prophet")
    def test_generate_forecast_success(self, mock_prophet):
        mock_model = Mock()
        mock_prophet.return_value = mock_model

        mock_forecast = pd.DataFrame(
            {
                "ds": [datetime(2024, 1, 2, i, 0, 0) for i in range(24)],
                "yhat": [100 + i for i in range(24)],
                "yhat_lower": [95 + i for i in range(24)],
                "yhat_upper": [105 + i for i in range(24)],
            }
        )
        mock_model.predict.return_value = mock_forecast

        self.pipeline.forecasting_pipeline.train_model(
            self.sample_inverter_data,
            self.sample_weather_data,
            self.sample_maintenance_data,
        )
        self.pipeline.models_trained["forecast"] = True

        result = self.pipeline.generate_forecast(periods=24)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 24)
        self.assertIn("timestamp", result.columns)
        self.assertIn("predicted_power", result.columns)

    def test_generate_forecast_not_trained(self):
        with self.assertRaises(MLPipelineError):
            self.pipeline.generate_forecast()

    @patch("src.models.anomaly.IsolationForest")
    @patch("src.models.anomaly.StandardScaler")
    def test_detect_anomalies_success(self, mock_scaler, mock_isolation_forest):
        mock_model = Mock()
        mock_scaler_instance = Mock()
        mock_isolation_forest.return_value = mock_model
        mock_scaler.return_value = mock_scaler_instance

        mock_model.predict.return_value = np.array([1, 1, -1, 1, 1])
        mock_model.decision_function.return_value = np.array(
            [-0.1, -0.2, -0.8, -0.3, -0.4]
        )
        mock_scaler_instance.transform.return_value = np.random.rand(5, 10)

        self.pipeline.anomaly_pipeline.train_model(
            self.sample_inverter_data,
            self.sample_weather_data,
            self.sample_maintenance_data,
        )
        self.pipeline.models_trained["anomaly"] = True

        result = self.pipeline.detect_anomalies(self.training_data)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("anomaly_score", result.columns)
        self.assertIn("is_anomaly", result.columns)

    def test_detect_anomalies_not_trained(self):
        with self.assertRaises(MLPipelineError):
            self.pipeline.detect_anomalies(self.training_data)

    @patch("src.models.forecasting.Prophet")
    @patch("src.models.anomaly.IsolationForest")
    @patch("src.models.anomaly.StandardScaler")
    @patch("builtins.open", create=True)
    def test_evaluate_models_success(
        self, mock_open, mock_scaler, mock_isolation_forest, mock_prophet
    ):
        mock_forecast_model = Mock()
        mock_anomaly_model = Mock()
        mock_scaler_instance = Mock()

        mock_prophet.return_value = mock_forecast_model
        mock_isolation_forest.return_value = mock_anomaly_model
        mock_scaler.return_value = mock_scaler_instance

        mock_forecast_predictions = pd.DataFrame(
            {"predicted_power": [100, 120, 150, 140, 130]}
        )
        mock_anomaly_predictions = pd.DataFrame(
            {
                "anomaly_score": [-0.1, -0.2, -0.8, -0.3, -0.4],
                "is_anomaly": [0, 0, 1, 0, 0],
            }
        )

        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file

        with patch.object(self.pipeline.forecasting_pipeline.forecaster, "save_model"):
            with patch.object(self.pipeline.anomaly_pipeline.detector, "save_model"):
                self.pipeline.forecasting_pipeline.train_model(
                    self.sample_inverter_data,
                    self.sample_weather_data,
                    self.sample_maintenance_data,
                )
                self.pipeline.anomaly_pipeline.train_model(
                    self.sample_inverter_data,
                    self.sample_weather_data,
                    self.sample_maintenance_data,
                )
                self.pipeline.models_trained["forecast"] = True
                self.pipeline.models_trained["anomaly"] = True

                with patch.object(
                    self.pipeline.forecasting_pipeline.forecaster,
                    "predict",
                    return_value=mock_forecast_predictions,
                ):
                    with patch.object(
                        self.pipeline.anomaly_pipeline,
                        "detect_anomalies",
                        return_value=mock_anomaly_predictions,
                    ):
                        results = self.pipeline.evaluate_models(self.training_data)

                        self.assertIn("forecast", results)
                        self.assertIn("anomaly", results)
                        self.assertIn("mse", results["forecast"])
                        self.assertIn("anomaly_count", results["anomaly"])

    def test_evaluate_models_no_models_trained(self):
        results = self.pipeline.evaluate_models(self.training_data)
        self.assertEqual(results, {})

    def test_get_model_status(self):
        status = self.pipeline.get_model_status()

        self.assertIn("models_trained", status)
        self.assertIn("model_paths", status)
        self.assertIn("forecast_info", status)
        self.assertIn("anomaly_info", status)

        self.assertEqual(status["models_trained"]["forecast"], False)
        self.assertEqual(status["models_trained"]["anomaly"], False)

    @patch("builtins.open", create=True)
    def test_load_models_success(self, mock_open):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as forecast_file:
            forecast_path = forecast_file.name
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as anomaly_file:
            anomaly_path = anomaly_file.name

        try:
            self.pipeline.model_paths["forecast"] = forecast_path
            self.pipeline.model_paths["anomaly"] = anomaly_path

            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file

            with patch.object(
                self.pipeline.forecasting_pipeline.forecaster, "save_model"
            ):
                with patch.object(
                    self.pipeline.anomaly_pipeline.detector, "save_model"
                ):
                    self.pipeline.forecasting_pipeline.forecaster.save_model(
                        forecast_path
                    )
                    self.pipeline.anomaly_pipeline.detector.save_model(anomaly_path)

            new_pipeline = SmartSolarMLPipeline(self.config)
            new_pipeline.set_feature_engineer(self.feature_engineer)
            new_pipeline.model_paths = self.pipeline.model_paths

            with patch.object(
                new_pipeline.forecasting_pipeline.forecaster, "load_model"
            ) as mock_load_forecast:
                with patch.object(
                    new_pipeline.anomaly_pipeline.detector, "load_model"
                ) as mock_load_anomaly:

                    def mock_load_forecast_side_effect(file_path):
                        new_pipeline.forecasting_pipeline.forecaster.is_trained = True
                        new_pipeline.forecasting_pipeline.forecaster.model_path = (
                            file_path
                        )

                    def mock_load_anomaly_side_effect(file_path):
                        new_pipeline.anomaly_pipeline.detector.is_trained = True
                        new_pipeline.anomaly_pipeline.detector.model_path = file_path

                    mock_load_forecast.side_effect = mock_load_forecast_side_effect
                    mock_load_anomaly.side_effect = mock_load_anomaly_side_effect

                    results = new_pipeline.load_trained_models()

                    self.assertTrue(results["forecast"])
                    self.assertTrue(results["anomaly"])
                    self.assertTrue(new_pipeline.models_trained["forecast"])
                    self.assertTrue(new_pipeline.models_trained["anomaly"])

        finally:
            for path in [forecast_path, anomaly_path]:
                if os.path.exists(path):
                    os.unlink(path)

    def test_load_models_missing_files(self):
        results = self.pipeline.load_trained_models()

        self.assertFalse(results["forecast"])
        self.assertFalse(results["anomaly"])

    @patch("src.models.forecasting.Prophet")
    @patch("src.models.anomaly.IsolationForest")
    @patch("src.models.anomaly.StandardScaler")
    @patch("builtins.open", create=True)
    def test_run_full_pipeline_success(
        self, mock_open, mock_scaler, mock_isolation_forest, mock_prophet
    ):
        mock_forecast_model = Mock()
        mock_anomaly_model = Mock()
        mock_scaler_instance = Mock()

        mock_prophet.return_value = mock_forecast_model
        mock_isolation_forest.return_value = mock_anomaly_model
        mock_scaler.return_value = mock_scaler_instance

        mock_forecast = pd.DataFrame(
            {
                "ds": [datetime(2024, 1, 2, i, 0, 0) for i in range(24)],
                "yhat": [100 + i for i in range(24)],
                "yhat_lower": [95 + i for i in range(24)],
                "yhat_upper": [105 + i for i in range(24)],
            }
        )
        mock_forecast_model.predict.return_value = mock_forecast

        mock_anomaly_model.predict.return_value = np.array([1, 1, -1, 1, 1])
        mock_anomaly_model.decision_function.return_value = np.array(
            [-0.1, -0.2, -0.8, -0.3, -0.4]
        )
        mock_scaler_instance.transform.return_value = np.random.rand(5, 10)

        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file

        with patch.object(self.pipeline.forecasting_pipeline.forecaster, "save_model"):
            with patch.object(self.pipeline.anomaly_pipeline.detector, "save_model"):
                self.pipeline.forecasting_pipeline.train_model(
                    self.sample_inverter_data,
                    self.sample_weather_data,
                    self.sample_maintenance_data,
                )
                self.pipeline.anomaly_pipeline.train_model(
                    self.sample_inverter_data,
                    self.sample_weather_data,
                    self.sample_maintenance_data,
                )
                self.pipeline.models_trained["forecast"] = True
                self.pipeline.models_trained["anomaly"] = True

                results = self.pipeline.run_full_pipeline(self.training_data)

                self.assertIn("forecast", results)
                self.assertIn("anomaly_data", results)
                self.assertIn("anomaly_summary", results)

    def test_run_full_pipeline_no_models_trained(self):
        results = self.pipeline.run_full_pipeline(self.training_data)
        self.assertEqual(results, {})

    def test_save_pipeline_state(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
            state_path = tmp_file.name

        try:
            self.pipeline.save_pipeline_state(state_path)

            self.assertTrue(os.path.exists(state_path))

            with open(state_path, "r") as f:
                state = json.load(f)

            self.assertIn("models_trained", state)
            self.assertIn("model_paths", state)
            self.assertIn("config", state)

        finally:
            if os.path.exists(state_path):
                os.unlink(state_path)

    def test_load_pipeline_state(self):
        test_state = {
            "models_trained": {"forecast": True, "anomaly": False},
            "model_paths": {
                "forecast": "test_forecast.pkl",
                "anomaly": "test_anomaly.pkl",
            },
            "config": self.config,
        }

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
            state_path = tmp_file.name

        try:
            with open(state_path, "w") as f:
                json.dump(test_state, f)

            self.pipeline.load_pipeline_state(state_path)

            self.assertEqual(self.pipeline.models_trained["forecast"], True)
            self.assertEqual(self.pipeline.models_trained["anomaly"], False)
            self.assertEqual(self.pipeline.model_paths["forecast"], "test_forecast.pkl")
            self.assertEqual(self.pipeline.model_paths["anomaly"], "test_anomaly.pkl")

        finally:
            if os.path.exists(state_path):
                os.unlink(state_path)


class TestMLPipelineError(unittest.TestCase):
    def test_ml_pipeline_error(self):
        error = MLPipelineError("Test error message")
        self.assertEqual(str(error), "Test error message")
        self.assertIsInstance(error, Exception)


if __name__ == "__main__":
    unittest.main()
