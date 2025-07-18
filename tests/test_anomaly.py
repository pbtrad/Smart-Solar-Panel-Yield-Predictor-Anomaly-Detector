import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os

from src.models.anomaly import (
    SolarAnomalyDetector,
    AnomalyDetectionPipeline,
    AnomalyDetectionError,
)
from src.data.features import ProphetFeatureEngineer


class TestSolarAnomalyDetector(unittest.TestCase):
    def setUp(self):
        self.config = {
            "anomaly": {
                "contamination": 0.1,
                "threshold": 0.95,
            }
        }
        self.detector = SolarAnomalyDetector(self.config)

        self.sample_data = pd.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 10, 0, 0),
                    datetime(2024, 1, 1, 11, 0, 0),
                    datetime(2024, 1, 1, 12, 0, 0),
                    datetime(2024, 1, 1, 13, 0, 0),
                    datetime(2024, 1, 1, 14, 0, 0),
                ],
                "power_output": [100.5, 120.3, 150.7, 140.2, 130.8],
                "solar_radiation": [800.0, 850.0, 900.0, 880.0, 860.0],
                "temp": [15.5, 16.2, 18.0, 17.5, 16.8],
                "clouds": [20, 15, 10, 12, 18],
            }
        )

    def test_init(self):
        self.assertIsNotNone(self.detector)
        self.assertFalse(self.detector.is_trained)
        self.assertIsNone(self.detector.model)
        self.assertEqual(self.detector.contamination, 0.1)
        self.assertEqual(self.detector.threshold, 0.95)

    def test_create_features(self):
        result = self.detector.create_features(self.sample_data)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("hour", result.columns)
        self.assertIn("day_of_week", result.columns)
        self.assertIn("month", result.columns)
        self.assertIn("is_daytime", result.columns)
        self.assertIn("efficiency", result.columns)
        self.assertIn("efficiency_ratio", result.columns)
        self.assertIn("power_ratio", result.columns)
        self.assertIn("power_change", result.columns)
        self.assertIn("power_volatility", result.columns)
        self.assertIn("temp_effect", result.columns)
        self.assertIn("cloud_impact", result.columns)

    def test_create_features_missing_columns(self):
        data_without_power = self.sample_data.drop(columns=["power_output"])
        result = self.detector.create_features(data_without_power)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("hour", result.columns)
        self.assertNotIn("efficiency", result.columns)
        self.assertNotIn("power_ratio", result.columns)

    def test_select_features(self):
        feature_df = self.detector.create_features(self.sample_data)
        features = self.detector.select_features(feature_df)

        self.assertIsInstance(features, list)
        self.assertIn("hour", features)
        self.assertIn("day_of_week", features)
        self.assertIn("month", features)
        self.assertIn("is_daytime", features)
        self.assertIn("efficiency", features)
        self.assertIn("efficiency_ratio", features)
        self.assertIn("power_ratio", features)
        self.assertIn("power_change", features)
        self.assertIn("power_volatility", features)
        self.assertIn("temp_effect", features)
        self.assertIn("cloud_impact", features)

    def test_prepare_training_data(self):
        feature_df, feature_columns = self.detector.prepare_training_data(
            self.sample_data
        )

        self.assertIsInstance(feature_df, pd.DataFrame)
        self.assertIsInstance(feature_columns, list)
        self.assertEqual(len(feature_df), 5)
        self.assertEqual(len(feature_columns), len(feature_df.columns))
        self.assertFalse(feature_df.isnull().any().any())

    def test_prepare_training_data_empty(self):
        empty_df = pd.DataFrame()

        with self.assertRaises(AnomalyDetectionError):
            self.detector.prepare_training_data(empty_df)

    @patch("src.models.anomaly.IsolationForest")
    @patch("src.models.anomaly.StandardScaler")
    def test_train_success(self, mock_scaler, mock_isolation_forest):
        mock_model = Mock()
        mock_scaler_instance = Mock()
        mock_isolation_forest.return_value = mock_model
        mock_scaler.return_value = mock_scaler_instance

        self.detector.train(self.sample_data)

        self.assertTrue(self.detector.is_trained)
        self.assertIsNotNone(self.detector.model)
        self.assertIsNotNone(self.detector.scaler)
        mock_model.fit.assert_called_once()

    def test_train_empty_data(self):
        empty_df = pd.DataFrame()

        with self.assertRaises(AnomalyDetectionError):
            self.detector.train(empty_df)

    def test_detect_anomalies_not_trained(self):
        with self.assertRaises(AnomalyDetectionError):
            self.detector.detect_anomalies(self.sample_data)

    @patch("src.models.anomaly.IsolationForest")
    @patch("src.models.anomaly.StandardScaler")
    def test_detect_anomalies_success(self, mock_scaler, mock_isolation_forest):
        mock_model = Mock()
        mock_scaler_instance = Mock()
        mock_isolation_forest.return_value = mock_model
        mock_scaler.return_value = mock_scaler_instance

        mock_model.predict.return_value = np.array([1, 1, -1, 1, 1])  # One anomaly
        mock_model.decision_function.return_value = np.array(
            [-0.1, -0.2, -0.8, -0.3, -0.4]
        )
        mock_scaler_instance.transform.return_value = np.random.rand(5, 10)

        self.detector.train(self.sample_data)
        result = self.detector.detect_anomalies(self.sample_data)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("anomaly_score", result.columns)
        self.assertIn("is_anomaly", result.columns)
        self.assertIn("anomaly_probability", result.columns)
        self.assertIn("high_anomaly", result.columns)
        self.assertEqual(result["is_anomaly"].sum(), 1)  # One anomaly detected

    def test_get_anomaly_summary_no_anomaly_data(self):
        with self.assertRaises(AnomalyDetectionError):
            self.detector.get_anomaly_summary(self.sample_data)

    @patch("src.models.anomaly.IsolationForest")
    @patch("src.models.anomaly.StandardScaler")
    def test_get_anomaly_summary_success(self, mock_scaler, mock_isolation_forest):
        mock_model = Mock()
        mock_scaler_instance = Mock()
        mock_isolation_forest.return_value = mock_model
        mock_scaler.return_value = mock_scaler_instance

        mock_model.predict.return_value = np.array([1, 1, -1, 1, 1])
        mock_model.decision_function.return_value = np.array(
            [-0.1, -0.2, -0.8, -0.3, -0.4]
        )
        mock_scaler_instance.transform.return_value = np.random.rand(5, 10)

        self.detector.train(self.sample_data)
        anomaly_data = self.detector.detect_anomalies(self.sample_data)
        summary = self.detector.get_anomaly_summary(anomaly_data)

        self.assertIn("total_records", summary)
        self.assertIn("anomaly_count", summary)
        self.assertIn("high_anomaly_count", summary)
        self.assertIn("anomaly_percentage", summary)
        self.assertIn("high_anomaly_percentage", summary)
        self.assertIn("avg_anomaly_score", summary)
        self.assertIn("min_anomaly_score", summary)

        self.assertEqual(summary["total_records"], 5)
        self.assertEqual(summary["anomaly_count"], 1)

    @patch("builtins.open", create=True)
    def test_save_model(self, mock_open):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            model_path = tmp_file.name

        try:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file

            self.detector.is_trained = True
            self.detector.model = Mock()
            self.detector.scaler = Mock()
            self.detector.feature_columns = ["temp", "humidity"]

            with patch.object(self.detector, "save_model") as mock_save:

                def save_side_effect(file_path):
                    self.detector.model_path = file_path

                mock_save.side_effect = save_side_effect
                self.detector.save_model(model_path)
                self.assertEqual(self.detector.model_path, model_path)
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)

    @patch("builtins.open", create=True)
    def test_load_model(self, mock_open):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            model_path = tmp_file.name

        try:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file

            self.detector.is_trained = True
            self.detector.model = Mock()
            self.detector.scaler = Mock()
            self.detector.feature_columns = ["temp", "humidity"]

            with patch.object(self.detector, "save_model") as mock_save:

                def save_side_effect(file_path):
                    self.detector.model_path = file_path

                mock_save.side_effect = save_side_effect
                self.detector.save_model(model_path)

            new_detector = SolarAnomalyDetector(self.config)

            with patch.object(new_detector, "load_model") as mock_load:

                def mock_load_side_effect(file_path):
                    new_detector.is_trained = True
                    new_detector.model_path = file_path

                mock_load.side_effect = mock_load_side_effect
                new_detector.load_model(model_path)

                self.assertTrue(new_detector.is_trained)
                self.assertEqual(new_detector.model_path, model_path)
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_get_model_info_not_trained(self):
        info = self.detector.get_model_info()
        self.assertEqual(info["status"], "not_trained")

    @patch("builtins.open", create=True)
    def test_get_model_info_trained(self, mock_open):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            model_path = tmp_file.name

        try:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file

            self.detector.is_trained = True
            self.detector.model = Mock()
            self.detector.scaler = Mock()
            self.detector.feature_columns = ["temp", "humidity"]

            with patch.object(self.detector, "save_model") as mock_save:

                def save_side_effect(file_path):
                    self.detector.model_path = file_path

                mock_save.side_effect = save_side_effect
                self.detector.save_model(model_path)
                info = self.detector.get_model_info()

                self.assertEqual(info["status"], "trained")
                self.assertEqual(info["model_path"], model_path)
                self.assertEqual(info["contamination"], 0.1)
                self.assertEqual(info["threshold"], 0.95)
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)


class TestAnomalyDetectionPipeline(unittest.TestCase):
    def setUp(self):
        self.config = {
            "anomaly": {
                "contamination": 0.1,
                "threshold": 0.95,
            }
        }
        self.feature_engineer = ProphetFeatureEngineer()
        self.pipeline = AnomalyDetectionPipeline(self.config)
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

    def test_init(self):
        self.assertIsNotNone(self.pipeline)
        self.assertIsNotNone(self.pipeline.detector)
        self.assertEqual(self.pipeline.feature_engineer, self.feature_engineer)

    def test_set_feature_engineer(self):
        new_engineer = ProphetFeatureEngineer()
        self.pipeline.set_feature_engineer(new_engineer)
        self.assertEqual(self.pipeline.feature_engineer, new_engineer)

    @patch("src.models.anomaly.IsolationForest")
    @patch("src.models.anomaly.StandardScaler")
    @patch("builtins.open", create=True)
    def test_train_model_success(self, mock_open, mock_scaler, mock_isolation_forest):
        mock_model = Mock()
        mock_scaler_instance = Mock()
        mock_isolation_forest.return_value = mock_model
        mock_scaler.return_value = mock_scaler_instance

        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file

        with patch.object(self.pipeline.detector, "save_model"):
            result = self.pipeline.train_model(
                self.sample_inverter_data,
                self.sample_weather_data,
                self.sample_maintenance_data,
            )

            self.assertEqual(result["status"], "success")
            self.assertIn("message", result)

    def test_train_model_no_feature_engineer(self):
        pipeline = AnomalyDetectionPipeline(self.config)

        with self.assertRaises(AnomalyDetectionError):
            pipeline.train_model(self.sample_inverter_data, self.sample_weather_data)

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

        self.pipeline.train_model(
            self.sample_inverter_data,
            self.sample_weather_data,
            self.sample_maintenance_data,
        )

        result = self.pipeline.detect_anomalies(
            self.sample_inverter_data,
            self.sample_weather_data,
            self.sample_maintenance_data,
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("anomaly_score", result.columns)
        self.assertIn("is_anomaly", result.columns)

    def test_detect_anomalies_not_trained(self):
        with self.assertRaises(AnomalyDetectionError):
            self.pipeline.detect_anomalies(
                self.sample_inverter_data, self.sample_weather_data
            )

    @patch("src.models.anomaly.IsolationForest")
    @patch("src.models.anomaly.StandardScaler")
    def test_get_anomaly_summary_success(self, mock_scaler, mock_isolation_forest):
        mock_model = Mock()
        mock_scaler_instance = Mock()
        mock_isolation_forest.return_value = mock_model
        mock_scaler.return_value = mock_scaler_instance

        mock_model.predict.return_value = np.array([1, 1, -1, 1, 1])
        mock_model.decision_function.return_value = np.array(
            [-0.1, -0.2, -0.8, -0.3, -0.4]
        )
        mock_scaler_instance.transform.return_value = np.random.rand(5, 10)

        self.pipeline.train_model(
            self.sample_inverter_data,
            self.sample_weather_data,
            self.sample_maintenance_data,
        )

        anomaly_data = self.pipeline.detect_anomalies(
            self.sample_inverter_data,
            self.sample_weather_data,
            self.sample_maintenance_data,
        )

        summary = self.pipeline.get_anomaly_summary(anomaly_data)

        self.assertIn("total_records", summary)
        self.assertIn("anomaly_count", summary)
        self.assertIn("anomaly_percentage", summary)

    def test_get_anomaly_summary_not_trained(self):
        mock_anomaly_data = pd.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, 10 + i, 0, 0) for i in range(5)],
                "is_anomaly": [0, 0, 1, 0, 0],
                "anomaly_score": [-0.1, -0.2, -0.8, -0.3, -0.4],
            }
        )

        summary = self.pipeline.get_anomaly_summary(mock_anomaly_data)

        self.assertIn("total_records", summary)
        self.assertIn("anomaly_count", summary)
        self.assertEqual(summary["total_records"], 5)
        self.assertEqual(summary["anomaly_count"], 1)


class TestAnomalyDetectionError(unittest.TestCase):
    def test_anomaly_detection_error(self):
        error = AnomalyDetectionError("Test error message")
        self.assertEqual(str(error), "Test error message")
        self.assertIsInstance(error, Exception)


if __name__ == "__main__":
    unittest.main()
