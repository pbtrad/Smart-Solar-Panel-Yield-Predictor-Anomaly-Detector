import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os

from src.models.forecasting import (
    SolarPowerForecaster,
    ForecastingPipeline,
    ForecastingError,
)
from src.data.features import ProphetFeatureEngineer


class TestSolarPowerForecaster(unittest.TestCase):
    def setUp(self):
        self.config = {
            "forecast": {
                "holidays": True,
                "seasonality_mode": "multiplicative",
                "interval_width": 0.95,
            }
        }
        self.forecaster = SolarPowerForecaster(self.config)

        self.sample_data = pd.DataFrame(
            {
                "ds": [
                    datetime(2024, 1, 1, 10, 0, 0),
                    datetime(2024, 1, 1, 11, 0, 0),
                    datetime(2024, 1, 1, 12, 0, 0),
                    datetime(2024, 1, 1, 13, 0, 0),
                    datetime(2024, 1, 1, 14, 0, 0),
                ],
                "y": [100.5, 120.3, 150.7, 140.2, 130.8],
                "temp": [15.5, 16.2, 18.0, 17.5, 16.8],
                "solar_radiation": [800.0, 850.0, 900.0, 880.0, 860.0],
                "efficiency": [0.125, 0.141, 0.167, 0.159, 0.152],
            }
        )

    def test_init(self):
        self.assertIsNotNone(self.forecaster)
        self.assertFalse(self.forecaster.is_trained)
        self.assertIsNone(self.forecaster.model)
        self.assertEqual(self.forecaster.config, self.config)

    def test_create_prophet_model(self):
        model = self.forecaster.create_prophet_model()

        self.assertIsNotNone(model)
        self.assertTrue(model.yearly_seasonality)
        self.assertTrue(model.weekly_seasonality)
        self.assertTrue(model.daily_seasonality)
        self.assertEqual(model.seasonality_mode, "multiplicative")
        self.assertEqual(model.interval_width, 0.95)

    def test_add_regressors(self):
        model = self.forecaster.create_prophet_model()
        model_with_regressors = self.forecaster.add_regressors(model, self.sample_data)

        self.assertIsNotNone(model_with_regressors)
        self.assertIn("temp", model_with_regressors.extra_regressors)
        self.assertIn("solar_radiation", model_with_regressors.extra_regressors)

    def test_prepare_data_valid(self):
        result = self.forecaster.prepare_training_data(self.sample_data)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 5)
        self.assertIn("ds", result.columns)
        self.assertIn("y", result.columns)
        self.assertFalse(result["y"].isnull().any())

    def test_prepare_data_empty(self):
        empty_df = pd.DataFrame()

        with self.assertRaises(ForecastingError):
            self.forecaster.prepare_training_data(empty_df)

    def test_prepare_data_with_nulls(self):
        data_with_nulls = self.sample_data.copy()
        data_with_nulls.loc[0, "y"] = np.nan

        result = self.forecaster.prepare_training_data(data_with_nulls)

        self.assertEqual(len(result), 4)

    @patch("src.models.forecasting.Prophet")
    def test_train_success(self, mock_prophet):
        mock_model = Mock()
        mock_prophet.return_value = mock_model

        self.forecaster.train(self.sample_data)

        self.assertTrue(self.forecaster.is_trained)
        self.assertIsNotNone(self.forecaster.model)
        mock_model.fit.assert_called_once()

    def test_train_empty_data(self):
        empty_df = pd.DataFrame()

        with self.assertRaises(ForecastingError):
            self.forecaster.train(empty_df)

    def test_predict_not_trained(self):
        with self.assertRaises(ForecastingError):
            self.forecaster.predict()

    @patch("src.models.forecasting.Prophet")
    def test_predict_success(self, mock_prophet):
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

        self.forecaster.train(self.sample_data)
        result = self.forecaster.predict(periods=24)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 24)
        self.assertIn("timestamp", result.columns)
        self.assertIn("predicted_power", result.columns)
        self.assertIn("lower_bound", result.columns)
        self.assertIn("upper_bound", result.columns)

    def test_evaluate_model_not_trained(self):
        test_data = self.sample_data.copy()

        with self.assertRaises(ForecastingError):
            self.forecaster.evaluate(test_data)

    @patch("src.models.forecasting.Prophet")
    def test_evaluate_model_success(self, mock_prophet):
        mock_model = Mock()
        mock_prophet.return_value = mock_model

        mock_predictions = pd.DataFrame({"predicted_power": [100, 120, 150, 140, 130]})

        with patch.object(self.forecaster, "predict", return_value=mock_predictions):
            self.forecaster.train(self.sample_data)
            metrics = self.forecaster.evaluate(self.sample_data)

            self.assertIn("mse", metrics)
            self.assertIn("rmse", metrics)
            self.assertIn("mae", metrics)
            self.assertIn("mape", metrics)
            self.assertIsInstance(metrics["mse"], float)
            self.assertIsInstance(metrics["rmse"], float)

    @patch("builtins.open", create=True)
    def test_save_model(self, mock_open):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            model_path = tmp_file.name

        try:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file

            self.forecaster.is_trained = True
            self.forecaster.model = Mock()
            self.forecaster.feature_columns = ["temp", "humidity"]

            with patch.object(self.forecaster, "save_model") as mock_save:

                def save_side_effect(file_path):
                    self.forecaster.model_path = file_path

                mock_save.side_effect = save_side_effect
                self.forecaster.save_model(model_path)
                self.assertEqual(self.forecaster.model_path, model_path)
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

            self.forecaster.is_trained = True
            self.forecaster.model = Mock()
            self.forecaster.feature_columns = ["temp", "humidity"]

            with patch.object(self.forecaster, "save_model") as mock_save:

                def save_side_effect(file_path):
                    self.forecaster.model_path = file_path

                mock_save.side_effect = save_side_effect
                self.forecaster.save_model(model_path)

            new_forecaster = SolarPowerForecaster(self.config)

            with patch.object(new_forecaster, "load_model") as mock_load:

                def mock_load_side_effect(file_path):
                    new_forecaster.is_trained = True
                    new_forecaster.model_path = file_path

                mock_load.side_effect = mock_load_side_effect
                new_forecaster.load_model(model_path)

                self.assertTrue(new_forecaster.is_trained)
                self.assertEqual(new_forecaster.model_path, model_path)
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)

    def test_get_model_info_not_trained(self):
        info = self.forecaster.get_model_info()
        self.assertEqual(info["status"], "not_trained")

    @patch("builtins.open", create=True)
    def test_get_model_info_trained(self, mock_open):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            model_path = tmp_file.name

        try:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file

            self.forecaster.is_trained = True
            self.forecaster.model = Mock()
            self.forecaster.feature_columns = ["temp", "humidity"]

            with patch.object(self.forecaster, "save_model") as mock_save:

                def save_side_effect(file_path):
                    self.forecaster.model_path = file_path

                mock_save.side_effect = save_side_effect
                self.forecaster.save_model(model_path)
                info = self.forecaster.get_model_info()

                self.assertEqual(info["status"], "trained")
                self.assertEqual(info["model_path"], model_path)
                self.assertEqual(info["config"], self.config)
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)


class TestForecastingPipeline(unittest.TestCase):
    def setUp(self):
        self.config = {
            "forecast": {
                "holidays": True,
                "seasonality_mode": "multiplicative",
            }
        }
        self.feature_engineer = ProphetFeatureEngineer()
        self.pipeline = ForecastingPipeline(self.config)
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
        self.assertIsNotNone(self.pipeline.forecaster)
        self.assertEqual(self.pipeline.feature_engineer, self.feature_engineer)

    def test_set_feature_engineer(self):
        new_engineer = ProphetFeatureEngineer()
        self.pipeline.set_feature_engineer(new_engineer)
        self.assertEqual(self.pipeline.feature_engineer, new_engineer)

    @patch("src.models.forecasting.Prophet")
    @patch("builtins.open", create=True)
    def test_train_model_success(self, mock_open, mock_prophet):
        mock_model = Mock()
        mock_prophet.return_value = mock_model

        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file

        with patch.object(self.pipeline.forecaster, "save_model"):
            result = self.pipeline.train_model(
                self.sample_inverter_data,
                self.sample_weather_data,
                self.sample_maintenance_data,
            )

            self.assertEqual(result["status"], "success")
            self.assertIn("message", result)

    def test_train_model_no_feature_engineer(self):
        pipeline = ForecastingPipeline(self.config)

        with self.assertRaises(ForecastingError):
            pipeline.train_model(self.sample_inverter_data, self.sample_weather_data)

    @patch("src.models.forecasting.Prophet")
    def test_generate_forecast_success(self, mock_prophet):
        mock_model = Mock()
        mock_prophet.return_value = mock_model

        self.pipeline.train_model(
            self.sample_inverter_data,
            self.sample_weather_data,
            self.sample_maintenance_data,
        )

        mock_forecast = pd.DataFrame(
            {
                "ds": [datetime(2024, 1, 2, i, 0, 0) for i in range(24)],
                "yhat": [100 + i for i in range(24)],
                "yhat_lower": [95 + i for i in range(24)],
                "yhat_upper": [105 + i for i in range(24)],
            }
        )
        mock_model.predict.return_value = mock_forecast

        result = self.pipeline.generate_forecast(periods=24)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 24)

    def test_generate_forecast_not_trained(self):
        with self.assertRaises(ForecastingError):
            self.pipeline.generate_forecast()

    @patch("src.models.forecasting.Prophet")
    def test_evaluate_model_success(self, mock_prophet):
        mock_model = Mock()
        mock_prophet.return_value = mock_model

        self.pipeline.train_model(
            self.sample_inverter_data,
            self.sample_weather_data,
            self.sample_maintenance_data,
        )

        mock_predictions = pd.DataFrame({"predicted_power": [100, 120, 150, 140, 130]})

        with patch.object(
            self.pipeline.forecaster, "predict", return_value=mock_predictions
        ):
            metrics = self.pipeline.evaluate_model(
                self.sample_inverter_data,
                self.sample_weather_data,
                self.sample_maintenance_data,
            )

            self.assertIn("mse", metrics)
            self.assertIn("rmse", metrics)
            self.assertIn("mae", metrics)
            self.assertIn("mape", metrics)


class TestForecastingError(unittest.TestCase):
    def test_forecasting_error(self):
        error = ForecastingError("Test error message")
        self.assertEqual(str(error), "Test error message")
        self.assertIsInstance(error, Exception)


if __name__ == "__main__":
    unittest.main()
