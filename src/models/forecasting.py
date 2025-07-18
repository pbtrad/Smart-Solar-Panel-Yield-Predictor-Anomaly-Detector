import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
import logging
from prophet import Prophet
import pickle

logger = logging.getLogger(__name__)


class ForecastingError(Exception):
    pass


class SolarPowerForecaster:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.feature_columns = []
        self.model_path = None
        self.is_trained = False

    def create_prophet_model(self) -> Prophet:
        model_params = self.config.get("forecast", {})

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode="multiplicative",
            interval_width=0.95,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
        )

        if model_params.get("holidays", False):
            model.add_country_holidays(country_name="US")

        return model

    def add_regressors(self, model: Prophet, df: pd.DataFrame) -> Prophet:
        regressor_columns = [
            "temp",
            "humidity",
            "wind_speed",
            "clouds",
            "solar_radiation",
            "efficiency",
            "temp_efficiency",
            "cloud_impact",
            "weather_efficiency",
            "maintenance_flag",
            "cleaning_effect",
        ]

        available_regressors = [col for col in regressor_columns if col in df.columns]

        for regressor in available_regressors:
            model.add_regressor(regressor)
            self.feature_columns.append(regressor)

        logger.info(f"Added {len(available_regressors)} regressors to Prophet model")
        return model

    def prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ForecastingError("No data provided for training")

        df = df.copy()

        df["ds"] = pd.to_datetime(df["ds"])
        df["y"] = pd.to_numeric(df["y"], errors="coerce")

        df = df.dropna(subset=["y"])

        if df.empty:
            raise ForecastingError("No valid data after cleaning")

        logger.info(f"Prepared {len(df)} records for training")
        return df

    def train(
        self, training_data: pd.DataFrame, model_path: Optional[str] = None
    ) -> None:
        try:
            if training_data.empty:
                raise ForecastingError("Training data is empty")

            self.model = self.create_prophet_model()
            training_df = self.prepare_training_data(training_data)

            self.model = self.add_regressors(self.model, training_df)

            self.model.fit(training_df)

            self.is_trained = True
            self.model_path = model_path

            if model_path:
                self.save_model(model_path)

            logger.info("Prophet model trained successfully")

        except Exception as e:
            raise ForecastingError(f"Failed to train Prophet model: {str(e)}")

    def predict(
        self,
        periods: int = 24,
        freq: str = "H",
        future_regressors: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if not self.is_trained or self.model is None:
            raise ForecastingError("Model must be trained before making predictions")

        try:
            future = self.model.make_future_dataframe(periods=periods, freq=freq)

            if future_regressors is not None and not future_regressors.empty:
                future = self._add_future_regressors(future, future_regressors)

            forecast = self.model.predict(future)

            result_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
            result_df.columns = [
                "timestamp",
                "predicted_power",
                "lower_bound",
                "upper_bound",
            ]

            logger.info(f"Generated {len(result_df)} predictions")
            return result_df

        except Exception as e:
            raise ForecastingError(f"Failed to generate predictions: {str(e)}")

    def _add_future_regressors(
        self, future: pd.DataFrame, regressors: pd.DataFrame
    ) -> pd.DataFrame:
        future = future.copy()
        regressors = regressors.copy()

        regressors["ds"] = pd.to_datetime(regressors["ds"])
        future["ds"] = pd.to_datetime(future["ds"])

        merged = pd.merge_asof(
            future.sort_values("ds"),
            regressors.sort_values("ds"),
            on="ds",
            direction="nearest",
            tolerance=pd.Timedelta("1H"),
        )

        for col in self.feature_columns:
            if col in merged.columns:
                merged[col] = merged[col].fillna(method="ffill").fillna(0)

        return merged

    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        if not self.is_trained or self.model is None:
            raise ForecastingError("Model must be trained before evaluation")

        try:
            predictions = self.predict(len(test_data))

            actual = test_data["y"].values
            predicted = predictions["predicted_power"].values[: len(actual)]

            mse = np.mean((actual - predicted) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(actual - predicted))
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100

            metrics = {"mse": mse, "rmse": rmse, "mae": mae, "mape": mape}

            logger.info(
                f"Model evaluation - RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%"
            )
            return metrics

        except Exception as e:
            raise ForecastingError(f"Failed to evaluate model: {str(e)}")

    def save_model(self, file_path: str) -> None:
        if not self.is_trained or self.model is None:
            raise ForecastingError("No trained model to save")

        try:
            with open(file_path, "wb") as f:
                pickle.dump(
                    {
                        "model": self.model,
                        "feature_columns": self.feature_columns,
                        "config": self.config,
                        "is_trained": self.is_trained,
                    },
                    f,
                )

            logger.info(f"Model saved to {file_path}")

        except Exception as e:
            raise ForecastingError(f"Failed to save model: {str(e)}")

    def load_model(self, file_path: str) -> None:
        try:
            with open(file_path, "rb") as f:
                model_data = pickle.load(f)

            self.model = model_data["model"]
            self.feature_columns = model_data["feature_columns"]
            self.config = model_data["config"]
            self.is_trained = model_data["is_trained"]
            self.model_path = file_path

            logger.info(f"Model loaded from {file_path}")

        except Exception as e:
            raise ForecastingError(f"Failed to load model: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        if not self.is_trained:
            return {"status": "not_trained"}

        return {
            "status": "trained",
            "feature_columns": self.feature_columns,
            "model_path": self.model_path,
            "config": self.config,
        }


class ForecastingPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.forecaster = SolarPowerForecaster(config)
        self.feature_engineer = None

    def set_feature_engineer(self, feature_engineer):
        self.feature_engineer = feature_engineer

    def train_model(
        self,
        inverter_data: pd.DataFrame,
        weather_data: pd.DataFrame,
        maintenance_data: Optional[pd.DataFrame] = None,
        model_path: Optional[str] = None,
    ) -> Dict[str, float]:
        try:
            if self.feature_engineer is None:
                raise ForecastingError("Feature engineer not set")

            processed_data = self.feature_engineer.process_for_prophet(
                inverter_data, weather_data, maintenance_data
            )

            if processed_data.empty:
                raise ForecastingError("No processed data available for training")

            self.forecaster.train(processed_data, model_path)

            return {"status": "success", "message": "Model trained successfully"}

        except Exception as e:
            raise ForecastingError(f"Training pipeline failed: {str(e)}")

    def generate_forecast(
        self,
        periods: int = 24,
        freq: str = "H",
        future_weather: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        try:
            if not self.forecaster.is_trained:
                raise ForecastingError("Model must be trained before forecasting")

            future_regressors = None
            if future_weather is not None and self.feature_engineer:
                future_regressors = self.feature_engineer.process_for_prophet(
                    pd.DataFrame(), future_weather
                )

            forecast = self.forecaster.predict(periods, freq, future_regressors)

            return forecast

        except Exception as e:
            raise ForecastingError(f"Forecasting pipeline failed: {str(e)}")

    def evaluate_model(
        self,
        test_inverter_data: pd.DataFrame,
        test_weather_data: pd.DataFrame,
        test_maintenance_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        try:
            if self.feature_engineer is None:
                raise ForecastingError("Feature engineer not set")

            processed_test_data = self.feature_engineer.process_for_prophet(
                test_inverter_data, test_weather_data, test_maintenance_data
            )

            if processed_test_data.empty:
                raise ForecastingError("No processed test data available")

            metrics = self.forecaster.evaluate(processed_test_data)

            return metrics

        except Exception as e:
            raise ForecastingError(f"Evaluation pipeline failed: {str(e)}")
