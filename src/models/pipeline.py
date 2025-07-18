import pandas as pd
from typing import Dict, Optional, Any
import logging
import json

from .forecasting import ForecastingPipeline
from .anomaly import AnomalyDetectionPipeline

logger = logging.getLogger(__name__)


class MLPipelineError(Exception):
    pass


class SmartSolarMLPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.forecasting_pipeline = ForecastingPipeline(config)
        self.anomaly_pipeline = AnomalyDetectionPipeline(config)
        self.feature_engineer = None

        self.models_trained = {"forecast": False, "anomaly": False}

        self.model_paths = {
            "forecast": config.get("model_paths", {}).get(
                "forecast", "models/forecast_model.pkl"
            ),
            "anomaly": config.get("model_paths", {}).get(
                "anomaly", "models/anomaly_model.pkl"
            ),
        }

    def set_feature_engineer(self, feature_engineer):
        self.feature_engineer = feature_engineer
        self.forecasting_pipeline.set_feature_engineer(feature_engineer)
        self.anomaly_pipeline.set_feature_engineer(feature_engineer)

    def train_all_models(
        self, training_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        results = {}

        try:
            inverter_data = training_data.get("inverter", pd.DataFrame())
            weather_data = training_data.get("weather", pd.DataFrame())
            maintenance_data = training_data.get("maintenance", pd.DataFrame())

            if inverter_data.empty or weather_data.empty:
                raise MLPipelineError("Inverter and weather data required for training")

            forecast_result = self.forecasting_pipeline.train_model(
                inverter_data,
                weather_data,
                maintenance_data,
                self.model_paths["forecast"],
            )

            if forecast_result["status"] == "success":
                self.models_trained["forecast"] = True
                results["forecast"] = forecast_result
                logger.info("Forecasting model trained successfully")

            anomaly_result = self.anomaly_pipeline.train_model(
                inverter_data,
                weather_data,
                maintenance_data,
                self.model_paths["anomaly"],
            )

            if anomaly_result["status"] == "success":
                self.models_trained["anomaly"] = True
                results["anomaly"] = anomaly_result
                logger.info("Anomaly detection model trained successfully")

            return results

        except Exception as e:
            raise MLPipelineError(f"Failed to train models: {str(e)}")

    def generate_forecast(
        self,
        periods: int = 24,
        freq: str = "H",
        future_weather: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if not self.models_trained["forecast"]:
            raise MLPipelineError("Forecasting model must be trained first")

        try:
            forecast = self.forecasting_pipeline.generate_forecast(
                periods, freq, future_weather
            )

            return forecast

        except Exception as e:
            raise MLPipelineError(f"Failed to generate forecast: {str(e)}")

    def detect_anomalies(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not self.models_trained["anomaly"]:
            raise MLPipelineError("Anomaly detection model must be trained first")

        try:
            inverter_data = data.get("inverter", pd.DataFrame())
            weather_data = data.get("weather", pd.DataFrame())
            maintenance_data = data.get("maintenance", pd.DataFrame())

            if inverter_data.empty or weather_data.empty:
                raise MLPipelineError(
                    "Inverter and weather data required for anomaly detection"
                )

            anomaly_data = self.anomaly_pipeline.detect_anomalies(
                inverter_data, weather_data, maintenance_data
            )

            return anomaly_data

        except Exception as e:
            raise MLPipelineError(f"Failed to detect anomalies: {str(e)}")

    def evaluate_models(self, test_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        results = {}

        try:
            inverter_data = test_data.get("inverter", pd.DataFrame())
            weather_data = test_data.get("weather", pd.DataFrame())
            maintenance_data = test_data.get("maintenance", pd.DataFrame())

            if inverter_data.empty or weather_data.empty:
                raise MLPipelineError(
                    "Inverter and weather data required for evaluation"
                )

            if self.models_trained["forecast"]:
                forecast_metrics = self.forecasting_pipeline.evaluate_model(
                    inverter_data, weather_data, maintenance_data
                )
                results["forecast"] = forecast_metrics

            if self.models_trained["anomaly"]:
                anomaly_data = self.anomaly_pipeline.detect_anomalies(
                    {
                        "inverter": inverter_data,
                        "weather": weather_data,
                        "maintenance": maintenance_data,
                    }
                )

                anomaly_summary = self.anomaly_pipeline.get_anomaly_summary(
                    anomaly_data
                )
                results["anomaly"] = anomaly_summary

            return results

        except Exception as e:
            raise MLPipelineError(f"Failed to evaluate models: {str(e)}")

    def get_model_status(self) -> Dict[str, Any]:
        return {
            "models_trained": self.models_trained,
            "model_paths": self.model_paths,
            "forecast_info": self.forecasting_pipeline.forecaster.get_model_info(),
            "anomaly_info": self.anomaly_pipeline.detector.get_model_info(),
        }

    def load_trained_models(self) -> Dict[str, bool]:
        results = {}

        try:
            if self.model_paths["forecast"]:
                self.forecasting_pipeline.forecaster.load_model(
                    self.model_paths["forecast"]
                )
                self.models_trained["forecast"] = True
                results["forecast"] = True
                logger.info("Forecasting model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load forecasting model: {str(e)}")
            results["forecast"] = False

        try:
            if self.model_paths["anomaly"]:
                self.anomaly_pipeline.detector.load_model(self.model_paths["anomaly"])
                self.models_trained["anomaly"] = True
                results["anomaly"] = True
                logger.info("Anomaly detection model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load anomaly detection model: {str(e)}")
            results["anomaly"] = False

        return results

    def run_full_pipeline(
        self,
        data: Dict[str, pd.DataFrame],
        generate_forecast: bool = True,
        detect_anomalies: bool = True,
    ) -> Dict[str, Any]:
        results = {}

        try:
            if generate_forecast and self.models_trained["forecast"]:
                forecast = self.generate_forecast()
                results["forecast"] = forecast
                logger.info(f"Generated forecast with {len(forecast)} predictions")

            if detect_anomalies and self.models_trained["anomaly"]:
                anomaly_data = self.detect_anomalies(data)
                anomaly_summary = self.anomaly_pipeline.get_anomaly_summary(
                    anomaly_data
                )
                results["anomaly_data"] = anomaly_data
                results["anomaly_summary"] = anomaly_summary
                logger.info(f"Detected {anomaly_summary['anomaly_count']} anomalies")

            return results

        except Exception as e:
            raise MLPipelineError(f"Failed to run full pipeline: {str(e)}")

    def save_pipeline_state(self, file_path: str) -> None:
        try:
            state = {
                "models_trained": self.models_trained,
                "model_paths": self.model_paths,
                "config": self.config,
            }

            with open(file_path, "w") as f:
                json.dump(state, f, indent=2, default=str)

            logger.info(f"Pipeline state saved to {file_path}")

        except Exception as e:
            raise MLPipelineError(f"Failed to save pipeline state: {str(e)}")

    def load_pipeline_state(self, file_path: str) -> None:
        try:
            with open(file_path, "r") as f:
                state = json.load(f)

            self.models_trained = state["models_trained"]
            self.model_paths = state["model_paths"]

            logger.info(f"Pipeline state loaded from {file_path}")

        except Exception as e:
            raise MLPipelineError(f"Failed to load pipeline state: {str(e)}")
