import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle

logger = logging.getLogger(__name__)


class AnomalyDetectionError(Exception):
    pass


class SolarAnomalyDetector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.model_path = None
        self.is_trained = False

        anomaly_config = config.get("anomaly", {})
        self.contamination = anomaly_config.get("contamination", 0.1)
        self.threshold = anomaly_config.get("threshold", 0.95)

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()

        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month

        df["is_daytime"] = ((df["hour"] >= 6) & (df["hour"] <= 18)).astype(int)

        if "power_output" in df.columns and "solar_radiation" in df.columns:
            df["efficiency"] = np.where(
                df["solar_radiation"] > 0,
                df["power_output"] / df["solar_radiation"],
                np.nan,
            )

            df["efficiency_ratio"] = (
                df["efficiency"] / df["efficiency"].rolling(24, min_periods=1).mean()
            )

        if "power_output" in df.columns:
            df["power_ratio"] = (
                df["power_output"]
                / df["power_output"].rolling(24, min_periods=1).mean()
            )
            df["power_change"] = df["power_output"].diff()
            df["power_volatility"] = df["power_output"].rolling(6).std()

        if "temp" in df.columns:
            df["temp_effect"] = df["temp"] * df.get("efficiency", 1)

        if "clouds" in df.columns:
            df["cloud_impact"] = (100 - df["clouds"]) / 100

        return df

    def select_features(self, df: pd.DataFrame) -> List[str]:
        base_features = ["hour", "day_of_week", "month", "is_daytime"]

        available_features = []
        for feature in base_features:
            if feature in df.columns:
                available_features.append(feature)

        if "efficiency" in df.columns:
            available_features.extend(["efficiency", "efficiency_ratio"])

        if "power_output" in df.columns:
            available_features.extend(
                ["power_ratio", "power_change", "power_volatility"]
            )

        if "temp" in df.columns:
            available_features.append("temp_effect")

        if "clouds" in df.columns:
            available_features.append("cloud_impact")

        return available_features

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        if df.empty:
            raise AnomalyDetectionError("No data provided for training")

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        df = self.create_features(df)

        feature_columns = self.select_features(df)

        if not feature_columns:
            raise AnomalyDetectionError("No features available for anomaly detection")

        feature_df = df[feature_columns].copy()

        feature_df = feature_df.fillna(method="ffill").fillna(method="bfill").fillna(0)

        self.feature_columns = feature_columns

        logger.info(
            f"Prepared {len(feature_df)} records with {len(feature_columns)} features"
        )
        return feature_df, feature_columns

    def train(
        self, training_data: pd.DataFrame, model_path: Optional[str] = None
    ) -> None:
        try:
            feature_df, feature_columns = self.prepare_training_data(training_data)

            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100,
                max_samples="auto",
            )

            scaled_features = self.scaler.fit_transform(feature_df)

            self.model.fit(scaled_features)

            self.is_trained = True
            self.model_path = model_path

            if model_path:
                self.save_model(model_path)

            logger.info("Anomaly detection model trained successfully")

        except Exception as e:
            raise AnomalyDetectionError(
                f"Failed to train anomaly detection model: {str(e)}"
            )

    def detect_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.is_trained or self.model is None:
            raise AnomalyDetectionError(
                "Model must be trained before anomaly detection"
            )

        try:
            df = data.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            df = self.create_features(df)

            feature_df = df[self.feature_columns].copy()
            feature_df = (
                feature_df.fillna(method="ffill").fillna(method="bfill").fillna(0)
            )

            scaled_features = self.scaler.transform(feature_df)

            anomaly_scores = self.model.decision_function(scaled_features)
            anomaly_predictions = self.model.predict(scaled_features)

            df["anomaly_score"] = anomaly_scores
            df["is_anomaly"] = (anomaly_predictions == -1).astype(int)
            df["anomaly_probability"] = 1 - np.exp(anomaly_scores)

            high_anomaly_threshold = np.percentile(
                anomaly_scores, (1 - self.threshold) * 100
            )
            df["high_anomaly"] = (anomaly_scores < high_anomaly_threshold).astype(int)

            logger.info(
                f"Detected {df['is_anomaly'].sum()} anomalies out of {len(df)} records"
            )
            return df

        except Exception as e:
            raise AnomalyDetectionError(f"Failed to detect anomalies: {str(e)}")

    def get_anomaly_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        if "is_anomaly" not in data.columns:
            raise AnomalyDetectionError(
                "Data must be processed for anomaly detection first"
            )

        total_records = len(data)
        anomaly_count = data["is_anomaly"].sum()
        high_anomaly_count = data.get("high_anomaly", pd.Series([0] * len(data))).sum()

        anomaly_percentage = (
            (anomaly_count / total_records) * 100 if total_records > 0 else 0
        )
        high_anomaly_percentage = (
            (high_anomaly_count / total_records) * 100 if total_records > 0 else 0
        )

        if anomaly_count > 0:
            avg_anomaly_score = data[data["is_anomaly"] == 1]["anomaly_score"].mean()
            min_anomaly_score = data[data["is_anomaly"] == 1]["anomaly_score"].min()
        else:
            avg_anomaly_score = 0
            min_anomaly_score = 0

        return {
            "total_records": total_records,
            "anomaly_count": anomaly_count,
            "high_anomaly_count": high_anomaly_count,
            "anomaly_percentage": anomaly_percentage,
            "high_anomaly_percentage": high_anomaly_percentage,
            "avg_anomaly_score": avg_anomaly_score,
            "min_anomaly_score": min_anomaly_score,
        }

    def save_model(self, file_path: str) -> None:
        if not self.is_trained or self.model is None:
            raise AnomalyDetectionError("No trained model to save")

        try:
            with open(file_path, "wb") as f:
                pickle.dump(
                    {
                        "model": self.model,
                        "scaler": self.scaler,
                        "feature_columns": self.feature_columns,
                        "config": self.config,
                        "is_trained": self.is_trained,
                    },
                    f,
                )

            logger.info(f"Anomaly detection model saved to {file_path}")

        except Exception as e:
            raise AnomalyDetectionError(f"Failed to save model: {str(e)}")

    def load_model(self, file_path: str) -> None:
        try:
            with open(file_path, "rb") as f:
                model_data = pickle.load(f)

            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.feature_columns = model_data["feature_columns"]
            self.config = model_data["config"]
            self.is_trained = model_data["is_trained"]
            self.model_path = file_path

            logger.info(f"Anomaly detection model loaded from {file_path}")

        except Exception as e:
            raise AnomalyDetectionError(f"Failed to load model: {str(e)}")

    def get_model_info(self) -> Dict[str, Any]:
        if not self.is_trained:
            return {"status": "not_trained"}

        return {
            "status": "trained",
            "feature_columns": self.feature_columns,
            "model_path": self.model_path,
            "contamination": self.contamination,
            "threshold": self.threshold,
        }


class AnomalyDetectionPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detector = SolarAnomalyDetector(config)
        self.feature_engineer = None

    def set_feature_engineer(self, feature_engineer):
        self.feature_engineer = feature_engineer

    def train_model(
        self,
        inverter_data: pd.DataFrame,
        weather_data: pd.DataFrame,
        maintenance_data: Optional[pd.DataFrame] = None,
        model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            if self.feature_engineer is None:
                raise AnomalyDetectionError("Feature engineer not set")

            processed_data = self.feature_engineer.process_for_anomaly_detection(
                inverter_data, weather_data, maintenance_data
            )

            if processed_data.empty:
                raise AnomalyDetectionError("No processed data available for training")

            self.detector.train(processed_data, model_path)

            return {
                "status": "success",
                "message": "Anomaly detection model trained successfully",
            }

        except Exception as e:
            raise AnomalyDetectionError(f"Training pipeline failed: {str(e)}")

    def detect_anomalies(
        self,
        inverter_data: pd.DataFrame,
        weather_data: pd.DataFrame,
        maintenance_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        try:
            if self.feature_engineer is None:
                raise AnomalyDetectionError("Feature engineer not set")

            processed_data = self.feature_engineer.process_for_anomaly_detection(
                inverter_data, weather_data, maintenance_data
            )

            if processed_data.empty:
                raise AnomalyDetectionError(
                    "No processed data available for anomaly detection"
                )

            anomaly_data = self.detector.detect_anomalies(processed_data)

            return anomaly_data

        except Exception as e:
            raise AnomalyDetectionError(f"Anomaly detection pipeline failed: {str(e)}")

    def get_anomaly_summary(self, anomaly_data: pd.DataFrame) -> Dict[str, Any]:
        try:
            summary = self.detector.get_anomaly_summary(anomaly_data)
            return summary

        except Exception as e:
            raise AnomalyDetectionError(f"Failed to get anomaly summary: {str(e)}")
