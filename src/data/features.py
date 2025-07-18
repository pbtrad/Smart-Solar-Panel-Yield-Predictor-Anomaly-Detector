import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureEngineeringError(Exception):
    pass


class ProphetFeatureEngineer:
    def __init__(self):
        self.weather_df = None
        self.maintenance_df = None

    def clean_inverter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()

        df["power_output"] = pd.to_numeric(df["power_output"], errors="coerce")
        df["voltage"] = pd.to_numeric(df["voltage"], errors="coerce")
        df["current"] = pd.to_numeric(df["current"], errors="coerce")

        df = df.dropna(subset=["power_output", "timestamp"])

        df = df[df["power_output"] >= 0]
        df = df[df["power_output"] <= 10000]

        df = df[df["voltage"] >= 0]
        df = df[df["voltage"] <= 1000]

        df = df[df["current"] >= 0]
        df = df[df["current"] <= 100]

        return df

    def clean_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()

        df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
        df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
        df["wind_speed"] = pd.to_numeric(df["wind_speed"], errors="coerce")
        df["clouds"] = pd.to_numeric(df["clouds"], errors="coerce")
        df["solar_radiation"] = pd.to_numeric(df["solar_radiation"], errors="coerce")

        df = df.dropna(subset=["temp", "timestamp"])

        df = df[df["temp"] >= -50]
        df = df[df["temp"] <= 80]

        df = df[df["humidity"] >= 0]
        df = df[df["humidity"] <= 100]

        df = df[df["wind_speed"] >= 0]
        df = df[df["wind_speed"] <= 100]

        df = df[df["clouds"] >= 0]
        df = df[df["clouds"] <= 100]

        df = df[df["solar_radiation"] >= 0]
        df = df[df["solar_radiation"] <= 1500]

        return df

    def merge_weather_inverter(
        self, inverter_df: pd.DataFrame, weather_df: pd.DataFrame
    ) -> pd.DataFrame:
        if inverter_df.empty or weather_df.empty:
            return pd.DataFrame()

        inverter_df = inverter_df.copy()
        weather_df = weather_df.copy()

        inverter_df["timestamp"] = pd.to_datetime(inverter_df["timestamp"])
        weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"])

        merged_df = pd.merge_asof(
            inverter_df.sort_values("timestamp"),
            weather_df.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("1H"),
        )

        return merged_df

    def create_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()

        df["efficiency"] = np.where(
            df["solar_radiation"] > 0,
            df["power_output"] / df["solar_radiation"],
            np.nan,
        )

        df["temp_efficiency"] = df["temp"] * df["efficiency"]

        df["cloud_impact"] = (100 - df["clouds"]) / 100

        df["weather_efficiency"] = df["efficiency"] * df["cloud_impact"]

        return df

    def add_maintenance_features(
        self, df: pd.DataFrame, maintenance_df: pd.DataFrame
    ) -> pd.DataFrame:
        if df.empty or maintenance_df.empty:
            return df

        df = df.copy()
        maintenance_df = maintenance_df.copy()

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        maintenance_df["timestamp"] = pd.to_datetime(maintenance_df["timestamp"])

        df["maintenance_flag"] = 0
        df["days_since_maintenance"] = np.nan

        for _, maintenance_row in maintenance_df.iterrows():
            maintenance_date = maintenance_row["timestamp"]
            maintenance_type = maintenance_row["maintenance_type"]

            mask = df["timestamp"] >= maintenance_date
            df.loc[mask, "maintenance_flag"] = 1

            if maintenance_type == "cleaning":
                mask = (df["timestamp"] >= maintenance_date) & (
                    df["timestamp"] <= maintenance_date + timedelta(days=7)
                )
                df.loc[mask, "cleaning_effect"] = 1
            else:
                df["cleaning_effect"] = 0

        return df

    def prepare_prophet_data(
        self, df: pd.DataFrame, target_column: str = "power_output"
    ) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        df = df.copy()

        df["ds"] = df["timestamp"]
        df["y"] = df[target_column]

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

        prophet_df = df[["ds", "y"] + available_regressors].copy()

        prophet_df = prophet_df.dropna(subset=["y"])

        return prophet_df

    def create_forecast_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()

        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["month"] = df["timestamp"].dt.month
        df["season"] = df["timestamp"].dt.month % 12 // 3 + 1

        df["is_daytime"] = ((df["hour"] >= 6) & (df["hour"] <= 18)).astype(int)

        df["peak_sun_hours"] = ((df["hour"] >= 10) & (df["hour"] <= 16)).astype(int)

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()

        numeric_columns = df.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            if col in ["y", "ds"]:
                continue

            if df[col].isnull().sum() > 0:
                df[col] = (
                    df[col].fillna(method="ffill").fillna(method="bfill").fillna(0)
                )

        return df

    def process_for_prophet(
        self,
        inverter_df: pd.DataFrame,
        weather_df: pd.DataFrame,
        maintenance_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        try:
            inverter_df = self.clean_inverter_data(inverter_df)
            weather_df = self.clean_weather_data(weather_df)

            if inverter_df.empty or weather_df.empty:
                logger.warning("No valid data after cleaning")
                return pd.DataFrame()

            merged_df = self.merge_weather_inverter(inverter_df, weather_df)

            if merged_df.empty:
                logger.warning("No data after merging")
                return pd.DataFrame()

            merged_df = self.create_efficiency_features(merged_df)
            merged_df = self.create_forecast_features(merged_df)

            if maintenance_df is not None and not maintenance_df.empty:
                merged_df = self.add_maintenance_features(merged_df, maintenance_df)

            prophet_df = self.prepare_prophet_data(merged_df)

            if prophet_df.empty:
                logger.warning("No data prepared for Prophet")
                return pd.DataFrame()

            prophet_df = self.handle_missing_values(prophet_df)

            logger.info(f"Prepared {len(prophet_df)} records for Prophet forecasting")
            return prophet_df

        except Exception as e:
            raise FeatureEngineeringError(
                f"Failed to process data for Prophet: {str(e)}"
            )


class DataQualityChecker:
    def __init__(self):
        self.quality_report = {}

    def check_data_quality(self, df: pd.DataFrame, source: str) -> Dict:
        if df.empty:
            return {"status": "empty", "message": f"{source} data is empty"}

        report = {
            "source": source,
            "total_records": len(df),
            "missing_values": {},
            "outliers": {},
            "data_types": {},
            "timestamp_range": {},
        }

        for column in df.columns:
            if column == "timestamp":
                report["timestamp_range"] = {
                    "start": df["timestamp"].min().isoformat(),
                    "end": df["timestamp"].max().isoformat(),
                }
                continue

            missing_count = df[column].isnull().sum()
            report["missing_values"][column] = missing_count

            if df[column].dtype in ["int64", "float64"]:
                if len(df) > 3:
                    q1 = df[column].quantile(0.25)
                    q3 = df[column].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr

                    outliers = df[
                        (df[column] < lower_bound) | (df[column] > upper_bound)
                    ]
                    report["outliers"][column] = len(outliers)
                else:
                    mean_val = df[column].mean()
                    std_val = df[column].std()
                    if std_val > 0:
                        outliers = df[abs(df[column] - mean_val) > 3 * std_val]
                        report["outliers"][column] = len(outliers)
                    else:
                        report["outliers"][column] = 0

            report["data_types"][column] = str(df[column].dtype)

        self.quality_report[source] = report
        return report

    def get_summary_report(self) -> Dict:
        return {
            "total_sources": len(self.quality_report),
            "sources": list(self.quality_report.keys()),
            "reports": self.quality_report,
        }
