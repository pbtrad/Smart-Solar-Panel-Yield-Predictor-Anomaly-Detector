import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.features import (
    ProphetFeatureEngineer,
    DataQualityChecker,
    FeatureEngineeringError,
)


class TestProphetFeatureEngineer(unittest.TestCase):
    def setUp(self):
        self.engineer = ProphetFeatureEngineer()

        self.sample_inverter_data = pd.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 10, 0, 0),
                    datetime(2024, 1, 1, 11, 0, 0),
                    datetime(2024, 1, 1, 12, 0, 0),
                ],
                "power_output": [100.5, 120.3, 150.7],
                "voltage": [240.0, 240.0, 240.0],
                "current": [0.42, 0.50, 0.63],
                "inverter_id": ["inv_001", "inv_001", "inv_001"],
            }
        )

        self.sample_weather_data = pd.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 10, 0, 0),
                    datetime(2024, 1, 1, 11, 0, 0),
                    datetime(2024, 1, 1, 12, 0, 0),
                ],
                "temp": [15.5, 16.2, 18.0],
                "humidity": [65, 60, 55],
                "wind_speed": [5.2, 4.8, 6.1],
                "clouds": [20, 15, 10],
                "solar_radiation": [800.0, 850.0, 900.0],
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

    def test_clean_inverter_data_valid(self):
        result = self.engineer.clean_inverter_data(self.sample_inverter_data)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertIn("power_output", result.columns)
        self.assertIn("voltage", result.columns)
        self.assertIn("current", result.columns)

    def test_clean_inverter_data_with_outliers(self):
        dirty_data = self.sample_inverter_data.copy()
        dirty_data.loc[0, "power_output"] = -100
        dirty_data.loc[1, "power_output"] = 50000
        dirty_data.loc[2, "voltage"] = 2000

        result = self.engineer.clean_inverter_data(dirty_data)

        self.assertEqual(len(result), 0)

    def test_clean_weather_data_valid(self):
        result = self.engineer.clean_weather_data(self.sample_weather_data)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertIn("temp", result.columns)
        self.assertIn("solar_radiation", result.columns)

    def test_clean_weather_data_with_outliers(self):
        dirty_data = self.sample_weather_data.copy()
        dirty_data.loc[0, "temp"] = -100
        dirty_data.loc[1, "humidity"] = 150
        dirty_data.loc[2, "solar_radiation"] = 2000

        result = self.engineer.clean_weather_data(dirty_data)

        self.assertEqual(len(result), 0)

    def test_merge_weather_inverter(self):
        result = self.engineer.merge_weather_inverter(
            self.sample_inverter_data, self.sample_weather_data
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertIn("power_output", result.columns)
        self.assertIn("temp", result.columns)
        self.assertIn("solar_radiation", result.columns)

    def test_merge_weather_inverter_empty_data(self):
        empty_df = pd.DataFrame()

        result = self.engineer.merge_weather_inverter(
            empty_df, self.sample_weather_data
        )
        self.assertTrue(result.empty)

        result = self.engineer.merge_weather_inverter(
            self.sample_inverter_data, empty_df
        )
        self.assertTrue(result.empty)

    def test_create_efficiency_features(self):
        merged_data = self.engineer.merge_weather_inverter(
            self.sample_inverter_data, self.sample_weather_data
        )

        result = self.engineer.create_efficiency_features(merged_data)

        self.assertIn("efficiency", result.columns)
        self.assertIn("temp_efficiency", result.columns)
        self.assertIn("cloud_impact", result.columns)
        self.assertIn("weather_efficiency", result.columns)

        self.assertTrue(all(result["efficiency"] > 0))
        self.assertTrue(all(result["cloud_impact"] >= 0))
        self.assertTrue(all(result["cloud_impact"] <= 1))

    def test_add_maintenance_features(self):
        merged_data = self.engineer.merge_weather_inverter(
            self.sample_inverter_data, self.sample_weather_data
        )

        result = self.engineer.add_maintenance_features(
            merged_data, self.sample_maintenance_data
        )

        self.assertIn("maintenance_flag", result.columns)
        self.assertIn("cleaning_effect", result.columns)

        self.assertTrue(all(result["maintenance_flag"] >= 0))
        self.assertTrue(all(result["cleaning_effect"] >= 0))

    def test_prepare_prophet_data(self):
        merged_data = self.engineer.merge_weather_inverter(
            self.sample_inverter_data, self.sample_weather_data
        )
        merged_data = self.engineer.create_efficiency_features(merged_data)

        result = self.engineer.prepare_prophet_data(merged_data)

        self.assertIn("ds", result.columns)
        self.assertIn("y", result.columns)
        self.assertIn("temp", result.columns)
        self.assertIn("solar_radiation", result.columns)
        self.assertIn("efficiency", result.columns)

        self.assertEqual(result["y"].iloc[0], 100.5)
        self.assertEqual(result["ds"].iloc[0], datetime(2024, 1, 1, 10, 0, 0))

    def test_create_forecast_features(self):
        merged_data = self.engineer.merge_weather_inverter(
            self.sample_inverter_data, self.sample_weather_data
        )

        result = self.engineer.create_forecast_features(merged_data)

        self.assertIn("hour", result.columns)
        self.assertIn("day_of_week", result.columns)
        self.assertIn("month", result.columns)
        self.assertIn("season", result.columns)
        self.assertIn("is_daytime", result.columns)
        self.assertIn("peak_sun_hours", result.columns)

        self.assertTrue(all(result["is_daytime"].isin([0, 1])))
        self.assertTrue(all(result["peak_sun_hours"].isin([0, 1])))

    def test_handle_missing_values(self):
        data_with_missing = self.sample_inverter_data.copy()
        data_with_missing.loc[0, "power_output"] = np.nan
        data_with_missing.loc[1, "voltage"] = np.nan

        result = self.engineer.handle_missing_values(data_with_missing)

        self.assertFalse(result["power_output"].isnull().any())
        self.assertFalse(result["voltage"].isnull().any())

    def test_process_for_prophet_success(self):
        result = self.engineer.process_for_prophet(
            self.sample_inverter_data,
            self.sample_weather_data,
            self.sample_maintenance_data,
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("ds", result.columns)
        self.assertIn("y", result.columns)
        self.assertIn("temp", result.columns)
        self.assertIn("solar_radiation", result.columns)
        self.assertIn("efficiency", result.columns)
        self.assertIn("maintenance_flag", result.columns)
        self.assertIn("cleaning_effect", result.columns)

    def test_process_for_prophet_empty_data(self):
        empty_df = pd.DataFrame()

        result = self.engineer.process_for_prophet(empty_df, self.sample_weather_data)
        self.assertTrue(result.empty)

        result = self.engineer.process_for_prophet(self.sample_inverter_data, empty_df)
        self.assertTrue(result.empty)

    def test_process_for_prophet_no_maintenance(self):
        result = self.engineer.process_for_prophet(
            self.sample_inverter_data, self.sample_weather_data
        )

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("ds", result.columns)
        self.assertIn("y", result.columns)
        self.assertNotIn("maintenance_flag", result.columns)


class TestDataQualityChecker(unittest.TestCase):
    def setUp(self):
        self.checker = DataQualityChecker()

        self.sample_data = pd.DataFrame(
            {
                "timestamp": [
                    datetime(2024, 1, 1, 10, 0, 0),
                    datetime(2024, 1, 1, 11, 0, 0),
                    datetime(2024, 1, 1, 12, 0, 0),
                ],
                "power_output": [100.5, 120.3, 150.7],
                "voltage": [240.0, 240.0, 240.0],
                "current": [0.42, 0.50, 0.63],
            }
        )

    def test_check_data_quality_valid(self):
        report = self.checker.check_data_quality(self.sample_data, "inverter")

        self.assertIn("source", report)
        self.assertIn("total_records", report)
        self.assertIn("missing_values", report)
        self.assertIn("outliers", report)
        self.assertIn("data_types", report)
        self.assertIn("timestamp_range", report)

        self.assertEqual(report["source"], "inverter")
        self.assertEqual(report["total_records"], 3)
        self.assertEqual(report["missing_values"]["power_output"], 0)

    def test_check_data_quality_empty_data(self):
        empty_df = pd.DataFrame()

        report = self.checker.check_data_quality(empty_df, "inverter")

        self.assertEqual(report["status"], "empty")
        self.assertIn("inverter data is empty", report["message"])

    def test_check_data_quality_with_missing_values(self):
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0, "power_output"] = np.nan

        report = self.checker.check_data_quality(data_with_missing, "inverter")

        self.assertEqual(report["missing_values"]["power_output"], 1)

    def test_check_data_quality_with_outliers(self):
        data_with_outliers = self.sample_data.copy()
        data_with_outliers.loc[0, "power_output"] = 1000000

        report = self.checker.check_data_quality(data_with_outliers, "inverter")

        self.assertIn("power_output", report["outliers"])
        self.assertIsInstance(report["outliers"]["power_output"], int)

    def test_get_summary_report(self):
        self.checker.check_data_quality(self.sample_data, "inverter")
        self.checker.check_data_quality(self.sample_data, "weather")

        summary = self.checker.get_summary_report()

        self.assertIn("total_sources", summary)
        self.assertIn("sources", summary)
        self.assertIn("reports", summary)

        self.assertEqual(summary["total_sources"], 2)
        self.assertIn("inverter", summary["sources"])
        self.assertIn("weather", summary["sources"])


if __name__ == "__main__":
    unittest.main()
