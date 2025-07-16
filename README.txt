# Smart Solar-Panel Yield Predictor & Anomaly Detector

The goal of this project is to develop a smart system that predicts solar panel output and detects anomalies in performance. This enables solar operators to identify faults or underperformance (like soiling, shading, or hardware issues) early and take corrective action. It also provides accurate short-term forecasts of power generation to help with energy planning and monitoring.

The system will:
- Predict hourly and daily solar energy output using weather and historical data.
- Detect when a panel or array is underperforming relative to expectations.
- Provide real-time monitoring through a dashboard.
- Send alerts via email or SMS when issues are detected.

## Technologies Plan

- **Data ingestion**: Python scripts, Kafka (optional), APIs for inverters and weather
- **Storage**: InfluxDB or TimescaleDB for time-series data, S3 for raw and model files
- **Processing & pipelines**: Pandas, Prefect or Airflow for scheduling
- **Forecasting models**: Prophet, XGBoost, or LSTM
- **Anomaly detection**: Isolation Forest, Autoencoder (scikit-learn or TensorFlow)
- **Serving**: FastAPI for model inference and data access
- **Dashboard**: Plotly Dash or React with Recharts
- **Notifications**: SMTP for email, Twilio or AWS SNS for SMS
- **DevOps**: Docker, AWS ECS