from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime


class PredictionRequest(BaseModel):
    periods: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Number of periods to forecast (1-168 hours)",
    )
    weather_data: Optional[List[Dict[str, Any]]] = Field(
        None, description="Future weather data for forecasting"
    )
    inverter_id: Optional[str] = Field(
        None, max_length=50, description="Specific inverter ID"
    )

    @validator("weather_data")
    def validate_weather_data(cls, v):
        if v is not None:
            if len(v) > 1000:  # Prevent large payload attacks
                raise ValueError("Weather data too large")
            for item in v:
                if not isinstance(item, dict):
                    raise ValueError("Weather data must be list of dictionaries")
        return v


class AnomalyRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="Data to analyze for anomalies")
    threshold: Optional[float] = Field(
        default=0.95, ge=0.0, le=1.0, description="Anomaly detection threshold"
    )

    @validator("data")
    def validate_data(cls, v):
        if len(v) > 10000:
            raise ValueError("Data too large")
        if len(v) == 0:
            raise ValueError("Data cannot be empty")
        return v


class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]] = Field(..., description="Forecast predictions")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class AnomalyResponse(BaseModel):
    anomalies: List[Dict[str, Any]] = Field(..., description="Detected anomalies")
    summary: Dict[str, Any] = Field(..., description="Anomaly summary statistics")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Error timestamp"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
