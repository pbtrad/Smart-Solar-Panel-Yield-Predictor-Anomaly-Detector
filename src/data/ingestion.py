import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestionError(Exception):
    pass

class BaseIngestionService(ABC):
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    @abstractmethod
    def fetch_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        pass
    
    def validate_response(self, response: requests.Response) -> None:
        if response.status_code != 200:
            raise DataIngestionError(f"API request failed: {response.status_code} - {response.text}")

class InverterIngestionService(BaseIngestionService):
    def __init__(self, api_url: str, api_key: str, inverter_ids: List[str]):
        super().__init__(api_url, api_key)
        self.inverter_ids = inverter_ids
    
    def fetch_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        all_data = []
        
        for inverter_id in self.inverter_ids:
            try:
                params = {
                    'inverter_id': inverter_id,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'granularity': 'hourly'
                }
                
                response = self.session.get(f"{self.api_url}/power_data", params=params)
                self.validate_response(response)
                
                data = response.json()
                df = pd.DataFrame(data['measurements'])
                df['inverter_id'] = inverter_id
                all_data.append(df)
                
                logger.info(f"Fetched {len(df)} records for inverter {inverter_id}")
                
            except Exception as e:
                logger.error(f"Failed to fetch data for inverter {inverter_id}: {str(e)}")
                continue
        
        if not all_data:
            raise DataIngestionError("No data fetched from any inverter")
        
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        combined_df = combined_df.sort_values('timestamp')
        
        return combined_df

class WeatherIngestionService(BaseIngestionService):
    def __init__(self, api_url: str, api_key: str, location: Dict[str, float]):
        super().__init__(api_url, api_key)
        self.lat = location['lat']
        self.lon = location['lon']
    
    def fetch_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        try:
            params = {
                'lat': self.lat,
                'lon': self.lon,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'units': 'metric'
            }
            
            response = self.session.get(f"{self.api_url}/weather/history", params=params)
            self.validate_response(response)
            
            data = response.json()
            df = pd.DataFrame(data['hourly'])
            df['timestamp'] = pd.to_datetime(df['dt'], unit='s')
            df = df.drop('dt', axis=1)
            
            logger.info(f"Fetched {len(df)} weather records")
            return df
            
        except Exception as e:
            raise DataIngestionError(f"Weather data fetch failed: {str(e)}")

class MaintenanceIngestionService:
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def fetch_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
            filtered_df = df[mask].copy()
            
            logger.info(f"Loaded {len(filtered_df)} maintenance records")
            return filtered_df
            
        except Exception as e:
            raise DataIngestionError(f"Maintenance data load failed: {str(e)}")

class DataIngestionOrchestrator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.services = {}
        self._initialize_services()
    
    def _initialize_services(self):
        inverter_config = self.config.get('inverter', {})
        if inverter_config.get('enabled', False):
            self.services['inverter'] = InverterIngestionService(
                api_url=inverter_config['api_url'],
                api_key=inverter_config['api_key'],
                inverter_ids=inverter_config['inverter_ids']
            )
        
        weather_config = self.config.get('weather', {})
        if weather_config.get('enabled', False):
            self.services['weather'] = WeatherIngestionService(
                api_url=weather_config['api_url'],
                api_key=weather_config['api_key'],
                location=weather_config['location']
            )
        
        maintenance_config = self.config.get('maintenance', {})
        if maintenance_config.get('enabled', False):
            self.services['maintenance'] = MaintenanceIngestionService(
                file_path=maintenance_config['file_path']
            )
    
    def ingest_all_data(self, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        results = {}
        
        for service_name, service in self.services.items():
            try:
                logger.info(f"Starting ingestion for {service_name}")
                data = service.fetch_data(start_date, end_date)
                results[service_name] = data
                logger.info(f"Completed ingestion for {service_name}: {len(data)} records")
                
            except Exception as e:
                logger.error(f"Failed to ingest {service_name} data: {str(e)}")
                continue
        
        return results
    
    def get_data_summary(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        summary = {}
        
        for source, df in data_dict.items():
            if not df.empty:
                summary[source] = {
                    'record_count': len(df),
                    'date_range': {
                        'start': df['timestamp'].min().isoformat(),
                        'end': df['timestamp'].max().isoformat()
                    },
                    'columns': list(df.columns)
                }
        
        return summary 