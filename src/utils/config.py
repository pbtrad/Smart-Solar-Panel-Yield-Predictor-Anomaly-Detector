import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "configs/app.yaml"
        self.config = self._load_config()
        self._load_env_vars()
    
    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'inverter': {
                'enabled': False,
                'api_url': os.getenv('INVERTER_API_URL', ''),
                'api_key': os.getenv('INVERTER_API_KEY', ''),
                'inverter_ids': []
            },
            'weather': {
                'enabled': False,
                'api_url': os.getenv('WEATHER_API_URL', ''),
                'api_key': os.getenv('WEATHER_API_KEY', ''),
                'location': {
                    'lat': float(os.getenv('WEATHER_LAT', '0.0')),
                    'lon': float(os.getenv('WEATHER_LON', '0.0'))
                }
            },
            'maintenance': {
                'enabled': False,
                'file_path': os.getenv('MAINTENANCE_FILE_PATH', '')
            },
            'storage': {
                'timeseries': {
                    'url': os.getenv('INFLUXDB_URL', 'http://localhost:8086'),
                    'token': os.getenv('INFLUXDB_TOKEN', ''),
                    'org': os.getenv('INFLUXDB_ORG', ''),
                    'bucket': os.getenv('INFLUXDB_BUCKET', 'solar_data')
                },
                'object': {
                    'endpoint_url': os.getenv('S3_ENDPOINT_URL', 'http://localhost:9000'),
                    'access_key': os.getenv('S3_ACCESS_KEY', ''),
                    'secret_key': os.getenv('S3_SECRET_KEY', ''),
                    'bucket_name': os.getenv('S3_BUCKET_NAME', 'solar-data')
                }
            },
            'models': {
                'forecast': {
                    'algorithm': os.getenv('FORECAST_ALGORITHM', 'prophet'),
                    'horizon_hours': int(os.getenv('FORECAST_HORIZON', '24')),
                    'retrain_frequency_hours': int(os.getenv('RETRAIN_FREQUENCY', '168'))
                },
                'anomaly': {
                    'algorithm': os.getenv('ANOMALY_ALGORITHM', 'isolation_forest'),
                    'threshold': float(os.getenv('ANOMALY_THRESHOLD', '0.95'))
                }
            },
            'api': {
                'host': os.getenv('API_HOST', '0.0.0.0'),
                'port': int(os.getenv('API_PORT', '8000')),
                'debug': os.getenv('API_DEBUG', 'False').lower() == 'true'
            },
            'notifications': {
                'enabled': os.getenv('NOTIFICATIONS_ENABLED', 'False').lower() == 'true',
                'email': {
                    'smtp_server': os.getenv('SMTP_SERVER', ''),
                    'smtp_port': int(os.getenv('SMTP_PORT', '587')),
                    'username': os.getenv('SMTP_USERNAME', ''),
                    'password': os.getenv('SMTP_PASSWORD', ''),
                    'from_email': os.getenv('FROM_EMAIL', ''),
                    'to_emails': os.getenv('TO_EMAILS', '').split(',')
                },
                'sms': {
                    'twilio_account_sid': os.getenv('TWILIO_ACCOUNT_SID', ''),
                    'twilio_auth_token': os.getenv('TWILIO_AUTH_TOKEN', ''),
                    'twilio_phone_number': os.getenv('TWILIO_PHONE_NUMBER', ''),
                    'to_phone_numbers': os.getenv('TO_PHONE_NUMBERS', '').split(',')
                }
            }
        }
    
    def _load_env_vars(self):
        for section in self.config:
            if isinstance(self.config[section], dict):
                for key in self.config[section]:
                    env_key = f"{section.upper()}_{key.upper()}"
                    env_value = os.getenv(env_key)
                    if env_value is not None:
                        if isinstance(self.config[section][key], bool):
                            self.config[section][key] = env_value.lower() == 'true'
                        elif isinstance(self.config[section][key], int):
                            self.config[section][key] = int(env_value)
                        elif isinstance(self.config[section][key], float):
                            self.config[section][key] = float(env_value)
                        else:
                            self.config[section][key] = env_value
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_ingestion_config(self) -> Dict[str, Any]:
        return {
            'inverter': self.config['inverter'],
            'weather': self.config['weather'],
            'maintenance': self.config['maintenance']
        }
    
    def get_storage_config(self) -> Dict[str, Any]:
        return {
            'timeseries': self.config['storage']['timeseries'],
            'object': self.config['storage']['object']
        }
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        return self.config['models'].get(model_type, {})
    
    def get_api_config(self) -> Dict[str, Any]:
        return self.config['api']
    
    def get_notification_config(self) -> Dict[str, Any]:
        return self.config['notifications']
    
    def validate_config(self) -> bool:
        required_sections = ['inverter', 'weather', 'storage', 'models', 'api']
        
        for section in required_sections:
            if section not in self.config:
                print(f"Missing required config section: {section}")
                return False
        
        if self.config['inverter']['enabled']:
            if not self.config['inverter']['api_url'] or not self.config['inverter']['api_key']:
                print("Inverter API configuration incomplete")
                return False
        
        if self.config['weather']['enabled']:
            if not self.config['weather']['api_url'] or not self.config['weather']['api_key']:
                print("Weather API configuration incomplete")
                return False
        
        return True
    
    def save_config(self, file_path: Optional[str] = None):
        if file_path is None:
            file_path = self.config_path
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False) 