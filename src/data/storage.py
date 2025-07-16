import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import json
import boto3
from io import StringIO

logger = logging.getLogger(__name__)

class StorageError(Exception):
    pass

class TimeSeriesStorage:
    def __init__(self, url: str, token: str, org: str, bucket: str):
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.bucket = bucket
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()
    
    def store_inverter_data(self, df: pd.DataFrame) -> None:
        try:
            points = []
            
            for _, row in df.iterrows():
                point = Point("solar_power") \
                    .tag("inverter_id", row['inverter_id']) \
                    .field("power_output", float(row['power_output'])) \
                    .field("voltage", float(row.get('voltage', 0))) \
                    .field("current", float(row.get('current', 0))) \
                    .time(row['timestamp'])
                
                points.append(point)
            
            self.write_api.write(bucket=self.bucket, record=points)
            logger.info(f"Stored {len(points)} inverter data points")
            
        except Exception as e:
            raise StorageError(f"Failed to store inverter data: {str(e)}")
    
    def store_weather_data(self, df: pd.DataFrame) -> None:
        try:
            points = []
            
            for _, row in df.iterrows():
                point = Point("weather") \
                    .field("temperature", float(row['temp'])) \
                    .field("humidity", float(row.get('humidity', 0))) \
                    .field("wind_speed", float(row.get('wind_speed', 0))) \
                    .field("cloud_cover", float(row.get('clouds', 0))) \
                    .field("solar_radiation", float(row.get('solar_radiation', 0))) \
                    .time(row['timestamp'])
                
                points.append(point)
            
            self.write_api.write(bucket=self.bucket, record=points)
            logger.info(f"Stored {len(points)} weather data points")
            
        except Exception as e:
            raise StorageError(f"Failed to store weather data: {str(e)}")
    
    def query_power_data(self, start_time: datetime, end_time: datetime, inverter_ids: Optional[List[str]] = None) -> pd.DataFrame:
        try:
            query = f'''
            from(bucket: "{self.bucket}")
                |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
                |> filter(fn: (r) => r._measurement == "solar_power")
            '''
            
            if inverter_ids:
                ids_filter = " or ".join([f'r.inverter_id == "{id}"' for id in inverter_ids])
                query += f' |> filter(fn: (r) => {ids_filter})'
            
            result = self.query_api.query_data_frame(query=query, data_frame_index=['_time'])
            
            if result.empty:
                return pd.DataFrame()
            
            result = result.pivot(index='_time', columns='_field', values='_value')
            result.reset_index(inplace=True)
            result.rename(columns={'_time': 'timestamp'}, inplace=True)
            
            return result
            
        except Exception as e:
            raise StorageError(f"Failed to query power data: {str(e)}")
    
    def query_weather_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        try:
            query = f'''
            from(bucket: "{self.bucket}")
                |> range(start: {start_time.isoformat()}, stop: {end_time.isoformat()})
                |> filter(fn: (r) => r._measurement == "weather")
            '''
            
            result = self.query_api.query_data_frame(query=query, data_frame_index=['_time'])
            
            if result.empty:
                return pd.DataFrame()
            
            result = result.pivot(index='_time', columns='_field', values='_value')
            result.reset_index(inplace=True)
            result.rename(columns={'_time': 'timestamp'}, inplace=True)
            
            return result
            
        except Exception as e:
            raise StorageError(f"Failed to query weather data: {str(e)}")
    
    def close(self):
        self.client.close()

class ObjectStorage:
    def __init__(self, endpoint_url: str, access_key: str, secret_key: str, bucket_name: str):
        self.s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        self.bucket_name = bucket_name
    
    def store_raw_data(self, data: pd.DataFrame, source: str, timestamp: datetime) -> str:
        try:
            key = f"raw_data/{source}/{timestamp.strftime('%Y/%m/%d')}/{timestamp.isoformat()}.csv"
            
            csv_buffer = StringIO()
            data.to_csv(csv_buffer, index=False)
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=csv_buffer.getvalue()
            )
            
            logger.info(f"Stored raw data: {key}")
            return key
            
        except Exception as e:
            raise StorageError(f"Failed to store raw data: {str(e)}")
    
    def store_model_artifacts(self, model_data: bytes, model_name: str, version: str) -> str:
        try:
            key = f"models/{model_name}/{version}/model.pkl"
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=model_data
            )
            
            logger.info(f"Stored model artifact: {key}")
            return key
            
        except Exception as e:
            raise StorageError(f"Failed to store model artifact: {str(e)}")
    
    def retrieve_model_artifacts(self, model_name: str, version: str) -> bytes:
        try:
            key = f"models/{model_name}/{version}/model.pkl"
            
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            model_data = response['Body'].read()
            
            logger.info(f"Retrieved model artifact: {key}")
            return model_data
            
        except Exception as e:
            raise StorageError(f"Failed to retrieve model artifact: {str(e)}")

class DataStorageManager:
    def __init__(self, ts_config: Dict[str, str], object_config: Dict[str, str]):
        self.ts_storage = TimeSeriesStorage(
            url=ts_config['url'],
            token=ts_config['token'],
            org=ts_config['org'],
            bucket=ts_config['bucket']
        )
        
        self.object_storage = ObjectStorage(
            endpoint_url=object_config['endpoint_url'],
            access_key=object_config['access_key'],
            secret_key=object_config['secret_key'],
            bucket_name=object_config['bucket_name']
        )
    
    def store_ingested_data(self, data_dict: Dict[str, pd.DataFrame], timestamp: datetime) -> Dict[str, str]:
        storage_keys = {}
        
        for source, df in data_dict.items():
            if not df.empty:
                try:
                    if source in ['inverter', 'weather']:
                        if source == 'inverter':
                            self.ts_storage.store_inverter_data(df)
                        else:
                            self.ts_storage.store_weather_data(df)
                    
                    raw_key = self.object_storage.store_raw_data(df, source, timestamp)
                    storage_keys[source] = raw_key
                    
                except Exception as e:
                    logger.error(f"Failed to store {source} data: {str(e)}")
                    continue
        
        return storage_keys
    
    def get_combined_data(self, start_time: datetime, end_time: datetime, inverter_ids: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        try:
            power_data = self.ts_storage.query_power_data(start_time, end_time, inverter_ids)
            weather_data = self.ts_storage.query_weather_data(start_time, end_time)
            
            return {
                'power': power_data,
                'weather': weather_data
            }
            
        except Exception as e:
            raise StorageError(f"Failed to retrieve combined data: {str(e)}")
    
    def close(self):
        self.ts_storage.close() 