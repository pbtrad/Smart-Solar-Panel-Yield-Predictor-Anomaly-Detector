#!/usr/bin/env python3

import argparse
import logging
from datetime import datetime

from src.data.ingestion import DataIngestionOrchestrator
from src.data.storage import DataStorageManager
from src.utils.config import ConfigManager

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_date_range(start_date: str, end_date: str) -> tuple:
    try:
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        return start, end
    except ValueError as e:
        raise ValueError(
            f"Invalid date format. Use ISO format (YYYY-MM-DDTHH:MM:SS): {e}"
        )


def main():
    parser = argparse.ArgumentParser(description="Solar Data Ingestion Pipeline")
    parser.add_argument(
        "--source",
        choices=["all", "inverter", "weather", "maintenance"],
        default="all",
        help="Data source to ingest",
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date (ISO format: YYYY-MM-DDTHH:MM:SS)",
    )
    parser.add_argument(
        "--end-date", required=True, help="End date (ISO format: YYYY-MM-DDTHH:MM:SS)"
    )
    parser.add_argument(
        "--config", default="configs/app.yaml", help="Configuration file path"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run without storing data"
    )

    args = parser.parse_args()

    try:
        config_manager = ConfigManager(args.config)

        if not config_manager.validate_config():
            logger.error("Configuration validation failed")
            return 1

        start_date, end_date = parse_date_range(args.start_date, args.end_date)

        logger.info(f"Starting data ingestion from {start_date} to {end_date}")
        logger.info(f"Sources: {args.source}")

        ingestion_config = config_manager.get_ingestion_config()
        storage_config = config_manager.get_storage_config()

        orchestrator = DataIngestionOrchestrator(ingestion_config)
        storage_manager = DataStorageManager(
            ts_config=storage_config["timeseries"],
            object_config=storage_config["object"],
        )

        if args.source == "all":
            data_dict = orchestrator.ingest_all_data(start_date, end_date)
        else:
            if args.source not in orchestrator.services:
                logger.error(f"Source '{args.source}' not configured")
                return 1

            service = orchestrator.services[args.source]
            data = service.fetch_data(start_date, end_date)
            data_dict = {args.source: data}

        if not data_dict:
            logger.warning("No data ingested from any source")
            return 0

        summary = orchestrator.get_data_summary(data_dict)
        logger.info("Data ingestion summary:")
        for source, info in summary.items():
            logger.info(f"  {source}: {info['record_count']} records")

        if not args.dry_run:
            storage_keys = storage_manager.store_ingested_data(
                data_dict, datetime.now()
            )
            logger.info(f"Data stored with keys: {storage_keys}")
        else:
            logger.info("Dry run completed - no data stored")

        storage_manager.close()
        logger.info("Data ingestion completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Data ingestion failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
