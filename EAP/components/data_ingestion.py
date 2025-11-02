import os
import sys
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

t
from EAP.entity.config_entity import DataIngestionConfig
from EAP.exception.exception import CustomException


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
       
        self.ingestion_config = config

    def initiate_data_ingestion(self) -> Path:
        try:
            print("Starting Data Ingestion component...")
            
            os.makedirs(self.ingestion_config.root_dir, exist_ok=True)
            print(f"Created artifact directory: {self.ingestion_config.root_dir}")
            raw_data_path_str = str(self.ingestion_config.raw_data_path)
            df = pd.read_csv(raw_data_path_str)
            print(f"Data loaded successfully from source: {raw_data_path_str}")
            ingested_file_path = Path(os.path.join(
                self.ingestion_config.root_dir, 
                self.ingestion_config.ingested_data_file
            ))
            
            
            df.to_csv(ingested_file_path, index=False, header=True)
            
            print(f"Raw data artifact saved to: {ingested_file_path}")
            
            
            return ingested_file_path

        except Exception as e:
            raise CustomException(e, sys)