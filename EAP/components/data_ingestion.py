import os
import sys
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

# Import the foundational classes we built
from EAP.entity.config_entity import DataIngestionConfig
from EAP.exception.exception import CustomException

# Note: In a full pipeline, DataIngestion might also perform the initial 
# train/test split. For simplicity here, we focus on loading and saving the raw data.

class DataIngestion:
    """
    Handles the process of reading raw data and saving it to the artifacts folder.
    """
    def __init__(self, config: DataIngestionConfig):
        # Initialize the component with the configuration settings
        self.ingestion_config = config

    def initiate_data_ingestion(self) -> Path:
        """
        Loads the data from the source path and saves it to the artifact directory.
        Returns the path to the ingested data file.
        """
        try:
            print("Starting Data Ingestion component...")
            
            # --- 1. Create Artifact Directory ---
            # Ensure the root directory for this component exists
            os.makedirs(self.ingestion_config.root_dir, exist_ok=True)
            print(f"Created artifact directory: {self.ingestion_config.root_dir}")

            # --- 2. Load Data from Source ---
            # Use the raw_data_path defined in config_entity.py
            raw_data_path_str = str(self.ingestion_config.raw_data_path)
            df = pd.read_csv(raw_data_path_str)
            print(f"Data loaded successfully from source: {raw_data_path_str}")

            # --- 3. Save Ingested Data Artifact ---
            # Construct the path for the saved artifact
            ingested_file_path = Path(os.path.join(
                self.ingestion_config.root_dir, 
                self.ingestion_config.ingested_data_file
            ))
            
            # Save the raw data copy into the artifacts folder
            df.to_csv(ingested_file_path, index=False, header=True)
            
            print(f"Raw data artifact saved to: {ingested_file_path}")
            
            # Return the path to the saved artifact for the next pipeline step
            return ingested_file_path

        except Exception as e:
            # Wrap any exception in our custom exception handler
            raise CustomException(e, sys)