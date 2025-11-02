import os
import sys
import pandas as pd
from pathlib import Path
from typing import List


from EAP.entity.config_entity import (
    DataIngestionConfig, 
    DataValidationConfig
)
from EAP.exception.exception import CustomException
from EAP.utils.utils import read_yaml 


class DataValidation:
    """
    Handles data integrity checks: schema validation, missing value check, 
    and feature drift check.
    """
    def __init__(self, validation_config: DataValidationConfig, 
                 ingestion_config: DataIngestionConfig):
        """Initializes DataValidation with configurations."""
        self.validation_config = validation_config
        self.ingestion_config = ingestion_config
        self.schema = read_yaml(self.validation_config.schema_file_path)

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Checks if the DataFrame columns and data types match the expected schema.
        """
        try:
            expected_columns = self.schema['columns']
            df_columns = df.columns
            validation_status = True
            
       
            missing_cols = [col for col in expected_columns if col not in df_columns]
            if missing_cols:
                print(f"Validation FAILED: Missing columns in data: {missing_cols}")
                validation_status = False

           
            extra_cols = [col for col in df_columns if col not in expected_columns]
            if extra_cols:
                print(f"Validation FAILED: Extra columns in data: {extra_cols}")
                validation_status = False

           
            for col, expected_dtype in expected_columns.items():
                if col in df_columns and str(df[col].dtype) != expected_dtype:
                    print(f"Validation FAILED: Column '{col}' has dtype '{df[col].dtype}', expected '{expected_dtype}'")
                    validation_status = False

            return validation_status

        except Exception as e:
            raise CustomException(e, sys)

    def is_valid_data(self, df: pd.DataFrame) -> bool:
        """
        Performs the complete validation check and updates the status file.
        """
        try:
            schema_status = self.validate_schema(df)
            missing_value_status = (df.isnull().sum().sum() == 0)
            if not missing_value_status:
                print("Validation FAILED: Missing values detected.")

            final_status = schema_status and missing_value_status
            
            
            os.makedirs(self.validation_config.root_dir, exist_ok=True)
            with open(self.validation_config.status_file, 'w') as f:
                f.write(f"Validation status: {final_status}")
            
            if final_status:
                print("Data Validation PASSED! Proceeding to Data Transformation.")
            
            return final_status

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self) -> Path:
        """
        Orchestrates the data validation process.
        """
        try:
            ingested_file_path = Path(os.path.join(
                self.ingestion_config.root_dir, 
                self.ingestion_config.ingested_data_file
            ))
            
            df = pd.read_csv(ingested_file_path)
            self.is_valid_data(df)
            return ingested_file_path
            
        except Exception as e:
            raise CustomException(e, sys)