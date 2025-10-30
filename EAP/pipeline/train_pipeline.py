import sys
import os
from pathlib import Path

# Import the foundational classes
from EAP.exception.exception import CustomException
from EAP.entity.config_entity import (
    DataIngestionConfig, 
    DataValidationConfig, 
    DataTransformationConfig, 
    ModelTrainerConfig
)

# Import the pipeline components
from EAP.components.data_ingestion import DataIngestion
from EAP.components.data_validation import DataValidation
from EAP.components.data_transformation import DataTransformation
from EAP.components.model_trainer import ModelTrainer


class TrainingPipeline:
    """
    Orchestrates the entire ML Training workflow by calling each component sequentially.
    """
    def __init__(self):
        # Initialize all configuration objects
        self.ingestion_config = DataIngestionConfig()
        self.validation_config = DataValidationConfig()
        self.transformation_config = DataTransformationConfig()
        self.trainer_config = ModelTrainerConfig()

    def run_pipeline(self):
        """
        Executes the full pipeline from data ingestion to model training.
        """
        try:
            print("🚀 Starting MLOps Training Pipeline...")

            # ----------------- 1. Data Ingestion -----------------
            ingestion = DataIngestion(config=self.ingestion_config)
            ingested_artifact_path = ingestion.initiate_data_ingestion()
            print("✅ Data Ingestion complete.")

            # ----------------- 2. Data Validation -----------------
            # Note: The validation component uses the same artifact path
            validation = DataValidation(
                validation_config=self.validation_config, 
                ingestion_config=self.ingestion_config # Passes config to locate file
            )
            # The initiate_data_validation function performs the check and updates the status file
            validated_artifact_path = validation.initiate_data_validation()

            # Optional: Add a check to halt if validation fails
            # with open(self.validation_config.status_file, 'r') as f:
            #     if 'False' in f.read():
            #         raise CustomException("Data Validation Failed. Halting pipeline.", sys)
                    
            print("✅ Data Validation complete.")


            # ----------------- 3. Data Transformation -----------------
            transformation = DataTransformation(
                transform_config=self.transformation_config,
                ingestion_config=self.ingestion_config # Needs ingestion config to find raw file
            )
            train_path, test_path = transformation.initiate_data_transformation()
            print("✅ Data Transformation & Train/Test Split complete.")

            # ----------------- 4. Model Trainer -----------------
            trainer = ModelTrainer(config=self.trainer_config)
            best_model_name, best_score = trainer.initiate_model_trainer()
            print(f"🎉 Model Training complete. Best Model: {best_model_name} (F1: {best_score:.4f})")
            
            print("\nPipeline execution finished successfully!")

        except CustomException as e:
            print(f"❌ Pipeline failed: {e}")
            raise e
        except Exception as e:
            # Catch any unexpected errors and wrap them in the CustomException
            raise CustomException(e, sys)