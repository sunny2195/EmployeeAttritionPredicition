import sys
import os
from pathlib import Path
from EAP.exception.exception import CustomException
from EAP.entity.config_entity import (
    DataIngestionConfig, 
    DataValidationConfig, 
    DataTransformationConfig, 
    ModelTrainerConfig
)
from EAP.components.data_ingestion import DataIngestion
from EAP.components.data_validation import DataValidation
from EAP.components.data_transformation import DataTransformation
from EAP.components.model_trainer import ModelTrainer


class TrainingPipeline:
    """
    Orchestrates the entire ML Training workflow by calling each component sequentially.
    """
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.validation_config = DataValidationConfig()
        self.transformation_config = DataTransformationConfig()
        self.trainer_config = ModelTrainerConfig()

    def run_pipeline(self):
        try:
            print("🚀 Starting MLOps Training Pipeline...")

            ingestion = DataIngestion(config=self.ingestion_config)
            ingested_artifact_path = ingestion.initiate_data_ingestion()
            print("✅ Data Ingestion complete.")

            validation = DataValidation(
                validation_config=self.validation_config, 
                ingestion_config=self.ingestion_config 
            )
            validated_artifact_path = validation.initiate_data_validation()
            print("✅ Data Validation complete.")
            
            transformation = DataTransformation(
                transform_config=self.transformation_config,
                ingestion_config=self.ingestion_config 
            )
            train_path, test_path = transformation.initiate_data_transformation()
            print("✅ Data Transformation & Train/Test Split complete.")

            trainer = ModelTrainer(config=self.trainer_config)
            best_model_name, best_score = trainer.initiate_model_trainer()
            print(f"🎉 Model Training complete. Best Model: {best_model_name} (F1: {best_score:.4f})")
            
            print("\nPipeline execution finished successfully!")

        except CustomException as e:
            print(f"❌ Pipeline failed: {e}")
            raise e
        except Exception as e:
            raise CustomException(e, sys)