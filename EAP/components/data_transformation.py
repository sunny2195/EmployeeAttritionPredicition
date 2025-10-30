import os
import sys
import pandas as pd
from pathlib import Path

# Scikit-learn preprocessing utilities
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split # Used to verify the final output structure

# Import the foundational classes we built
from EAP.entity.config_entity import (
    DataIngestionConfig, 
    DataTransformationConfig
)
from EAP.exception.exception import CustomException
from EAP.utils.utils import save_object 

class DataTransformation:
    """
    Handles all data transformation steps: dropping columns, encoding, and scaling.
    """
    def __init__(self, transform_config: DataTransformationConfig, 
                 ingestion_config: DataIngestionConfig):
        """Initializes DataTransformation with configurations."""
        self.transform_config = transform_config
        self.ingestion_config = ingestion_config

    def initiate_data_transformation(self) -> tuple[Path, Path]:
        """
        Orchestrates the data transformation process and returns paths to 
        the train and test artifacts.
        """
        try:
            print("Starting Data Transformation component...")
            
            # --- 1. Load Data ---
            # Get the path to the raw data artifact from the Ingestion component
            ingested_file_path = Path(os.path.join(
                self.ingestion_config.root_dir, 
                self.ingestion_config.ingested_data_file
            ))
            
            # Read the data (assuming validation has passed)
            df = pd.read_csv(ingested_file_path)
            print(f"Data loaded for transformation: {df.shape}")

            # --- 2. Drop Irrelevant Columns ---
            df.drop(columns=self.transform_config.columns_to_drop, inplace=True)
            print(f"Dropped non-informative columns: {self.transform_config.columns_to_drop}")

            # --- 3. Label Encoding (Binary and Target) ---
            le = LabelEncoder()
            for col in self.transform_config.label_encode_cols:
                if col in df.columns:
                    df[col] = le.fit_transform(df[col])
            print("Label Encoding applied to binary and target columns.")

            # --- 4. One-Hot Encoding (Nominal Categorical) ---
            df = pd.get_dummies(
                df, 
                columns=self.transform_config.one_hot_encode_cols, 
                drop_first=True, 
                dtype=int
            )
            print("One-Hot Encoding applied to nominal categorical features.")

            # --- 5. Standard Scaling (Numerical/Ordinal) ---
            scaler = StandardScaler()
            
            # Apply the scaler to the selected columns
            df[self.transform_config.standard_scale_cols] = scaler.fit_transform(
                df[self.transform_config.standard_scale_cols]
            )
            print("Standard Scaling applied to numerical and ordinal features.")

            
            scaler_path = Path(os.path.join(self.transform_config.root_dir, "scaler.pkl"))

            # Save the scaler object for later use in prediction
            save_object(file_path=scaler_path, obj=scaler)
            print(f"Scaler object saved to: {scaler_path}")
            
            # --- 6. Data Splitting (Train/Test) ---
            # Separate Features (X) and Target (y)
            X = df.drop(columns=[self.transform_config.label_encode_cols[0]]) # 'Attrition'
            y = df[self.transform_config.label_encode_cols[0]]

            # Use ModelTrainerConfig for split ratios, as this is the next step's configuration
            from EAP.entity.config_entity import ModelTrainerConfig
            trainer_config = ModelTrainerConfig()
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=trainer_config.test_size, 
                random_state=trainer_config.random_state,
                stratify=y # Critical for imbalanced data
            )
            print(f"Data split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples).")

            # --- 7. Save Transformed Artifacts ---
            
            # Create the Transformation artifact directory
            os.makedirs(self.transform_config.root_dir, exist_ok=True)
            
            # Define output paths for train/test data
            train_path = Path(os.path.join(self.transform_config.root_dir, "train_data.csv"))
            test_path = Path(os.path.join(self.transform_config.root_dir, "test_data.csv"))
            
            # Save the split data
            X_train.to_csv(train_path, index=False)
            X_test.to_csv(test_path, index=False)
            y_train.to_csv(Path(os.path.join(self.transform_config.root_dir, "train_target.csv")), index=False)
            y_test.to_csv(Path(os.path.join(self.transform_config.root_dir, "test_target.csv")), index=False)
            
            print(f"Transformed train/test artifacts saved to: {self.transform_config.root_dir}")
            
            return train_path, test_path

        except Exception as e:
            raise CustomException(e, sys)