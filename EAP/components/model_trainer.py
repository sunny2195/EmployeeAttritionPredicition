import os
import sys
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

# Scikit-learn models and metrics
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# Import the foundational classes and utilities
from EAP.entity.config_entity import (
    ModelTrainerConfig
)
from EAP.exception.exception import CustomException
from EAP.utils.utils import save_object # To save the final model

# ====================================================================
# HELPER FUNCTION: Model Evaluation
# ====================================================================

def evaluate_models(X_train, y_train, X_test, y_test, models: Dict[str, Any], target_names: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Trains and evaluates a dictionary of models, returning performance metrics.
    Focuses on Precision, Recall, and F1-Score for the minority class (Attrition: Yes).
    """
    report = {}

    for name, model in models.items():
        try:
            # Train model
            model.fit(X_train, y_train)

            # Predict on test data
            y_test_pred = model.predict(X_test)

            # Calculate key metrics (focused on the minority class: Attrition=1)
            # Use average='binary' and pos_label=1 to focus on the 'Yes' class (Attrition)
            
            f1 = f1_score(y_test, y_test_pred, pos_label=1, average='binary')
            precision = precision_score(y_test, y_test_pred, pos_label=1, average='binary')
            recall = recall_score(y_test, y_test_pred, pos_label=1, average='binary')
            
            report[name] = {
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'full_report': classification_report(y_test, y_test_pred, target_names=target_names)
            }

        except Exception as e:
            raise CustomException(f"Error evaluating model {name}: {e}", sys)
    
    return report

# ====================================================================
# MAIN COMPONENT: ModelTrainer
# ====================================================================

class ModelTrainer:
    """
    Handles training, evaluation, comparison, and saving of the best model.
    """
    def __init__(self, config: ModelTrainerConfig):
        """Initializes ModelTrainer with configurations."""
        self.trainer_config = config

    def initiate_model_trainer(self):
        """
        Orchestrates model training, selection, and artifact saving.
        """
        try:
            print("Starting Model Trainer component...")

            # --- 1. Load Train/Test Data Artifacts ---
            
            # Use ModelTrainerConfig to determine file locations 
            root_dir = Path("artifacts/data_transformation")
            X_train = pd.read_csv(Path(os.path.join(root_dir, "train_data.csv")))
            X_test = pd.read_csv(Path(os.path.join(root_dir, "test_data.csv")))
            y_train = pd.read_csv(Path(os.path.join(root_dir, "train_target.csv")))['Attrition']
            y_test = pd.read_csv(Path(os.path.join(root_dir, "test_target.csv")))['Attrition']
            
            print("Train/Test data artifacts loaded successfully.")
            
            # --- 2. Initialize Models for Comparison ---
            
            # Use the defined models, applying class_weight='balanced' to handle imbalance
            models = {
                "LogisticRegression": LogisticRegression(
                    random_state=self.trainer_config.random_state, 
                    solver=self.trainer_config.solver,
                    class_weight='balanced' # CRITICAL for imbalanced data
                ),
                "DecisionTreeClassifier": DecisionTreeClassifier(
                    random_state=self.trainer_config.random_state, 
                    **self.trainer_config.models_to_train['DecisionTreeClassifier']
                ),
                "RandomForestClassifier": RandomForestClassifier(
                    random_state=self.trainer_config.random_state, 
                    class_weight='balanced', # CRITICAL for imbalanced data
                    **self.trainer_config.models_to_train['RandomForestClassifier']
                )
            }
            
            # --- 3. Evaluate Models ---
            target_names = ['No Attrition (0)', 'Attrition (1)']
            model_report = evaluate_models(
                X_train, y_train, X_test, y_test, models, target_names
            )

            # --- 4. Select Best Model ---
            
            # Find the model with the highest F1-Score (since it balances Precision and Recall)
            best_model_score = max(report['f1_score'] for report in model_report.values())
            best_model_name = [
                name for name, report in model_report.items() 
                if report['f1_score'] == best_model_score
            ][0]
            best_model = models[best_model_name]

            print(f"\n--- Model Comparison Results ---")
            for name, report in model_report.items():
                 print(f"Model: {name} | F1-Score (Attrition): {report['f1_score']:.4f}")
            
            print(f"\nBest Model Found: {best_model_name}")
            print(f"Best F1-Score (Attrition): {best_model_score:.4f}")

            # --- 5. Save the Best Model Artifact ---
            
            # Create Model Trainer artifact directory
            os.makedirs(self.trainer_config.root_dir, exist_ok=True)
            
            # Save the final best model (e.g., as a pickle file)
            save_object(
                file_path=Path(os.path.join(self.trainer_config.root_dir, self.trainer_config.trained_model_file)),
                obj=best_model
            )
            
            print(f"Best model artifact saved successfully to: {self.trainer_config.root_dir}")
            
            return best_model_name, best_model_score

        except Exception as e:
            raise CustomException(e, sys)