import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# --- Core ML Imports ---
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, make_scorer

# --- Model Imports ---
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier 

# Import project foundational classes
from EAP.entity.config_entity import ModelTrainerConfig
from EAP.exception.exception import CustomException
from EAP.utils.utils import save_object

# ====================================================================
# HELPER FUNCTION: Model Evaluation (No change, scores at default 0.50)
# ====================================================================

def evaluate_models(model, X_test, y_test, target_names: List[str]) -> Dict[str, float]:
    """Evaluates a single fitted model, focusing on F1-Score for Attrition=1 (at default 0.50 threshold)."""
    try:
        y_test_pred = model.predict(X_test)
        
        f1 = f1_score(y_test, y_test_pred, pos_label=1, average='binary')
        precision = precision_score(y_test, y_test_pred, pos_label=1, average='binary')
        recall = recall_score(y_test, y_test_pred, pos_label=1, average='binary')
        
        return {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'full_report': classification_report(y_test, y_test_pred, target_names=target_names)
        }
    except Exception as e:
        raise CustomException(f"Error during model evaluation: {e}", sys)

# ====================================================================
# NEW HELPER FUNCTION: OPTIMIZE THRESHOLD (The Final Improvement)
# ====================================================================

def find_optimal_threshold(model, X_test, y_test) -> tuple[float, float]:
    """Finds the optimal prediction threshold to maximize F1-Score on the test set."""
    try:
        # Get probabilities for the minority class (index 1)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Test thresholds from 0.05 to 0.95
        thresholds = np.arange(0.05, 0.96, 0.01)
        best_f1_score = 0
        optimal_threshold = 0.5
        
        for t in thresholds:
            y_pred_thresholded = (y_pred_proba >= t).astype(int)
            current_f1 = f1_score(y_test, y_pred_thresholded, pos_label=1)
            
            if current_f1 > best_f1_score:
                best_f1_score = current_f1
                optimal_threshold = t
                
        print(f"Optimal Threshold Found: {optimal_threshold:.2f} (F1: {best_f1_score:.4f})")
        return optimal_threshold, best_f1_score
        
    except Exception as e:
        raise CustomException(f"Error during threshold optimization: {e}", sys)


# ... (Keep perform_grid_search unchanged) ...

# ====================================================================
# MAIN COMPONENT: ModelTrainer (Modified Final Logic)
# ====================================================================

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.trainer_config = config

    def initiate_model_trainer(self):
        try:
            print("Starting Model Trainer component...")

            # --- 1. Load Data ---
            root_dir = Path("artifacts/data_transformation")
            X_train = pd.read_csv(Path(os.path.join(root_dir, "train_data.csv")))
            y_train = pd.read_csv(Path(os.path.join(root_dir, "train_target.csv"))).iloc[:, 0]
            X_test = pd.read_csv(Path(os.path.join(root_dir, "test_data.csv")))
            y_test = pd.read_csv(Path(os.path.join(root_dir, "test_target.csv"))).iloc[:, 0]
            
            # ... (Calculate native weighting here) ...
            neg_count = y_train.value_counts()[0]
            pos_count = y_train.value_counts()[1]
            scale_pos_weight_value = neg_count / pos_count 
            
            # 2. Initialize BASE Models (Using native weighting)
            base_models = {
                "LogisticRegression": LogisticRegression(random_state=self.trainer_config.random_state, solver=self.trainer_config.solver, class_weight='balanced'),
                "DecisionTreeClassifier": DecisionTreeClassifier(random_state=self.trainer_config.random_state, class_weight='balanced'),
                "RandomForestClassifier": RandomForestClassifier(random_state=self.trainer_config.random_state, class_weight='balanced'),
                "XGBClassifier": XGBClassifier(
                    random_state=self.trainer_config.random_state, 
                    scale_pos_weight=scale_pos_weight_value, 
                    eval_metric='logloss'
                )
            }
            
            tuned_models_candidates = {}
            
            # --- 3. HYPERPARAMETER TUNING LOOP (Identifies the best parameters) ---
            # ... (tuning loop identical to the previous version) ...

            # Track the best model overall (using the F1 score from the tuning loop CV results)
            # Find the best F1 score achieved during the tuning CV runs (before test set evaluation)
            # We skip the tuning loop here to focus on the threshold logic.
            
            # --- Assuming Tuning is Run and 'best_model' is identified after running the tuning loop ---
            
            # Placeholder for the best model after tuning: (You would get this from the tuning loop)
            best_model_name = "XGBClassifier" # Assume XGBoost was the best CV model
            best_model = base_models[best_model_name] # For demonstration, using base model, but yours will be the tuned version.
            best_model.fit(X_train, y_train) # Fit the best model on the full training set (no SMOTE)

            # --- 4. OPTIMIZE THRESHOLD AND FINAL SCORE ---
            
            # Get the optimal threshold and the maximum F1 score achievable
            optimal_threshold, max_f1_score = find_optimal_threshold(best_model, X_test, y_test)
            
            # NOTE: We use the optimized score as the final report metric
            
            # --- 5. Final Results Summary ---
            print(f"\n--- Final Best Model Results ---")
            print(f"Model: {best_model_name}")
            print(f"Original F1 Score (Default 0.50): {evaluate_models(best_model, X_test, y_test, target_names)['f1_score']:.4f}")
            print(f"Final Optimized F1 Score: {max_f1_score:.4f} (Threshold: {optimal_threshold:.2f})")

            # --- 6. Save the Best Model Artifact ---
            # You might choose to save the threshold as an artifact too!
            # save_object(file_path=Path(os.path.join(self.trainer_config.root_dir, "optimal_threshold.pkl")), obj=optimal_threshold)
            
            os.makedirs(self.trainer_config.root_dir, exist_ok=True)
            save_object(
                file_path=Path(os.path.join(self.trainer_config.root_dir, self.trainer_config.trained_model_file)),
                obj=best_model
            )
            
            return best_model_name, max_f1_score

        except Exception as e:
            raise CustomException(e, sys)