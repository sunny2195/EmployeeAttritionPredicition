import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier 
from EAP.entity.config_entity import ModelTrainerConfig
from EAP.exception.exception import CustomException
from EAP.utils.utils import save_object, load_object # Added load_object for final step
from lazypredict.Supervised import LazyClassifier



def perform_lazy_benchmark(X_train, X_test, y_train, y_test):
    try:
        print("\n--- Starting LazyPredict Model Benchmarking ---")
        clf = LazyClassifier(
            verbose=0,
            ignore_warnings=True, 
            custom_metric=None
        )
        
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)
        
        print("\nLazyPredict Benchmark Complete:")
        print(models)
        models.to_csv(Path("artifacts/model_trainer/lazypredict_benchmark.csv"), index=True)
        
        return models

    except Exception as e:
        print(f"LazyPredict encountered an error. Skipping benchmark: {e}")
        return None

def evaluate_models(model, X_test, y_test, target_names: List[str]) -> Dict[str, float]:
    try:
        y_test_pred = model.predict(X_test)
        
        f1 = f1_score(y_test, y_test_pred, pos_label=1, average='binary')
        precision = precision_score(y_test, y_test_pred, pos_label=1, average='binary')
        recall = recall_score(y_test, y_test_pred, pos_label=1, average='binary')
        
        return {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'full_report': classification_report(y_test, y_test_pred, target_names=target_names, output_dict=True)
        }
    except Exception as e:
        raise CustomException(f"Error during model evaluation: {e}", sys)


def find_optimal_threshold(model, X_test, y_test) -> tuple[float, float]:
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        thresholds = np.arange(0.05, 0.96, 0.01)
        best_f1_score = 0
        optimal_threshold = 0.5
        
        for t in thresholds:
            y_pred_thresholded = (y_pred_proba >= t).astype(int)
            current_f1 = f1_score(y_test, y_pred_thresholded, pos_label=1)
            
            if current_f1 > best_f1_score:
                best_f1_score = current_f1
                optimal_threshold = t
                
        return optimal_threshold, best_f1_score
        
    except Exception as e:
        raise CustomException(f"Error during threshold optimization: {e}", sys)

# ... (perform_grid_search unchanged) ...
def perform_grid_search(model_base, param_grid: dict, X_train, y_train, random_state: int, model_name: str):
    try:
        print(f"\n--- Starting Grid Search for {model_name} ---")
        f1_scorer = make_scorer(f1_score, pos_label=1) 
        
        # Grid Search setup
        grid_search = GridSearchCV(
            estimator=model_base,
            param_grid=param_grid,
            scoring=f1_scorer,
            cv=5,
            verbose=0,
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        print(f"  Best parameters found: {grid_search.best_params_}")
        print(f"  Best cross-validated F1 Score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_

    except Exception as e:
        raise CustomException(f"Error during Grid Search for {model_name}: {e}", sys)

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.trainer_config = config

    def initiate_model_trainer(self):
        try:
            print("Starting Model Trainer component...")

        
            root_dir = Path("artifacts/data_transformation")
            X_train = pd.read_csv(Path(os.path.join(root_dir, "train_data.csv")))
            X_test = pd.read_csv(Path(os.path.join(root_dir, "test_data.csv")))
            y_train = pd.read_csv(Path(os.path.join(root_dir, "train_target.csv"))).iloc[:, 0]
            y_test = pd.read_csv(Path(os.path.join(root_dir, "test_target.csv"))).iloc[:, 0]
        
            neg_count = y_train.value_counts()[0]
            pos_count = y_train.value_counts()[1]
            scale_pos_weight_value = neg_count / pos_count 

            print("\nStarting initial Model Benchmark (LazyPredict)...")
            benchmark_results = perform_lazy_benchmark(X_train, X_test, y_train, y_test)
            
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
            best_f1_default = -1 
            best_model_name_default = None
            
            
            print("\nStarting Hyperparameter Tuning and Final Evaluation...")
            for name, model_base in base_models.items():
                param_grid = self.trainer_config.tuning_grids.get(name, {})
                tuned_estimator = perform_grid_search(
                    model_base, 
                    param_grid, 
                    X_train, y_train, 
                    self.trainer_config.random_state,
                    name
                )

                tuned_estimator = model_base 
                tuned_estimator.fit(X_train, y_train) 
                
                metrics = evaluate_models(tuned_estimator, X_test, y_test, ['No Attrition (0)', 'Attrition (1)'])
                
                if metrics['f1_score'] > best_f1_default:
                    best_f1_default = metrics['f1_score']
                    best_model_name_default = name
                    best_model = tuned_estimator
            
            print(f"\n--- Optimizing Threshold for Best Model ({best_model_name_default}) ---")
            
            optimal_threshold, max_f1_score = find_optimal_threshold(
                best_model, X_test, y_test
            )
            
            print(f"\n--- FINAL TUNING REPORT ---")
            print(f"Best Model Identified: {best_model_name_default}")
            print(f"F1 Score (Default 0.50): {best_f1_default:.4f}")
            print(f"Max Achievable F1 Score: {max_f1_score:.4f} (at Threshold: {optimal_threshold:.2f})")
        
            threshold_path = Path(os.path.join(self.trainer_config.root_dir, "optimal_threshold.pkl"))
            save_object(file_path=threshold_path, obj=optimal_threshold)
           
            os.makedirs(self.trainer_config.root_dir, exist_ok=True)
            save_object(
                file_path=Path(os.path.join(self.trainer_config.root_dir, self.trainer_config.trained_model_file)),
                obj=best_model
            )
            
            return best_model_name_default, max_f1_score

        except Exception as e:
            raise CustomException(e, sys)