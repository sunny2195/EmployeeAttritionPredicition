from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass(frozen=True) 
class DataIngestionConfig:
    root_dir: Path = Path('artifacts/data_ingestion')
    raw_data_path: Path = Path(r'D:\Projects\Employee Attiriton Prediction\Data\EmployeeData.csv')
    ingested_data_file: str = "EmployeeData.csv"

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path = Path('artifacts/data_validation')
    data_to_validate_path: Path = Path('artifacts/data_ingestion/EmployeeData.csv')
    status_file: Path = Path('artifacts/data_validation/validation_status.txt')
    schema_file_path: Path = Path("schema.yaml")
    expected_column_count: int = 35

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path = Path('artifacts/data_transformation')
    ingested_data_file: str = DataIngestionConfig().ingested_data_file
    transformed_data_file: str = "EmployeeData_Scaled.csv"
    columns_to_drop: list = field(default_factory=lambda: [
        'EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'
    ])
    label_encode_cols: list = field(default_factory=lambda: [
        'Attrition', 'Gender', 'OverTime'
    ])
    one_hot_encode_cols: list = field(default_factory=lambda: [
        'BusinessTravel', 'Department', 'EducationField', 'JobRole', 
        'MaritalStatus'
    ])
    standard_scale_cols: list = field(default_factory=lambda: [
        'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
        'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 
        'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 
        'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 
        'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 
        'YearsWithCurrManager'
    ])
@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path = Path('artifacts/model_trainer')
    transformed_data_file: str = DataTransformationConfig().transformed_data_file
    trained_model_file: str = "best_model.pkl"
    target_column: str = "Attrition"
    test_size: float = 0.3
    random_state: int = 42
    solver: str = 'liblinear'
    models_to_train: dict = field(default_factory=lambda: {
        "LogisticRegression": {},
        "DecisionTreeClassifier": {},
        "RandomForestClassifier": {},
        "XGBClassifier": {} 
    })
    tuning_grids: dict = field(default_factory=lambda: {
        "LogisticRegression": {
            'C': [0.01, 0.1, 1.0, 10.0],  
            'penalty': ['l2'],
            'solver': ['liblinear'] 
        },
        "DecisionTreeClassifier": {
            'max_depth': [5, 8, 12],
            'min_samples_split': [10, 20],
            'criterion': ['gini', 'entropy'],
            'class_weight': ['balanced', None]
        },
        "RandomForestClassifier": {
            'n_estimators': [50, 100, 200],
            'max_depth': [8, 12, None],
            'min_samples_leaf': [2, 4, 8],
            'class_weight': ['balanced', None]
        },
        "XGBClassifier": {
            'learning_rate': [0.05, 0.1, 0.2],
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'scale_pos_weight': [1, 5] 
        }
    })