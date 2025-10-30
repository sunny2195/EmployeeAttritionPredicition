from dataclasses import dataclass, field
from pathlib import Path
import os

# --- Configuration for Data Ingestion ---
@dataclass(frozen=True) # frozen=True makes the object immutable after creation
class DataIngestionConfig:
    """Configuration for reading and locating the raw data."""
    root_dir: Path = Path('artifacts/data_ingestion')
    # Use the Windows path structure for demonstration, but Path converts it safely.
    raw_data_path: Path = Path(r'D:\Projects\Employee Attiriton Prediction\Data\EmployeeData.csv')
    
    # Store the filename that we'll create in the artifacts folder
    ingested_data_file: str = "EmployeeData.csv"

@dataclass(frozen=True)
class DataValidationConfig:
    """Configuration for data validation checks."""
    root_dir: Path = Path('artifacts/data_validation')
    
    # Path to the data file from the ingestion step
    data_to_validate_path: Path = Path('artifacts/data_ingestion/EmployeeData.csv')
    
    # File to store the validation status (e.g., "PASSED" or "FAILED")
    status_file: Path = Path('artifacts/data_validation/validation_status.txt')
    
    # Define the expected schema (all 35 original columns)
    schema_file_path: Path = Path("schema.yaml")
    
    # Expected number of columns
    expected_column_count: int = 35

# --- Configuration for Data Transformation ---
@dataclass(frozen=True)
class DataTransformationConfig:
    """Configuration for cleaning, encoding, and scaling the data."""
    root_dir: Path = Path('artifacts/data_transformation')
    
    # Input file from Data Ingestion (path will be constructed)
    ingested_data_file: str = DataIngestionConfig().ingested_data_file

    # Output file name for the final ML-ready, scaled data
    transformed_data_file: str = "EmployeeData_Scaled.csv"
    
    # Columns to be dropped
    columns_to_drop: list = field(default_factory=lambda: [
        'EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'
    ])
    
    # Columns for Label Encoding (Binary and Target)
    label_encode_cols: list = field(default_factory=lambda: [
        'Attrition', 'Gender', 'OverTime'
    ])
    
    # Columns for One-Hot Encoding
    one_hot_encode_cols: list = field(default_factory=lambda: [
        'BusinessTravel', 'Department', 'EducationField', 'JobRole', 
        'MaritalStatus'
    ])

    # Columns for Standard Scaling (numerical/ordinal)
    standard_scale_cols: list = field(default_factory=lambda: [
        'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
        'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 
        'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 
        'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 
        'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 
        'YearsWithCurrManager'
    ])

# --- Configuration for Model Trainer ---
@dataclass(frozen=True)
class ModelTrainerConfig:
    """Configuration for model training and selection."""
    root_dir: Path = Path('artifacts/model_trainer')
    
    # Input file from Data Transformation
    transformed_data_file: str = DataTransformationConfig().transformed_data_file
    
    # Output file name for the trained model (using a common format like .pkl)
    trained_model_file: str = "best_model.pkl"

    # Training Parameters
    target_column: str = "Attrition"
    test_size: float = 0.3
    random_state: int = 42
    
    solver: str = 'liblinear'

    models_to_train: dict = field(default_factory=lambda: {
        "LogisticRegression": {},
        "DecisionTreeClassifier": {'max_depth': 5},
        "RandomForestClassifier": {'n_estimators': 100, 'max_depth': 8}
    })