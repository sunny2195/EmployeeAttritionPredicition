import os
import sys
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from pathlib import Path

# Import custom utilities and exception handling
from EAP.utils.utils import load_object
from EAP.exception.exception import CustomException

FEATURE_COLUMNS = [
    
    'Age', 
    'DailyRate', 
    'DistanceFromHome', 
    'Education', 
    'EnvironmentSatisfaction', 
    'Gender', 
    'HourlyRate', 
    'JobInvolvement', 
    'JobLevel', 
    'JobSatisfaction', 
    'MonthlyIncome', 
    'MonthlyRate', 
    'NumCompaniesWorked', 
    'OverTime', 
    'PercentSalaryHike', 
    'PerformanceRating', 
    'RelationshipSatisfaction', 
    'StockOptionLevel', 
    'TotalWorkingYears', 
    'TrainingTimesLastYear', 
    'WorkLifeBalance', 
    'YearsAtCompany', 
    'YearsInCurrentRole', 
    'YearsSinceLastPromotion', 
    'YearsWithCurrManager', 
    
    # --- One-Hot Encoded Features (Start Here) ---
    'BusinessTravel_Travel_Frequently', 
    'BusinessTravel_Travel_Rarely', 
    'Department_Research & Development', 
    'Department_Sales', 
    'EducationField_Life Sciences', 
    'EducationField_Marketing', 
    'EducationField_Medical', 
    'EducationField_Other', 
    'EducationField_Technical Degree', 
    'JobRole_Human Resources', 
    'JobRole_Laboratory Technician', 
    'JobRole_Manager', 
    'JobRole_Manufacturing Director', 
    'JobRole_Research Director', 
    'JobRole_Research Scientist', 
    'JobRole_Sales Executive', 
    'JobRole_Sales Representative', 
    'MaritalStatus_Married', 
    'MaritalStatus_Single'
]

# --- Configuration & Artifact Paths ---
MODEL_PATH = Path('artifacts/model_trainer/best_model.pkl')
SCALER_PATH = Path('artifacts/data_transformation/scaler.pkl')



# --- Initialization ---
app = Flask(__name__)
model = None # The model will be loaded upon startup
scaler = None

def load_artifacts():
    """Load the trained model and scaler object."""
    global model, scaler
    try:
        model = load_object(file_path=MODEL_PATH)
        scaler = load_object(file_path=SCALER_PATH) 
        print("✅ Model and Scaler loaded successfully for serving.")
    except Exception as e:
        print(f"❌ Failed to load artifacts: {e}") 

load_artifacts() 

# --- Core Prediction Logic ---

def prepare_input_data(raw_data: dict, feature_cols: list, scaler_obj) -> pd.DataFrame:
    """
    Converts raw user input into the standardized, 44-feature DataFrame required by the model.
    NOTE: This is a simplified transformation. A full solution requires implementing ALL 
    encoding steps here.
    """
    # 1. Start with the raw inputs from the user (e.g., 4 features from index.html)
    raw_df = pd.DataFrame([raw_data])
    
    # 2. Add all missing features (dummy variables) and initialize them to 0
    # The model expects 44 features in a specific order.
    final_df = pd.DataFrame(0, index=[0], columns=feature_cols)

    # 3. Map user inputs to the correct columns in the final_df
    for key, value in raw_df.iloc[0].items():
        if key in final_df.columns:
            final_df[key] = float(value) # Ensure numerical inputs are floats

    # 4. Handle Categorical/Encoding (Simplified)
    # Since we are using a minimal HTML form, we'll manually set the one-hot/label encoded value.
    # Example for OverTime: '0' or '1' is the input from the form, which matches our Label Encoding.
    
    # 5. Apply Scaling to the Numerical Columns (the critical step)
    # NOTE: You need the exact list of columns that were scaled during training.
    # For this example, we use the original numerical columns list:
    scaled_cols_for_pred = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
        'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 
        'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 
        'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 
        'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 
        'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
    
    # Check if the scaler was loaded before attempting to transform
    if scaler_obj is not None:
        # Scale only the columns that were scaled during training
        final_df[scaled_cols_for_pred] = scaler_obj.transform(final_df[scaled_cols_for_pred])
    
    return final_df


# --- Prediction Endpoint ---

@app.route('/', methods=['GET', 'POST'])
def predict_employee_attrition():
    if request.method == 'GET':
        # FIX: Ensure this always renders the base template for the initial page load.
        return render_template('index.html') 

    elif request.method == 'POST':
        # Initial check 1: Artifact readiness
        if model is None or scaler is None:
            return render_template('index.html', 
                                   prediction_result="Error: Artifacts not loaded. Rerun pipeline.", 
                                   css_class="high-risk"), 503

        try:
            # --- 1. GET RAW INPUT DATA ---
            # This line defines raw_data
            raw_data = request.form.to_dict()
            
            # --- 2. PREPARE AND SCALE INPUT ---
            # THIS LINE DEFINES input_df_scaled and must run BEFORE prediction
            input_df_scaled = prepare_input_data(raw_data, FEATURE_COLUMNS, scaler)
            
            # --- 3. PREDICT ---
            prediction_proba = model.predict_proba(input_df_scaled)[0]
            prediction_score = prediction_proba[1] 
            
            # --- 4. DEFINE OUTCOME VARIABLES ---
            if prediction_score > 0.5:
                 result = "HIGH RISK OF ATTRITION (Recommend Intervention)"
                 css_class = "high-risk"
            else:
                 result = "LOW RISK (Likely Retained)"
                 css_class = "low-risk"
            
            # --- 5. RETURN SUCCESSFUL RESPONSE ---
            return render_template( 
                'index.html', 
                prediction_result=result, 
                score=f"{prediction_score*100:.2f}%",
                css_class=css_class,
                user_input=raw_data
            )

        except Exception as e:
            # 6. RETURN ERROR RESPONSE
            print(f"Prediction logic failed: {e}")
            return render_template('index.html', 
                                   prediction_result=f"Critical Error: {e}", 
                                   score="N/A", # Ensure score is defined in error path
                                   css_class="high-risk"), 500

    # This final return acts as a safety net, although the logic above should cover everything.
    return render_template('index.html')

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=5000, debug=True)