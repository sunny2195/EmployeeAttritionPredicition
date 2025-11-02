import os
import sys
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from pathlib import Path


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

MODEL_PATH = Path('artifacts/model_trainer/best_model.pkl')
SCALER_PATH = Path('artifacts/data_transformation/scaler.pkl')
THRESHOLD_PATH = Path('artifacts/model_trainer/optimal_threshold.pkl')


app = Flask(__name__)
model = None
scaler = None
optimal_threshold = 0.5

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


def prepare_input_data(raw_data: dict, feature_cols: list, scaler_obj) -> pd.DataFrame:
    raw_df = pd.DataFrame([raw_data])
    
    
    final_df = pd.DataFrame(0, index=[0], columns=feature_cols)

    for key, value in raw_df.iloc[0].items():
        if key in final_df.columns:
            final_df[key] = float(value) 
    
    scaled_cols_for_pred = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
        'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 
        'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 
        'RelationshipSatisfaction', 'StockOptionLevel', 'TotalWorkingYears', 
        'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 
        'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

    if scaler_obj is not None:
        final_df[scaled_cols_for_pred] = scaler_obj.transform(final_df[scaled_cols_for_pred])
    
    return final_df

@app.route('/', methods=['GET', 'POST'])
def predict_employee_attrition():
    if request.method == 'GET':
        return render_template('index.html') 

    elif request.method == 'POST':
        if model is None or scaler is None:
            return render_template('index.html', 
                                   prediction_result="Error: Artifacts not loaded. Rerun pipeline.", 
                                   css_class="high-risk"), 503

        try:
            raw_data = request.form.to_dict()
            input_df_scaled = prepare_input_data(raw_data, FEATURE_COLUMNS, scaler)
            prediction_proba = model.predict_proba(input_df_scaled)[0]
            prediction_score = prediction_proba[1] 
            if prediction_score > 0.5:
                 result = "HIGH RISK OF ATTRITION (Recommend Intervention)"
                 css_class = "high-risk"
            else:
                 result = "LOW RISK (Likely Retained)"
                 css_class = "low-risk"
            return render_template( 
                'index.html', 
                prediction_result=result, 
                score=f"{prediction_score*100:.2f}%",
                css_class=css_class,
                user_input=raw_data
            )

        except Exception as e:
            print(f"Prediction logic failed: {e}")
            return render_template('index.html', 
                                   prediction_result=f"Critical Error: {e}", 
                                   score="N/A", 
                                   css_class="high-risk"), 500
    return render_template('index.html')

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=5000, debug=True)