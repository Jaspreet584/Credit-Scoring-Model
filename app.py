import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

app = FastAPI(title="Credit Scoring API", version="1.0.0")

# Global variables
DATASET: Optional[pd.DataFrame] = None
MODELS: Dict = {}
SCALERS: Dict = {}
METRICS: Dict = {}
FEATURE_NAMES = [
    "Age", "Annual_Income", "Total_Debt", "Missed_Payments", "Savings", 
    "Credit_Card_Limit", "Credit_Card_Balance", "Credit_History_Length_Years",
    "Debt_to_Income_Ratio", "Credit_Utilization_Ratio", "Savings_to_Income_Ratio"
]

def generate_credit_dataset(n_samples: int = 1000, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)
    
    # Base applicant demographic and financial variables
    age = np.random.randint(18, 70, size=n_samples)
    annual_income = np.random.lognormal(mean=11.0, sigma=0.5, size=n_samples)
    annual_income = np.clip(annual_income, 15000, 250000).round(-2)
    
    # Debt is partially correlated with income
    total_debt = annual_income * np.random.beta(a=2, b=5, size=n_samples) * 1.5
    total_debt = np.clip(total_debt, 0, 180000).round(-2)
    
    # Payment behavior: count of missed payments in the last 2 years
    missed_payments = np.random.exponential(scale=0.8, size=n_samples).astype(int)
    missed_payments = np.clip(missed_payments, 0, 10)
    
    # Savings buffer
    savings = annual_income * np.random.beta(a=1, b=3, size=n_samples) * 0.4
    savings = np.clip(savings, 500, 120000).round(-2)
    
    # Credit Card Limit and Balance
    credit_card_limit = np.clip(annual_income * 0.25 * np.random.uniform(0.5, 1.5, size=n_samples), 1000, 40000).round(-2)
    credit_card_balance = credit_card_limit * np.random.beta(a=1.5, b=2.5, size=n_samples)
    credit_card_balance = np.clip(credit_card_balance, 0, credit_card_limit).round(-2)
    
    # Length of credit history in years
    credit_history_years = np.clip((age - 18) * np.random.uniform(0.3, 0.9, size=n_samples), 1, 35).astype(int)
    
    df = pd.DataFrame({
        "Age": age,
        "Annual_Income": annual_income,
        "Total_Debt": total_debt,
        "Missed_Payments": missed_payments,
        "Savings": savings,
        "Credit_Card_Limit": credit_card_limit,
        "Credit_Card_Balance": credit_card_balance,
        "Credit_History_Length_Years": credit_history_years
    })
    
    # Feature Engineering
    df["Debt_to_Income_Ratio"] = (df["Total_Debt"] / df["Annual_Income"]).round(4)
    df["Credit_Utilization_Ratio"] = (df["Credit_Card_Balance"] / df["Credit_Card_Limit"]).round(4)
    df["Savings_to_Income_Ratio"] = (df["Savings"] / df["Annual_Income"]).round(4)
    
    # Target Construction (Credit Status: 0 = Approved, 1 = Denied)
    risk_index = (
        df["Debt_to_Income_Ratio"] * 45 + 
        df["Credit_Utilization_Ratio"] * 25 + 
        (df["Missed_Payments"] / 10) * 50 - 
        (df["Savings_to_Income_Ratio"] * 30) - 
        (df["Credit_History_Length_Years"] / 35) * 15
    )
    
    noise = np.random.normal(loc=0, scale=8.0, size=n_samples)
    final_risk = risk_index + noise
    
    denial_threshold = np.percentile(final_risk, 75)
    df["Credit_Status"] = (final_risk >= denial_threshold).astype(int)
    
    return df

def train_models():
    global DATASET, MODELS, SCALERS, METRICS
    
    DATASET = generate_credit_dataset()
    X = DATASET[FEATURE_NAMES]
    y = DATASET["Credit_Status"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    
    # Standardizer for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    SCALERS["scaler"] = scaler
    
    models_dict = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    }
    
    for name, clf in models_dict.items():
        if name == "logistic_regression":
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        else:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]
            
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        # ROC Curve Points
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        # Downsample ROC points for ease of visualization
        step = max(1, len(fpr) // 50)
        roc_points = [{"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr[::step], tpr[::step])]
        if roc_points[-1]["fpr"] != 1.0 or roc_points[-1]["tpr"] != 1.0:
            roc_points.append({"fpr": 1.0, "tpr": 1.0})
            
        MODELS[name] = clf
        METRICS[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc),
            "roc_curve": roc_points
        }

@app.on_event("startup")
def startup():
    train_models()

# Schema for applicant prediction
class PredictInput(BaseModel):
    age: int = Field(..., ge=18, le=80)
    annual_income: float = Field(..., ge=1000)
    total_debt: float = Field(..., ge=0)
    missed_payments: int = Field(..., ge=0)
    savings: float = Field(..., ge=0)
    credit_card_limit: float = Field(..., ge=500)
    credit_card_balance: float = Field(..., ge=0)
    credit_history_length_years: int = Field(..., ge=0)
    model_name: str = Field("random_forest")

@app.get("/api/performance")
def get_performance():
    if not METRICS:
        train_models()
    return METRICS

@app.post("/api/predict")
def predict_credit(data: PredictInput):
    global MODELS, SCALERS
    if not MODELS:
        train_models()
        
    model_name = data.model_name
    if model_name not in MODELS:
        raise HTTPException(status_code=400, detail="Invalid model selection")
        
    dti = round(data.total_debt / data.annual_income, 4) if data.annual_income > 0 else 0.0
    util = round(data.credit_card_balance / data.credit_card_limit, 4) if data.credit_card_limit > 0 else 0.0
    sav_ratio = round(data.savings / data.annual_income, 4) if data.annual_income > 0 else 0.0
    
    features = {
        "Age": data.age,
        "Annual_Income": data.annual_income,
        "Total_Debt": data.total_debt,
        "Missed_Payments": data.missed_payments,
        "Savings": data.savings,
        "Credit_Card_Limit": data.credit_card_limit,
        "Credit_Card_Balance": data.credit_card_balance,
        "Credit_History_Length_Years": data.credit_history_length_years,
        "Debt_to_Income_Ratio": dti,
        "Credit_Utilization_Ratio": util,
        "Savings_to_Income_Ratio": sav_ratio
    }
    
    X_df = pd.DataFrame([features])[FEATURE_NAMES]
    clf = MODELS[model_name]
    
    if model_name == "logistic_regression":
        scaler = SCALERS["scaler"]
        X_scaled = scaler.transform(X_df)
        pred = int(clf.predict(X_scaled)[0])
        prob = float(clf.predict_proba(X_scaled)[0][1])
    else:
        pred = int(clf.predict(X_df)[0])
        prob = float(clf.predict_proba(X_df)[0][1])
        
    # Mapping to credit scores (300 to 850)
    credit_score = int(300 + (1.0 - prob) * 550)
    
    if credit_score < 580:
        risk_cat = "Poor"
    elif credit_score < 670:
        risk_cat = "Fair"
    elif credit_score < 740:
        risk_cat = "Good"
    elif credit_score < 800:
        risk_cat = "Very Good"
    else:
        risk_cat = "Exceptional"
        
    return {
        "decision": "Approved" if pred == 0 else "Denied",
        "score": credit_score,
        "probability_of_default": prob,
        "risk_category": risk_cat,
        "calculated_features": {
            "dti": dti,
            "utilization": util,
            "savings_ratio": sav_ratio
        }
    }

# Static assets hosting
static_path = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_path):
    os.makedirs(static_path)

app.mount("/static", StaticFiles(directory=static_path), name="static")

@app.get("/", response_class=HTMLResponse)
def get_index():
    index_file = os.path.join(static_path, "index.html")
    if not os.path.exists(index_file):
        return HTMLResponse("<h3>Web dashboard index.html is missing.</h3>", status_code=404)
    with open(index_file, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
