# RiskFlow // Credit Scoring & Risk Intelligence Dashboard

RiskFlow is an interactive web-based dashboard designed to predict creditworthiness using machine learning classification models. It trains multiple models (Logistic Regression, Decision Trees, and Random Forests) on applicant financial features, compares their performance, and offers a real-time risk estimator calculator.

## Key Features

- **Machine Learning Classifiers**: Trains and compares three classification models:
  - **Logistic Regression** (with feature standardization)
  - **Decision Tree Classifier**
  - **Random Forest Classifier**
- **Automated Performance Analytics**: Computes and compares model metrics side-by-side using Chart.js:
  - Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
  - Interactive **ROC Curves** comparison charts.
- **Feature Engineering**: Calculates critical ratios from applicant profiles:
  - **Debt-to-Income (DTI)**
  - **Credit Card Utilization Ratio**
  - **Savings-to-Income Ratio**
- **Real-Time Credit Estimator**: Input applicant financial history parameters using range sliders to calculate:
  - FICO equivalent score (300 - 850)
  - Risk category level (Exceptional, Very Good, Good, Fair, Poor)
  - Approval / Denial status

## Technology Stack

- **Backend**: Python 3, FastAPI, scikit-learn, pandas, numpy, uvicorn
- **Frontend**: HTML5, Vanilla CSS3, JavaScript (ES6+), Chart.js, FontAwesome Icons

## Project Structure

```text
├── app.py              # FastAPI server, data simulation, and model pipelines
├── static/
│   ├── index.html      # Responsive dashboard structure
│   ├── styles.css      # Custom dark-theme CSS style sheets
│   └── app.js          # Chart.js renderers and form events handlers
├── .gitignore          # File exclusions list for Git tracking
└── README.md           # Documentation
```

## Getting Started

### Prerequisites

Ensure you have Python 3 installed. Install the required libraries:

```bash
pip install pandas scikit-learn numpy fastapi uvicorn pydantic python-multipart
```

### Running the Application

1. Clone or download the repository.
2. Launch the server from the root folder:

```bash
python app.py
```

3. Open your browser and navigate to:
   [http://localhost:8000](http://localhost:8000)

## Project Background

This credit assessment pipeline is engineered based on classic credit risk features:
- **Income & Debt**: Determines capability to service obligations.
- **Payment History**: Tracks missed installments, which correlates strongly with default probabilities.
- **Savings Balance**: Represents liquidity buffer in financial stress.
- **Credit Card Utilization**: Monitors card balance levels relative to overall limits.
