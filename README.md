<<<<<<< HEAD
# Diabetes-Prevention-
=======
# Diabetes Prevention and Risk Assessment (Young Adults)

Python Flask web app that predicts diabetes risk from lifestyle and stress factors and generates AI-backed recommendations using Gemini.

## Features
- Home info portal on stress-diabetes connection
- Assessment form with BMI auto-calculation
- ML prediction using a trained model (with heuristic fallback)
- Result page with color-coded score, Plotly visualization, and Gemini recommendations

## Tech
- Flask, Jinja2, Bootstrap 5
- scikit-learn, pandas, numpy, joblib
- Plotly for charts
- Gemini API for recommendations

## Setup
1. Create environment and install dependencies
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Configure environment
```bash
copy .env.example .env
# edit .env to set GEMINI_API_KEY optionally
```

3. Run the app
```bash
python app.py
```
Open `http://localhost:5000`.

## Training
Prepare a CSV with columns:
`Age, Gender, BMI, Diet_Type, Physical_Activity, Stress_Level, Sleep_Quality, Family_History, Hypertension, Cholesterol_Level, Diabetes`

- `Diabetes` is the binary target (0/1).
- You can merge the Kaggle dataset and your survey data into this schema.

Train and save artifacts:
```bash
python train_model.py --data path\to\merged.csv --outdir models
```
This produces `models/diabetes_model.joblib`, `models/label_encoders.joblib`, `models/meta.joblib`.

## Dataset
Kaggle: `https://www.kaggle.com/datasets/ankushpanday1/diabetes-in-youth-vs-adult-in-india/data`

## Notes
- If no model is found, the app uses a heuristic aligned to the requested risk bands.
- The tool is educational and not a medical diagnosis.
>>>>>>> 96c8e7c (Diabetes)
