# AI-Driven Digital Twin — Smart Manufacturing Operations

BTech Data Science Capstone Project

---

## Project Structure

```
smart_manufacturing/
├── data/
│   ├── ai4i2020.csv
│   └── preprocess.py
├── models/
│   ├── tune.py
│   ├── train.py
│   ├── evaluate.py
│   ├── best_model.pkl       # generated after training
│   └── scaler.pkl           # generated after preprocessing
├── simulation/
│   ├── factory.py
│   └── predictor.py
├── llm/
│   └── alert_generator.py
├── dashboard/
│   └── app.py
├── logs/
│   └── alerts.log
├── mlruns/                  # auto-created by MLflow
├── config.py
├── main.py
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Gemini API key

```bash
# Linux / macOS
export GEMINI_API_KEY="your_key_here"

# Windows (PowerShell)
$env:GEMINI_API_KEY = "your_key_here"
```

Get a free key at: https://aistudio.google.com/app/apikey

### 3. Place the dataset

```
data/ai4i2020.csv
```

---

## Run

### Full pipeline (preprocess → tune → train → dashboard)

```bash
python main.py
```

> Optuna tuning takes ~20–40 minutes depending on hardware.

### Run dashboard independently (after training)

```bash
streamlit run dashboard/app.py
# Open: http://localhost:8501
```

### View MLflow experiments

```bash
mlflow ui
# Open: http://localhost:5000
```

---

## Modules

| Module | File | Description |
|---|---|---|
| Preprocessing | `data/preprocess.py` | SMOTE, scaling, 80/20 split |
| Tuning | `models/tune.py` | Optuna for RF, XGBoost, MLP |
| Training | `models/train.py` | MLflow-tracked training |
| Evaluation | `models/evaluate.py` | Metrics, confusion matrix, reports |
| Digital Twin | `simulation/factory.py` | SimPy 3-machine simulation |
| Predictor | `simulation/predictor.py` | Real-time inference bridge |
| LLM Alerts | `llm/alert_generator.py` | Gemini Flash 2.0 alerts |
| Dashboard | `dashboard/app.py` | Streamlit real-time UI |

---

## Models Trained

| Model | MLflow Run Name |
|---|---|
| Random Forest | `RF_Optuna_Tuned` |
| XGBoost | `XGB_Optuna_Tuned` |
| PyTorch MLP | `MLP_Optuna_Tuned` |

The best model by macro F1 is automatically saved as `models/best_model.pkl`.

---

## Evaluation Metrics

All metrics computed per model and logged to MLflow:
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)
- ROC-AUC
- Confusion matrix (PNG artifact)
- Classification report (TXT artifact)
