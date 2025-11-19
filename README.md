# Medical Appointment No-Show Risk Predictor

End-to-end ML project that predicts the probability a patient will **miss** their scheduled medical appointment.

Built as a portfolio project to demonstrate data science + software engineering skills:
- Data cleaning & feature engineering on a real healthcare dataset (~110K appointments)
- Model training & evaluation with scikit-learn
- Production-style pipeline saved with `joblib`
- FastAPI service with Pydantic-validated JSON inputs
- Streamlit UI for interactive demo

> ⚠️ **Disclaimer:** This project is for learning and demonstration only.  
> It is **not** intended for real clinical decision-making.

---

## 1. Problem & Motivation

Missed appointments (no-shows) waste clinic capacity, delay care, and cost money.  
If clinics can **predict no-show risk ahead of time**, they can:

- Overbook risky time slots
- Proactively call or text high-risk patients
- Offer telehealth or rescheduling options

This project uses a public Brazilian hospital dataset to estimate **no-show probability** from basic appointment and patient features.

---

## 2. Dataset

- Source: Kaggle “Medical Appointment No Shows” dataset  
- Size: ~110K appointments
- Key fields:
  - `Age`
  - `Gender`
  - `Scholarship` (government assistance)
  - `Hipertension`, `Diabetes`, `Alcoholism`, `Handcap`
  - `ScheduledDay`, `AppointmentDay`
  - `SMS_received`
  - `No-show` (target)

### Engineered features

In `notebooks/01_eda.ipynb` and `notebooks/02_modeling.ipynb`:

- `WaitingDays` – days between `ScheduledDay` and `AppointmentDay`
- `ApptWeekday` – day of the week of the appointment
- `NoShow` – binary target (`1` = no-show, `0` = showed)

---

## 3. Modeling

The main model is a scikit-learn `Pipeline`:

- **Preprocessing**
  - Standard scaling for numeric features (`Age`, `WaitingDays`)
  - Pass-through binary dummies (comorbidities, scholarship, SMS)
  - One-hot encoding for `Gender` and `ApptWeekday`
- **Model**
  - `LogisticRegression(max_iter=1000, class_weight='balanced')`

Train/test split:

- `train_test_split(test_size=0.2, stratify=y, random_state=42)`

### Performance (test set)

- **ROC AUC:** ~0.67  
- **Accuracy:** ~0.67  
- **No-show class (1):**
  - Precision ≈ 0.32
  - Recall ≈ 0.57
  - F1 ≈ 0.41

Interpretation:

- Baseline model that always predicts “show” has **0 recall** for no-shows.
- This model captures **~57% of no-shows**, at the cost of some false positives.
- Suitable as a **decision-support tool** (e.g., prioritizing outreach) rather than an automated gatekeeper.

The trained pipeline is saved to `models/no_show_model.joblib` using `joblib.dump`.

---

## 4. Project Structure

```text
no-show-predictor/
├── data/
│   ├── raw/
│   │   └── noshowappointments.csv
│   └── processed/           # (optional for future use)
├── notebooks/
│   ├── 01_eda.ipynb         # Exploratory analysis & feature engineering
│   └── 02_modeling.ipynb    # Model training, evaluation, and saving
├── models/
│   └── no_show_model.joblib # Saved scikit-learn Pipeline
├── src/
│   ├── __init__.py
│   ├── predict.py           # Helper to load model and run predictions
│   └── api/
│       ├── __init__.py
│       └── main.py          # FastAPI app exposing /predict
├── streamlit_app.py         # Streamlit UI using src.predict
├── requirements.txt
└── README.md
