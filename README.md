# 🚚 Delivery Delay Prediction Analysis
## 4-Tier Analytics Ladder | Logistics & Supply Chain ML Project

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Problem Statement

Delivery delays represent one of the most significant operational and customer-experience challenges in last-mile logistics. An undetected delay leads to SLA breaches, customer refund requests, brand erosion, and costly reactive recovery — typically 3–5× more expensive than proactive intervention.

This project builds a **production-ready end-to-end analytics and ML system** that:
- Identifies **when, where, and why** delays occur (Descriptive & Diagnostic Analytics)
- Predicts **whether a specific shipment will be delayed** before it departs (Predictive ML)
- Prescribes **actionable, resource-constrained operational interventions** (Prescriptive Strategy)

---

## 📦 Dataset

| Property | Value |
|----------|-------|
| **File** | "https://www.kaggle.com/datasets/muhammadahmaddaar/delivery-logistics-dataset-india-multi-partner?resource=download" |
| **Records** | 25,000 delivery transactions |
| **Features** | 15 raw columns → 9 ML features after cleaning |
| **Target** | `Delay_Flag` (0 = On-Time, 1 = Delayed) |
| **Overall Delay Rate** | 26.68% (6,669 delayed out of 25,000) |

**Columns in the dataset:**
`delivery_id`, `delivery_partner`, `package_type`, `vehicle_type`, `delivery_mode`, `region`, `weather_condition`, `distance_km`, `package_weight_kg`, `delivery_time_hours`, `expected_time_hours`, `delayed`, `delivery_status`, `delivery_rating`, `delivery_cost`

---

## 🏗️ 4-Tier Analytics Ladder

```
┌─────────────────────────────────────────────────────────────────┐
│  TIER 4 — PRESCRIPTIVE       What should we do?                  │
│  Resource-constrained dispatch rules, SLA renegotiation,         │
│  vehicle-distance optimisation, weather-adaptive protocols       │
├─────────────────────────────────────────────────────────────────┤
│  TIER 3 — PREDICTIVE         What will happen?                   │
│  Random Forest classifier (ROC-AUC: 0.9668, Recall: 96.7%)      │
│  Shipment-level delay probability scoring at dispatch time       │
├─────────────────────────────────────────────────────────────────┤
│  TIER 2 — DIAGNOSTIC         Why did it happen?                  │
│  Weather impact, delivery mode SLA mismatch, distance bands,     │
│  vehicle-type fit, partner performance root-cause analysis       │
├─────────────────────────────────────────────────────────────────┤
│  TIER 1 — DESCRIPTIVE        What happened?                      │
│  Overall delay rate (26.68%), partner rankings, regional KPIs,   │
│  cost analysis, customer rating distribution                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤖 ML Model Performance

| Metric | Random Forest | Notes |
|--------|--------------|-------|
| **Accuracy** | 88.50% | Test set (5,000 samples) |
| **Precision** | 70.84% | % of flagged delays that are real |
| **Recall** | **96.70%** | % of real delays caught ← Primary objective |
| **F1-Score** | 81.77% | Harmonic mean |
| **ROC-AUC** | **0.9668** | Outstanding discrimination |
| **5-Fold CV AUC** | 0.9667 ± 0.0017 | Stable — not overfit |

### Confusion Matrix (Test Set: 5,000 samples)

|  | Predicted On-Time | Predicted Delayed |
|--|-------------------|-------------------|
| **Actual On-Time** | TN = 3,135 ✅ | FP = 531 ⚠️ |
| **Actual Delayed** | FN = **44** 🚨 | TP = 1,290 ✅ |

> **Key Trade-Off:** Only 44 genuine delays escape detection (False Negatives = highest business cost), while 531 on-time shipments are flagged unnecessarily (False Positives = manageable overhead). The model is deliberately tuned for maximum Recall to minimise undetected delays.

---

## 🔑 Key Findings

1. **Express Mode is the #1 Delay Driver**: Express mode shows a **73.78% delay rate** — the SLA commitment is systematically over-committed relative to available capacity. Delivery mode features contribute >52% of total model feature importance.

2. **Weather Amplifies Risk by 2.6×**: Stormy conditions produce a **41.45% delay rate** vs. 16–17% under clear/cold weather. Weather-adaptive dispatch protocols can proactively protect SLA.

3. **Standard/Two-Day Modes are Near-Zero Risk**: Two-Day (0.43%) and Standard (0.0%) delivery modes show negligible delay rates — adequate lead time eliminates operational pressure.

4. **Partner Variation is Narrow**: Delay rates across 9 partners range only 24.8%–28.3%, suggesting systemic issues rather than individual partner failures.

5. **Regional Risk is Secondary**: Region explains far less variance than delivery mode or weather — regional investment should be targeted at infrastructure, not partner replacement.

---

## 🎯 Prescriptive Operational Rule

When delivery workload is high and available capacity is limited:

```
Priority Score = 0.60 × Delay Probability + 0.40 × (Mode Urgency / 4)

Risk Tiers:
  ≥ 0.60 → HIGH RISK   → ESCALATE: Premium partner + immediate customer alert
  0.35–0.60 → MEDIUM   → MONITOR: Reliable partner + real-time tracking
  < 0.35  → LOW RISK   → STANDARD: Normal dispatch

Vehicle-Distance Rules:
  < 60 km  → Bike / EV Bike / Scooter
  60–150km → Any vehicle (weight-based selection)
  > 150km  → Truck / Van / EV Van (mandatory)
```

---

## 🛠️ Technologies Used

| Category | Technology |
|----------|-----------|
| Language | Python 3.9+ |
| Data Processing | pandas, numpy |
| Machine Learning | scikit-learn (RandomForest, LogisticRegression, Pipeline) |
| Visualisation | matplotlib, seaborn |
| Dashboard | Streamlit |
| Model Serialisation | joblib |
| Reporting | python-docx |
| Notebook | Jupyter (nbformat, nbconvert) |

---

## 🚀 Setup & Run Instructions

### Prerequisites
- Python 3.9 or higher
- pip package manager

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate All Artifacts (Model + Charts + Metrics)

```bash
python run_analysis.py
```

This generates:
- `rf_delay_model.pkl` — trained Random Forest pipeline
- `model_metrics.json` — all performance metrics
- `Delivery_Logistics_Scored.csv` — dataset with delay probabilities
- All chart PNG files

### 3. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

### 4. Run the Jupyter Notebook (Interactive)

```bash
jupyter notebook Delivery_Delay_Prediction_Analysis.ipynb
```

---

## 📁 Project Structure

```
Delivery Delay Prediction/
│
├── 📄 Delivery_Logistics.csv                         ← Raw dataset
│
├── 📓 Delivery_Delay_Prediction_Analysis.ipynb       ← Complete analysis notebook
├── 🖥️  app.py                                         ← Streamlit dashboard
├── 🐍 run_analysis.py                                 ← Artifact generation script
├── 🐍 generate_report.py                              ← DOCX report generator
├── 📋 requirements.txt                                ← Python dependencies
│
├── 🤖 rf_delay_model.pkl                              ← Trained ML model
├── 📊 model_metrics.json                              ← Model performance metrics
├── 📈 Delivery_Logistics_Scored.csv                   ← Dataset + risk scores
│
├── 🖼️  chart1_partner_vehicle.png                     ← EDA Chart 1
├── 🖼️  chart2_region_weather.png                      ← EDA Chart 2
├── 🖼️  chart3_distance_mode.png                       ← EDA Chart 3
├── 🖼️  chart4_package_rating.png                      ← EDA Chart 4
├── 🖼️  chart5_heatmap_cost.png                        ← EDA Chart 5
├── 🖼️  chart_roc_confusion.png                        ← ML Evaluation Chart
├── 🖼️  chart_feature_importance.png                   ← Feature Importance Chart
│
├── 📝 Delivery_Delay_Prediction_Analysis_Report.docx  ← Full project report
└── 📖 README.md                                       ← This file
```

---

## 📊 Dashboard Pages

The Streamlit dashboard (`app.py`) includes **8 interactive pages**:

| Page | Content |
|------|---------|
| 🏠 Executive Overview | KPI cards, methodology summary, key findings |
| 📊 EDA — Partner & Vehicle | Chart 1 + performance tables + insights |
| 🌍 EDA — Region & Weather | Chart 2 + regional/weather tables + insights |
| 📦 EDA — Distance & Mode | Chart 3 + delivery mode analysis |
| ⭐ EDA — Package & Rating | Chart 4 + package type breakdown |
| 🔥 EDA — Risk Heatmap | Chart 5 + interactive Region × Weather heatmap |
| 🤖 ML Model Performance | Confusion matrix, ROC curve, feature importance, leakage policy |
| 🎯 Prescriptive Strategies | Live dispatch simulator + 7 business recommendations |

---

## 📋 Deliverables Summary

| Deliverable | Status | Description |
|-------------|--------|-------------|
| `Delivery_Delay_Prediction_Analysis.ipynb` | ✅ | Complete 4-tier project notebook |
| `app.py` | ✅ | 8-page interactive Streamlit dashboard |
| `requirements.txt` | ✅ | All library dependencies |
| `Delivery_Delay_Prediction_Analysis_Report.docx` | ✅ | Comprehensive project documentation |
| `README.md` | ✅ | This file |

---

## ⚠️ Target Leakage Prevention Policy

The following columns were **strictly excluded** from ML features to prevent data leakage:

| Column | Reason |
|--------|--------|
| `delayed` | Direct source of the target variable |
| `delivery_status` | Post-event target derivative |
| `delivery_rating` | Post-delivery outcome — unavailable at dispatch |
| `delivery_id` | Raw identifier — no predictive signal |
| `delivery_time_hours` | All-zero column — zero variance |
| `expected_time_hours` | All-zero column — zero variance |

All retained features are verifiably available at dispatch time.

---

## 📌 Business Impact Summary

- **96.7% of genuine delays are predicted** before the shipment leaves, enabling proactive intervention
- **Express mode rebalancing** alone could reduce overall delay rate by an estimated 15–20 percentage points
- **Weather-adaptive protocols** can prevent SLA breaches during 6 adverse-weather scenarios
- **Vehicle-distance optimisation** reduces both delay probability and operational cost for mismatched assignments
- **Real-time ML integration** into TMS/WMS enables shipment-level risk scoring at zero marginal analysis cost

---

## 👤 Author

**Principal Data Analyst & Machine Learning Engineer**  
Domain: Logistics & Supply Chain Management  
Methodology: 4-Tier Analytics Ladder (Descriptive → Diagnostic → Predictive → Prescriptive)

---

*For questions about the methodology, model architecture, or business recommendations, refer to `Delivery_Delay_Prediction_Analysis_Report.docx` for full documentation.*
