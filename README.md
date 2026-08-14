# Maternal Satisfaction with Delivery Care — ML Explorer

An interactive **Streamlit** application for exploring maternal satisfaction with vaginal and cesarean-section delivery care at Bahir Dar city health facilities, Northwest Ethiopia, and for training/inspecting two **logistic regression** machine-learning models built on that data.

🔗 **Live app:** _add your Streamlit Cloud URL here after deploying_

---

## 📖 About the project

This project analyzes survey data from **894 mothers** who recently delivered at health facilities in Bahir Dar city, examining how satisfied they were with the care they received and what factors predict that satisfaction.

Two logistic regression models are built and compared:

| Model | Description | Test AUC |
|---|---|---|
| **Model A (full)** | Includes all predictors, including service-domain satisfaction items (cleanliness, privacy, waiting time, etc.) | 0.991 |
| **Model B (conservative)** | Excludes service-domain satisfaction items — uses only socio-demographic, obstetric, and facility-level predictors | 0.841 |

Model B is the more policy-actionable model, since it identifies independent, structural predictors of satisfaction (e.g. facility ownership, parity, ANC attendance) rather than relying on sub-ratings of satisfaction itself.

---

## ✨ App features

The Streamlit app has four tabs:

- **📊 Overview** — headline stats (sample size, overall satisfaction rate), numeric variable summary, raw data preview
- **📈 Descriptive Charts** — satisfaction distribution, satisfaction by mode of delivery, age histogram, income by satisfaction, plus an interactive dropdown to cross-tab satisfaction against any categorical variable
- **🧠 Model Performance** — toggle between Model A and Model B; view accuracy/precision/recall/F1/AUC/Brier score, ROC curve, confusion matrix, odds-ratio and permutation-importance charts
- **🔮 Predict** — fill in a form describing a hypothetical mother and get a live predicted probability of satisfaction

---

## 🗂️ Project structure

```
.
├── streamlit_app.py                    # Main Streamlit application
├── Delivery_care_satisfaction.sav      # Source dataset (SPSS format)
├── requirements.txt                    # Python dependencies
├── 01_descriptive.py                   # Descriptive statistics script
├── 02_charts.py                        # Chart-generation script
├── 03_modeling.py                      # Model A (full) — training, evaluation, deployment
├── 04_sensitivity_model.py             # Model B (conservative) — sensitivity analysis
└── README.md
```

---

## 🚀 Running locally

**1. Clone the repository**
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

**2. Create and activate a virtual environment** (recommended)
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux
```

**3. Install dependencies**
```bash
python -m pip install -r requirements.txt
```

**4. Run the app**
```bash
python -m streamlit run streamlit_app.py
```
The app opens automatically at `http://localhost:8501`. If `Delivery_care_satisfaction.sav` is in the same folder, it loads automatically — otherwise upload it via the sidebar.

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push this repository to GitHub (done ✅)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **"New app"**, select this repository, branch `main`, and file `streamlit_app.py`
4. Click **Deploy**

---

## 🧪 Methods summary

- **Data cleaning:** dropped high-missingness/redundant variables (distance-in-hours duplicates, pre-binned recodes), imputed 5 non-response cases in religion with the modal category
- **Preprocessing:** numeric features standardized, categorical features one-hot encoded, all inside a scikit-learn `Pipeline`/`ColumnTransformer` to prevent test-set leakage
- **Train/test split:** 75%/25%, stratified by outcome
- **Model:** L2-regularized logistic regression with `class_weight='balanced'`
- **Evaluation:** accuracy, precision, recall, F1 score, AUC-ROC, Brier score on a held-out test set
- **Feature importance:** logistic regression coefficients (odds ratios) and permutation importance (model-agnostic)

---

## 📦 Requirements

```
streamlit
plotly
pandas
pyreadstat
scikit-learn
```

---

## ⚠️ Data note

`Delivery_care_satisfaction.sav` contains de-identified survey data. If this repository is public, confirm the dataset is appropriate to share publicly before pushing it, or remove it from the repo and rely on the in-app file uploader instead.

---

## 📄 License

_Add a license of your choice (e.g. MIT) if you want others to be able to reuse this code._

## 🙏 Acknowledgments

Dataset based on the facility-based comparative cross-sectional study of maternal satisfaction with delivery care in Bahir Dar city, Northwest Ethiopia.
