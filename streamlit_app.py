"""
Streamlit app — Maternal Satisfaction with Delivery Care (Bahir Dar, Ethiopia)
================================================================================
Interactive companion to Project 2 Part 1. Lets you:
  1. Explore the dataset (descriptive stats, charts)
  2. Inspect two logistic-regression models:
       Model A - full model (includes service-domain satisfaction items)
       Model B - conservative model (socio-demographic/obstetric/facility only)
  3. Score a new/hypothetical mother's predicted satisfaction interactively

Run with:  streamlit run streamlit_app.py
Place "Delivery_care_satisfaction.sav" in the same folder, or upload it
from the sidebar when the app opens.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, brier_score_loss,
                              confusion_matrix, roc_curve)
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
DEFAULT_PATH = Path(__file__).parent / "Delivery_care_satisfaction.sav"

st.set_page_config(page_title="Maternal Satisfaction ML Explorer", layout="wide")

# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(file) -> pd.DataFrame:
    df = pd.read_spss(file)
    # Some SPSS categorical columns come back with mixed-type categories
    # (e.g. a stray numeric non-response code alongside string labels like
    # "muslim"), which breaks Streamlit's Arrow-based table renderer with
    # "tried to convert to double". Normalize all categorical columns to
    # plain strings, keeping real missing values as NaN.
    for col in df.select_dtypes(include='category').columns:
        df[col] = df[col].astype(object)
        df[col] = df[col].where(df[col].isna(), df[col].astype(str))
    return df

st.sidebar.title("📁 Data")
uploaded = st.sidebar.file_uploader("Upload Delivery_care_satisfaction.sav", type=["sav"])

if uploaded is not None:
    df = load_data(uploaded)
elif DEFAULT_PATH.exists():
    df = load_data(str(DEFAULT_PATH))
else:
    st.warning("Upload Delivery_care_satisfaction.sav from the sidebar to get started.")
    st.stop()

st.sidebar.success(f"Loaded {df.shape[0]} mothers, {df.shape[1]} variables")

# ---------------------------------------------------------------------------
# 2. CLEANING / FEATURE ENGINEERING (mirrors 03_modeling.py / 04_sensitivity_model.py)
# ---------------------------------------------------------------------------
SATISFACTION_DOMAIN_ITEMS = [
    'waitingtimerecode', 'privacyrecode', 'encourageandsupportatdeliveryrecode',
    'politenessrecode', 'availabilityofmedicalequipmentsrecode', 'counselingrecode',
    'overallcleanessrecode', 'accessandcleanessoftoiletrecode',
    'waitingareacleanessandconfort', 'availabilityofbedintheward',
    'q111timeta', 'q112waitpl', 'q113privac', 'q301greeti', 'q302respec',
]
BASE_DROP = ['q1092disth', 'wolkingdistancerecodeinhrs']
DUPLICATE_RAW = ['ageafterrecod', 'monthlyincomerecode', 'gestationaagerecode',
                  'birthweightofbaby', 'distancerecode1']

@st.cache_data
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    d = df.drop(columns=[c for c in BASE_DROP + DUPLICATE_RAW if c in df.columns])
    if 'q103religi' in d.columns:
        # after load_data's string normalization, the stray non-response
        # code can show up as 88.0, 88, or '88.0' / '88' - catch all forms
        d['q103religi'] = d['q103religi'].replace({88.0: np.nan, 88: np.nan, '88.0': np.nan, '88': np.nan})
        d['q103religi'] = d['q103religi'].astype('object')
        d['q103religi'] = d['q103religi'].fillna(d['q103religi'].mode()[0])
    return d

df_model = clean_data(df)

@st.cache_resource
def fit_model(df_model: pd.DataFrame, exclude_domain_items: bool):
    d = df_model.copy()
    if exclude_domain_items:
        d = d.drop(columns=[c for c in SATISFACTION_DOMAIN_ITEMS if c in d.columns])

    y = (d['satisfaction'] == 'satisfied').astype(int)
    X = d.drop(columns=['satisfaction'])
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

    preprocess = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
    ])
    pipe = Pipeline([
        ('preprocess', preprocess),
        ('clf', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight='balanced')),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_proba),
        'Brier score': brier_score_loss(y_test, y_proba),
    }
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    feature_names = (numeric_features +
                      list(pipe.named_steps['preprocess'].named_transformers_['cat']
                           .get_feature_names_out(categorical_features)))
    coefs = pipe.named_steps['clf'].coef_[0]
    coef_table = pd.DataFrame({'feature': feature_names, 'coefficient': coefs})
    coef_table['odds_ratio'] = np.exp(coef_table['coefficient'])
    coef_table = coef_table.reindex(coef_table['coefficient'].abs().sort_values(ascending=False).index)

    perm = permutation_importance(pipe, X_test, y_test, n_repeats=15,
                                   random_state=RANDOM_STATE, scoring='roc_auc')
    perm_table = pd.DataFrame({'feature': X.columns, 'importance': perm.importances_mean}).sort_values(
        'importance', ascending=False)

    return {
        'pipeline': pipe, 'metrics': metrics, 'cm': cm, 'fpr': fpr, 'tpr': tpr,
        'coef_table': coef_table, 'perm_table': perm_table,
        'numeric_features': numeric_features, 'categorical_features': categorical_features,
        'X': X, 'X_test': X_test, 'y_test': y_test,
    }

model_a = fit_model(df_model, exclude_domain_items=False)
model_b = fit_model(df_model, exclude_domain_items=True)

# ---------------------------------------------------------------------------
# 3. LAYOUT
# ---------------------------------------------------------------------------
st.title("🤰 Maternal Satisfaction with Delivery Care — Bahir Dar, Ethiopia")
st.caption("Interactive exploration of survey data and two logistic-regression ML models "
           "(n = {} mothers)".format(len(df)))

tab_overview, tab_charts, tab_models, tab_predict = st.tabs(
    ["📊 Overview", "📈 Descriptive Charts", "🧠 Model Performance", "🔮 Predict"])

# --- TAB 1: OVERVIEW ---------------------------------------------------
with tab_overview:
    c1, c2, c3 = st.columns(3)
    sat_rate = (df['satisfaction'] == 'satisfied').mean()
    c1.metric("Mothers surveyed", f"{len(df):,}")
    c2.metric("Overall satisfaction", f"{sat_rate*100:.1f}%")
    c3.metric("Variables", f"{df.shape[1]}")

    st.subheader("Numeric variable summary")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.dataframe(df[numeric_cols].describe().T.round(2), use_container_width=True)

    st.subheader("Raw data preview")
    st.dataframe(df.head(50), use_container_width=True)

# --- TAB 2: DESCRIPTIVE CHARTS -----------------------------------------
with tab_charts:
    col1, col2 = st.columns(2)

    with col1:
        counts = df['satisfaction'].value_counts().reset_index()
        counts.columns = ['satisfaction', 'count']
        fig = px.bar(counts, x='satisfaction', y='count', color='satisfaction',
                     text='count', title="Distribution of maternal satisfaction")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'modeofdeliveryrecode' in df.columns:
            ct = pd.crosstab(df['modeofdeliveryrecode'], df['satisfaction'], normalize='index') * 100
            ct = ct.reset_index().melt(id_vars='modeofdeliveryrecode', var_name='satisfaction', value_name='pct')
            fig = px.bar(ct, x='modeofdeliveryrecode', y='pct', color='satisfaction', barmode='group',
                         title="Satisfaction by mode of delivery (%)")
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if 'q101age' in df.columns:
            fig = px.histogram(df, x='q101age', nbins=15, title="Maternal age distribution")
            st.plotly_chart(fig, use_container_width=True)
    with col4:
        if 'q107mincom' in df.columns:
            fig = px.box(df, x='satisfaction', y='q107mincom', color='satisfaction',
                         title="Monthly income by satisfaction status")
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Explore any categorical variable")
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    chosen = st.selectbox("Choose a variable", [c for c in cat_cols if c != 'satisfaction'])
    if chosen:
        ct = pd.crosstab(df[chosen], df['satisfaction'], normalize='index') * 100
        ct = ct.reset_index().melt(id_vars=chosen, var_name='satisfaction', value_name='pct')
        fig = px.bar(ct, x=chosen, y='pct', color='satisfaction', barmode='group',
                     title=f"Satisfaction by {chosen} (%)")
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: MODEL PERFORMANCE ------------------------------------------
with tab_models:
    model_choice = st.radio(
        "Choose model",
        ["Model A — full (includes service-domain satisfaction items)",
         "Model B — conservative (socio-demographic / obstetric / facility only)"],
        horizontal=False,
    )
    result = model_a if model_choice.startswith("Model A") else model_b

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    for col, (name, val) in zip([m1, m2, m3, m4, m5, m6], result['metrics'].items()):
        col.metric(name, f"{val:.3f}")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result['fpr'], y=result['tpr'], mode='lines',
                                  name=f"AUC = {result['metrics']['AUC-ROC']:.3f}"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'),
                                  name='Chance'))
        fig.update_layout(title="ROC curve", xaxis_title="False Positive Rate",
                           yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        cm = result['cm']
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                         x=['not satisfied', 'satisfied'], y=['not satisfied', 'satisfied'],
                         labels=dict(x="Predicted", y="Actual"), title="Confusion matrix")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top predictors (odds ratio)")
    top = result['coef_table'].head(10).iloc[::-1]
    fig = px.bar(top, x='odds_ratio', y='feature', orientation='h',
                 color=(top['coefficient'] > 0).map({True: 'increases odds', False: 'decreases odds'}),
                 title="Top 10 predictors by odds ratio")
    fig.add_vline(x=1, line_dash='dash', line_color='gray')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Permutation importance (drop in AUC when shuffled)")
    topp = result['perm_table'].head(10).iloc[::-1]
    fig = px.bar(topp, x='importance', y='feature', orientation='h',
                 title="Top 10 predictors by permutation importance")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full coefficient / odds-ratio table"):
        st.dataframe(result['coef_table'].round(3), use_container_width=True)

# --- TAB 4: PREDICT ------------------------------------------------------
with tab_predict:
    st.subheader("Score a hypothetical mother")
    predict_model_choice = st.radio(
        "Model to use for prediction",
        ["Model A — full", "Model B — conservative"], horizontal=True, key="predict_model")
    result = model_a if predict_model_choice.startswith("Model A") else model_b
    X_ref = result['X']

    N_TOP = 12
    # Rank original (pre-encoding) columns by permutation importance so the
    # quick form only asks about the features that actually move the
    # prediction - showing all 23-38 raw fields at once overwhelms users.
    ranked_features = result['perm_table'].sort_values('importance', ascending=False)['feature'].tolist()
    top_features = ranked_features[:N_TOP]
    remaining_features = [c for c in X_ref.columns if c not in top_features]

    st.caption(f"Showing the top {N_TOP} most influential features for this model. "
               f"The remaining {len(remaining_features)} are set to the dataset's typical "
               f"(median/most-common) value - expand 'More options' below to adjust them too.")

    def render_field(col_name, target):
        if col_name in result['numeric_features']:
            default = float(X_ref[col_name].median())
            return target.number_input(col_name, value=default, key=f"{predict_model_choice}_{col_name}")
        else:
            options = sorted(X_ref[col_name].dropna().unique().tolist())
            return target.selectbox(col_name, options, index=0, key=f"{predict_model_choice}_{col_name}")

    input_data = {}
    cols = st.columns(3)
    for i, col_name in enumerate(top_features):
        input_data[col_name] = render_field(col_name, cols[i % 3])

    # Defaults for the fields not shown in the quick form
    for col_name in remaining_features:
        if col_name in result['numeric_features']:
            input_data[col_name] = float(X_ref[col_name].median())
        else:
            input_data[col_name] = X_ref[col_name].mode()[0]

    with st.expander(f"More options ({len(remaining_features)} additional features)"):
        adv_cols = st.columns(3)
        for i, col_name in enumerate(remaining_features):
            input_data[col_name] = render_field(col_name, adv_cols[i % 3])

    if st.button("Predict satisfaction", type="primary"):
        new_row = pd.DataFrame([input_data])[X_ref.columns]  # preserve original column order
        proba = result['pipeline'].predict_proba(new_row)[0, 1]
        label = "Satisfied 😊" if proba >= 0.5 else "Not satisfied 😟"
        st.success(f"**Prediction: {label}**")
        st.metric("Predicted probability of satisfaction", f"{proba*100:.1f}%")
        st.progress(min(max(proba, 0.0), 1.0))

st.sidebar.markdown("---")
st.sidebar.caption("Project 2 — Maternal Satisfaction ML Explorer")
