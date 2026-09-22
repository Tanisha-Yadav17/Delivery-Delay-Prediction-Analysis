"""
Delivery Delay Prediction — Interactive Streamlit Dashboard
4-Tier Analytics Ladder | Logistics & Supply Chain Analytics
"""

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import json
import os
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Delivery Delay Prediction Dashboard",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .metric-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 16px 20px;
        border-left: 4px solid #3b82d4;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        margin-bottom: 10px;
    }
    .metric-card.red  { border-left-color: #E05C5C; }
    .metric-card.green { border-left-color: #28a745; }
    .metric-card.orange { border-left-color: #fd7e14; }
    .metric-card.purple { border-left-color: #7c5cd8; }
    .metric-label { font-size: 13px; color: #6c757d; font-weight: 500; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: 700; color: #1f2328; }
    .metric-sub   { font-size: 12px; color: #6c757d; margin-top: 2px; }
    .section-header {
        font-size: 18px; font-weight: 700; color: #1f2328;
        border-bottom: 2px solid #e5e7eb; padding-bottom: 6px;
        margin: 20px 0 14px 0;
    }
    .insight-box {
        background: #f0f4ff; border-left: 4px solid #3b82d4;
        border-radius: 4px; padding: 12px 16px; margin: 8px 0;
        font-size: 13.5px; color: #1f2328; line-height: 1.6;
    }
    .insight-box.warning { background: #fff8e1; border-left-color: #fd7e14; }
    .insight-box.danger  { background: #fff0f0; border-left-color: #E05C5C; }
    .insight-box.success { background: #f0fff4; border-left-color: #28a745; }
    .rec-card {
        background: #ffffff; border: 1px solid #e5e7eb;
        border-radius: 8px; padding: 14px 18px; margin: 8px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .rec-title { font-weight: 700; font-size: 14px; color: #1f2328; }
    .rec-body  { font-size: 13px; color: #4a5568; margin-top: 4px; line-height: 1.5; }
    .tier-badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 11px; font-weight: 600; margin-right: 6px;
    }
    .badge-1 { background: #dbeafe; color: #1e40af; }
    .badge-2 { background: #dcfce7; color: #166534; }
    .badge-3 { background: #fef3c7; color: #92400e; }
    .badge-4 { background: #f3e8ff; color: #6b21a8; }
    h1 { color: #1f2328 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data & Model Loading
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('Delivery_Logistics.csv')
    # Apply same cleaning
    df.drop(columns=['delivery_time_hours','expected_time_hours'], inplace=True)
    str_cols = df.select_dtypes(include='object').columns.tolist()
    for col in str_cols:
        df[col] = df[col].str.lower().str.strip()
    df['Delay_Flag'] = (df['delayed'] == 'yes').astype(int)
    df.drop(columns=['delayed','delivery_status'], inplace=True)
    df = df.reset_index(drop=True)
    df['delivery_id'] = df.index + 1
    return df

@st.cache_resource
def load_model():
    if os.path.exists('rf_delay_model.pkl'):
        return joblib.load('rf_delay_model.pkl')
    return None

@st.cache_data
def load_metrics():
    if os.path.exists('model_metrics.json'):
        with open('model_metrics.json') as f:
            return json.load(f)
    return {}

df      = load_data()
model   = load_model()
metrics = load_metrics()

delay_rate = df['Delay_Flag'].mean() * 100

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/delivery.png", width=64)
st.sidebar.title("Delivery Delay\nPrediction")
st.sidebar.caption("4-Tier Analytics Dashboard")

pages = [
    "🏠 Executive Overview",
    "📊 EDA — Partner & Vehicle",
    "🌍 EDA — Region & Weather",
    "📦 EDA — Distance & Mode",
    "⭐ EDA — Package & Rating",
    "🔥 EDA — Risk Heatmap",
    "🤖 ML Model Performance",
    "🎯 Prescriptive Strategies"
]
page = st.sidebar.radio("Navigate", pages)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset Overview**")
st.sidebar.metric("Total Records", f"{metrics.get('total_records', 25000):,}")
st.sidebar.metric("Overall Delay Rate", f"{metrics.get('delay_rate', 26.68):.2f}%")
st.sidebar.metric("Model ROC-AUC", f"{metrics.get('roc_auc', 0.9668):.4f}")
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · Random Forest · scikit-learn")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: plot_style
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'DejaVu Sans', 'axes.titlesize': 12, 'axes.labelsize': 10,
})

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — EXECUTIVE OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Executive Overview":
    st.title("🚚 Delivery Delay Prediction Dashboard")
    st.markdown("""
    <div style='background:#eef4ff;border-left:4px solid #3b82d4;padding:12px 18px;border-radius:4px;margin-bottom:20px'>
    <b>4-Tier Analytics Ladder</b> applied to 25,000 delivery records in the Logistics & Supply Chain domain.
    <span class='tier-badge badge-1'>Tier 1: Descriptive</span>
    <span class='tier-badge badge-2'>Tier 2: Diagnostic</span>
    <span class='tier-badge badge-3'>Tier 3: Predictive</span>
    <span class='tier-badge badge-4'>Tier 4: Prescriptive</span>
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards Row 1
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Total Deliveries</div>
        <div class='metric-value'>{metrics.get('total_records',25000):,}</div>
        <div class='metric-sub'>Full dataset records</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card red'>
        <div class='metric-label'>Delayed Deliveries</div>
        <div class='metric-value'>{metrics.get('delayed_count',6669):,}</div>
        <div class='metric-sub'>{metrics.get('delay_rate',26.68):.2f}% of all shipments</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card green'>
        <div class='metric-label'>On-Time Deliveries</div>
        <div class='metric-value'>{metrics.get('ontime_count',18331):,}</div>
        <div class='metric-sub'>{100-metrics.get('delay_rate',26.68):.2f}% of all shipments</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card purple'>
        <div class='metric-label'>Avg Delivery Cost</div>
        <div class='metric-value'>₹{metrics.get('avg_cost',864.94):,.0f}</div>
        <div class='metric-sub'>Avg distance: {metrics.get('avg_distance',150.39):.1f} km</div></div>""", unsafe_allow_html=True)

    # KPI Cards Row 2
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>Model Accuracy</div>
        <div class='metric-value'>{metrics.get('accuracy',0.885)*100:.1f}%</div>
        <div class='metric-sub'>Random Forest classifier</div></div>""", unsafe_allow_html=True)
    with c6:
        st.markdown(f"""<div class='metric-card green'>
        <div class='metric-label'>ROC-AUC Score</div>
        <div class='metric-value'>{metrics.get('roc_auc',0.9668):.4f}</div>
        <div class='metric-sub'>5-Fold CV: {metrics.get('cv_auc_mean',0.9667):.4f} ± {metrics.get('cv_auc_std',0.0017):.4f}</div></div>""", unsafe_allow_html=True)
    with c7:
        st.markdown(f"""<div class='metric-card orange'>
        <div class='metric-label'>Model Recall</div>
        <div class='metric-value'>{metrics.get('recall',0.967)*100:.1f}%</div>
        <div class='metric-sub'>Catches 96.7% of real delays</div></div>""", unsafe_allow_html=True)
    with c8:
        st.markdown(f"""<div class='metric-label' style='padding:16px;background:#fff;border-radius:8px;border:1px solid #e5e7eb;'>
        <div class='metric-label'>Avg Customer Rating</div>
        <div class='metric-value' style='font-size:28px;font-weight:700;color:#1f2328'>{metrics.get('avg_rating',3.67):.2f} / 5</div>
        <div class='metric-sub'>Across all 25,000 deliveries</div></div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Methodology: 4-Tier Analytics Ladder</div>", unsafe_allow_html=True)
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown("""<div style='background:#dbeafe;border-radius:8px;padding:14px;text-align:center'>
        <b style='color:#1e40af'>Tier 1 — Descriptive</b><br>
        <small>What happened? Overall delay rate (26.68%), partner rankings, volume distribution, regional breakdown.</small>
        </div>""", unsafe_allow_html=True)
    with t2:
        st.markdown("""<div style='background:#dcfce7;border-radius:8px;padding:14px;text-align:center'>
        <b style='color:#166534'>Tier 2 — Diagnostic</b><br>
        <small>Why it happened? Weather impact, distance bands, vehicle-type fit, mode SLA alignment.</small>
        </div>""", unsafe_allow_html=True)
    with t3:
        st.markdown("""<div style='background:#fef3c7;border-radius:8px;padding:14px;text-align:center'>
        <b style='color:#92400e'>Tier 3 — Predictive</b><br>
        <small>What will happen? Random Forest model with 96.6% AUC predicts shipment-level delay probability.</small>
        </div>""", unsafe_allow_html=True)
    with t4:
        st.markdown("""<div style='background:#f3e8ff;border-radius:8px;padding:14px;text-align:center'>
        <b style='color:#6b21a8'>Tier 4 — Prescriptive</b><br>
        <small>What to do? Risk-scored dispatch prioritisation, SLA renegotiation, vehicle-distance rules.</small>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Key Findings at a Glance</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box danger'>
    <b>Delivery Mode is the #1 Delay Driver:</b> Express mode shipments show a <b>73.78% delay rate</b> vs 0% for Standard.
    Delivery mode contributes >50% of total model feature importance, indicating systematic SLA over-commitment in express channels.
    </div>
    <div class='insight-box warning'>
    <b>Weather Amplifies Risk Significantly:</b> Stormy conditions produce a <b>41.45% delay rate</b> vs 16.02% in cold/clear conditions.
    Weather-adaptive dispatch protocols can reduce unnecessary SLA breaches during peak adverse events.
    </div>
    <div class='insight-box success'>
    <b>High-Recall Model Protects SLA:</b> The Random Forest achieves <b>96.7% Recall</b> — correctly flagging 1,290 out of 1,334 real delays.
    Only 44 delayed shipments (False Negatives) escape detection, minimising surprise SLA breaches.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Partner & Vehicle
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 EDA — Partner & Vehicle":
    st.title("📊 Chart 1 — Partner & Vehicle Delay Performance")
    st.caption("Tier 1 (Descriptive) + Tier 2 (Diagnostic)")

    if os.path.exists('chart1_partner_vehicle.png'):
        st.image('chart1_partner_vehicle.png', use_container_width=True)
    else:
        st.warning("Chart image not found. Run run_analysis.py first.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-header'>Delivery Partner Performance</div>", unsafe_allow_html=True)
        partner_data = metrics.get('partner_delay_rates', {})
        partner_df = pd.DataFrame.from_dict(partner_data, orient='index', columns=['Delay Rate (%)'])
        partner_df = partner_df.sort_values('Delay Rate (%)', ascending=False)
        partner_df['vs. Average'] = (partner_df['Delay Rate (%)'] - metrics.get('delay_rate', 26.68)).round(2)
        partner_df['Status'] = partner_df['Delay Rate (%)'].apply(
            lambda x: '🔴 Above Avg' if x > metrics.get('delay_rate', 26.68) else '🟢 Below Avg')
        st.dataframe(partner_df, use_container_width=True)

    with c2:
        st.markdown("<div class='section-header'>Vehicle Type Performance</div>", unsafe_allow_html=True)
        vehicle_data = metrics.get('vehicle_delay_rates', {})
        vehicle_df = pd.DataFrame.from_dict(vehicle_data, orient='index', columns=['Delay Rate (%)'])
        vehicle_df = vehicle_df.sort_values('Delay Rate (%)', ascending=False)
        vehicle_df['vs. Average'] = (vehicle_df['Delay Rate (%)'] - metrics.get('delay_rate', 26.68)).round(2)
        vehicle_df['Status'] = vehicle_df['Delay Rate (%)'].apply(
            lambda x: '🔴 Above Avg' if x > metrics.get('delay_rate', 26.68) else '🟢 Below Avg')
        st.dataframe(vehicle_df, use_container_width=True)

    st.markdown("""
    <div class='insight-box'>
    <b>Empirical Observations:</b><br>
    • <b>xpressbees</b> leads the delay ranking at 28.27%, while <b>delhivery</b> performs best at 24.80%.<br>
    • The spread across partners (3.47 pp) is narrow, suggesting systemic issues rather than purely partner-specific failures.<br>
    • Vehicle type delay rates cluster tightly (26.1%–27.0%), implying vehicle choice alone has limited independent impact on delays.
    </div>
    <div class='insight-box warning'>
    <b>Correlation vs. Causation:</b> Partner delay rates are <i>associations</i> — not proof of partner negligence.
    Confounding factors such as regional coverage, package mix assigned to each partner, and distance band allocations
    all influence measured outcomes. A controlled experiment or regression controlling for these factors would be needed
    to establish causal responsibility.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Region & Weather
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🌍 EDA — Region & Weather":
    st.title("🌍 Chart 2 — Regional & Weather Delay Performance")
    st.caption("Tier 1 (Descriptive) + Tier 2 (Diagnostic)")

    if os.path.exists('chart2_region_weather.png'):
        st.image('chart2_region_weather.png', use_container_width=True)
    else:
        st.warning("Chart image not found. Run run_analysis.py first.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='section-header'>Regional Delay Rates</div>", unsafe_allow_html=True)
        region_data = metrics.get('region_delay_rates', {})
        region_df   = pd.DataFrame.from_dict(region_data, orient='index', columns=['Delay Rate (%)'])
        region_df   = region_df.sort_values('Delay Rate (%)', ascending=False).reset_index()
        region_df.columns = ['Region', 'Delay Rate (%)']
        region_df['Risk Level'] = region_df['Delay Rate (%)'].apply(
            lambda x: '🔴 High' if x > 27 else ('🟡 Medium' if x > 26 else '🟢 Low'))
        st.dataframe(region_df, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("<div class='section-header'>Weather Condition Delay Rates</div>", unsafe_allow_html=True)
        weather_data = metrics.get('weather_delay_rates', {})
        weather_df   = pd.DataFrame.from_dict(weather_data, orient='index', columns=['Delay Rate (%)'])
        weather_df   = weather_df.sort_values('Delay Rate (%)', ascending=False).reset_index()
        weather_df.columns = ['Weather', 'Delay Rate (%)']
        weather_df['Risk Level'] = weather_df['Delay Rate (%)'].apply(
            lambda x: '🔴 High' if x > 30 else ('🟡 Medium' if x > 20 else '🟢 Low'))
        st.dataframe(weather_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class='insight-box danger'>
    <b>Weather is the Strongest Binary Risk Amplifier:</b><br>
    • <b>Stormy</b> conditions: 41.45% delay rate (55% above average)<br>
    • <b>Rainy</b> conditions: 37.35% delay rate (40% above average)<br>
    • <b>Clear/Hot/Cold</b> conditions: 16–17% delay rate — significantly below average
    </div>
    <div class='insight-box'>
    <b>Regional Insight:</b> Regional variation (25.8%–27.25%) is much smaller than weather variation,
    suggesting that region primarily acts as a proxy for infrastructure quality and partner coverage,
    while weather is the more operationally actionable variable.
    </div>
    <div class='insight-box warning'>
    <b>Correlation vs. Causation:</b> While stormy/rainy weather <i>correlates</i> strongly with delays,
    a causal chain is plausible (road hazards, reduced visibility, transit disruptions) but the dataset
    does not include route-level data to confirm the mechanism. Confounders: regions prone to storms
    may also have weaker infrastructure.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — Distance & Mode
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📦 EDA — Distance & Mode":
    st.title("📦 Chart 3 — Distance Distribution & Delivery Mode")
    st.caption("Tier 1 (Descriptive) + Tier 2 (Diagnostic)")

    if os.path.exists('chart3_distance_mode.png'):
        st.image('chart3_distance_mode.png', use_container_width=True)
    else:
        st.warning("Chart image not found. Run run_analysis.py first.")

    st.markdown("<div class='section-header'>Delivery Mode Delay Rates</div>", unsafe_allow_html=True)
    mode_data = metrics.get('mode_delay_rates', {})
    mode_df   = pd.DataFrame.from_dict(mode_data, orient='index', columns=['Delay Rate (%)'])
    mode_df   = mode_df.sort_values('Delay Rate (%)', ascending=False).reset_index()
    mode_df.columns = ['Delivery Mode', 'Delay Rate (%)']
    mode_df['Interpretation'] = [
        'Express SLAs systematically over-committed vs. capacity',
        'Same-day commitments strain last-mile routing',
        'Sufficient lead time — near-zero delays',
        'Maximum flexibility — zero recorded delays'
    ]
    st.dataframe(mode_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class='insight-box danger'>
    <b>Critical Finding — Delivery Mode Drives Most Delays:</b><br>
    • <b>Express mode: 73.78% delay rate</b> — this is extraordinarily high and indicates the express SLA
    commitment is not backed by sufficient operational capacity.<br>
    • <b>Standard mode: 0.0% delay rate</b> — adequate lead time eliminates virtually all delays.<br>
    • This single factor explains why delivery_mode features account for >52% of the model's total feature importance.
    </div>
    <div class='insight-box warning'>
    <b>Distance Observation:</b> Mean distance for delayed shipments is marginally higher than on-time,
    but the distribution overlap is substantial. Distance alone is a weak predictor — its effect is
    mediated by delivery mode and vehicle type assignment.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — Package & Rating
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⭐ EDA — Package & Rating":
    st.title("⭐ Chart 4 — Package Type & Customer Rating Analysis")
    st.caption("Tier 1 (Descriptive) + Tier 2 (Diagnostic)")

    if os.path.exists('chart4_package_rating.png'):
        st.image('chart4_package_rating.png', use_container_width=True)
    else:
        st.warning("Chart image not found. Run run_analysis.py first.")

    st.markdown("<div class='section-header'>Package Type Delay Rates</div>", unsafe_allow_html=True)
    pkg_data = metrics.get('package_delay_rates', {})
    pkg_df   = pd.DataFrame.from_dict(pkg_data, orient='index', columns=['Delay Rate (%)'])
    pkg_df   = pkg_df.sort_values('Delay Rate (%)', ascending=False).reset_index()
    pkg_df.columns = ['Package Type', 'Delay Rate (%)']
    pkg_df['Risk Flag'] = pkg_df['Delay Rate (%)'].apply(
        lambda x: '🔴 Above Avg' if x > metrics.get('delay_rate', 26.68) else '🟢 Below Avg')
    st.dataframe(pkg_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class='insight-box'>
    <b>Package Type Observations:</b><br>
    • Package type delay rates cluster tightly (24.76%–27.54%), suggesting package type has limited independent predictive power.<br>
    • <b>Pharmacy</b> (27.54%) and <b>Groceries</b> (27.44%) show slightly elevated risk — both are time-sensitive categories
    where delays have disproportionate customer impact.<br>
    • <b>Furniture</b> (24.76%) — typically Standard or Two-Day mode — has the lowest delay rate, consistent with adequate lead time.
    </div>
    <div class='insight-box warning'>
    <b>Rating-Delay Correlation:</b> While delayed deliveries tend to attract lower ratings, the relationship is imperfect.
    Some customers rate delayed deliveries highly (proactive communication) while some on-time deliveries receive poor ratings
    (damaged packaging, wrong items). Rating should NOT be used as a proxy for delay in predictive models — it is a post-event outcome.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — Risk Heatmap
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔥 EDA — Risk Heatmap":
    st.title("🔥 Chart 5 — Region × Weather Risk Heatmap & Cost Analysis")
    st.caption("Tier 2 (Diagnostic) — Multi-dimensional risk identification")

    if os.path.exists('chart5_heatmap_cost.png'):
        st.image('chart5_heatmap_cost.png', use_container_width=True)
    else:
        st.warning("Chart image not found. Run run_analysis.py first.")

    # Interactive heatmap table
    st.markdown("<div class='section-header'>Region × Weather Delay Rate Table (%)</div>", unsafe_allow_html=True)
    heatmap_data = df.groupby(['region','weather_condition'])['Delay_Flag'].mean()*100
    heatmap_pivot = heatmap_data.unstack(fill_value=0).round(1)
    st.dataframe(heatmap_pivot.style.background_gradient(cmap='RdYlGn_r', axis=None), use_container_width=True)

    # Find top risk cells
    top_cells = heatmap_data.sort_values(ascending=False).head(5).reset_index()
    top_cells.columns = ['Region','Weather','Delay Rate (%)']
    top_cells['Delay Rate (%)'] = top_cells['Delay Rate (%)'].round(1)

    st.markdown("<div class='section-header'>Top 5 Highest-Risk Region × Weather Combinations</div>", unsafe_allow_html=True)
    st.dataframe(top_cells, use_container_width=True, hide_index=True)

    st.markdown(f"""
    <div class='insight-box danger'>
    <b>Operational Risk Hotspots:</b><br>
    • The highest-risk combinations are concentrated around <b>stormy and rainy weather across all regions</b>.<br>
    • These cells should trigger automatic SLA buffer extensions and proactive customer notifications.<br>
    • Targeting the top 5 risk cells alone covers a significant proportion of avoidable delays.
    </div>
    <div class='insight-box'>
    <b>Cost vs. Delay:</b> The boxplot shows that delivery cost distributions for delayed vs. on-time shipments overlap
    substantially. Delayed shipments are not predominantly expensive — cost is driven by delivery mode and distance,
    not delay status. Higher-cost deliveries do not have structurally higher delay rates once mode is controlled.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 7 — ML Model Performance
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 ML Model Performance":
    st.title("🤖 Tier 3 — Predictive ML Model Performance")
    st.caption("Random Forest Classifier | Target: Delay_Flag | Train/Test Split: 80/20 Stratified")

    # Metrics Row
    c1, c2, c3, c4, c5 = st.columns(5)
    metric_config = [
        (c1, "Accuracy",  metrics.get('accuracy', 0.885),  "", ""),
        (c2, "Precision", metrics.get('precision', 0.7084), "", "orange"),
        (c3, "Recall",    metrics.get('recall', 0.967),    "", "green"),
        (c4, "F1-Score",  metrics.get('f1', 0.8177),       "", ""),
        (c5, "ROC-AUC",   metrics.get('roc_auc', 0.9668),  "", "purple"),
    ]
    for col, label, val, sub, color in metric_config:
        card_class = f"metric-card {color}" if color else "metric-card"
        with col:
            st.markdown(f"""<div class='{card_class}'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{val:.4f}</div>
            </div>""", unsafe_allow_html=True)

    # Charts
    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("<div class='section-header'>Confusion Matrix & ROC Curve</div>", unsafe_allow_html=True)
        if os.path.exists('chart_roc_confusion.png'):
            st.image('chart_roc_confusion.png', use_container_width=True)

    with c_right:
        st.markdown("<div class='section-header'>Feature Importance (Top 20)</div>", unsafe_allow_html=True)
        if os.path.exists('chart_feature_importance.png'):
            st.image('chart_feature_importance.png', use_container_width=True)

    # Confusion matrix breakdown
    st.markdown("<div class='section-header'>Confusion Matrix Breakdown</div>", unsafe_allow_html=True)
    tn_v = metrics.get('tn', 3135)
    fp_v = metrics.get('fp', 531)
    fn_v = metrics.get('fn', 44)
    tp_v = metrics.get('tp', 1290)

    cm_c1, cm_c2, cm_c3, cm_c4 = st.columns(4)
    with cm_c1:
        st.markdown(f"""<div class='metric-card green'>
        <div class='metric-label'>True Positives (TP)</div>
        <div class='metric-value'>{tp_v:,}</div>
        <div class='metric-sub'>Delayed → correctly flagged</div></div>""", unsafe_allow_html=True)
    with cm_c2:
        st.markdown(f"""<div class='metric-card red'>
        <div class='metric-label'>False Negatives (FN)</div>
        <div class='metric-value'>{fn_v:,}</div>
        <div class='metric-sub'>Delayed → missed (HIGH COST)</div></div>""", unsafe_allow_html=True)
    with cm_c3:
        st.markdown(f"""<div class='metric-card orange'>
        <div class='metric-label'>False Positives (FP)</div>
        <div class='metric-value'>{fp_v:,}</div>
        <div class='metric-sub'>On-time → flagged as delayed</div></div>""", unsafe_allow_html=True)
    with cm_c4:
        st.markdown(f"""<div class='metric-card'>
        <div class='metric-label'>True Negatives (TN)</div>
        <div class='metric-value'>{tn_v:,}</div>
        <div class='metric-sub'>On-time → correctly cleared</div></div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class='insight-box danger'>
    <b>False Negatives (FN = 44) — HIGH BUSINESS COST:</b><br>
    A shipment predicted as on-time but actually delayed results in an <b>unannounced SLA breach</b>.
    This causes customer complaints, refund liability, potential churn, and brand damage.
    In logistics, reactive recovery (re-routing, penalty payments, customer appeasement) is 3–5x
    more expensive than proactive intervention. <b>Minimising FN is the #1 model objective.</b>
    </div>
    <div class='insight-box warning'>
    <b>False Positives (FP = 531) — MANAGEABLE OVERHEAD:</b><br>
    A shipment predicted as delayed but delivered on time results in an <b>unnecessary proactive alert</b>
    or expedite action. This creates operational overhead (extra capacity allocated, spurious customer
    notifications) but is far cheaper than an undetected FN. With 96.7% Recall, the model is highly
    tuned toward FN minimisation — the right trade-off for logistics SLA management.
    </div>
    <div class='insight-box success'>
    <b>Cross-Validation:</b> 5-Fold Stratified CV ROC-AUC = {cv_auc_mean:.4f} ± {cv_auc_std:.4f} —
    model is stable and not overfit to the test set.
    </div>
    """.format(
        cv_auc_mean=metrics.get('cv_auc_mean', 0.9667),
        cv_auc_std=metrics.get('cv_auc_std', 0.0017)
    ), unsafe_allow_html=True)

    # Top Features Table
    st.markdown("<div class='section-header'>Top 10 Predictive Features</div>", unsafe_allow_html=True)
    top_feat = metrics.get('top_features', {})
    feat_df  = pd.DataFrame.from_dict(top_feat, orient='index', columns=['Importance Score'])
    feat_df  = feat_df.sort_values('Importance Score', ascending=False).reset_index()
    feat_df.columns = ['Feature', 'Importance Score']
    feat_df['Importance Score'] = feat_df['Importance Score'].round(4)
    feat_df['Rank'] = range(1, len(feat_df)+1)
    st.dataframe(feat_df[['Rank','Feature','Importance Score']], use_container_width=True, hide_index=True)

    # Leakage protection note
    with st.expander("Target Leakage Prevention Policy"):
        st.markdown("""
        The following columns were explicitly **excluded** from the predictor feature set to prevent data leakage:

        | Column | Reason for Exclusion |
        |--------|---------------------|
        | `delayed` | Direct source of the target variable (`Delay_Flag`) |
        | `delivery_status` | Direct derivative of the target — encodes delay/delivered/failed post-event |
        | `delivery_rating` | Post-delivery customer rating — unavailable at dispatch time |
        | `delivery_id` | Raw identifier with no predictive signal |
        | `delivery_time_hours` | All-zero column — zero variance, no information |
        | `expected_time_hours` | All-zero column — zero variance, no information |

        **Retained features** (`delivery_cost`, `distance_km`, `package_weight_kg`, and all categorical fields)
        are all known at dispatch time and do not encode post-delivery outcomes.
        """)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 8 — Prescriptive Strategies
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🎯 Prescriptive Strategies":
    st.title("🎯 Tier 4 — Prescriptive Strategies & Operational Levers")
    st.caption("Resource-constrained dispatch rules and risk mitigation strategies")

    st.markdown("""
    <div style='background:#f3e8ff;border-left:4px solid #7c5cd8;padding:12px 18px;border-radius:4px;margin-bottom:20px'>
    <b>Operational Rule:</b> When delivery workload is high and available capacity is limited,
    prioritise high-risk or time-sensitive deliveries and assign the most suitable available
    delivery partners and vehicles based on <b>distance band, region, vehicle type,
    weather condition, and predicted delay probability</b>.
    </div>
    """, unsafe_allow_html=True)

    # Priority Scoring Logic
    with st.expander("Priority Score Formula", expanded=True):
        st.markdown("""
        **Composite Priority Score = 0.60 × Delay Probability + 0.40 × (Mode Urgency / 4)**

        | Delivery Mode | Urgency Weight |
        |---------------|---------------|
        | Same Day      | 4 (highest)   |
        | Express       | 3             |
        | Two Day       | 2             |
        | Standard      | 1             |

        Shipments are sorted by priority score (descending) and processed within capacity limits.
        """)

    # Live Dispatch Simulator
    st.markdown("<div class='section-header'>Live Dispatch Risk Simulator</div>", unsafe_allow_html=True)
    st.caption("Configure a sample shipment to get real-time delay probability and recommended action")

    if model:
        col1, col2, col3 = st.columns(3)
        with col1:
            partner    = st.selectbox("Delivery Partner", sorted(df['delivery_partner'].unique()))
            pkg_type   = st.selectbox("Package Type", sorted(df['package_type'].unique()))
            vehicle    = st.selectbox("Vehicle Type", sorted(df['vehicle_type'].unique()))
        with col2:
            mode       = st.selectbox("Delivery Mode", sorted(df['delivery_mode'].unique()))
            region     = st.selectbox("Region", sorted(df['region'].unique()))
            weather    = st.selectbox("Weather Condition", sorted(df['weather_condition'].unique()))
        with col3:
            distance   = st.slider("Distance (km)", 5, 300, 150)
            weight     = st.slider("Package Weight (kg)", 1, 50, 20)
            cost       = st.slider("Delivery Cost (₹)", 100, 1650, 800)

        if st.button("Predict Delay Risk", type="primary"):
            input_df = pd.DataFrame([{
                'delivery_partner': partner.lower(),
                'package_type':     pkg_type.lower(),
                'vehicle_type':     vehicle.lower(),
                'delivery_mode':    mode.lower(),
                'region':           region.lower(),
                'weather_condition':weather.lower(),
                'distance_km':      distance,
                'package_weight_kg':weight,
                'delivery_cost':    cost
            }])
            prob  = model.predict_proba(input_df)[0, 1]
            label = "DELAYED" if prob >= 0.50 else "ON-TIME"

            if prob >= 0.60:
                risk_color = "#E05C5C"
                risk_label = "HIGH RISK"
                action = "ESCALATE: Assign premium partner + send proactive customer alert immediately"
            elif prob >= 0.35:
                risk_color = "#fd7e14"
                risk_label = "MEDIUM RISK"
                action = "MONITOR: Assign reliable partner; enable real-time tracking; prepare buffer"
            else:
                risk_color = "#28a745"
                risk_label = "LOW RISK"
                action = "STANDARD: Normal dispatch; no additional intervention required"

            # Vehicle suitability check
            vehicle_note = ""
            if distance > 150 and vehicle in ['bike','ev bike','scooter']:
                vehicle_note = " | ⚠️ VEHICLE MISMATCH: Long-haul requires truck/van — reassign"
            elif distance <= 50 and vehicle in ['truck','van','ev van']:
                vehicle_note = " | 💡 OPTIMISE: Short-haul — consider bike/scooter to reduce cost"

            st.markdown(f"""
            <div style='background:{risk_color}15;border:2px solid {risk_color};border-radius:8px;padding:20px;margin-top:10px'>
                <div style='font-size:24px;font-weight:700;color:{risk_color}'>{risk_label} — {prob*100:.1f}% Delay Probability</div>
                <div style='font-size:16px;color:#1f2328;margin-top:6px'>Predicted outcome: <b>{label}</b></div>
                <div style='font-size:14px;color:#4a5568;margin-top:8px'><b>Recommended Action:</b> {action}{vehicle_note}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("ML model not found. Run run_analysis.py to generate rf_delay_model.pkl")

    # Business Recommendations
    st.markdown("<div class='section-header'>7 Specific Business Recommendations</div>", unsafe_allow_html=True)

    recs = [
        ("1. High-Risk Partner SLA Renegotiation", "#3b82d4",
         "Partners with delay rates >30% above the fleet average (currently xpressbees at 28.27%, ekart at 27.42%) "
         "must enter quarterly SLA performance reviews. Implement financial penalties at 0.5% of contract value per "
         "percentage point above agreed delay threshold. Trigger contract renegotiation after 2 consecutive quarters of breach."),
        ("2. Weather-Adaptive Dispatch Protocol", "#fd7e14",
         "For all shipments dispatched during stormy (41.45% delay rate) or rainy (37.35%) conditions: "
         "automatically extend estimated delivery time by 20–35%, trigger pre-emptive customer notifications "
         "within 2 hours of dispatch, and reduce same-day delivery acceptance to 70% of normal capacity to protect SLA commitments."),
        ("3. Express Mode Capacity Rebalancing", "#E05C5C",
         "Express mode's 73.78% delay rate signals systematic over-commitment. Immediate actions: "
         "(a) Cap express order acceptance at 85% of verified real-time partner capacity, "
         "(b) Introduce dynamic pricing (+15%) for express during peak adverse-weather days, "
         "(c) Conduct root-cause analysis — identify if delays occur in first-mile, mid-mile, or last-mile."),
        ("4. Vehicle-Distance Optimisation Rule", "#28a745",
         "Enforce hard routing rules in the TMS: Bikes/ev-bikes/scooters assigned ONLY for routes <60 km. "
         "Trucks/vans/ev-vans required for routes >150 km. Mixed-haul (60–150 km): allow any vehicle. "
         "Mismatched assignments correlate with elevated delays and unnecessary fuel/maintenance cost."),
        ("5. Real-Time Risk Scoring at Dispatch", "#7c5cd8",
         "Integrate the Random Forest model (ROC-AUC 0.9668) into the TMS/WMS dispatch workflow. "
         "Flag all shipments with delay_probability >0.60 for mandatory human review before dispatch confirmation. "
         "Estimated impact: prevents ~96.7% of delay SLA breaches (model Recall) with early intervention."),
        ("6. Pharmacy & Perishables Priority Protocol", "#fd7e14",
         "Pharmacy (27.54% delay rate) and Groceries (27.44%) are time-sensitive package types. "
         "When delay_probability >0.40 for these categories, automatically: (a) upgrade to next faster delivery mode, "
         "(b) assign only top-3 performers from the partner pool, (c) set proactive contact threshold at T-2 hours before ETA."),
        ("7. High-Risk Region Infrastructure Investment", "#3b82d4",
         "Central (27.25%) and West (26.95%) regions show above-average delay rates. "
         "Recommended investments: (a) establish micro-fulfillment hubs in high-demand zip codes, "
         "(b) diversify partner contracts to at least 3 active partners per region, "
         "(c) negotiate dedicated last-mile capacity contracts with at least one regional carrier per zone."),
    ]

    for title, color, body in recs:
        st.markdown(f"""
        <div class='rec-card' style='border-left:4px solid {color}'>
            <div class='rec-title' style='color:{color}'>{title}</div>
            <div class='rec-body'>{body}</div>
        </div>
        """, unsafe_allow_html=True)

    # Risk tier summary
    if os.path.exists('Delivery_Logistics_Scored.csv'):
        st.markdown("<div class='section-header'>Current Shipment Risk Tier Distribution</div>", unsafe_allow_html=True)
        scored = pd.read_csv('Delivery_Logistics_Scored.csv')
        risk_counts = scored['risk_tier'].value_counts().reset_index()
        risk_counts.columns = ['Risk Tier', 'Count']
        risk_counts['% of Total'] = (risk_counts['Count'] / len(scored) * 100).round(1)
        st.dataframe(risk_counts, use_container_width=True, hide_index=True)
