import streamlit as st
import pandas as pd
import pickle

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Marketing Campaign Suite", layout="centered")

# ── LOAD MODELS ──────────────────────────────────────────────────────────────
# We load both models once and cache them so the app doesn't reload on every click

@st.cache_resource
def load_revenue_model():
    with open("revenue_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_classifier():
    with open("profit_loss_classifier.pkl", "rb") as f:
        return pickle.load(f)

try:
    revenue_model = load_revenue_model()
except FileNotFoundError:
    st.error("⚠️ revenue_model.pkl not found — run pipeline.ipynb first!")
    st.stop()

try:
    classifier = load_classifier()
except FileNotFoundError:
    st.error("⚠️ profit_loss_classifier.pkl not found — run pipeline.ipynb first!")
    st.stop()

# ── ENCODING MAPS ────────────────────────────────────────────────────────────
# These must match exactly what was used in pipeline.ipynb during training

brand_map    = {"Nykaa": 0, "Purplle": 1, "Tira": 2}
camp_map     = {"Email": 0, "Influencer": 1, "Paid Ads": 2, "SEO": 3, "Social Media": 4}
audience_map = {"College Students": 0, "Premium Shoppers": 1,
                "Tier 2 City Customers": 2, "Working Women": 3, "Youth": 4}
lang_map     = {"Bengali": 0, "English": 1, "Hindi": 2, "Tamil": 3}
channel_opts = ['Email', 'Facebook', 'Google', 'Instagram', 'WhatsApp', 'YouTube']

# ── HEADER ───────────────────────────────────────────────────────────────────
st.title("Marketing Campaign Predictor")
st.caption("Nykaa · Purplle · Tira")
st.divider()

# ── TABS — Revenue comes first, then Profit/Loss ──────────────────────────────
tab1, tab2 = st.tabs(["📈 Revenue Prediction", "📊 Profit / Loss Classification"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — REVENUE PREDICTION
# Model: RandomForestRegressor (revenue_model.pkl)
# Features: 17 (same as classifier but WITHOUT Revenue — Revenue is the target)
# Output: Predicted Revenue in ₹
# ════════════════════════════════════════════════════════════════════════════════
with tab1:

    st.markdown("Enter campaign details — the model will predict the expected **Revenue (₹)**.")
    st.divider()

    # Row 1 — Dropdowns
    c1, c2, c3, c4 = st.columns(4)
    brand_r    = c1.selectbox("Brand",           list(brand_map),    key="brand_r")
    camp_r     = c2.selectbox("Campaign Type",   list(camp_map),     key="camp_r")
    audience_r = c3.selectbox("Target Audience", list(audience_map), key="aud_r")
    language_r = c4.selectbox("Language",        list(lang_map),     key="lang_r")

    # Row 2 — Numbers
    st.markdown("### Campaign Metrics")
    left_r, right_r = st.columns(2)

    with left_r:
        duration_r    = st.number_input("Duration (days)", min_value=1,   value=15,    key="dur_r")
        impressions_r = st.number_input("Impressions",     min_value=0,   value=50000, step=1000, key="imp_r")
        clicks_r      = st.number_input("Clicks",          min_value=0,   value=3000,  step=100,  key="clk_r")
        leads_r       = st.number_input("Leads",           min_value=0,   value=500,   step=50,   key="ld_r")
        conversions_r = st.number_input("Conversions",     min_value=0,   value=200,   step=10,   key="conv_r")

    with right_r:
        # NOTE: No Revenue field here — Revenue is what we're predicting
        acq_cost_r  = st.number_input("Acquisition Cost (₹)", min_value=0.0, value=200.0, step=10.0, key="ac_r")
        eng_score_r = st.number_input("Engagement Score",     min_value=0.0, value=13.0,  step=0.5,  key="es_r")
        channels_r  = st.multiselect("Channels Used", channel_opts,
                                      default=["Instagram", "Facebook"], key="ch_r")

    st.divider()

    if st.button("Predict Revenue", use_container_width=True, type="primary", key="btn_rev"):

        # Build input DataFrame — 17 features, Revenue excluded (it's the target)
        rev_input = pd.DataFrame([[
            brand_map[brand_r], camp_map[camp_r], audience_map[audience_r], lang_map[language_r],
            duration_r, impressions_r, clicks_r, leads_r, conversions_r,
            acq_cost_r, eng_score_r,
            1 if "Email"     in channels_r else 0,
            1 if "Facebook"  in channels_r else 0,
            1 if "Google"    in channels_r else 0,
            1 if "Instagram" in channels_r else 0,
            1 if "WhatsApp"  in channels_r else 0,
            1 if "YouTube"   in channels_r else 0,
        ]], columns=[
            "Campaign_Name", "Campaign_Type", "Target_Audience", "Language",
            "Duration", "Impressions", "Clicks", "Leads", "Conversions",
            "Acquisition_Cost", "Engagement_Score",
            "Email", "Facebook", "Google", "Instagram", "WhatsApp", "YouTube"
        ])

        predicted_revenue = revenue_model.predict(rev_input)[0]

        st.markdown("### 📌 Predicted Revenue")
        st.success(f"₹ {predicted_revenue:,.2f}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Brand",         brand_r)
        m2.metric("Campaign Type", camp_r)
        m3.metric("Duration",      f"{duration_r} days")
        m4.metric("Channels",      len(channels_r))


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — PROFIT / LOSS CLASSIFICATION
# Model: RandomForestClassifier (profit_loss_classifier.pkl)
# Features: 18 (includes Revenue — needed to predict if campaign is profitable)
# Output: Profit (1) or Loss (0) with confidence %
# ════════════════════════════════════════════════════════════════════════════════
with tab2:

    st.markdown("Enter campaign details — the model will predict **Profit ✅ or Loss ❌**.")
    st.divider()

    # Row 1 — Dropdowns
    c1, c2, c3, c4 = st.columns(4)
    brand_c    = c1.selectbox("Brand",           list(brand_map),    key="brand_c")
    camp_c     = c2.selectbox("Campaign Type",   list(camp_map),     key="camp_c")
    audience_c = c3.selectbox("Target Audience", list(audience_map), key="aud_c")
    language_c = c4.selectbox("Language",        list(lang_map),     key="lang_c")

    # Row 2 — Numbers
    st.markdown("### Campaign Metrics")
    left_c, right_c = st.columns(2)

    with left_c:
        duration_c    = st.number_input("Duration (days)", min_value=1,   value=15,     key="dur_c")
        impressions_c = st.number_input("Impressions",     min_value=0,   value=50000,  step=1000, key="imp_c")
        clicks_c      = st.number_input("Clicks",          min_value=0,   value=3000,   step=100,  key="clk_c")
        leads_c       = st.number_input("Leads",           min_value=0,   value=500,    step=50,   key="ld_c")
        conversions_c = st.number_input("Conversions",     min_value=0,   value=200,    step=10,   key="conv_c")

    with right_c:
        # NOTE: Revenue IS included here — it helps predict if the campaign was profitable
        revenue_c   = st.number_input("Revenue (₹)",          min_value=0.0, value=500000.0, step=10000.0, key="rev_c")
        acq_cost_c  = st.number_input("Acquisition Cost (₹)", min_value=0.0, value=200.0,    step=10.0,    key="ac_c")
        eng_score_c = st.number_input("Engagement Score",     min_value=0.0, value=13.0,     step=0.5,     key="es_c")
        channels_c  = st.multiselect("Channels Used", channel_opts,
                                      default=["Instagram", "Facebook"], key="ch_c")

    st.divider()

    if st.button("Predict Profit / Loss", use_container_width=True, type="primary", key="btn_clf"):

        # Build input DataFrame — 18 features, Revenue included
        clf_input = pd.DataFrame([[
            brand_map[brand_c], camp_map[camp_c], audience_map[audience_c], lang_map[language_c],
            duration_c, impressions_c, clicks_c, leads_c, conversions_c,
            revenue_c, acq_cost_c, eng_score_c,
            1 if "Email"     in channels_c else 0,
            1 if "Facebook"  in channels_c else 0,
            1 if "Google"    in channels_c else 0,
            1 if "Instagram" in channels_c else 0,
            1 if "WhatsApp"  in channels_c else 0,
            1 if "YouTube"   in channels_c else 0,
        ]], columns=[
            "Campaign_Name", "Campaign_Type", "Target_Audience", "Language",
            "Duration", "Impressions", "Clicks", "Leads", "Conversions",
            "Revenue", "Acquisition_Cost", "Engagement_Score",
            "Email", "Facebook", "Google", "Instagram", "WhatsApp", "YouTube"
        ])

        pred  = classifier.predict(clf_input)[0]
        proba = classifier.predict_proba(clf_input)[0]

        st.markdown("### 📌 Result")
        if pred == 1:
            conf = proba[1] * 100
            st.success(f"✅  PROFIT  —  Confidence: {conf:.1f}%")
        else:
            conf = proba[0] * 100
            st.error(f"❌  LOSS  —  Confidence: {conf:.1f}%")

        st.progress(int(conf))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Brand",         brand_c)
        m2.metric("Campaign Type", camp_c)
        m3.metric("Revenue",       f"₹{revenue_c:,.0f}")
        m4.metric("Channels",      len(channels_c))

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("Marketing Campaign Performance Prediction  |  RandomForest  |  Classifier Accuracy: 96.26%")
