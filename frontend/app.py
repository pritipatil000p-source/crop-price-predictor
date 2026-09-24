"""
app.py
------
Streamlit frontend for the Crop Price Predictor.

Run with:
    streamlit run frontend/app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, timedelta

import api_client  # the helper module in the same frontend/ folder

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="🌾 Crop Price Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a clean look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {font-size: 2rem; font-weight: 700; color: #1f7a1f; margin-bottom: 0.2rem;}
    .sub-header  {font-size: 1rem; color: #57606a; margin-bottom: 1.5rem;}
    .metric-card {background: #f7f8fa; border-radius: 8px; padding: 1rem;
                  border: 1px solid #e5e7eb; text-align: center;}
    .predict-box {background: #f0fff0; border-radius: 10px; padding: 1.5rem;
                  border: 2px solid #1f7a1f; text-align: center;}
    .price-big   {font-size: 2.5rem; font-weight: 800; color: #1f7a1f;}
    .disclaimer  {font-size: 0.8rem; color: #888; font-style: italic;}
    .section-title {font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "selected_state" not in st.session_state:
    st.session_state.selected_state = None
if "selected_district" not in st.session_state:
    st.session_state.selected_district = None


# ---------------------------------------------------------------------------
# Helper: show backend error
# ---------------------------------------------------------------------------
def show_backend_error(msg: str):
    st.error(
        f"⚠️ **Backend Error**\n\n{msg}\n\n"
        "_Make sure the FastAPI server is running:_\n"
        "`uvicorn backend.app.main:app --reload`",
        icon="🔴",
    )


# ---------------------------------------------------------------------------
# Check backend health
# ---------------------------------------------------------------------------
backend_ok = api_client.check_health()

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-header">🌾 Crop Price Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Predict Indian Mandi Commodity Prices Using Machine Learning</div>',
    unsafe_allow_html=True,
)

if not backend_ok:
    st.warning(
        "⚠️ The backend API is not reachable. "
        "Start it with: `uvicorn backend.app.main:app --reload`",
        icon="⚠️",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Load dropdown data from the backend (cached per session)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)  # refresh every 5 minutes
def load_states():
    try:
        return api_client.get_states()
    except RuntimeError as e:
        return []

@st.cache_data(ttl=300)
def load_districts(state):
    try:
        return api_client.get_districts(state=state)
    except RuntimeError:
        return []

@st.cache_data(ttl=300)
def load_markets(state, district):
    try:
        return api_client.get_markets(state=state, district=district)
    except RuntimeError:
        return []

@st.cache_data(ttl=300)
def load_commodities():
    try:
        return api_client.get_commodities()
    except RuntimeError:
        return []

@st.cache_data(ttl=300)
def load_varieties():
    try:
        return api_client.get_varieties()
    except RuntimeError:
        return []

@st.cache_data(ttl=300)
def load_grades():
    try:
        return api_client.get_grades()
    except RuntimeError:
        return []

states = load_states()
commodities = load_commodities()
varieties = load_varieties()
grades = load_grades()

if not states:
    st.error(
        "No states found. Please train the model first:\n`python ml/train.py`"
    )
    st.stop()

# ---------------------------------------------------------------------------
# SIDEBAR — Selection form
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("🔍 Select Parameters")
    st.markdown("---")

    # 1. State
    state_options = ["-- Select State --"] + states
    selected_state_idx = 0
    selected_state = st.selectbox("📍 State", state_options, index=0)

    # 2. District — filtered by state
    if selected_state and selected_state != "-- Select State --":
        districts = load_districts(selected_state)
    else:
        districts = []

    district_options = ["-- Select District --"] + districts
    selected_district = st.selectbox(
        "🏘️ District",
        district_options,
        disabled=(not districts),
    )

    # 3. Market — filtered by state + district
    if (
        selected_district
        and selected_district != "-- Select District --"
        and selected_state != "-- Select State --"
    ):
        markets = load_markets(selected_state, selected_district)
    else:
        markets = []

    market_options = ["-- Select Market --"] + markets
    selected_market = st.selectbox(
        "🏪 Market",
        market_options,
        disabled=(not markets),
    )

    # 4. Commodity
    commodity_options = ["-- Select Commodity --"] + commodities
    selected_commodity = st.selectbox("🌽 Commodity", commodity_options)

    # 5. Variety
    variety_options = ["-- Select Variety --"] + varieties
    selected_variety = st.selectbox("🌿 Variety", variety_options)

    # 6. Grade
    grade_options = ["-- Select Grade --"] + grades
    selected_grade = st.selectbox("⭐ Grade", grade_options)

    # 7. Prediction Date
    st.markdown("---")
    min_future_date = date.today() + timedelta(days=1)
    prediction_date = st.date_input(
        "📅 Prediction Date",
        value=min_future_date,
        min_value=date(2000, 1, 1),
        help="Select the future date for which you want to predict the price.",
    )

    st.markdown("---")
    # Predict button
    predict_clicked = st.button(
        "🤖 Predict Price",
        type="primary",
        use_container_width=True,
        disabled=(
            selected_state == "-- Select State --"
            or selected_district == "-- Select District --"
            or selected_market == "-- Select Market --"
            or selected_commodity == "-- Select Commodity --"
            or selected_variety == "-- Select Variety --"
            or selected_grade == "-- Select Grade --"
        ),
    )

# ---------------------------------------------------------------------------
# Validate selections
# ---------------------------------------------------------------------------
selections_complete = (
    selected_state not in (None, "-- Select State --")
    and selected_district not in (None, "-- Select District --")
    and selected_market not in (None, "-- Select Market --")
    and selected_commodity not in (None, "-- Select Commodity --")
    and selected_variety not in (None, "-- Select Variety --")
    and selected_grade not in (None, "-- Select Grade --")
)

# ---------------------------------------------------------------------------
# Handle Predict button click
# ---------------------------------------------------------------------------
if predict_clicked and selections_complete:
    with st.spinner("⏳ Getting prediction from the model..."):
        try:
            result = api_client.predict_price(
                state=selected_state,
                district=selected_district,
                market=selected_market,
                commodity=selected_commodity,
                variety=selected_variety,
                grade=selected_grade,
                prediction_date=str(prediction_date),
            )
            st.session_state.prediction_result = result
        except RuntimeError as e:
            show_backend_error(str(e))
            st.session_state.prediction_result = None

# ---------------------------------------------------------------------------
# Load historical data for the selected commodity + market
# ---------------------------------------------------------------------------
hist_data = None
hist_error = None

if selections_complete:
    try:
        hist_resp = api_client.get_historical_prices(
            commodity=selected_commodity,
            market=selected_market,
            limit=200,
        )
        if hist_resp and hist_resp.get("records"):
            hist_data = pd.DataFrame(hist_resp["records"])
            hist_data["date"] = pd.to_datetime(hist_data["date"])
            hist_data = hist_data.sort_values("date")
    except RuntimeError as e:
        hist_error = str(e)

# ---------------------------------------------------------------------------
# SECTION 1: 📊 Market Overview
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📊 Market Overview")

if not selections_complete:
    st.info("👈 Please select all parameters in the sidebar to see market data.")
elif hist_error:
    show_backend_error(hist_error)
elif hist_data is None or hist_data.empty:
    st.warning(
        f"No historical data found for **{selected_commodity}** in **{selected_market}**. "
        "Try a different combination."
    )
else:
    latest = hist_data.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🌽 Commodity",
            value=selected_commodity,
        )
    with col2:
        st.metric(
            label="💰 Latest Modal Price",
            value=f"₹ {latest['modal_price']:,.0f}",
            help="Most recent recorded modal price (INR/quintal)",
        )
    with col3:
        st.metric(
            label="📉 Min Price",
            value=f"₹ {hist_data['min_price'].min():,.0f}",
        )
    with col4:
        st.metric(
            label="📈 Max Price",
            value=f"₹ {hist_data['max_price'].max():,.0f}",
        )

    st.caption(f"Based on {len(hist_data):,} historical records for this combination.")

# ---------------------------------------------------------------------------
# SECTION 2: 📈 Historical Price Chart
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📈 Historical Price Trend")

if selections_complete and hist_data is not None and not hist_data.empty:
    fig = go.Figure()

    # Shaded area between min and max
    fig.add_trace(
        go.Scatter(
            x=pd.concat([hist_data["date"], hist_data["date"][::-1]]),
            y=pd.concat([hist_data["max_price"], hist_data["min_price"][::-1]]),
            fill="toself",
            fillcolor="rgba(59,130,212,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Min–Max Range",
            showlegend=True,
        )
    )

    # Modal price line
    fig.add_trace(
        go.Scatter(
            x=hist_data["date"],
            y=hist_data["modal_price"],
            mode="lines+markers",
            name="Modal Price",
            line=dict(color="#1f7a1f", width=2),
            marker=dict(size=4),
        )
    )

    fig.update_layout(
        title=f"{selected_commodity} — Modal Price in {selected_market}",
        xaxis_title="Date",
        yaxis_title="Price (INR / quintal)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=400,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)
elif selections_complete:
    st.info("No historical data available to plot for the selected combination.")

# ---------------------------------------------------------------------------
# SECTION 3: 🤖 Price Prediction
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🤖 Price Prediction")

if not selections_complete:
    st.info("👈 Complete all sidebar selections and click **Predict Price** to see a prediction.")
else:
    col_info, col_result = st.columns([1, 1])

    with col_info:
        st.markdown("**Your Selection:**")
        st.write(f"- **Commodity:** {selected_commodity}")
        st.write(f"- **Market:** {selected_market}")
        st.write(f"- **State:** {selected_state} / {selected_district}")
        st.write(f"- **Variety:** {selected_variety} | **Grade:** {selected_grade}")
        st.write(f"- **Prediction Date:** {prediction_date}")

    with col_result:
        if st.session_state.prediction_result:
            pred = st.session_state.prediction_result
            price = pred.get("predicted_price", 0)

            st.markdown(
                f"""
                <div class="predict-box">
                    <div style="font-size:1rem; color:#57606a;">Predicted Price</div>
                    <div class="price-big">₹ {price:,.2f}</div>
                    <div style="font-size:0.9rem; color:#444;">per quintal (INR)</div>
                    <br>
                    <div class="disclaimer">
                        ⚠️ {pred.get('disclaimer', 'This is an ML estimate, not a guaranteed market price.')}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background:#f7f8fa; border-radius:10px; padding:2rem;
                            border:1px dashed #ccc; text-align:center; color:#888;">
                    Click <strong>Predict Price</strong> in the sidebar to see the prediction.
                </div>
                """,
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------------------------
# SECTION 4: 📊 Model Performance
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📊 Model Performance")

try:
    metrics = api_client.get_model_metrics()
    col_m1, col_m2, col_m3 = st.columns(3)

    with col_m1:
        st.metric(
            label="MAE (Mean Absolute Error)",
            value=f"₹ {metrics['MAE']:,.2f}",
            help="Average absolute difference between predicted and actual prices (lower is better).",
        )
    with col_m2:
        st.metric(
            label="RMSE (Root Mean Squared Error)",
            value=f"₹ {metrics['RMSE']:,.2f}",
            help="Root mean squared error — penalizes large errors more (lower is better).",
        )
    with col_m3:
        r2 = metrics["R2"]
        r2_pct = f"{r2 * 100:.1f}%"
        st.metric(
            label="R² Score",
            value=r2_pct,
            help="Proportion of variance explained by the model (higher is better; 1.0 = perfect).",
        )

    # R² interpretation bar
    r2_color = "#1f7a1f" if r2 >= 0.7 else ("#e6a817" if r2 >= 0.5 else "#c0392b")
    st.markdown(
        f"""
        <div style="background:#e5e7eb; border-radius:6px; height:12px; margin-top:0.5rem;">
            <div style="background:{r2_color}; width:{max(0, min(100, r2*100)):.1f}%;
                        height:100%; border-radius:6px;"></div>
        </div>
        <div style="font-size:0.8rem; color:#57606a; margin-top:0.3rem;">
            R² = {r2:.4f} — Model explains {r2*100:.1f}% of price variance on the test set
        </div>
        """,
        unsafe_allow_html=True,
    )

    if metrics.get("train_size") and metrics.get("test_size"):
        st.caption(
            f"Model trained on {metrics['train_size']:,} records · "
            f"Evaluated on {metrics['test_size']:,} records (chronological split)"
        )

except RuntimeError as e:
    st.warning(
        f"Model metrics not available. Train the model first: `python ml/train.py`\n\n_{e}_"
    )

# ---------------------------------------------------------------------------
# SECTION 5: 📋 Historical Data Table
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📋 Recent Historical Records")

if selections_complete and hist_data is not None and not hist_data.empty:
    # Show most recent 50 records
    display_df = hist_data.tail(50).copy()
    display_df["date"] = display_df["date"].dt.strftime("%Y-%m-%d")
    display_df = display_df.rename(
        columns={
            "date": "Date",
            "commodity": "Commodity",
            "market": "Market",
            "state": "State",
            "min_price": "Min Price (₹)",
            "max_price": "Max Price (₹)",
            "modal_price": "Modal Price (₹)",
        }
    )[["Date", "Commodity", "Market", "State", "Min Price (₹)", "Max Price (₹)", "Modal Price (₹)"]]

    st.dataframe(
        display_df.sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Showing most recent {len(display_df)} records out of {len(hist_data)} total.")
elif selections_complete:
    st.info("No historical records to display for the selected combination.")
else:
    st.info("Select a commodity and market to see historical data.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; color:#888; font-size:0.8rem; padding:1rem 0;">
        🌾 Crop Price Predictor · Built with Streamlit + FastAPI + scikit-learn ·
        Data: Kaggle Indian Mandi Price Dataset ·
        <em>Predictions are ML estimates, not guaranteed market prices.</em>
    </div>
    """,
    unsafe_allow_html=True,
)
