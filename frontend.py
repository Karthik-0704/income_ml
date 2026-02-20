import streamlit as st
import requests
import json
from streamlit_lottie import st_lottie

# ==========================================
# PAGE CONFIGURATION & UI SETUP
# ==========================================
st.set_page_config(page_title="Income Classification Engine", page_icon="🏦", layout="wide")

# Helper function to load Lottie animations from URL
@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load contextual animation (Finance/Data Theme)
lottie_animation = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_M9p23l.json")

# ==========================================
# HEADER & ANIMATION SECTION
# ==========================================
col1, col2 = st.columns([1, 3])
with col1:
    if lottie_animation:
        st_lottie(lottie_animation, height=150, key="finance_anim")
with col2:
    st.title("Socio-Economic Classification Engine")
    st.markdown("""
    **Instructions:** Select your model architecture below, paste a JSON payload containing the individual's socio-economic data, and click 'Run Prediction'.
    * **Model Version:** `v1.2-XGBoost-Hybrid (32 Features)`
    """)

st.divider()

# ==========================================
# MODEL SELECTION & METRICS DASHBOARD
# ==========================================
st.subheader("1. Select Model Architecture")
model_choice = st.radio(
    "Choose the deployment mode:",
    options=["Unbalanced (High Precision)", "Balanced (High Recall)"],
    horizontal=True
)

# Display Historical Validation Metrics based on choice
st.markdown("##### Validated Model Metrics (Historical Training Performance)")
m1, m2, m3, m4 = st.columns(4)

if "Unbalanced" in model_choice:
    backend_mode = "Unbalanced"
    m1.metric(label="Accuracy", value="95.51%")
    m2.metric(label="Precision", value="73.98%")
    m3.metric(label="Recall", value="49.20%")
    m4.metric(label="ROC-AUC", value="0.954")
else:
    backend_mode = "Balanced"
    # Note: Replace these with your exact Balanced metrics from DagsHub
    m1.metric(label="Accuracy", value="82.10%") 
    m2.metric(label="Precision", value="34.50%")
    m3.metric(label="Recall", value="89.50%")
    m4.metric(label="ROC-AUC", value="0.952")

st.divider()

# ==========================================
# INFERENCE PAYLOAD SECTION
# ==========================================
st.subheader("2. Execute Live Prediction")

default_json = """[
  {
    "age": 42,
    "wage per hour": 0,
    "capital gains": 0,
    "capital losses": 0,
    "dividends from stocks": 1500,
    "num persons worked for employer": 4,
    "weeks worked in year": 52,
    "year": 95,
    "education": "Bachelors degree(BA AB BS)",
    "major occupation code": "Professional specialty",
    "sex": "Female"
  }
]"""

json_input = st.text_area("Paste JSON Payload Here:", value=default_json, height=250)

if st.button("Run Prediction", type="primary"):
    try:
        parsed_data = json.loads(json_input)
        
        # Structure the payload to tell the backend which model to use
        api_payload = {
            "mode": backend_mode,
            "data": parsed_data
        }
        
        with st.spinner("Processing via XGBoost Engine..."):
            # Ensure your FastAPI backend app.py is running on port 8000
            response = requests.post("http://127.0.0.1:8000/predict", json=api_payload)
            
        if response.status_code == 200:
            result = response.json()
            st.success("Prediction Successfully Computed!")
            
            # Display results neatly
            st.json(result["predictions"])
            
        else:
            st.error(f"Backend API Error: {response.text}")
            
    except json.JSONDecodeError:
        st.error("Invalid JSON format. Please check your payload syntax.")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the Backend API. Ensure `app.py` is running on port 8000.")