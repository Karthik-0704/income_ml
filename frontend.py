import streamlit as st
import requests
import json
from streamlit_lottie import st_lottie

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Income Classification Engine", page_icon="🏦", layout="wide")

@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200: return None
    return r.json()

lottie_animation = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_M9p23l.json")

# ==========================================
# HEADER SECTION
# ==========================================
col1, col2 = st.columns([1, 3])
with col1:
    if lottie_animation:
        st_lottie(lottie_animation, height=150, key="finance_anim")
with col2:
    st.title("Socio-Economic Classification Engine")
    st.markdown("""
    **Production-Grade Inference Portal**: Toggle between different model architectures and class-balancing strategies to observe performance trade-offs.
    """)

st.divider()

# ==========================================
# MODEL SELECTION & METRICS
# ==========================================
st.subheader("1. Configure Model Engine")

c1, c2 = st.columns(2)
with c1:
    algo_choice = st.selectbox("Select Algorithm Family:", ["XGBoost", "Random Forest"])
with c2:
    mode_choice = st.radio("Class Balancing Strategy:", ["Unbalanced", "Balanced"], horizontal=True)

# Define the Metrics Database from your provided results
metrics_data = {
    "XGBoost": {
        "Balanced": {"Acc": "86.73%", "Prec": "32.03%", "Rec": "90.24%", "AUC": "0.953"},
        "Unbalanced": {"Acc": "95.54%", "Prec": "74.48%", "Rec": "49.27%", "AUC": "0.955"}
    },
    "Random Forest": {
        "Balanced": {"Acc": "93.89%", "Prec": "52.96%", "Rec": "65.62%", "AUC": "0.947"},
        "Unbalanced": {"Acc": "95.40%", "Prec": "81.16%", "Rec": "39.33%", "AUC": "0.950"}
    }
}

current_metrics = metrics_data[algo_choice][mode_choice]

st.markdown(f"##### Dashboard: {algo_choice} ({mode_choice} Mode) Performance")
m1, m2, m3, m4 = st.columns(4)
m1.metric(label="Accuracy", value=current_metrics["Acc"])
m2.metric(label="Precision", value=current_metrics["Prec"])
m3.metric(label="Recall", value=current_metrics["Rec"])
m4.metric(label="Test ROC-AUC", value=current_metrics["AUC"])

# Logic for Backend Mode Mapping
# Format: "XGBoost_Balanced", etc.
backend_mode = f"{algo_choice}_{mode_choice}"

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

json_input = st.text_area("Input JSON Features:", value=default_json, height=200)

if st.button("Run Prediction", type="primary"):
    try:
        parsed_data = json.loads(json_input)
        
        # Send the specific algorithm and mode to the FastAPI backend
        api_payload = {
            "mode": backend_mode, 
            "data": parsed_data
        }
        
        with st.spinner(f"Requesting {algo_choice} Inference..."):
            # Replace with your EC2 Public IP once ready!
            response = requests.post("http://127.0.0.1:8000/predict", json=api_payload)
            
        if response.status_code == 200:
            result = response.json()
            st.success("Analysis Complete")
            st.json(result["predictions"])
        else:
            st.error(f"Error: {response.text}")
            
    except json.JSONDecodeError:
        st.error("Invalid JSON format.")
    except requests.exceptions.ConnectionError:
        st.error("Backend Offline. Cannot connect to API on Port 8000.")