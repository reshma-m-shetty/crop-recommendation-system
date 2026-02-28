import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# Page Configuration
# ======================
st.set_page_config(page_title="Crop Recommendation Dashboard", layout="wide")

st.title("🌾 Crop Recommendation Dashboard")
st.markdown("Predict the most suitable crop based on soil and weather conditions.")

# ======================
# Load Model & Data
# ======================
model = joblib.load("crop.pkl")
data = pd.read_csv("Crop_recommendation.csv")

# ======================
# Sidebar Inputs
# ======================
st.sidebar.header("🌱 Input Features")

N = st.sidebar.number_input("Nitrogen (N)", min_value=0, max_value=200, value=50)
P = st.sidebar.number_input("Phosphorus (P)", min_value=0, max_value=200, value=50)
K = st.sidebar.number_input("Potassium (K)", min_value=0, max_value=200, value=50)

temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 60.0)
ph = st.sidebar.slider("pH Value", 0.0, 14.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 500.0, 100.0)

# ======================
# Prediction Section
# ======================
input_data = pd.DataFrame({
    'N': [N],
    'P': [P],
    'K': [K],
    'temperature': [temperature],
    'humidity': [humidity],
    'ph': [ph],
    'rainfall': [rainfall]
})

prediction = model.predict(input_data)

st.subheader("🌾 Recommended Crop")
st.success(prediction[0])

# Optional: Show confidence if model supports it
if hasattr(model, "predict_proba"):
    probability = model.predict_proba(input_data)
    confidence = probability.max() * 100
    st.info(f"Model Confidence: {confidence:.2f}%")

# ======================
# Dataset Overview
# ======================
st.subheader("📊 Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.write("First 5 Rows")
    st.dataframe(data.head())

with col2:
    st.write("Summary Statistics")
    st.dataframe(data.describe())

# ======================
# Visualizations
# ======================
st.subheader("📈 Data Visualizations")

col3, col4 = st.columns(2)

with col3:
    fig, ax = plt.subplots()
    sns.histplot(data["rainfall"], kde=True, ax=ax)
    ax.set_title("Rainfall Distribution")
    st.pyplot(fig)

with col4:
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x="temperature", y="humidity", data=data, ax=ax2)
    ax2.set_title("Temperature vs Humidity")
    st.pyplot(fig2)

# ======================
# Model Info
# ======================
st.subheader("📌 Model Information")

st.metric("Model Used", "Random Forest Classifier")
st.metric("Prediction Type", "Crop Recommendation")