import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open("life_expectancy_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the saved scaler
with open("ss.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit App
st.title("🌍 Life Expectancy Prediction App")
st.write("Predict the expected lifespan based on various health, economic, and social factors.")

# Sidebar for user inputs
st.sidebar.header("Enter Input Features")

# User input fields (Make sure these match training features)
status = st.sidebar.selectbox("Country Status", ["Developing", "Developed"])
adult_mortality = st.sidebar.number_input("Adult Mortality Rate", min_value=0, max_value=1000, step=1)
infant_deaths = st.sidebar.number_input("Infant Deaths", min_value=0, max_value=500, step=1)
alcohol = st.sidebar.number_input("Alcohol Consumption (liters per capita)", min_value=0.0, max_value=20.0, step=0.1)
percentage_expenditure = st.sidebar.number_input("Health Expenditure (% of GDP)", min_value=0.0, max_value=10000.0, step=0.1)
schooling = st.sidebar.number_input("Average Years of Schooling", min_value=0.0, max_value=20.0, step=0.1)
GDP = st.sidebar.number_input("GDP per Capita (USD)", min_value=0.0, max_value=100000.0, step=100.0)

# Dummy values for missing features (Modify based on actual features used)
# You must include all 21 features used during training
dummy_features = [0] * 14  # Placeholder for missing features

# Convert categorical variables
status_encoded = 1 if status == "Developing" else 0

# Prepare input data (Including dummy values for missing features)
input_data = np.array([[status_encoded, adult_mortality, infant_deaths, alcohol, 
                        percentage_expenditure, schooling, GDP] + dummy_features])

# Scale input data using the loaded scaler
input_data_scaled = scaler.transform(input_data)

# Predict life expectancy when button is clicked
if st.sidebar.button("Predict Life Expectancy"):
    prediction = model.predict(input_data_scaled)[0]
    st.success(f"Predicted Life Expectancy: **{prediction:.2f} years**")
