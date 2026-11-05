
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="",
    layout="centered"
)

@st.cache_resource
def load_model():
    model = joblib.load("knn_heart_model.pkl")
    scaler = joblib.load("heart_scaler.pkl")
    expected_columns = joblib.load("heart_columns.pkl")
    return model, scaler, expected_columns


try:
    model, scaler, expected_columns = load_model()

except Exception as e:
    st.error("Unable to load the model files. Please check your project files.")
    st.stop()


# Application Title
st.title("Heart Disease Risk Prediction System")

st.markdown("Please provide the following clinical details to assess your heart stroke risk.")
    

st.divider()


st.subheader("Personal Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)

with col2:
    sex = st.selectbox("Sex", ["M", "F"])


st.subheader("Health Information")

col1, col2 = st.columns(2)

with col1:
    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=80,
        max_value=200,
        value=120
    )

    cholesterol = st.number_input(
        "Cholesterol (mg/dL)",
        min_value=100,
        max_value=600,
        value=200
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        [0, 1]
    )

with col2:
    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    max_hr = st.slider(
        "Maximum Heart Rate",
        60, 220, 150
    )

    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["Y", "N"]
    )

    oldpeak = st.slider(
        "Oldpeak (ST Depression)",
        0.0, 6.0, 1.0, 0.1
    )


st.subheader("Additional Information")

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)


st.divider()


# Prediction
if st.button("Predict Heart Disease Risk", use_container_width=True):

    try:
        raw_input = {
            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak,
            "Sex_" + sex: 1,
            "ChestPainType_" + chest_pain: 1,
            "RestingECG_" + resting_ecg: 1,
            "ExerciseAngina_" + exercise_angina: 1,
            "ST_Slope_" + st_slope: 1
        }

        input_df = pd.DataFrame([raw_input])

        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns
        input_df = input_df[expected_columns]

        # Scale the input
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)[0]

        # Display result
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")

        elif prediction == 0:
            st.success("✅ Low Risk of Heart Disease")

        else:
            st.warning("Unexpected prediction result.")

    except Exception as e:
        st.error("An error occurred while making the prediction.")
        st.caption(str(e))


# Medical Disclaimer
st.divider()

st.caption(
    "Disclaimer: This application is developed for educational "
    "purposes only. It is not a substitute for professional medical advice."
)


st.caption("Heart Disease Prediction | KNN Machine Learning Project")
