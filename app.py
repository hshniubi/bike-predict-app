
import streamlit as st
import pandas as pd
import joblib

# 加载模型
model = joblib.load("xgb_model.pkl")

st.title("Seoul Bike Demand Predictor")
st.markdown("Enter the conditions below to forecast hourly bike rental demand:")

# 构建输入
hour = st.slider("Hour", 0, 23, 8)
temp = st.slider("Temperature(℃)", -10.0, 40.0, 20.0)
humidity = st.slider("Humidity(%)", 0, 100, 50)
wind = st.slider("Wind speed (m/s)", 0.0, 10.0, 2.0)
visibility = st.slider("Visibility (10m)", 0, 2000, 1000)
dew_point = st.slider("Dew point temperature(℃)", -20.0, 30.0, 10.0)
solar = st.slider("Solar Radiation (MJ/m2)", 0.0, 3.0, 0.5)
rainfall = st.slider("Rainfall(mm)", 0.0, 100.0, 0.0)
snowfall = st.slider("Snowfall (cm)", 0.0, 10.0, 0.0)
season = st.selectbox("Season", {"Spring": 0, "Summer": 1, "Autumn": 2, "Winter": 3})
holiday = st.radio("Is Holiday?", ["No", "Yes"]) == "Yes"
functioning = st.radio("Is Functioning Day?", ["Yes", "No"]) == "Yes"
weekday = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
month = st.slider("Month", 1, 12, 5)

# 构建输入 DataFrame
input_df = pd.DataFrame([{
    "Hour": hour,
    "Temperature(℃)": temp,
    "Humidity(%)": humidity,
    "Wind speed (m/s)": wind,
    "Visibility (10m)": visibility,
    "Dew point temperature(℃)": dew_point,
    "Solar Radiation (MJ/m2)": solar,
    "Rainfall(mm)": rainfall,
    "Snowfall (cm)": snowfall,
    "Seasons": season,
    "Holiday": int(holiday),
    "Functioning Day": int(functioning),
    "Weekday": weekday,
    "Month": month
}])

# 预测
if st.button("Predict"):
    input_df = input_df.astype(float).reindex(columns=model.feature_names_in_)
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted bike rental count: {int(prediction)}")
