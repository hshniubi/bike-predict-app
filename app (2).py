
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
season = st.selectbox("Season", [0, 1, 2, 3], format_func=lambda x: ["Spring", "Summer", "Autumn", "Winter"][x])
holiday = 1 if st.radio("Is Holiday?", ["No", "Yes"]) == "Yes" else 0
functioning = 1 if st.radio("Is Functioning Day?", ["Yes", "No"]) == "Yes" else 0
weekday = st.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 2)
month = st.slider("Month", 1, 12, 5)

# 构建输入 DataFrame
input_df = pd.DataFrame([{
    "Hour": float(hour),
    "Temperature(℃)": float(temp),
    "Humidity(%)": float(humidity),
    "Wind speed (m/s)": float(wind),
    "Visibility (10m)": float(visibility),
    "Dew point temperature(℃)": float(dew_point),
    "Solar Radiation (MJ/m2)": float(solar),
    "Rainfall(mm)": float(rainfall),
    "Snowfall (cm)": float(snowfall),
    "Seasons": float(season),
    "Holiday": float(holiday),
    "Functioning Day": float(functioning),
    "Weekday": float(weekday),
    "Month": float(month)
}])

# 预测
if st.button("Predict"):
    input_df = input_df.reindex(columns=model.feature_names_in_)
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted bike rental count: {int(prediction)}")
