# import streamlit as st
# import numpy as np
# import pandas as pd
# import time
# import warnings
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor

# warnings.filterwarnings("ignore")

# # App Title
# st.title("🏋️‍♂️ Personal Fitness Tracker")
# st.write(
#     "This WebApp predicts the estimated calories burned based on your physical attributes and exercise parameters."
# )

# # Sidebar for User Input
# st.sidebar.header("🔹 User Input Parameters")

# def get_user_input():
#     age = st.sidebar.slider("Age", 10, 100, 30)
#     bmi = st.sidebar.slider("BMI", 15, 40, 22)
#     duration = st.sidebar.slider("Exercise Duration (min)", 0, 35, 15)
#     heart_rate = st.sidebar.slider("Heart Rate (bpm)", 60, 130, 85)
#     body_temp = st.sidebar.slider("Body Temperature (°C)", 36, 42, 37)
#     gender = st.sidebar.radio("Gender", ("Male", "Female"))

#     gender_encoded = 1 if gender == "Male" else 0

#     return pd.DataFrame(
#         {
#             "Age": [age],
#             "BMI": [bmi],
#             "Duration": [duration],
#             "Heart_Rate": [heart_rate],
#             "Body_Temp": [body_temp],
#             "Gender_male": [gender_encoded],  # 1 for male, 0 for female
#         }
#     )

# user_data = get_user_input()

# # Display User Input
# st.write("### 📌 Your Input Parameters")
# st.dataframe(user_data)

# # Load Data
# @st.cache_data
# def load_data():
#     exercise = pd.read_csv("exercise.csv")
#     calories = pd.read_csv("calories.csv")
#     df = exercise.merge(calories, on="User_ID").drop(columns="User_ID")
#     df["BMI"] = round(df["Weight"] / ((df["Height"] / 100) ** 2), 2)
#     return df

# exercise_df = load_data()

# # Train-Test Split
# train_data, test_data = train_test_split(exercise_df, test_size=0.2, random_state=42)

# # Feature Selection
# features = ["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp"]
# train_data = train_data[features + ["Calories"]]
# test_data = test_data[features + ["Calories"]]

# # One-Hot Encoding
# train_data = pd.get_dummies(train_data, drop_first=True)
# test_data = pd.get_dummies(test_data, drop_first=True)

# X_train, y_train = train_data.drop(columns="Calories"), train_data["Calories"]
# X_test, y_test = test_data.drop(columns="Calories"), test_data["Calories"]

# # Train Model
# @st.cache_resource
# def train_model():
#     model = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6, random_state=42)
#     model.fit(X_train, y_train)
#     return model

# model = train_model()

# # Align user input with training data columns
# user_data = user_data.reindex(columns=X_train.columns, fill_value=0)

# # Prediction
# st.write("### 🔮 Predicted Calories Burned")
# progress_bar = st.progress(0)
# for i in range(100):
#     progress_bar.progress(i + 1)
#     time.sleep(0.01)

# prediction = model.predict(user_data)[0]
# st.write(f"💪 **{round(prediction, 2)} kilocalories**")

# # Find Similar Results
# st.write("### 🔍 Similar Results")
# similar_data = exercise_df[
#     (exercise_df["Calories"] >= prediction - 10) & (exercise_df["Calories"] <= prediction + 10)
# ]
# st.dataframe(similar_data.sample(5))

# # General Insights
# st.write("### 📊 General Insights")
# st.write(
#     f"📅 Your age is higher than **{round((exercise_df['Age'] < user_data['Age'].values[0]).mean() * 100, 2)}%** of people."
# )
# st.write(
#     f"⏳ Your exercise duration is longer than **{round((exercise_df['Duration'] < user_data['Duration'].values[0]).mean() * 100, 2)}%** of people."
# )
# st.write(
#     f"💓 Your heart rate is higher than **{round((exercise_df['Heart_Rate'] < user_data['Heart_Rate'].values[0]).mean() * 100, 2)}%** of people."
# )
# st.write(
#     f"🌡️ Your body temperature is higher than **{round((exercise_df['Body_Temp'] < user_data['Body_Temp'].values[0]).mean() * 100, 2)}%** of people."
# )












import streamlit as st
import pandas as pd
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# Set Streamlit Page Config
st.set_page_config(page_title="Personal Fitness Tracker", layout="wide")

# Header Section
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🏋️‍♂️ Personal Fitness Tracker</h1>
    <h4 style='text-align: center;'>Estimate your calories burned based on your activity parameters.</h4>
    """,
    unsafe_allow_html=True,
)

st.write("---")

# Sidebar: User Input
st.sidebar.header("🔹 Enter Your Details")

def get_user_input():
    with st.sidebar:
        age = st.slider("Age", 10, 100, 30)
        bmi = st.slider("BMI", 15, 40, 22)
        duration = st.slider("Exercise Duration (min)", 0, 35, 15)
        heart_rate = st.slider("Heart Rate (bpm)", 60, 130, 85)
        body_temp = st.slider("Body Temperature (°C)", 36, 42, 37)
        gender = st.radio("Gender", ["Male", "Female"])

    return pd.DataFrame(
        {
            "Age": [age],
            "BMI": [bmi],
            "Duration": [duration],
            "Heart_Rate": [heart_rate],
            "Body_Temp": [body_temp],
            "Gender_male": [1 if gender == "Male" else 0],
        }
    )

user_data = get_user_input()

# Display User Input Parameters
st.markdown("### 📌 Your Input Parameters")
st.dataframe(user_data.style.format("{:.2f}"), use_container_width=True)

# Load Data
@st.cache_data
def load_data():
    exercise = pd.read_csv("exercise.csv")
    calories = pd.read_csv("calories.csv")
    df = exercise.merge(calories, on="User_ID").drop(columns="User_ID")
    df["BMI"] = round(df["Weight"] / ((df["Height"] / 100) ** 2), 2)
    return df

exercise_df = load_data()

# Train-Test Split
train_data, test_data = train_test_split(exercise_df, test_size=0.2, random_state=42)

# Feature Selection
features = ["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp"]
train_data = train_data[features + ["Calories"]]
test_data = test_data[features + ["Calories"]]

# One-Hot Encoding
train_data = pd.get_dummies(train_data, drop_first=True)
test_data = pd.get_dummies(test_data, drop_first=True)

X_train, y_train = train_data.drop(columns="Calories"), train_data["Calories"]
X_test, y_test = test_data.drop(columns="Calories"), test_data["Calories"]

# Train Model
@st.cache_resource
def train_model():
    model = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

# Align user input with training data columns
user_data = user_data.reindex(columns=X_train.columns, fill_value=0)

# Prediction
st.markdown("### 🔮 Predicted Calories Burned")
with st.spinner("Analyzing data..."):
    time.sleep(1)
    prediction = model.predict(user_data)[0]

st.success(f"🔥 **{round(prediction, 2)} kilocalories** burned")

# Similar Results
st.markdown("### 🔍 Similar Results")
similar_data = exercise_df[
    (exercise_df["Calories"] >= prediction - 10) & (exercise_df["Calories"] <= prediction + 10)
].sample(5)

st.dataframe(similar_data, use_container_width=True)

st.write("---")

# Visualization Section
st.markdown("### 📊 Data Insights")

st.markdown("""
    <style>
        .small-font {
            font-size:14px !important;
        }
        .medium-font {
            font-size:16px !important;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h3 class="medium-font">📊 Data Insights</h3>', unsafe_allow_html=True)

# Use columns to display multiple elements in the same row
col1, col2 = st.columns(2)

with col1:
    st.markdown('<h4 class="small-font">📉 Calories Distribution</h4>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.histplot(exercise_df["Calories"], kde=True, bins=20, color="royalblue")
    ax.axvline(prediction, color="red", linestyle="dashed", linewidth=2, label="Your Prediction")
    plt.legend()
    st.pyplot(fig)

with col2:
    st.markdown('<h3 class="small-font">📌 Your Stats vs Others</h3>', unsafe_allow_html=True)
    st.metric(label="📅  Age Rank ", value=f"{round((exercise_df['Age'] < user_data['Age'].values[0]).mean() * 100, 2)}% Higher")
    st.metric(label="⏳ Duration Rank", value=f"{round((exercise_df['Duration'] < user_data['Duration'].values[0]).mean() * 100, 2)}% Higher")
    st.metric(label="💓 Heart Rate Rank", value=f"{round((exercise_df['Heart_Rate'] < user_data['Heart_Rate'].values[0]).mean() * 100, 2)}% Higher")
    st.metric(label="🌡️ Body Temp Rank", value=f"{round((exercise_df['Body_Temp'] < user_data['Body_Temp'].values[0]).mean() * 100, 2)}% Higher")
