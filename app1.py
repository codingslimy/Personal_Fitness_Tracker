# import pandas as pd
# import numpy as np
# import time
# import streamlit as st
# # 🔹 Initialize session state to avoid KeyError
# if 'latest_prediction' not in st.session_state:
#     st.session_state['latest_prediction'] = {
#         "SVM": "N/A",
#         "Random Forest": "N/A",
#         "Logistic Regression": "N/A",
#         "XGBoost": "N/A"
#     }
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier

# # 🔹 Generate a simulated dataset
# def generate_dataset(n_samples=1000):
#     data = {
#         'heart_rate': np.random.randint(60, 180, n_samples),
#         'steps': np.random.randint(0, 20000, n_samples),
#         'calories_burned': np.random.randint(100, 3000, n_samples),
#         'activity_label': np.random.choice(['Walking', 'Running', 'Cycling', 'Rest'], n_samples)
#     }
#     return pd.DataFrame(data)

# df = generate_dataset()

# # 🔹 Preprocessing
# X = df.drop(columns=['activity_label'])  # Features
# y = df['activity_label']  # Labels
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)  # Encode labels (0, 1, 2, 3)

# # 🔹 Standardizing features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # 🔹 Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# # 🔹 Train models
# xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
# svm_model = SVC(probability=True)
# rf_model = RandomForestClassifier(n_estimators=100)
# log_reg = LogisticRegression()

# xgb_model.fit(X_train, y_train)
# svm_model.fit(X_train, y_train)
# rf_model.fit(X_train, y_train)
# log_reg.fit(X_train, y_train)



# # 🔹 Streamlit UI
# st.title("🏋️ Personal Fitness Tracker")
# st.write("Live Activity Prediction Using ML Models")

# prediction_container = st.empty()  # Placeholder for live updates

# # 🔹 Simulate live tracking (real-time updates)
# for _ in range(100):  # Simulating 100 updates
#     new_data = pd.DataFrame(
#         np.random.randint(60, 180, size=(1, X.shape[1])),
#         columns=X.columns
#     )  
#     # Generate random feature values

#     new_data_scaled = scaler.transform(new_data)

#     # Get predictions from models
#     xgb_pred = label_encoder.inverse_transform(xgb_model.predict(new_data_scaled))[0]
#     svm_pred = label_encoder.inverse_transform(svm_model.predict(new_data_scaled))[0]
#     rf_pred = label_encoder.inverse_transform(rf_model.predict(new_data_scaled))[0]
#     log_reg_pred = label_encoder.inverse_transform(log_reg.predict(new_data_scaled))[0]

#    # 🔹 Update session state safely
#     st.session_state['latest_prediction'] = {
#         "SVM": svm_pred,
#         "Random Forest": rf_pred,
#         "Logistic Regression": log_reg_pred,
#         "XGBoost": xgb_pred
#     }

#     prediction_container.subheader("📊 Latest Prediction")
#     prediction_container.write(st.session_state['latest_prediction'])


#     time.sleep(1)  # Pause to simulate real-time effect
















import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import time

import warnings
warnings.filterwarnings('ignore')

st.write("## Personal Fitness Tracker")
#st.image("", use_column_width=True)
st.write("In this WebApp you will be able to observe your predicted calories burned in your body. Pass your parameters such as `Age`, `Gender`, `BMI`, etc., into this WebApp and then you will see the predicted value of kilocalories burned.")

st.sidebar.header("User Input Parameters: ")

def user_input_features():
    age = st.sidebar.slider("Age: ", 10, 100, 30)
    bmi = st.sidebar.slider("BMI: ", 15, 40, 20)
    duration = st.sidebar.slider("Duration (min): ", 0, 35, 15)
    heart_rate = st.sidebar.slider("Heart Rate: ", 60, 130, 80)
    body_temp = st.sidebar.slider("Body Temperature (C): ", 36, 42, 38)
    gender_button = st.sidebar.radio("Gender: ", ("Male", "Female"))

    gender = 1 if gender_button == "Male" else 0

    # Use column names to match the training data
    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender  # Gender is encoded as 1 for male, 0 for female
    }

    features = pd.DataFrame(data_model, index=[0])
    return features

df = user_input_features()

st.write("---")
st.header("Your Parameters: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)
st.write(df)

# Load and preprocess data
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

# Add BMI column to both training and test sets
for data in [exercise_train_data, exercise_test_data]:
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"], 2)

# Prepare the training and testing sets
exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

# Separate features and labels
X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]

X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

# Train the model
random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(X_train, y_train)

# Align prediction data columns with training data
df = df.reindex(columns=X_train.columns, fill_value=0)

# Make prediction
prediction = random_reg.predict(df)

st.write("---")
st.header("Prediction: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

st.write(f"{round(prediction[0], 2)} **kilocalories**")

st.write("---")
st.header("Similar Results: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

# Find similar results based on predicted calories
calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
st.write(similar_data.sample(5))

st.write("---")
st.header("General Information: ")

# Boolean logic for age, duration, etc., compared to the user's input
boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()

st.write("You are older than", round(sum(boolean_age) / len(boolean_age), 2) * 100, "% of other people.")
st.write("Your exercise duration is higher than", round(sum(boolean_duration) / len(boolean_duration), 2) * 100, "% of other people.")
st.write("You have a higher heart rate than", round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100, "% of other people during exercise.")
st.write("You have a higher body temperature than", round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100, "% of other people during exercise.")



























































