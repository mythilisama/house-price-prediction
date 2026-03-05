# ==============================
# 🏠 House Price Prediction App
# ==============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ==============================
# 1️⃣ Page Config
# ==============================

st.set_page_config(page_title="House Price Prediction", layout="wide")

st.title("🏠 House Price Prediction")
st.write("Predicting house price based on house features")

# ==============================
# 2️⃣ Load Dataset
# ==============================

df = pd.read_csv("house_price_regression_dataset_inr.csv")

# Data Cleaning
df.drop_duplicates(inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)

# ==============================
# 3️⃣ Define Input & Output
# ==============================

X = df.drop("price_inr", axis=1)
y = df["price_inr"]

# ==============================
# 4️⃣ Train Model
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open("house_model.pkl", "wb") as f:
    pickle.dump(model, f)

# ==============================
# 5️⃣ Layout Columns
# ==============================

col1, col2 = st.columns(2)

# ==============================
# 6️⃣ Dataset Preview
# ==============================

with col1:
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

# ==============================
# 8️⃣ Model Evaluation
# ==============================

st.subheader("📌 Model Performance")

y_pred = model.predict(X_test)

st.write("Mean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 2))
st.write("R2 Score:", round(r2_score(y_test, y_pred), 3))

# ==============================
# 9️⃣ User Input Section
# ==============================

st.subheader("🔢 Enter House Details")

user_inputs = []

input_col1, input_col2 = st.columns(2)

columns_list = list(X.columns)

for i, column in enumerate(columns_list):

    current_col = input_col1 if i % 2 == 0 else input_col2

    with current_col:

        # Integer columns → no decimals
        if pd.api.types.is_integer_dtype(df[column]):
            value = st.number_input(
                f"Enter {column}",
                min_value=int(df[column].min()),
                max_value=int(df[column].max()),
                step=1
            )
        else:
            value = st.number_input(
                f"Enter {column}",
                min_value=float(df[column].min())
            )

        user_inputs.append(value)

# ==============================
# 🔟 Prediction
# ==============================

if st.button("Predict Price"):

    prediction = model.predict([user_inputs])

    st.success(f"🏠 Predicted House Price: ₹ {prediction[0]:,.2f}")
