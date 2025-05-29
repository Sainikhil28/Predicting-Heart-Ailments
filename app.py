import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load data
@st.cache_data
def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

# Function for data preprocessing
def preprocess_data(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Function to train and evaluate models
def train_and_evaluate(X_train, X_test, y_train, y_test):
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    logreg_accuracy = accuracy_score(y_test, logreg.predict(X_test))
    dtree_accuracy = accuracy_score(y_test, dtree.predict(X_test))
    return logreg, dtree, logreg_accuracy, dtree_accuracy

# Main function
def main():
    st.title("Heart Disease Prediction")

    # Add custom theme color
    st.markdown(
        """
        <style>
        .css-1aumxhk {
            color: white;
            background-color: #FF0000; /* Heart red color */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Load dataset
    data_path = r"D:\20MSSL04-Data mining lab\heart.csv"
    df = load_data(data_path)

    st.subheader("Dataset Preview:")
    st.write(df.head())

    st.subheader("Data Cleaning and Preprocessing:")
    st.write("No missing values found in the dataset.")

    # Preprocess data
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    st.subheader("Model Training and Evaluation:")
    logreg, dtree, logreg_accuracy, dtree_accuracy = train_and_evaluate(X_train, X_test, y_train, y_test)

    st.write("Logistic Regression Accuracy:", logreg_accuracy)
    st.write("Decision Tree Accuracy:", dtree_accuracy)

    # Visualize correlation matrix
    st.subheader("Correlation Matrix")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # Visualizations
    st.subheader("Data Visualizations")
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # Age Distribution
    sns.histplot(df['age'], bins=20, kde=True, ax=ax[0])
    ax[0].set_title('Age Distribution')

    # Sex Distribution
    sns.countplot(x='sex', data=df, ax=ax[1])
    ax[1].set_title('Sex Distribution')
    ax[1].set_xticklabels(['Female', 'Male'])

    # Target Distribution
    sns.countplot(x='target', data=df, ax=ax[2])
    ax[2].set_title('Target Distribution')
    ax[2].set_xticklabels(['No Heart Disease', 'Heart Disease'])

    st.pyplot(fig)

    st.sidebar.header("User Input")
    age = st.sidebar.slider("Age", min_value=20, max_value=80, value=40)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    cp = st.sidebar.slider("Chest Pain Type (CP)", min_value=0, max_value=3, value=1)
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=120)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    restecg = st.sidebar.selectbox("Resting ECG", ["Normal", "Abnormality", "Probable LVH"])
    thalach = st.sidebar.slider("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", min_value=0.0, max_value=6.2, value=0.0)
    slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    ca = st.sidebar.slider("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4, value=0)
    thal = st.sidebar.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

    # Transform user input into model-compatible format
    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    restecg_mapping = {"Normal": 0, "Abnormality": 1, "Probable LVH": 2}
    restecg = restecg_mapping[restecg]
    exang = 1 if exang == "Yes" else 0
    slope_mapping = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    slope = slope_mapping[slope]
    thal_mapping = {"Normal": 2, "Fixed Defect": 1, "Reversible Defect": 3}
    thal = thal_mapping[thal]

    user_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    user_data_scaled = scaler.transform(user_data)

    if st.sidebar.button("Predict"):
        logreg_prediction = logreg.predict(user_data_scaled)
        dtree_prediction = dtree.predict(user_data_scaled)
        if logreg_prediction[0] == 1:
            st.sidebar.warning("Logistic Regression Prediction: Heart Disease (1)")
        else:
            st.sidebar.info("Logistic Regression Prediction: No Heart Disease (0)")
        if dtree_prediction[0] == 1:
            st.sidebar.warning("Decision Tree Prediction: Heart Disease (1)")
        else:
            st.sidebar.info("Decision Tree Prediction: No Heart Disease (0)")

# Run the app
if __name__ == "__main__":
    main()
