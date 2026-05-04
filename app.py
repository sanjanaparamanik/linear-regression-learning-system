import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="Linear Regression Learning System", layout="wide")

st.title("📊 Linear Regression Learning System")

# ================= DATA INPUT =================
file = st.file_uploader("Upload CSV File", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.header(" Dataset Input")
    st.write(df.head())
    st.write("Shape:", df.shape)
    st.write("Data Types:", df.dtypes)

    # ================= PREPROCESSING =================
    st.header(" Preprocessing")

    # Missing values
    if st.checkbox("Handle Missing Values (Mean Imputation)"):
        st.info("WHY: Missing values can break model learning.")
        st.write("Before:")
        st.write(df.isnull().sum())

        df = df.fillna(df.mean(numeric_only=True))

        st.write("After:")
        st.write(df.isnull().sum())

    # Encoding
    cat_cols = df.select_dtypes(include='object').columns
    if len(cat_cols) > 0:
        method = st.selectbox("Encoding Method", ["None", "Label Encoding", "One Hot"])

        if method == "Label Encoding":
            st.info("WHY: ML models need numeric data")
            before = df.copy()
            le = LabelEncoder()
            for col in cat_cols:
                df[col] = le.fit_transform(df[col])
            st.write("Before:", before.head())
            st.write("After:", df.head())

        elif method == "One Hot":
            st.info("WHY: Avoids wrong ordinal relationship")
            before = df.copy()
            df = pd.get_dummies(df, drop_first=True)
            st.write("Before:", before.head())
            st.write("After:", df.head())

    # ================= TARGET & FEATURES =================
    target = st.selectbox("Select Target Column", df.columns)

    features = df.drop(columns=[target]).columns
    selected_features = st.multiselect(
        "Select Features",
        list(features),
        default=list(features)
    )

    if len(selected_features) == 0:
        st.error("Please select at least one feature")
        st.stop()

    X = df[selected_features]
    y = df[target]

    # ================= SCALING =================
    scaling = st.selectbox("Scaling Method", ["None", "Standardization", "Normalization"])

    if scaling != "None":
        st.info("WHY: Improves convergence of gradient descent")

        before = X.copy()

        if scaling == "Standardization":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        X = pd.DataFrame(scaler.fit_transform(X), columns=selected_features)

        st.write("Before:", before.head())
        st.write("After:", X.head())

    # ================= EDA =================
    st.header(" Exploratory Data Analysis")

    if st.checkbox("Show Histogram"):
        st.info("Shows distribution of features")
        df.hist(figsize=(10,6))
        st.pyplot(plt)

    if st.checkbox("Correlation Heatmap"):
        plt.figure(figsize=(8,5))
        sns.heatmap(df.corr(), annot=True)
        st.pyplot(plt)

    if st.checkbox("Feature vs Target Plot"):
        feature = st.selectbox("Select Feature", selected_features)
        plt.figure()
        plt.scatter(df[feature], df[target])
        plt.xlabel(feature)
        plt.ylabel(target)
        st.pyplot(plt)

    # ================= LEARNING MODULE =================
    st.header("Linear Regression Learning")

    st.subheader("Hypothesis")
    st.latex("y = w_1x_1 + w_2x_2 + ... + b")

    st.subheader("Cost Function")
    st.latex("MSE = \\frac{1}{n} \\sum (y - y_{pred})^2")

    st.subheader("Error Computation")
    st.write("Error = Actual - Predicted")

    st.subheader("Parameter Learning")
    st.write("Weights are learned using optimization (Gradient Descent or Normal Equation)")

    # ================= TRAINING =================
    st.header(" Training Configuration")

    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
    st.info(f"{int((1-test_size)*100)}% training data, {int(test_size*100)}% testing data")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Cross Validation
    st.subheader("Cross Validation")
    if st.checkbox("Enable K-Fold Cross Validation"):
        k = st.slider("K value", 2, 10, 5)
        model_cv = LinearRegression()
        scores = cross_val_score(model_cv, X, y, cv=KFold(n_splits=k))
        st.write("Cross-validation scores:", scores)
        st.write("Mean score:", np.mean(scores))

    # ================= MODEL =================
    st.header(" Model Training & Visualization")

    model = LinearRegression()
    model.fit(X_train, y_train)

    st.write("Coefficients:", model.coef_)
    st.write("Intercept:", model.intercept_)

    y_pred = model.predict(X_test)

    # Regression line (only if 1 feature)
    if len(selected_features) == 1:
        plt.figure()
        plt.scatter(X_test, y_test)
        plt.plot(X_test, y_pred)
        st.pyplot(plt)

    # ================= EVALUATION =================
    st.header(" Evaluation Metrics")

    st.write("MSE:", mean_squared_error(y_test, y_pred))
    st.write("MAE:", mean_absolute_error(y_test, y_pred))
    st.write("R² Score:", r2_score(y_test, y_pred))

    # ================= PREDICTION =================
    st.header(" Prediction & Explanation")

    inputs = []
    for col in selected_features:
        val = st.number_input(f"Enter {col}")
        inputs.append(val)

    if st.button("Predict"):
        result = model.intercept_

        st.write("### Step-by-step calculation:")
        for i in range(len(inputs)):
            calc = model.coef_[i] * inputs[i]
            st.write(f"{model.coef_[i]} × {inputs[i]} = {calc}")
            result += calc

        st.write("+ Intercept:", model.intercept_)
        st.success(f"Final Prediction = {result}")

    # ================= EDUCATIONAL =================
    st.header(" Educational Insights")

    st.info("WHY scaling? → Faster convergence and better performance")
    st.info("WHY train-test split? → Evaluate model on unseen data")
    st.info("WHY encoding? → Convert categorical data into numeric")

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    st.pyplot(fig)