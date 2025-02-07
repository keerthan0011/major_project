import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from svm import load_and_preprocess_data as svm_load_and_preprocess_data, train_svm_model, make_prediction as svm_make_prediction
from knn import load_and_preprocess_data as knn_load_and_preprocess_data, train_knn_model, make_prediction as knn_make_prediction
from voting import load_and_preprocess_data as voting_load_and_preprocess_data, train_voting_model, make_prediction as voting_make_prediction

# Function to load and preprocess data for the selected model
def load_and_preprocess_data(file_path, model):
    if model == 'SVM':
        return svm_load_and_preprocess_data("C:\\Users\\asus\\OneDrive\\Desktop\\final project\\svm.py")
    elif model == 'KNN':
        return knn_load_and_preprocess_data("C:\\Users\\asus\\OneDrive\\Desktop\\final project\\knn.py")
    elif model == 'Voting':
        return voting_load_and_preprocess_data("C:\\Users\\asus\\OneDrive\\Desktop\\final project\\voting.py")
    else:
        raise ValueError("Invalid model selection.")

# Function to train the selected model
def train_model(X_selected, y, model):
    if model == 'SVM':
        return train_svm_model(X_selected, y)
    elif model == 'KNN':
        return train_knn_model(X_selected, y)
    elif model == 'Voting':
        return train_voting_model(X_selected, y)
    else:
        raise ValueError("Invalid model selection.")

# Function to make predictions using the selected model
def make_prediction(clf, scaler, input_data, model, pca=None):
    if model == 'SVM':
        return svm_make_prediction(clf, scaler,pca, input_data)
    elif model == 'KNN':
        return knn_make_prediction(clf, scaler, pca, input_data)
    elif model == 'Voting':
        return voting_make_prediction(clf, scaler, pca, input_data)
    else:
        raise ValueError("Invalid model selection.")

# Streamlit app
st.title("Breast Cancer Prediction")

# Sidebar for model selection
model_option = st.sidebar.selectbox("Select Model", ["SVM", "KNN","Voting"])

# Load and preprocess data based on the selected model
file_path = 'C:\\Users\\asus\\OneDrive\\Desktop\\final project\\data_clean.csv'
X_selected, y, selected_feature_names, scaler, additional = load_and_preprocess_data(file_path, model_option)

# Train and evaluate the model based on the selected model
clf, classifier_score, cm, report, X_test, y_test, y_pred = train_model(X_selected, y, model_option)

# Display selected features and accuracy score
st.sidebar.title("Model Configuration")
st.markdown(f"*Selected Features:* {', '.join(selected_feature_names)}")
st.markdown(f"*Classifier accuracy score:* {classifier_score:.2f}")

# User input for selected features
st.sidebar.title("Enter Input Features:")
input_data = []
for feature in selected_feature_names:
    value = st.sidebar.number_input(f"{feature}:", step=0.01)
    input_data.append(value)

# Make prediction based on the selected model
if st.sidebar.button("Predict"):
    if model_option == 'SVM':
        prediction, prediction_proba = make_prediction(clf, scaler, input_data, model_option,pca=additional)
    elif model_option == 'KNN':
        prediction, prediction_proba = make_prediction(clf, scaler, input_data, model_option, pca=additional)
    elif model_option == 'Voting':
        prediction, prediction_proba = make_prediction(clf, scaler, input_data, model_option, pca=additional)


    result = "Malignant" if prediction[0] == 1 else "Benign"
    st.markdown(f"*Prediction:* {result}")
    st.markdown(f"*Probability:* {prediction_proba[0][prediction[0]]:.2f}")
    st.markdown(f"*Accuracy:* {classifier_score:.2f}")

    # Save predictions to a text file
    with open("predictions.txt", "a") as file:
        file.write(f"Input: {input_data}, Prediction: {result}, Probability: {prediction_proba[0][prediction[0]]:.2f}\n")

    # Display classification report as table
    st.subheader("Classification Report:")
    df_report = pd.DataFrame(report).transpose()  # Convert report to DataFrame
    st.table(df_report)

    # Display confusion matrix
    st.subheader("Confusion Matrix:")
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    # Plot the prediction probabilities
    st.subheader("Prediction Probability:")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=['Benign', 'Malignant'], y=prediction_proba[0], palette='viridis', ax=ax)
    ax.set_title('Prediction Probability')
    ax.set_ylabel('Probability')
    ax.set_xlabel('Diagnosis')
    st.pyplot(fig)

# Visualizations
st.title("Data Visualizations")

# Load the raw data for visualizations
df = pd.read_csv(file_path)
df.drop('Unnamed: 0', axis=1, inplace=True)

# Histograms
st.subheader("Histograms of Selected Features")
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
for ax, feature in zip(axes.flatten(), selected_feature_names):
    sns.histplot(df[feature], kde=True, ax=ax)
    ax.set_title(f'Histogram of {feature}')
st.pyplot(fig)

# Density Plots
st.subheader("Density Plots of Selected Features")
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
for ax, feature in zip(axes.flatten(), selected_feature_names):
    sns.kdeplot(df[feature], ax=ax)
    ax.set_title(f'Density Plot of {feature}')
st.pyplot(fig)

# Box Plots
st.subheader("Box Plots of Selected Features")
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
for ax, feature in zip(axes.flatten(), selected_feature_names):
    sns.boxplot(y=df[feature], ax=ax)
    ax.set_title(f'Box Plot of {feature}')
st.pyplot(fig)