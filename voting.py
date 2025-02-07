import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def load_and_preprocess_data(file_path):
    # Load data
    data = pd.read_csv("C:\\Users\\asus\\OneDrive\\Desktop\\final project\\data_clean.csv")
    data.drop(columns=['Unnamed: 0'], inplace=True)
    
    # Encode the diagnosis column
    label_encoder = LabelEncoder()
    data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])

    # Select features based on correlation with target variable
    corr_matrix = data.corr()
    corr_with_target = abs(corr_matrix['diagnosis']).sort_values(ascending=False)
    selected_features = list(corr_with_target[1:6].index)  # Select top 5 features based on correlation (excluding 'diagnosis')

    # Split data into features and labels
    X = data[selected_features]
    y = data['diagnosis']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dimensionality reduction using PCA
    pca = PCA(n_components=5)  # Adjust the number of components as needed
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, y, selected_features, scaler, pca

def train_voting_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define individual classifiers
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    svm_classifier = SVC(probability=True, random_state=42)

    # Create a Voting Classifier
    voting_model = VotingClassifier(estimators=[
        ('knn', knn_classifier),
        ('rf', rf_classifier),
        ('svm', svm_classifier)
    ], voting='soft')

    # Train the Voting Classifier
    voting_model.fit(X_train, y_train)

    # Make predictions and calculate metrics
    y_pred = voting_model.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred)
    # recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    classifier_score = voting_model.score(X_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return voting_model,classifier_score,cm,report,X_test,y_test,y_pred

def make_prediction(clf, scaler,pca, input_data):
    input_data = np.array(input_data).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    input_data_pca=pca.transform(input_data_scaled)
    prediction = clf.predict(input_data_scaled)
    prediction_proba = clf.predict_proba(input_data_scaled)

    return prediction, prediction_proba