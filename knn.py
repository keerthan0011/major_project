import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv("C:\\Users\\asus\\OneDrive\\Desktop\\final project\\data_clean.csv")
    df.drop(columns=['Unnamed: 0'], inplace=True)

    # Encode the diagnosis column
    label_encoder = LabelEncoder()
    df['diagnosis'] = label_encoder.fit_transform(df['diagnosis'])

    # Calculate correlation matrix
    corr_matrix = df.corr()
    corr_with_diagnosis = corr_matrix['diagnosis'].abs().sort_values(ascending=False)

    # Select top 5 features with highest correlation to the target variable
    selected_features = corr_with_diagnosis.index[1:6]  # excluding 'diagnosis' itself
    print("Selected Features:", selected_features)

    # Split data into features and labels
    X = df[selected_features]
    y = df['diagnosis']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dimensionality reduction using PCA
    pca = PCA(n_components=5)  # Adjust the number of components as needed
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, y, selected_features, scaler, pca

def train_knn_model(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

    # Train a KNN model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    # Analyze accuracy of predictions on the test set
    classifier_score = knn.score(X_test, y_test)
    y_pred = knn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return knn, classifier_score, cm, report, X_test, y_test, y_pred

def make_prediction(clf, scaler, pca, input_data):
    input_data = np.array(input_data).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    input_data_pca = pca.transform(input_data_scaled)
    prediction = clf.predict(input_data_pca)
    prediction_proba = clf.predict_proba(input_data_pca)

    return prediction, prediction_proba