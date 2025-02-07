import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA

def load_and_preprocess_data(file_path):
    # Load data
    df = pd.read_csv("C:\\Users\\asus\\OneDrive\\Desktop\\final project\\data_clean.csv")
    df.drop('Unnamed: 0', axis=1, inplace=True)

    # Encode the diagnosis column
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis'])

    # Calculate correlation matrix
    corr_matrix = df.corr()
    corr_with_diagnosis = corr_matrix['diagnosis'].abs().sort_values(ascending=False)

    # Select top 5 features with highest correlation to the target variable
    selected_features = corr_with_diagnosis.index[1:6]  # excluding 'diagnosis' itself
    print("Selected Features:", selected_features)

    # Split data into features and labels
    X = df[selected_features]
    y = df['diagnosis']

    # Normalize the selected features (center around 0 and scale to remove the variance)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dimensionality reduction using PCA
    pca = PCA(n_components=5)  # Adjust the number of components as needed
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, y, selected_features, scaler, pca

def train_svm_model(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2, stratify=y)
    
    # Create an SVM classifier and train it on 70% of the data set
    clf = SVC(probability=True)
    clf.fit(X_train, y_train)

    # Analyze accuracy of predictions on 30% of the holdout test sample
    classifier_score = clf.score(X_test, y_test)

    # Get average of 5-fold cross-validation score using an SVC estimator
    n_folds = 5
    cv_error = np.average(cross_val_score(SVC(), X, y, cv=n_folds))

    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return clf, classifier_score, cm, report, X_test, y_test, y_pred

def make_prediction(clf, scaler, pca, input_data):
    input_data = np.array(input_data).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    input_data_pca = pca.transform(input_data_scaled)
    prediction = clf.predict(input_data_pca)
    prediction_proba = clf.predict_proba(input_data_pca)

    return prediction, prediction_proba