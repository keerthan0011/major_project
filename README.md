# Breast Cancer Detection Using Voting Classifier

## Overview
This project implements a **Breast Cancer Detection Model** using machine learning techniques. The model applies **Principal Component Analysis (PCA)** for dimensionality reduction and utilizes an **ensemble voting classifier** consisting of **K-Nearest Neighbors (KNN), Random Forest (RF), and Support Vector Machine (SVM)** to classify tumors as malignant or benign.

## Dataset
The model is trained on a **preprocessed dataset** (`data_clean.csv`), where:
- Features are selected based on correlation with the target variable (diagnosis).
- The **'diagnosis'** column is **label-encoded** (0 for benign, 1 for malignant).
- Features are **standardized** using `StandardScaler`.
- **PCA** is applied to reduce dimensionality before classification.

## Dependencies
Ensure you have the following libraries installed:

```bash
pip install pandas numpy scikit-learn
```

## File Structure
```
📂 Breast_Cancer_Detection
│── 📄 main.py                  # Main script for training and testing the model
│── 📄 data_clean.csv            # Preprocessed dataset
│── 📄 README.md                 # Documentation
```

## Implementation
### 1. Load and Preprocess Data
```python
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data.drop(columns=['Unnamed: 0'], inplace=True)
    label_encoder = LabelEncoder()
    data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])
    
    corr_matrix = data.corr()
    corr_with_target = abs(corr_matrix['diagnosis']).sort_values(ascending=False)
    selected_features = list(corr_with_target[1:6].index)  # Top 5 correlated features

    X = data[selected_features]
    y = data['diagnosis']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca, y, selected_features, scaler, pca
```

### 2. Train a Voting Classifier
```python
def train_voting_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    svm_classifier = SVC(probability=True, random_state=42)
    
    voting_model = VotingClassifier(estimators=[
        ('knn', knn_classifier),
        ('rf', rf_classifier),
        ('svm', svm_classifier)
    ], voting='soft')
    
    voting_model.fit(X_train, y_train)
    y_pred = voting_model.predict(X_test)
    classifier_score = voting_model.score(X_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return voting_model, classifier_score, cm, report, X_test, y_test, y_pred
```

### 3. Make Predictions
```python
def make_prediction(clf, scaler, pca, input_data):
    input_data = np.array(input_data).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    input_data_pca = pca.transform(input_data_scaled)
    prediction = clf.predict(input_data_pca)
    prediction_proba = clf.predict_proba(input_data_pca)
    return prediction, prediction_proba
```

## Usage
### Train and Evaluate Model
```python
X_pca, y, selected_features, scaler, pca = load_and_preprocess_data("data_clean.csv")
voting_model, score, cm, report, X_test, y_test, y_pred = train_voting_model(X_pca, y)
print("Model Accuracy:", score)
```

### Make Predictions on New Data
```python
input_data = [value1, value2, value3, value4, value5]  # Replace with actual feature values
prediction, prediction_proba = make_prediction(voting_model, scaler, pca, input_data)
print("Prediction:", prediction)
print("Probability:", prediction_proba)
```

## Evaluation Metrics
- **Accuracy Score**
- **F1 Score**
- **Confusion Matrix**
- **Classification Report**

## Future Improvements
- Increase feature selection for better accuracy
- Implement a hyperparameter tuning method
- Deploy as a web app using Flask or Streamlit

## License
This project is for educational purposes. Modify and use as needed.

