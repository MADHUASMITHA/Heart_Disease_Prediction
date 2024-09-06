
# Heart Disease Prediction using Machine Learning

This project implements heart disease prediction using three machine learning models: Support Vector Machine (SVM), Decision Tree, and Logistic Regression. The dataset used in this project contains various health-related features, and the goal is to predict whether a person has heart disease or not (binary classification).

## Prerequisites

Ensure that you have Python 3.7+ installed on your machine. You'll also need the following Python libraries:

### Required Libraries:
1. pandas
2. scikit-learn

To install the necessary libraries, run the following command in your terminal:

\`\`\`bash
pip install pandas scikit-learn
\`\`\`

## Dataset

The dataset used in this project is \`Heart_Disease_Dataset.csv\`. It includes various features such as age, sex, blood pressure, cholesterol levels, and more, with the target column indicating the presence of heart disease.

### Features:

The dataset should have columns such as:

- \`age\`: Age of the patient
- \`sex\`: Gender (1 = male, 0 = female)
- \`cp\`: Chest pain type (categorical)
- \`trestbps\`: Resting blood pressure (in mm Hg)
- \`chol\`: Serum cholesterol (in mg/dl)
- \`fbs\`: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- \`restecg\`: Resting electrocardiographic results (categorical)
- \`thalach\`: Maximum heart rate achieved
- \`exang\`: Exercise induced angina (1 = yes, 0 = no)
- \`oldpeak\`: ST depression induced by exercise relative to rest
- \`slope\`: The slope of the peak exercise ST segment (categorical)
- \`ca\`: Number of major vessels (0-3) colored by fluoroscopy
- \`thal\`: Thalassemia (categorical)
- \`target\`: Whether the patient has heart disease (1 = heart disease, 0 = no heart disease)

## Running the Project

1. Clone the repository or download the code to your local machine.
2. Make sure you have the dataset \`Heart_Disease_Dataset.csv\` in your working directory.
3. Run the Python script.

\`\`\`bash
python heart_disease_prediction.py
\`\`\`

### Code Overview:

1. **Loading the dataset**:
   The heart disease dataset is loaded using \`pandas\`.

   \`\`\`python
   heart_df = pd.read_csv('/path/to/Heart_Disease_Dataset.csv')
   \`\`\`

2. **Data Preprocessing**:
   The dataset is split into features (X) and target (y). Then, the data is divided into training and testing sets using the \`train_test_split\` function from \`sklearn.model_selection\`.

   \`\`\`python
   X_train, X_test, y_train, y_test = train_test_split(heart_df.drop('target', axis=1), heart_df['target'], test_size=0.2, random_state=42)
   \`\`\`

3. **Model Creation and Training**:
   Three different classifiers are created and trained: Support Vector Machine (SVM), Decision Tree, and Logistic Regression.

   - **SVM Classifier**: 
     \`\`\`python
     svm_clf = SVC(kernel='linear', random_state=42)
     svm_clf.fit(X_train, y_train)
     \`\`\`

   - **Decision Tree Classifier**: 
     \`\`\`python
     dt_clf = DecisionTreeClassifier(random_state=42)
     dt_clf.fit(X_train, y_train)
     \`\`\`

   - **Logistic Regression Classifier**: 
     \`\`\`python
     lr_clf = LogisticRegression(random_state=42)
     lr_clf.fit(X_train, y_train)
     \`\`\`

4. **Model Predictions**:
   Each model is used to make predictions on the test data.

   \`\`\`python
   svm_preds = svm_clf.predict(X_test)
   dt_preds = dt_clf.predict(X_test)
   lr_preds = lr_clf.predict(X_test)
   \`\`\`

5. **Accuracy Calculation**:
   The accuracy of each model is calculated using \`accuracy_score\` from \`sklearn.metrics\`.

   \`\`\`python
   svm_acc = accuracy_score(y_test, svm_preds)
   dt_acc = accuracy_score(y_test, dt_preds)
   lr_acc = accuracy_score(y_test, lr_preds)
   \`\`\`

6. **Output**:
   The accuracy of each classifier is printed in percentage form.

   \`\`\`python
   print("Accuracy of SVM: {:.2f}%".format(svm_acc*100))
   print("Accuracy of Decision Tree: {:.2f}%".format(dt_acc*100))
   print("Accuracy of Logistic Regression: {:.2f}%".format(lr_acc*100))
   \`\`\`

## Example Output:

\`\`\`bash
Accuracy of SVM: 85.00%
Accuracy of Decision Tree: 80.00%
Accuracy of Logistic Regression: 83.50%
\`\`\`

## License

This project is licensed under the MIT License.
