# Titanic Survival Prediction
## 1. Introduction
The Titanic dataset is a classic dataset used in machine learning and data science for binary classification tasks. The objective of this project is to predict whether a passenger survived the Titanic disaster based on various features such as age, sex, class, and fare. Logistic regression, a robust and interpretable classification algorithm, is used for this purpose.

## 2. Dataset Description
The Titanic dataset contains information about passengers on the Titanic, including:

Pclass: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
Sex: Gender of the passenger (male, female)
Age: Age of the passenger
SibSp: Number of siblings/spouses aboard the Titanic
Parch: Number of parents/children aboard the Titanic
Fare: Ticket fare
Embarked: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
Survived: Survival status (0 = No, 1 = Yes)

## 3. Methodology
### 3.1. Data Loading
The Titanic dataset is loaded from a CSV file into a pandas DataFrame. Basic information and the first few rows of the dataset are displayed to understand its structure and contents.

### 3.2. Data Preprocessing
Several preprocessing steps are applied to clean and prepare the data for modeling:

Handling Missing Values: Missing values in the 'Age', 'Embarked', and 'Fare' columns are filled with the median or mode values to ensure no missing data remains.
Encoding Categorical Variables: The categorical variables 'Sex' and 'Embarked' are encoded into numeric values using label encoding. This converts string labels into numeric format required for modeling.

### 3.3. Feature Selection and Target Definition
The features (independent variables) and target (dependent variable) are defined:

Features: 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'
Target: 'Survived'

### 3.4. Data Splitting
The dataset is split into training and testing sets with a 70-30 split, ensuring that 30% of the data is reserved for evaluating the model. The split is performed using a function from the scikit-learn library with a fixed random state to ensure reproducibility.

### 3.5. Data Standardization
Standardizing the features is crucial for logistic regression to perform well. The features are standardized by removing the mean and scaling to unit variance using a scaler from the scikit-learn library. The scaler is fitted on the training data and then applied to both the training and testing data.

### 3.6. Model Training
A logistic regression model is trained using the standardized training data. The logistic regression model from the scikit-learn library is used, with the number of iterations set to 200 to ensure convergence.

### 3.7. Model Evaluation
The model's performance is evaluated on the test data:

Accuracy Score: The proportion of correctly classified samples is calculated.
Classification Report: A detailed report providing precision, recall, and F1-score for each class (survived, not survived).
Confusion Matrix: A confusion matrix is plotted to visualize the true vs. predicted classifications.

## 4. Results
### 4.1. Model Performance
The logistic regression model achieves a high accuracy, indicating its effectiveness in predicting survival on the Titanic. The classification report provides detailed metrics for each class, showing good precision, recall, and F1-scores, indicating that the model performs well across both classes.

### 4.2. Visualization
Confusion Matrix
The confusion matrix is visualized using a heatmap, showing the true vs. predicted classifications. This helps in understanding the model's performance in terms of false positives, false negatives, true positives, and true negatives.

## 5. Conclusion
This project successfully demonstrates the application of logistic regression for predicting survival on the Titanic based on various passenger features. The model achieves high accuracy and performs well across all classes, as shown by the detailed classification report. The visualizations provide additional insights into the model's performance and areas for potential improvement. Future work could involve exploring more complex models or additional preprocessing steps to further enhance the classification performance.


