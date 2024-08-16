# Sales Prediction Project Report


## 1. Introduction
This project focuses on predicting sales based on advertising expenditures across different media channels: TV, Radio, and Newspaper. The dataset used contains historical data of these expenditures and corresponding sales figures. The goal is to build a predictive model using linear regression to help businesses optimize their advertising strategies and maximize sales.

## 2. Dataset Description
The dataset used in this project includes the following columns:

TV: Advertising expenditure on TV (in thousands of dollars)
Radio: Advertising expenditure on Radio (in thousands of dollars)
Newspaper: Advertising expenditure on Newspaper (in thousands of dollars)
Sales: Sales figures (in thousands of units)

## 3. Methodology
### 3.1. Data Loading and Preparation
The dataset is loaded using Pandas from a CSV file located at C:\Users\Santhosh\Sale Prediction dataset\advertising (2).csv. The features (X) include 'TV', 'Radio', and 'Newspaper', while the target variable (y) is 'Sales'.

### 3.2. Data Splitting
The dataset is split into training and testing sets using a test size of 20% and a random state of 42 for reproducibility.

### 3.3. Model Selection and Training
A linear regression model is chosen for its simplicity and interpretability. It is trained on the training data (X_train, y_train) to learn the relationship between advertising expenditures and sales.


### 3.4. Model Evaluation
The model's performance is evaluated using Mean Squared Error (MSE) and R-squared (R²) on the test data (X_test, y_test). These metrics assess how well the model predicts sales based on the advertising expenditures.

### 3.5. Visualization
Actual vs Predicted Sales Scatter Plot: This plot visually compares the actual sales (y_test) against the predicted sales (y_pred), showing how closely the predictions match the actual data points.
Residuals Distribution Plot: The distribution of residuals (the differences between actual and predicted sales) is visualized to check if the residuals are normally distributed, which is a key assumption of linear regression.

## 4. Results
### 4.1. Model Performance
Mean Squared Error (MSE): Provides a measure of the average squared difference between actual and predicted values. Lower values indicate better model performance.
R-squared (R²): Indicates the proportion of the variance in the dependent variable (sales) that is predictable from the independent variables (advertising expenditures). A value closer to 1 indicates a better fit.
### 4.2. Predictions
Users can input new advertising expenditures for TV, Radio, and Newspaper, and the model predicts the corresponding sales. For instance, after entering specific expenditures, the model predicts sales with a high degree of accuracy.

## 5. Conclusion
This project successfully demonstrates the application of linear regression for sales prediction based on advertising expenditures. The model's performance, as evaluated by MSE and R², indicates its effectiveness in capturing the relationship between advertising spends across various media channels and sales outcomes. Businesses can leverage this model to optimize their advertising budgets and maximize sales potential.


