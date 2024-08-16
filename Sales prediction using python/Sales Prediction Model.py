import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv(r"C:\Users\Santhosh\Sale Prediction dataset\advertising (2).csv")

# Features and target variable
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# Visualize Actual vs Predicted Sales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='w', linewidth=0.5)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.show()

# Visualize Residuals Distribution
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.title('Residuals Distribution')
plt.xlabel('Residuals')
plt.show()

# Predict sales for unknown data based on user input
def predict_sales(tv, radio, newspaper):
    # Create a DataFrame for the new input
    new_data = pd.DataFrame({'TV': [tv], 'Radio': [radio], 'Newspaper': [newspaper]})
    
    # Predict sales using the trained model
    predicted_sales = model.predict(new_data)
    return predicted_sales[0]

# Get user input
print("Enter the advertising expenditures for prediction:")
tv = float(input("TV advertising expenditure: "))
radio = float(input("Radio advertising expenditure: "))
newspaper = float(input("Newspaper advertising expenditure: "))

# Predict and display the result
predicted_sales = predict_sales(tv, radio, newspaper)
print(f"Predicted Sales for TV={tv}, Radio={radio}, Newspaper={newspaper}: ${predicted_sales:.2f}")
