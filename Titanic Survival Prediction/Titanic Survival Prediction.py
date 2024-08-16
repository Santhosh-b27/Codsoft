import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the Titanic dataset
df = pd.read_csv(r"D:\Project Codsoft\Titanic Survival Prediction\Titanic-Dataset.csv")

# Display basic info and first few rows
print("Basic Info:")
print(df.info())
print("\nFirst Few Rows:")
print(df.head())

# Descriptive statistics for numerical features
print("\nDescriptive Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Encode categorical variables
label_encoders = {}
for column in ['Sex', 'Embarked']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Define features and target
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model
logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = logistic_model.predict(X_test_scaled)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

# Plot confusion matrix for Logistic Regression
print("Confusion Matrix for Logistic Regression:")
plot_confusion_matrix(y_test, y_pred, title='Logistic Regression Confusion Matrix')

# Exploratory Data Analysis (EDA)
# Distribution of target variable
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=df)
plt.title('Distribution of Survival')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Not Survived', 'Survived'])
plt.show()

# Distribution of categorical features
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

sns.countplot(x='Pclass', data=df, ax=axes[0, 0])
axes[0, 0].set_title('Distribution of Passenger Class')

sns.countplot(x='Sex', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Distribution by Sex')

sns.countplot(x='Embarked', data=df, ax=axes[1, 0])
axes[1, 0].set_title('Distribution of Embarked')

# Visualizing Age distribution
sns.histplot(df['Age'].dropna(), bins=30, kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Age Distribution')

plt.tight_layout()
plt.show()

# Survival by Pclass
plt.figure(figsize=(10, 6))
sns.barplot(x='Pclass', y='Survived', data=df)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()

# Survival by Sex
plt.figure(figsize=(10, 6))
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Sex')
plt.xlabel('Sex')
plt.ylabel('Survival Rate')
plt.show()

# Survival by Embarked
plt.figure(figsize=(10, 6))
sns.barplot(x='Embarked', y='Survived', data=df)
plt.title('Survival Rate by Embarked')
plt.xlabel('Embarked')
plt.ylabel('Survival Rate')

# Dynamically set the labels based on unique values in 'Embarked'
embarked_labels = df['Embarked'].unique()
plt.xticks(ticks=np.arange(len(embarked_labels)), labels=embarked_labels)
plt.show()

# Correlation heatmap of numerical features
plt.figure(figsize=(10, 8))
corr = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
