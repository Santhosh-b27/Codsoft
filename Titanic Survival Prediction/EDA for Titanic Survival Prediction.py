import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
