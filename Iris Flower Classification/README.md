# Iris Flower Classification
## 1. Introduction
The Iris dataset is a well-known dataset in the machine learning community, often used for benchmarking classification algorithms. The objective of this project is to classify Iris flowers into three species—Setosa, Versicolor, and Virginica—based on their sepal and petal measurements. We use logistic regression, a simple yet effective classification algorithm, to achieve this goal.

## 2. Dataset Description
The Iris dataset contains 150 samples of iris flowers with the following features:

Sepal Length (cm)
Sepal Width (cm)
Petal Length (cm)
Petal Width (cm)
The target variable, species, has three classes:

0: Setosa
1: Versicolor
2: Virginica

## 3. Methodology
### 3.1. Data Loading
The Iris dataset is loaded using a function from the scikit-learn library. The data is then converted into a pandas DataFrame for ease of manipulation and analysis. The species column is mapped to their respective string labels for better interpretability.

### 3.2. Data Splitting
The dataset is divided into training and testing sets. 30% of the data is set aside for testing, ensuring a good balance between training and validation data. This split is done using a function from the scikit-learn library with a fixed random state to ensure reproducibility.

### 3.3. Data Standardization
Standardizing the features is crucial for logistic regression to perform well. The features are standardized by removing the mean and scaling to unit variance using a scaler from the scikit-learn library. The scaler is fitted on the training data and then applied to both the training and testing data.

### 3.4. Model Training
A logistic regression model is trained using the standardized training data. The logistic regression model from the scikit-learn library is used, with the number of iterations set to 200 to ensure convergence.

### 3.5. Model Evaluation
The model's performance is evaluated on the test data. The accuracy score is calculated, which measures the proportion of correctly classified samples. Additionally, a classification report is generated to provide detailed metrics such as precision, recall, and F1-score for each class.

### 3.6. Visualization
Two types of visualizations are created to better understand the data and model performance:

Pair Plot: A pair plot is created using the seaborn library to visualize the relationships between features, colored by species. This helps in understanding the separation between different species based on feature pairs.
Correlation Matrix: A heatmap of the correlation matrix is generated to show the linear relationships between different features. The heatmap is created using the seaborn library and helps in identifying which features are strongly correlated.

## 4. Results
### 4.1. Model Performance
The logistic regression model achieves a high accuracy, indicating its effectiveness in classifying the Iris species. The classification report provides detailed metrics for each class, showing high precision, recall, and F1-scores, indicating that the model performs well across all species.

### 4.2. Visualizations
#### Pair Plot
The pair plot visualizes the relationships between the features and the separation between different species. This plot helps in visually assessing how well the features distinguish between the three species.

#### Correlation Matrix
The correlation matrix heatmap shows the strength and direction of the linear relationships between the features. This visualization helps in understanding which features are highly correlated and can provide insights into feature selection and model interpretation.

## 5. Conclusion
This project successfully demonstrates the application of logistic regression for classifying Iris species based on their sepal and petal measurements. The model achieves a high accuracy, and the detailed classification report shows that it performs well across all classes. The visualizations provide additional insights into the feature relationships and their correlations. Future work could involve exploring more complex models or additional preprocessing steps to further improve the classification performance.
