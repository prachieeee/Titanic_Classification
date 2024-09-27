# Titanic Survival Prediction

This repository contains a machine learning project to predict passenger survival on the Titanic using multiple classification models. The dataset used is the famous Titanic dataset, which includes features like age, sex, ticket class, and more to classify whether a passenger survived the disaster.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Features](#features)
- [Models](#models)
- [Results](#results)
- [Visualization](#visualization)
- [License](#license)

## Installation

To run this project, you will need to install the following libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Dataset

The dataset used is the Titanic dataset (`titanic.csv`), which can be obtained from [Kaggle](https://www.kaggle.com/c/titanic/data). The dataset includes the following columns:

- `Survived`: The target variable (0 = No, 1 = Yes)
- `Pclass`: Passenger class (1st, 2nd, 3rd)
- `Sex`: Gender of the passenger
- `Age`: Age of the passenger
- `SibSp`: Number of siblings/spouses aboard the Titanic
- `Parch`: Number of parents/children aboard the Titanic
- `Fare`: Fare paid by the passenger

Some columns like `PassengerId`, `Name`, `Ticket`, `Cabin`, and `Embarked` were dropped as they do not contribute to survival prediction.

## Features

Several preprocessing steps were applied to clean and prepare the dataset:

1. **Handling Missing Data**:
   - Filled missing values in the `Age` and `Fare` columns with the median.
   
2. **Categorical Encoding**:
   - Converted the `Sex` column to binary values (0 for female, 1 for male).
   - Applied one-hot encoding to the `Pclass`, `SibSp`, and `Parch` columns.

## Models

Three classification models were trained to predict passenger survival:

1. **Logistic Regression**: A linear model used for binary classification.
   - The maximum number of iterations (`max_iter`) was increased to address convergence issues.
2. **Decision Tree**: A tree-based model that splits the data based on feature values.
3. **Random Forest**: An ensemble model that creates multiple decision trees and combines their predictions.

The dataset was split into training (80%) and testing (20%) sets using `train_test_split`.

## Results

Each model was evaluated based on the following metrics:

- **Accuracy Score**: Measures the proportion of correctly predicted instances.
- **Confusion Matrix**: A heatmap to visualize true positives, false positives, true negatives, and false negatives.
- **Classification Report**: Provides detailed precision, recall, and F1-score for each class (survived and not survived).

### Model Performance:

- **Logistic Regression**:
  - Accuracy: `0.8045`
  - **Classification Report**:
    ```
              precision    recall  f1-score   support

           0       0.81      0.87      0.84       105
           1       0.79      0.72      0.75        74

    accuracy                           0.80       179
   macro avg       0.80      0.79      0.80       179
weighted avg       0.80      0.80      0.80       179
    ```

- **Decision Tree**:
  - Accuracy: `0.7374`
  - **Classification Report**:
    ```
              precision    recall  f1-score   support

           0       0.78      0.77      0.78       105
           1       0.68      0.69      0.68        74

    accuracy                           0.74       179
   macro avg       0.73      0.73      0.73       179
weighted avg       0.74      0.74      0.74       179
    ```

- **Random Forest**:
  - Accuracy: `0.7877`
  - **Classification Report**:
    ```
              precision    recall  f1-score   support

           0       0.81      0.83      0.82       105
           1       0.75      0.73      0.74        74

    accuracy                           0.79       179
   macro avg       0.78      0.78      0.78       179
weighted avg       0.79      0.79      0.79       179
    ```

## Visualization

For each model, a confusion matrix is plotted to visualize the model's performance:

![Confusion Matrix](confusion_matrix.png)

## License

This project is licensed under the MIT License.
