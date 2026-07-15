# Titanic Survival Prediction

A complete machine learning classification project that predicts whether a Titanic passenger survived.

## Project Objective

The goal of this project is to build a clean, reproducible, and reliable machine learning workflow using the Titanic dataset.

The project focuses on:

- preventing data leakage
- organizing preprocessing inside pipelines
- comparing multiple classification models
- using cross-validation
- evaluating performance with several metrics
- checking overfitting
- tuning the Random Forest model
- saving the final trained pipeline

## Dataset

Expected file name:

```text
Titanic-Dataset.csv
```

Main columns used:

- `Pclass`
- `Sex`
- `Age`
- `SibSp`
- `Parch`
- `Fare`
- `Embarked`
- `Survived`

## Project Workflow

1. Load and inspect the dataset
2. Assess missing values and data quality
3. Perform survival-focused exploratory data analysis
4. Create additional features
5. Split the data into training and testing sets
6. Build numerical and categorical preprocessing pipelines
7. Train multiple classification models
8. Apply 5-fold cross-validation
9. Compare test-set performance
10. Plot and interpret the confusion matrix
11. Check overfitting
12. Tune the Random Forest model with GridSearchCV
13. Predict survival for a new passenger
14. Save the final machine learning pipeline

## Feature Engineering

### Family Size

```python
FamilySize = SibSp + Parch + 1
```

### Is Alone

```python
IsAlone = 1 if FamilySize == 1 else 0
```

### Passenger Title

Titles are extracted from passenger names and grouped into common and rare categories.

## Preprocessing

The project uses `Pipeline` and `ColumnTransformer` so preprocessing is learned from the training data only.

### Numerical Features

- missing values filled using the median
- values standardized using `StandardScaler`

### Categorical Features

- missing values filled using the most frequent category
- categories encoded using `OneHotEncoder`

This prevents data leakage and keeps preprocessing consistent during training and prediction.

## Models Compared

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

The models are also compared using 5-fold cross-validation with F1 Score.

## Hyperparameter Tuning

`GridSearchCV` is used to tune the Random Forest model.

Parameters tested include:

- number of trees
- maximum tree depth
- minimum samples required to split
- minimum samples required at a leaf

The tuned model is evaluated again on the untouched test set.

## Overfitting Check

Training and testing F1 Scores are compared. A small difference suggests that the model generalizes well, while a large difference may indicate overfitting.

## New Passenger Prediction

The final pipeline accepts readable values such as:

```python
{
    "Pclass": 3,
    "Sex": "male",
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 8.05,
    "Embarked": "S",
    "FamilySize": 1,
    "IsAlone": 1
}
```

## Saved Model

The complete final pipeline is saved as:

```text
titanic_survival_pipeline.joblib
```

## Installation

```bash
pip install pandas matplotlib scikit-learn joblib jupyter
```

## How to Run

1. Place `Titanic-Dataset.csv` in the same folder as the notebook.
2. Open the notebook in Jupyter Notebook, VS Code, or Google Colab.
3. Select **Restart Kernel and Run All Cells**.
4. Wait until all outputs, metrics, and plots are displayed.
5. Save the notebook before uploading it to GitHub.

## Files

```text
Titanic_Project.ipynb
Titanic-Dataset.csv
titanic_survival_pipeline.joblib
README.md
```

## Main Findings

The exploratory analysis shows that survival was strongly related to:

- gender
- passenger class
- age
- family-related features

The final model is selected using cross-validation and test-set evidence rather than accuracy alone.

## Limitations

- The dataset is relatively small.
- Several passenger values are missing.
- The data represents one historical event.
- Results should not be generalized beyond this dataset.
- Additional interpretation techniques such as SHAP could improve explainability.

## Future Improvements

- feature importance visualization
- SHAP explanations
- ROC and Precision-Recall curves
- Streamlit prediction interface
- automated input validation
- model deployment through FastAPI

## Author

**Layan Mousa**
