# Titanic Survival Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A complete machine learning classification project that predicts whether a Titanic passenger survived using a clean, reproducible, and leakage-safe workflow.

## Project Objective

The goal of this project is to build a reliable binary-classification workflow using the Titanic dataset.

The project focuses on:

- preventing data leakage
- organizing preprocessing inside reusable pipelines
- comparing multiple classification models
- using stratified cross-validation
- evaluating performance with multiple metrics
- checking for overfitting
- tuning a Random Forest model
- selecting the final model using training-set cross-validation
- saving the trained pipeline and its inference metadata

## Dataset

Expected file name:

```text
Titanic-Dataset.csv
```

Main columns used:

- `Pclass`
- `Name`
- `Sex`
- `Age`
- `SibSp`
- `Parch`
- `Fare`
- `Embarked`
- `Survived`

The notebook validates the expected dataset schema before training begins.

## Machine Learning Workflow

```text
Dataset
   ↓
Data Quality Assessment
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Stratified Train-Test Split
   ↓
Preprocessing Pipeline
   ↓
Cross-Validation
   ↓
Random Forest Hyperparameter Tuning
   ↓
Final Model Selection
   ↓
Untouched Test-Set Evaluation
   ↓
New Passenger Prediction
   ↓
Model Bundle Saving
```

The full workflow is:

1. Load and validate the dataset
2. Inspect missing values, duplicates, and data quality
3. Perform survival-focused exploratory data analysis
4. Create additional passenger-level features
5. Split the data using a stratified train-test split
6. Build numerical and categorical preprocessing pipelines
7. Compare Logistic Regression, Decision Tree, and Random Forest
8. Apply five-fold stratified cross-validation using F1-score
9. Tune the Random Forest model using `GridSearchCV`
10. Select the final model using cross-validation results only
11. Fit the selected model on the complete training set
12. Evaluate it once on the untouched test set
13. Plot and interpret the confusion matrix
14. Compare training and testing F1-scores
15. Predict survival for a new passenger
16. Save the trained model and inference metadata

## Feature Engineering

### Family Size

```python
FamilySize = SibSp + Parch + 1
```

This combines the passenger, siblings, spouses, parents, and children into one family-size feature.

### Is Alone

```python
IsAlone = 1 if FamilySize == 1 else 0
```

This identifies passengers who traveled without immediate family members.

### Passenger Title

Titles are extracted from passenger names and normalized into common groups such as:

- `Mr`
- `Miss`
- `Mrs`
- `Master`
- `Rare`

Rare and uncommon titles are grouped together to reduce unnecessary category fragmentation.

## Preprocessing

The project uses `Pipeline` and `ColumnTransformer` so all learned preprocessing steps are fitted using the training data only.

### Numerical Features

- missing values filled using the median
- values standardized using `StandardScaler`

### Categorical Features

- missing values filled using the most frequent category
- categories encoded using `OneHotEncoder`
- unseen categories handled safely during prediction

Using a pipeline provides several advantages:

- prevents data leakage
- keeps training and prediction preprocessing consistent
- improves reproducibility
- makes the final model easier to reuse and deploy

## Models Compared

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Tuned Random Forest Classifier

## Model Selection Strategy

The final model is selected using mean five-fold stratified cross-validation F1-score on the training data.

The test set is not used to choose the model or tune hyperparameters. It is kept untouched until the final model has been selected and fitted.

This avoids test-set leakage and provides a more reliable estimate of generalization performance.

## Evaluation Metrics

The project reports:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Mean cross-validation F1-score
- Cross-validation standard deviation
- Training-test F1 gap

F1-score is used as the main model-selection metric because it balances precision and recall.

## Hyperparameter Tuning

`GridSearchCV` is used to tune the Random Forest model using the training data only.

Parameters tested include:

- number of trees
- maximum tree depth
- minimum samples required to split
- minimum samples required at a leaf

The tuned model is then compared with the baseline models using the same cross-validation strategy.

## Confusion Matrix Interpretation

The confusion matrix reports:

- **True Negatives:** passengers correctly predicted not to survive
- **False Positives:** passengers incorrectly predicted to survive
- **False Negatives:** surviving passengers incorrectly predicted not to survive
- **True Positives:** passengers correctly predicted to survive

False negatives are especially important because they represent surviving passengers missed by the model.

## Overfitting Check

Training and testing F1-scores are compared.

A small absolute difference suggests that the model generalizes reasonably well. A large difference may indicate overfitting.

This comparison is treated as a diagnostic check rather than the main model-selection criterion.

## Reproducibility

The notebook uses:

```python
RANDOM_STATE = 42
```

This keeps the train-test split, cross-validation behavior, and model training reproducible across repeated runs.

## New Passenger Prediction

The notebook starts with readable passenger information:

```python
{
    "Pclass": 3,
    "Name": "Example, Mr. Passenger",
    "Sex": "male",
    "Age": 25,
    "SibSp": 0,
    "Parch": 0,
    "Fare": 8.05,
    "Embarked": "S"
}
```

The same reusable feature-engineering function used during training creates:

- `FamilySize`
- `IsAlone`
- `Title`

The preprocessing and classification pipeline then generates the final prediction and survival probability.

## Saved Model Bundle

The final trained model and its inference metadata are saved together as:

```text
titanic_survival_bundle.joblib
```

The bundle includes:

- trained preprocessing and classification pipeline
- final model name
- expected feature columns
- random-state value

The file is generated when the notebook is executed and should normally be excluded from version control.

## Installation

Install the required packages using:

```bash
pip install -r requirements.txt
```

Alternatively:

```bash
pip install pandas matplotlib scikit-learn joblib notebook
```

## How to Run

1. Clone or download the repository.
2. Place `Titanic-Dataset.csv` in the repository root or beside the notebook.
3. Open `Titanic_Project_.ipynb` in Jupyter Notebook, JupyterLab, VS Code, or Google Colab.
4. Select **Restart Kernel and Run All Cells**.
5. Confirm that all cells run without errors.
6. Review the generated metrics, plots, conclusion, and saved model bundle.

## Repository Structure

```text
Titanic---ML/
├── Titanic_Project_.ipynb
├── Titanic-Dataset.csv
├── README.md
├── requirements.txt
└── .gitignore
```

Generated model files such as `*.joblib` should be ignored by Git and recreated by running the notebook.

## Main Findings

The exploratory analysis indicates that survival was strongly associated with:

- gender
- passenger class
- age
- family-related features
- passenger title

The final model is selected using cross-validation rather than test-set accuracy alone.

The notebook then performs one final evaluation on the untouched test set.

## Limitations

- The dataset is relatively small.
- Several passenger fields contain missing values.
- The data represents one historical event.
- The relationships are observational and should not be interpreted as causal.
- Performance may vary with a different sample or validation strategy.
- Results should not be generalized beyond this dataset.

## Future Improvements

Possible extensions include:

- permutation feature importance
- SHAP explanations
- ROC and Precision-Recall curves
- probability calibration
- fairness analysis
- automated tests for feature engineering
- Streamlit prediction interface
- model deployment through FastAPI

## Author

**Layan Mousa**
