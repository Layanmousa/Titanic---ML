# Titanic Survival Prediction

<p align="center">
  <img src="titanic.png" width="850">
</p>

A Machine Learning project that predicts whether a passenger survived the Titanic disaster using demographic and travel information. The project demonstrates a complete machine learning workflow including data exploration, preprocessing, feature engineering, model training, evaluation, visualization, and prediction.

---

# Project Overview

This project uses the famous Titanic dataset to build classification models capable of predicting passenger survival.

The notebook follows the standard Machine Learning pipeline from loading the data to evaluating multiple models and making predictions for new passengers.

---

# Dataset

Source:
https://www.kaggle.com/competitions/titanic

Target Variable:

**Survived**

- 0 → Passenger did not survive
- 1 → Passenger survived

Original Features include:

- PassengerId
- Pclass
- Name
- Sex
- Age
- SibSp
- Parch
- Ticket
- Fare
- Cabin
- Embarked

---

# Project Workflow

- Import Libraries
- Load Dataset
- Explore Data
- Handle Missing Values
- Feature Engineering
- Encode Categorical Features
- Split Training and Testing Data
- Train Multiple Machine Learning Models
- Evaluate Performance
- Predict New Passenger Survival

---

# Data Preprocessing

The preprocessing stage included:

- Handling missing values
- Encoding categorical variables
- Removing unnecessary columns
- Creating new informative features

Removed Columns

- PassengerId
- Name
- Ticket
- Cabin

---

# Feature Engineering

Two additional features were created:

### FamilySize

```python
FamilySize = SibSp + Parch + 1
```

Represents the total number of family members traveling together.

### IsAlone

```python
IsAlone = 1 if FamilySize == 1 else 0
```

Indicates whether the passenger was traveling alone.

---

# Models Used

The following Machine Learning algorithms were implemented:

- Logistic Regression
- Decision Tree
- Random Forest

---

# Model Performance

| Model | Accuracy |
|--------|----------|
| Logistic Regression | **(Your Score)** |
| Decision Tree | **(Your Score)** |
| Random Forest | **(Your Score)** |

---

# Confusion Matrix

<p align="center">
  <img src="confusion_matrix.png" width="700">
</p>

The confusion matrix shows the number of correctly and incorrectly classified passengers, providing deeper insight into model performance beyond accuracy alone.

---

# Model Comparison

<p align="center">
  <img src="model_comparison.png" width="700">
</p>

The comparison illustrates the performance differences between the implemented machine learning models.

---

# Predicting a New Passenger

The trained model was tested using manually created passenger information.

Example:

```python
Age = 25
Sex = Male
Pclass = 3
Fare = 8.05
```

The model predicts whether this passenger would survive or not.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook

---

# Project Structure

```
Titanic-ML/
│
├── TitanicProject.ipynb
├── Titanic-Dataset.csv
├── README.md
├── titanic.png
├── confusion_matrix.png
└── model_comparison.png
```

---

# Future Improvements

Potential enhancements include:

- Hyperparameter tuning
- Cross-validation
- Feature selection
- XGBoost
- LightGBM
- ROC Curve and AUC analysis
- Precision, Recall, and F1-score evaluation

---

# Conclusion

This project demonstrates a complete end-to-end machine learning classification workflow.

Starting from raw Titanic passenger data, the project performs preprocessing, feature engineering, model training, evaluation, and prediction.

The comparison between Logistic Regression, Decision Tree, and Random Forest highlights how different algorithms perform on the same dataset while emphasizing the importance of feature engineering in improving predictive accuracy.

---

## Author

**Layan Mousa**


GitHub:
https://github.com/Layanmousa
