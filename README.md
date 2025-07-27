# AIML_PA_17.1_Comparing_Classifiers

--- 

# Overview : Term Deposit Subscription Prediction using Classification Models

This project explores various supervised learning classifiers to predict whether a client will subscribe to a term deposit, based on data from a Portuguese banking institution. The work includes data preprocessing, baseline modeling, hyperparameter tuning, performance visualization and interpretation of results.

--- 

## Objective

- Predict client subscription (`y`: yes/no) using demographic and marketing attributes
- Benchmark different classifiers (KNN, Logistic Regression, Decision Tree, SVM)
- Apply GridSearchCV for hyperparameter optimization
- Interpret influential features using Logistic Regression

---
## Important Links

[Bank Term Deposit Marketing Dataset](https://github.com/amitkushwaha2000/AIML_PA_17.1_Comparing_Classifiers/blob/main/bank-additional-full.csv)

[Jupyter Notebook](https://github.com/amitkushwaha2000/AIML_PA_17.1_Comparing_Classifiers/blob/main/prompt_III_AK.ipynb)

---

## Dataset Overview

- **Source**: [UCI ML Repository – Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- **Observations**: 41,188
- **Target Variable**: `y` (subscription to term deposit)
- **Class Balance**: Highly imbalanced (~11.3% positive cases)

---

## Preprocessing

- One-hot encoding for categorical features
- PCA on numeric variables (optional)
- Standard scaling
- Stratified train-test split (80-20)

> Baseline classifier always predicted "no" — accuracy was high (0.8873) but **F1 Score was 0.0**, highlighting the need for smarter models.

---

## Model Training & Default Evaluation

|	Model              | 	Train Time (s)  | 	Train Accuracy 	|Test Accuracy  |
|--------------------|------------------|-------------------|---------------|
|Logistic Regression | 	0.2162          |      0.8998       |    0.9011     |
|Decision Tree 	     |  1.6074          |     	0.9954      | 	0.8410      |
|K-Nearest Neighbors |	0.0156 	        |    0.9132 	      |  0.8945       |
|SVM 	               |    175.8509      |       	0.9061 	  |  0.9024       |

> **Observation**:  
> - KNN & Logistic Regression were the fastest and most generalizable  
> - Decision Tree overfit heavily  
> - SVM showed higher accuracy but required excessive computation  

---

## Hyperparameter Tuning (GridSearchCV)

| Model                 | GridSearch Time (s) | Best Parameters                                    | Test Accuracy | F1 Score |
|----------------------|----------------------|----------------------------------------------------|---------------|----------|
| **K-Nearest Neighbors** | 647.92             | `{'n_neighbors': 5}`                               | 0.8945        | 0.3762   |
| **Logistic Regression** | 2.26              | `{'C': 100, 'penalty': 'l2', 'solver': 'lbfgs'}`    | 0.9013        | 0.3417   |
| **Decision Tree**    | 7.15                 | `{'criterion': 'gini', 'max_depth': 5, 'min_samples_split': 2}` | 0.9016        | 0.3336   |
| **SVM**              | 847.93               | `{'C': 0.1, 'kernel': 'linear'}`                   | 0.8977        | 0.3073   |

---

## Model Performance Visualization

![Model Comparison - Accuracy, F1 Score, ROC AUC](images/5a706e83-8378-484a-bffa-b2eeb841e1ae.png)

> Logistic Regression remained the most balanced model across all metrics  
> KNN improved F1 but was slow  
> ROC AUC for SVM (default) was **NaN** due to absence of `probability=True` during fitting (which was done delierately to reduce the computational requirements)

---

## Feature Importance from Logistic Regression

The top 10 influential features (by absolute coefficient value):

![Top Logistic Regression Features](https://github.com/amitkushwaha2000/AIML_PA_17.1_Comparing_Classifiers/blob/main/LogReg%20To%20Influential%20Features.png)

**Key Interpretations:**

- **Positive Influencers**:
  - `cat__month_mar` and `cat__poutcome_success` significantly increased likelihood of subscription.
  - `cat__poutcome_nonexistent`, `cat__job_retired` also positively influenced the outcome.

- **Negative Influencers**:
  - Contact via telephone (`cat__contact_telephone`)
  - Campaigns in May and September (`cat__month_may`, `cat__month_sep`)
  - Some PCA components had influence but were not directly interpretable

> Logistic Regression’s transparency makes it valuable for actionable marketing strategy.

---

## Final Recommendations

| Model               | Recommended Use |
|--------------------|------------------|
| **Logistic Regression** | Best balance of accuracy, speed, interpretability |
| **KNN**                | Improved recall; consider if memory is not a constraint |
| **Decision Tree**      | Only viable with pruning (max depth tuning) |
| **SVM**                | Computationally expensive with only marginal gain |

---

## Next Steps

- Try ensemble methods (Random Forest, XGBoost)
- Apply SMOTE or class weights for better handling imbalance
- Use SHAP/LIME for interpretable ML insights

---

## Acknowledgements

- [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- scikit-learn, pandas, matplotlib, seaborn
