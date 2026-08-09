# Machine Learning Assignment 2 - DigitLens Classifier

> **Important:** Replace the two placeholder links below with your own GitHub and deployed Streamlit URLs before submission. Run the notebook/scripts in the **BITS Virtual Lab** and take the required screenshot there.

## a. Problem statement

The objective is to build and compare multiple supervised classification models for recognizing handwritten digits, evaluate them using the six metrics required by the assignment, and expose the trained models through an interactive Streamlit application. The app supports CSV upload, model selection, evaluation metrics, a confusion matrix/classification report, and visible predictions on test data.

## b. Dataset description

**Dataset:** Optical Recognition of Handwritten Digits, UCI Machine Learning Repository (Dataset ID 80).  
**UCI DOI:** https://doi.org/10.24432/C50P49  
**Implementation data:** scikit-learn's bundled 1,797-row copy of the UCI test portion.  
**Features:** 64 integer pixel-intensity features (8 x 8 image), values 0-16.  
**Target:** digit class 0-9 (10 classes).  
**Missing values:** none in the source data.  
**Train/test split:** 75% / 25%, stratified, `random_state=314`.  

The assignment requires at least 12 features and 500 instances; this implementation uses 64 features and 1,797 instances.

## c. GitHub Repository Link

**GitHub Repository:** https://github.com/suditpatnaik-coder/ml-assignment-2

## Live Streamlit App Link

**Streamlit App:** `REPLACE_WITH_YOUR_STREAMLIT_APP_LINK`

## d. Models used and comparison table

The assignment PDF explicitly lists five models. It also contains the phrase "all the 6 ML models", but the model list and rubric table contain only the following five, so this solution implements those five named models.

| Model               |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:--------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression |     0.98   | 0.9996 |      0.9804 |   0.98   | 0.98   | 0.9778 |
| Decision Tree       |     0.8556 | 0.9197 |      0.8602 |   0.8556 | 0.8554 | 0.8401 |
| kNN                 |     0.9822 | 0.9998 |      0.9826 |   0.9822 | 0.9822 | 0.9803 |
| Naive Bayes         |     0.8622 | 0.9805 |      0.875  |   0.8622 | 0.8623 | 0.8485 |
| Random Forest       |     0.98   | 0.9998 |      0.9802 |   0.98   | 0.98   | 0.9778 |

### Metric definitions used for multiclass evaluation

- Accuracy: fraction of correctly classified test rows.
- AUC: weighted one-vs-rest (OvR) ROC AUC using `predict_proba`.
- Precision: weighted average across classes.
- Recall: weighted average across classes.
- F1: weighted average across classes.
- MCC: multiclass Matthews Correlation Coefficient.

## Model observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong linear baseline. Standardization helped optimization; it achieved high accuracy and a near-perfect multiclass AUC, showing that digit classes are largely separable in the 64-dimensional pixel space. |
| Decision Tree | The single tree was easy to interpret but clearly overfit relative to the stronger models. Its accuracy, F1 and MCC were the lowest group, although performance remained substantially above chance. |
| kNN | Best overall on this split. Distance weighting plus standardized pixel features produced the highest accuracy, weighted F1 and MCC, with an almost perfect OvR AUC. |
| Naive Bayes | Fast and simple, with a strong AUC but lower class-decision accuracy than the top models. The conditional-independence/Gaussian assumptions are imperfect for correlated pixel intensities. |
| Random Forest | Excellent ensemble performance and more stable than a single tree. It matched Logistic Regression on accuracy while delivering a very high AUC and strong MCC. |
| **Overall Winner** | **kNN**, based primarily on the highest weighted F1 (and also top accuracy/MCC on this split). |

## Repository structure

```text
ML_Assignment_2_End_to_End/
├── app.py
├── train_models.py
├── utils.py
├── requirements.txt
├── README.md
├── test_data.csv
├── data/
│   └── digits_full_dataset.csv
├── model/
│   ├── logistic_regression_model.py
│   ├── decision_tree_model.py
│   ├── knn_model.py
│   ├── naive_bayes_model.py
│   ├── random_forest_model.py
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── random_forest.joblib
│   ├── metrics.csv
│   ├── metrics.json
│   └── metadata.json
├── notebooks/
│   └── ML_Assignment_2_Experiment.ipynb
└── docs/
    ├── Submission_Report_Draft.docx
    └── Submission_Report_Draft.pdf
```

## How to run in BITS Virtual Lab

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

For the required BITS Virtual Lab proof, run either `python train_models.py` or the notebook and take **one screenshot that visibly shows the BITS Virtual Lab environment and successful execution/output**.

## Streamlit features implemented

1. CSV upload option, with the bundled `test_data.csv` used by default.
2. Model-selection dropdown for all five required models.
3. Display of Accuracy, AUC, Precision, Recall, F1 and MCC.
4. Confusion matrix and classification report.
5. Comparison table for all models.
6. Prediction explorer that renders the 8 x 8 digit and shows class probabilities.

## Deployment on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Create a new app from your repository.
4. Select the `main` branch and `app.py` as the entry point.
5. Deploy and wait until the app status is healthy.
6. Open the public app URL in an incognito/private browser window to verify it is accessible.
7. Put the final GitHub and Streamlit links into this README and into the submission PDF.

## Suggested Git commit history

Because the assignment states that commit history is reviewed, do not upload everything as one unexplained final commit. A reasonable history is:

1. `Initialize assignment repository and dataset preparation`
2. `Add five classification model implementations`
3. `Add multiclass evaluation metrics and test split export`
4. `Build Streamlit evaluation interface`
5. `Add confusion matrix, comparison and prediction explorer`
6. `Finalize README, requirements and submission report`

## Reproducibility

Run `python train_models.py` to regenerate the serialized models, `test_data.csv`, full local dataset CSV and metric files using the fixed stratified split.

## Dataset citation

Alpaydin, E. & Kaynak, C. (1998). *Optical Recognition of Handwritten Digits* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C50P49.
