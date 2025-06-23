# Employee Attrition Prediction Using Logistic Regression

This project uses a logistic regression model to predict whether an employee is likely to leave the company based on various HR attributes such as salary level, department, satisfaction, and more.

---

## 📊 Dataset

- Source: [Kaggle HR Analytics Dataset](https://www.kaggle.com/giripujar/hr-analytics)
- File: `HR_comma_sep.csv`
- Columns include:
  - `satisfaction_level`
  - `last_evaluation`
  - `number_project`
  - `average_montly_hours`
  - `time_spend_company`
  - `Work_accident`
  - `promotion_last_5years`
  - `Department`
  - `salary`
  - `left` (target: 1 if employee left, 0 if retained)

---

## 🔍 Objectives

- Perform exploratory data analysis (EDA) to understand attrition trends.
- Visualize how salary and department affect employee retention.
- Preprocess categorical features using label encoding.
- Use `SelectKBest` to choose the top 5 features.
- Train a logistic regression model.
- Evaluate model performance using accuracy score.

---

## 🛠️ How to Run

1. Clone this repository or download the files.
2. Install required libraries:
   ```bash
   pip install -r requirements.txt

