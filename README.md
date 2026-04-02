Student Performance Tracking & Attrition Predictor
📌 Project Overview
Student attrition (dropouts) costs universities millions of dollars in lost tuition and severely impacts institutional reputation. The goal of this project is to build an end-to-end Machine Learning pipeline that acts as an Early Warning System, predicting which students are at high risk of failing or dropping out before it happens.

By analyzing behavioral metrics (attendance, participation, and study hours) of 1,000,000 students, this project translates predictive models into actionable academic intervention strategies, boasting a potential $18.6 Million ROI per semester.

📊 The Data & Methodology
Dataset Size: 1,000,000 students, 6 features.

Target Variable Creation: Created a custom is_at_risk binary target based on failing final grades.

Severe Class Imbalance: Only 5.1% of the student body was actively failing. Handled using algorithm-level class weighting (class_weight='balanced').

Data Leakage Prevention: Dropped total_score from the training features to ensure the model only predicts using mid-semester behavioral metrics, not end-of-year grades.

💡 Key Advanced Insights (EDA)
During the Exploratory Data Analysis phase, two massive behavioral patterns were uncovered:

The "10-Hour Tipping Point": A clear threshold exists where the risk of dropping out plummets. Students studying fewer than 5 hours a week are in the extreme danger zone, but crossing the 10-hour/week threshold effectively drops attrition risk to near zero.

The "Hidden Dropouts" (False Safety Nets): Found over 8,000 students who had >95% attendance but still failed. This proved that attendance alone is a false safety net for Academic Advisors if not paired with self-study hours.

🤖 Machine Learning Models & Evaluation
Because the dataset is heavily imbalanced (only 5.1% at-risk), Accuracy was discarded as a metric in favor of Recall (to ensure no at-risk student falls through the cracks).

Two models were tested using 5-Fold Stratified Cross-Validation (with Scikit-Learn Pipelines and Standard Scalers to prevent leakage):

Random Forest Classifier: 37.34% Recall

Logistic Regression: 91.23% Recall 🏆

Why Logistic Regression Won: In a heavily imbalanced dataset where minority class behaviors overlap with majority behaviors, the heavily-penalized Logistic Regression pushed its probability threshold aggressively, capturing almost all at-risk students while remaining highly interpretable for academic advisors.

💰 Business Impact & ROI
By prioritizing Recall, the Logistic Regression model successfully flags over 91% of at-risk students.

Hypothetical ROI Calculation:

Testing on a student body of 200,000.

Model flags ~9,300 true at-risk students.

Assuming a conservative 20% success rate for advisor interventions = 1,860 students saved.

At $10,000 retained tuition per student = $18,600,000 in saved revenue per semester.

🚀 How to Run the Code
Clone the repository.

Ensure you have the following packages installed: pandas, numpy, matplotlib, seaborn, scikit-learn.

Open the student_attrition_predictor.ipynb Jupyter Notebook and run all cells.
