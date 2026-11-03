# Student Learning Behaviour Analysis & Recommendation System

A beginner-friendly ML project using Python, pandas, scikit-learn, matplotlib/seaborn and Streamlit.

## 1. What the project does
- Analyses study hours, attendance, completion, quiz scores, interests and preferred formats.
- Preprocesses numerical and categorical data.
- Creates EDA charts.
- Trains a Random Forest classifier to predict a suitable learning-resource format.
- Evaluates the model using accuracy, precision, recall, F1-score and a confusion matrix.
- Provides an interactive dashboard for recommendations.

**Important:** `generate_data.py` creates synthetic demonstration data. The target label is generated from a simple illustrative rule, so its evaluation scores are not evidence of real-world effectiveness. Replace it with consented, appropriately governed student data and a defensible target (for example, resource usefulness ratings or subsequent learning gain).

## 2. Setup
Install Python 3.10 or newer. Open a terminal in this folder:

```bash
python -m venv .venv
```

Activate it:

Windows:
```bash
.venv\\Scripts\\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install packages:
```bash
pip install -r requirements.txt
```

## 3. Generate sample data
```bash
python generate_data.py
```
This creates `students.csv`. Open it in Excel to inspect the columns.

## 4. Train and evaluate
```bash
python train.py
```
This prints evaluation metrics, saves `learning_model.joblib`, and creates plots in `outputs/`.

## 5. Run the dashboard
```bash
streamlit run app.py
```
A browser tab opens. Enter a student profile in the sidebar and click **Generate recommendations**.

## 6. Understanding the workflow
1. **Data collection:** collect only necessary learning data with appropriate permission.
2. **Preprocessing:** remove duplicates, impute missing values, scale numeric columns and one-hot encode categories.
3. **EDA:** inspect distributions and relationships before modelling.
4. **Model:** Random Forest predicts the resource format label.
5. **Evaluation:** use a held-out test set; review precision, recall, F1 and confusion matrix, not accuracy alone.
6. **Recommendation:** use the predicted format together with score-based learning guidance and topic resource suggestions.
7. **Iteration:** compare recommendations with student feedback and measured learning improvement.

## 7. Using your own CSV
Your CSV should contain these columns:
`student_id, age, weekly_study_hours, attendance_pct, completion_pct, quiz_score, interest_area, preferred_format, topic, current_level, best_resource_format`

Update category values consistently. The target column `best_resource_format` should come from meaningful observations such as student ratings or instructor-approved labels—not an arbitrary guess. Then rerun `python train.py`.

## 8. Suggested report sections
- Problem statement and objectives
- Dataset description and data dictionary
- Data cleaning and preprocessing
- EDA with at least three charts
- Model choice and training process
- Evaluation metrics and confusion matrix
- Example student profile and recommendation
- Limitations, privacy and fairness considerations
- Future improvements (collaborative filtering, learning-gain tracking, feedback loop)

## 9. Ethical use
Do not use predictions to grade, rank, punish or limit students. Keep student identifiers out of charts, restrict access, obtain appropriate consent, check performance across groups, and allow students/teachers to override recommendations.
