
# Student Placement Predictor — Flask + Machine Learning

This is a complete beginner-friendly Flask project that:

- reads `students_record.csv`
- cleans the data
- converts categorical columns using `pd.get_dummies()`
- splits the dataset into train/test data
- scales features with `StandardScaler`
- trains a `LogisticRegression` model
- calculates model accuracy
- accepts student details from a Flask web form
- predicts placement
- shows the result with celebration emojis

## Project Structure

```text
placement_predictor_flask/
│
├── app.py
├── model_training.py
├── students_record.csv
├── requirements.txt
├── README.md
│
├── templates/
│   ├── index.html
│   └── result.html
│
└── static/
    └── style.css
```

## Step 1 — Open terminal in the project folder

Example:

```bash
cd placement_predictor_flask
```

## Step 2 — Create a virtual environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

## Step 3 — Install libraries

```bash
pip install -r requirements.txt
```

## Step 4 — Run Flask

```bash
python app.py
```

## Step 5 — Open browser

Go to:

```text
http://127.0.0.1:5000
```

## Using your own CSV

The included `students_record.csv` is a demo dataset so that the project runs immediately.

To use your real dataset, replace the included CSV with your own file named:

```text
students_record.csv
```

Your CSV should contain these columns:

```text
student_id
name
registration_date
gender
branch
city
cgpa
attendance_percent
technical_skills
projects_completed
coding_problems_solved
communication_skill
internship
has_backlog
tech_events_attended
salary_expectation
placed
```

The target column should contain:

```text
Yes
No
```

## Important ML Concept

During training, `pd.get_dummies()` creates many columns such as:

```text
branch_cse
branch_it
city_Delhi
internship_Yes
```

A new student may not naturally create all of the same dummy columns.

That is why the project uses:

```python
new_student_encoded = new_student_encoded.reindex(
    columns=feature_columns,
    fill_value=0
)
```

This makes the new student's columns exactly match the columns used while training the model.

## Teaching Flow

For students, explain the app in this order:

1. Load data
2. Clean data
3. Convert categories into numbers
4. Separate X and Y
5. Train/test split
6. Scale X
7. Train Logistic Regression
8. Test accuracy
9. Take data from HTML form
10. Convert it into a DataFrame
11. Apply the same encoding
12. Apply the same scaler
13. Predict
14. Show the result in Flask
