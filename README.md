# 🎓 Student Placement Predictor

A Flask web app that uses a **Logistic Regression** model to predict whether a student is likely to get placed. It takes a student's academic and skill details through a web form and returns the prediction along with the placement probability.

## ✨ Features

- Trains the model once when the Flask server starts, using `students_record.csv`
- Cleans the data and encodes categorical columns with `pd.get_dummies()`
- Splits the data into train/test sets and scales features with `StandardScaler`
- Trains a `LogisticRegression` model and shows its accuracy on the home and result pages
- Web form to enter a student's details
- Shows the **placement probability (%)** along with the prediction
- Fun result messages with emojis 🎉 for placed and 💪 for not placed yet
- Shows an error message if the input is invalid

## 🛠️ Tech Stack

- Python, Flask
- pandas, NumPy
- scikit-learn
- HTML, CSS

## 📁 Project Structure

```text
placement_predictor_flask/
│
├── app.py                  # Flask routes: home page and /predict
├── model_training.py       # train_model() and predict_student()
├── students_record.csv     # Dataset
├── requirements.txt
├── README.md
│
├── templates/
│   ├── index.html          # Input form
│   └── result.html         # Prediction result
│
└── static/
    └── style.css
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
python app.py
```

### 5. Open in your browser

```text
http://127.0.0.1:5000
```

## 📝 Input Fields

The form asks for these details of a student:

| Field | Type |
|---|---|
| Gender | Category |
| Branch | Category |
| City | Category |
| CGPA | Number |
| Attendance percent | Number |
| Technical skills | Number |
| Projects completed | Number |
| Coding problems solved | Number |
| Communication skill | Number |
| Internship (Yes/No) | Category |
| Has backlog (Yes/No) | Category |
| Tech events attended | Number |
| Salary expectation | Number |

## 📊 Dataset

The included `students_record.csv` is a demo dataset so the project runs right away. To use your own data, replace it with a file of the same name containing these columns:

```text
student_id, name, registration_date, gender, branch, city, cgpa,
attendance_percent, technical_skills, projects_completed,
coding_problems_solved, communication_skill, internship, has_backlog,
tech_events_attended, salary_expectation, placed
```

The target column `placed` must contain `Yes` or `No`. The columns `student_id`, `name` and `registration_date` are identifiers and are not used as inputs by the model.

## 🧠 How It Works

1. Load the data
2. Clean the data
3. Convert categories into numbers (`get_dummies`)
4. Separate features (X) and target (y)
5. Train/test split
6. Scale X with `StandardScaler`
7. Train Logistic Regression
8. Check accuracy on test data
9. Take input from the HTML form
10. Convert it into a DataFrame
11. Apply the same encoding and the same scaler
12. Predict the result and the placement probability
13. Show the result in the browser

### Why `reindex` matters

`pd.get_dummies()` creates columns like `branch_cse`, `city_Delhi`, `internship_Yes`. A single new student won't produce all of these, so the input is aligned to the training columns:

```python
new_student_encoded = new_student_encoded.reindex(
    columns=feature_columns,
    fill_value=0
)
```

This guarantees the new student's columns match exactly what the model was trained on.

## 🔮 Future Improvements

- Try to improve accuracy
- Add stronger input validation on the form
- Deploy online (Render, PythonAnywhere)

