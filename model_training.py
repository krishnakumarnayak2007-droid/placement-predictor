
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


CATEGORICAL_COLUMNS = [
    "gender",
    "branch",
    "city",
    "internship",
    "has_backlog",
]


def get_canonical_branch(branch_name):
    canonical_branch_names = {
        "cse": "cse",
        "computer science": "cse",
        "c.s.e": "cse",
        "it": "it",
        "i.t.": "it",
        "i.t": "it",
        "information technology": "it",
        "ece": "ece",
        "e.c.e": "ece",
        "electronics": "ece",
        "eee": "eee",
        "e.e.e": "eee",
        "electrical": "eee",
        "mech": "mech",
        "mechanical": "mech",
        "civil": "civil",
    }

    if pd.isna(branch_name):
        return "unknown"

    branch_name = str(branch_name).strip().lower()
    return canonical_branch_names.get(branch_name, branch_name)


def train_model(csv_path="students_record.csv"):
    # -------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------
    df = pd.read_csv(csv_path)

    # -------------------------------------------------------
    # 2. BASIC CLEANING
    # -------------------------------------------------------
    df = df.drop_duplicates()

    # Fix CGPA values like "7,91" -> "7.91"
    df["cgpa"] = (
        df["cgpa"]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )
    df["cgpa"] = pd.to_numeric(df["cgpa"], errors="coerce")

    # Normalize branch names
    df["branch"] = df["branch"].map(get_canonical_branch)

    # Turn impossible values into missing values
    df.loc[df["attendance_percent"] > 100, "attendance_percent"] = np.nan
    df.loc[df["cgpa"] > 10, "cgpa"] = np.nan
    df.loc[df["salary_expectation"] > 5_000_000, "salary_expectation"] = np.nan
    df.loc[df["coding_problems_solved"] > 1000, "coding_problems_solved"] = np.nan

    # Fill missing numeric values using median
    numeric_columns_with_gaps = [
        "attendance_percent",
        "communication_skill",
        "tech_events_attended",
        "salary_expectation",
        "cgpa",
        "coding_problems_solved",
    ]

    for column_name in numeric_columns_with_gaps:
        column_median = df[column_name].median()
        df[column_name] = df[column_name].fillna(column_median)

    # Fill missing categorical value
    df["city"] = df["city"].fillna("Unknown")

    # -------------------------------------------------------
    # 3. DROP COLUMNS THAT DO NOT HELP THE MODEL
    # -------------------------------------------------------
    columns_to_drop = [
        "student_id",
        "name",
        "registration_date",
    ]

    model_df = df.drop(columns=columns_to_drop)

    # -------------------------------------------------------
    # 4. CONVERT CATEGORICAL COLUMNS INTO NUMBERS
    # -------------------------------------------------------
    model_df = pd.get_dummies(
        model_df,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True,
    )

    # -------------------------------------------------------
    # 5. SEPARATE X AND Y
    # -------------------------------------------------------
    target_map = {
        "No": 0,
        "Yes": 1,
    }

    Y = model_df["placed"].map(target_map)
    X = model_df.drop(columns=["placed"])

    # -------------------------------------------------------
    # 6. TRAIN / TEST SPLIT
    # -------------------------------------------------------
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=0.20,
        random_state=42,
        stratify=Y,
    )

    # -------------------------------------------------------
    # 7. SCALING
    # -------------------------------------------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------------------------------------
    # 8. TRAIN MODEL
    # -------------------------------------------------------
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, Y_train)

    # -------------------------------------------------------
    # 9. SIMPLE ACCURACY TEST
    # -------------------------------------------------------
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(Y_test, y_pred)

    return model, scaler, X.columns, accuracy


def prepare_new_student(student_data, feature_columns):
    # Make one-row DataFrame
    new_student_df = pd.DataFrame([student_data])

    # Keep branch formatting same as training
    new_student_df["branch"] = new_student_df["branch"].map(get_canonical_branch)

    # Convert categorical columns into dummy columns
    new_student_encoded = pd.get_dummies(
        new_student_df,
        columns=CATEGORICAL_COLUMNS,
    )

    # IMPORTANT:
    # Make the new student's columns exactly match training X columns.
    # Missing columns become 0 and extra columns are removed.
    new_student_encoded = new_student_encoded.reindex(
        columns=feature_columns,
        fill_value=0,
    )

    return new_student_encoded


def predict_student(model, scaler, feature_columns, student_data):
    new_student_encoded = prepare_new_student(
        student_data,
        feature_columns,
    )

    new_student_scaled = scaler.transform(new_student_encoded)

    prediction = model.predict(new_student_scaled)[0]
    probability = model.predict_proba(new_student_scaled)[0][1]

    return int(prediction), float(probability)
