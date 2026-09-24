
from pathlib import Path

from flask import Flask, render_template, request

from model_training import train_model, predict_student


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "students_record.csv"

# Train once when Flask starts
model, scaler, feature_columns, model_accuracy = train_model(CSV_PATH)


@app.route("/")
def home():
    return render_template(
        "index.html",
        model_accuracy=round(model_accuracy * 100, 2),
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        student_data = {
            "gender": request.form["gender"],
            "branch": request.form["branch"],
            "city": request.form["city"],
            "cgpa": float(request.form["cgpa"]),
            "attendance_percent": float(request.form["attendance_percent"]),
            "technical_skills": int(request.form["technical_skills"]),
            "projects_completed": int(request.form["projects_completed"]),
            "coding_problems_solved": int(request.form["coding_problems_solved"]),
            "communication_skill": float(request.form["communication_skill"]),
            "internship": request.form["internship"],
            "has_backlog": request.form["has_backlog"],
            "tech_events_attended": float(request.form["tech_events_attended"]),
            "salary_expectation": float(request.form["salary_expectation"]),
        }

        prediction, probability = predict_student(
            model,
            scaler,
            feature_columns,
            student_data,
        )

        placement_probability = round(probability * 100, 1)

        if prediction == 1:
            result_title = "Congratulations! 🎉🥳🚀"
            result_message = "The model predicts that this student may get placed."
            result_class = "success"
            emoji = "🎓✨💼"
        else:
            result_title = "Keep Improving! 💪📚🚀"
            result_message = "The model predicts that this student may not get placed yet."
            result_class = "warning"
            emoji = "🌱🔥🎯"

        return render_template(
            "result.html",
            prediction=prediction,
            result_title=result_title,
            result_message=result_message,
            result_class=result_class,
            emoji=emoji,
            placement_probability=placement_probability,
            model_accuracy=round(model_accuracy * 100, 2),
            student=student_data,
        )

    except Exception as error:
        return render_template(
            "result.html",
            error=str(error),
            result_class="error",
            model_accuracy=round(model_accuracy * 100, 2),
        )


if __name__ == "__main__":
    app.run(debug=True)
