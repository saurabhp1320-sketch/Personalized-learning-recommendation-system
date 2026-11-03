import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 500
topics = ["Mathematics", "Science", "Programming", "English", "History"]
formats = ["Video", "Article", "Quiz", "Interactive"]
interests = ["Technology", "Science", "Arts", "Business", "Sports"]

rows = []
for i in range(1, n + 1):
    interest = rng.choice(interests)
    preferred = rng.choice(formats)
    age = int(rng.integers(15, 23))
    study_hours = round(float(rng.uniform(1, 18)), 1)
    attendance = int(rng.integers(55, 101))
    completion = int(np.clip(rng.normal(72 + study_hours * 1.2, 15), 20, 100))
    quiz = int(np.clip(rng.normal(45 + study_hours * 2.5 + (attendance-70)*0.35, 16), 10, 100))
    difficulty = rng.choice(["Beginner", "Intermediate", "Advanced"], p=[.35,.45,.20])
    topic = rng.choice(topics)
    # Demonstration label: the resource type that best matches preference + learning need.
    if quiz < 50:
        recommended = "Video" if preferred in ["Video", "Interactive"] else "Article"
    elif quiz < 75:
        recommended = preferred if preferred != "Quiz" else "Interactive"
    else:
        recommended = "Quiz" if preferred in ["Quiz", "Interactive"] else preferred
    rows.append([f"S{i:04}", age, study_hours, attendance, completion, quiz,
                 interest, preferred, topic, difficulty, recommended])

df = pd.DataFrame(rows, columns=[
    "student_id","age","weekly_study_hours","attendance_pct","completion_pct",
    "quiz_score","interest_area","preferred_format","topic","current_level",
    "best_resource_format"
])
df.to_csv("students.csv", index=False)
print(f"Created students.csv with {len(df)} rows")
