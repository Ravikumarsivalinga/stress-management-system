# stress-management-system
import gradio as gr
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# Generate a dataset
data = {
    "age": np.random.randint(18, 60, 500),
    "work_category": np.random.choice(["IT", "Healthcare", "Education", "Finance", "Other"], 500),
    "sleep_hours": np.random.randint(4, 10, 500),
    "exercise_hours": np.random.randint(0, 5, 500),
    "caffeine_intake": np.random.randint(0, 5, 500),
    "screen_time": np.random.randint(1, 10, 500),
    "social_interaction": np.random.choice(["Low", "Medium", "High"], 500),
    "workload": np.random.randint(1, 10, 500),
    "diet_quality": np.random.choice(["Poor", "Average", "Good"], 500),
    "hydration": np.random.randint(1, 10, 500),
    "hobbies": np.random.choice(["Rarely", "Occasionally", "Frequently"], 500),
    "mindfulness_practice": np.random.choice(["Never", "Sometimes", "Regularly"], 500),
    "outdoor_activity": np.random.choice(["Never", "Sometimes", "Regularly"], 500),
    "financial_stress": np.random.choice(["Low", "Medium", "High"], 500),
    "relationship_stress": np.random.choice(["Low", "Medium", "High"], 500),
    "stress_level": np.random.choice(["Low", "Medium", "High"], 500)
}
df = pd.DataFrame(data)

# Encode categorical values
label_encoders = {}
for col in ["work_category", "stress_level", "social_interaction", "diet_quality", "hobbies",
            "mindfulness_practice", "outdoor_activity", "financial_stress", "relationship_stress"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Train and save model
model_path = "stress_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    X = df.drop("stress_level", axis=1)
    y = df["stress_level"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)

# Store user data
user_data = {}
correct_password = "secure123"

def encode_input(features):
    encoded_features = []
    for i, col in enumerate(X.columns):
        try:
            if col in label_encoders:
                encoded_features.append(label_encoders[col].transform([features[i]])[0])
            else:
                encoded_features.append(features[i])
        except IndexError:
            return None  # Return None if features are missing
    return np.array(encoded_features).reshape(1, -1)





def predict_stress(*args):
    try:
        name = args[0]
        features = args[1:]

        if len(features) != len(X.columns):
            return f"Error: Expected {len(X.columns)} input features, but got {len(features)}.", ""

        input_data = encode_input(features)
        if input_data is None:
            return "Error: Missing or incorrect feature values.", ""

        stress_pred = model.predict(input_data)[0]
        stress_label = label_encoders["stress_level"].inverse_transform([stress_pred])[0]

        if name not in user_data:
            user_data[name] = []
        user_data[name].append(list(features) + [stress_label])

        return f"{name}, your predicted stress level is: {stress_label}", ""
    except Exception as e:
        return f"Error: {str(e)}", ""



def show_results(password, name):
    if not password:
        return "⚠️ Please enter the password first.", None, None
    elif password != correct_password:
        return "❌ Incorrect password. Try again.", None, None
    elif name not in user_data or len(user_data[name]) == 0:
        return f"✅ Password accepted. No data available for {name}.", None, None
    else:
        df_results = pd.DataFrame(user_data[name], columns=[
            "Age", "Work Category", "Sleep Hours", "Exercise Hours", "Caffeine Intake",
            "Screen Time", "Social Interaction", "Workload", "Diet Quality", "Hydration",
            "Hobbies", "Mindfulness Practice", "Outdoor Activity", "Financial Stress",
            "Relationship Stress", "Predicted Stress Level"
        ])

        stress_counts = df_results["Predicted Stress Level"].value_counts()
        fig, ax = plt.subplots()
        stress_counts.plot(kind="bar", ax=ax, color=["green", "orange", "red"])
        ax.set_title(f"Stress Level Distribution for {name}")
        ax.set_xlabel("Stress Level")
        ax.set_ylabel("Count")

        return "✅ Password accepted. Displaying results.", df_results, fig

# Gradio UI
with gr.Blocks(theme="default") as app:
    gr.Markdown("# Stress Management System", elem_id="title")

    with gr.Row():
        name = gr.Textbox(label="Name")
        age = gr.Number(label="Age")
        work_category = gr.Dropdown(choices=["IT", "Healthcare", "Education", "Finance", "Other"], label="Work Category")

    with gr.Row():
        sleep_hours = gr.Slider(4, 10, step=1, label="Sleep Hours")
        exercise_hours = gr.Slider(0, 5, step=1, label="Exercise Hours")
        caffeine_intake = gr.Slider(0, 5, step=1, label="Caffeine Intake")

    with gr.Row():
        screen_time = gr.Slider(1, 10, step=1, label="Screen Time")
        social_interaction = gr.Dropdown(choices=["Low", "Medium", "High"], label="Social Interaction")
        workload = gr.Slider(1, 10, step=1, label="Workload")

    with gr.Row():
       diet_quality = gr.Dropdown(choices=["Poor", "Average", "Good"], label="Diet Quality")
       hydration = gr.Slider(1, 10, step=1, label="Hydration")
       hobbies = gr.Dropdown(choices=["Rarely", "Occasionally", "Frequently"], label="Hobbies")

    with gr.Row():
       mindfulness_practice = gr.Dropdown(choices=["Never", "Sometimes", "Regularly"], label="Mindfulness Practice")
       outdoor_activity = gr.Dropdown(choices=["Never", "Sometimes", "Regularly"], label="Outdoor Activity")
       financial_stress = gr.Dropdown(choices=["Low", "Medium", "High"], label="Financial Stress")


    relationship_stress = gr.Dropdown(choices=["Low", "Medium", "High"], label="Relationship Stress")

    submit_btn = gr.Button("Predict Stress Level")
    output, reset_password = gr.Textbox(), gr.Textbox(visible=False)

    submit_btn.click(predict_stress, inputs=[name, age, work_category, sleep_hours, exercise_hours, caffeine_intake,
                                         screen_time, social_interaction, workload, diet_quality, hydration, hobbies,
                                         mindfulness_practice, outdoor_activity, financial_stress, relationship_stress],
                 outputs=[output, reset_password])


    gr.Markdown("## 🔐 Admin Panel: View Results")

    password_input = gr.Textbox(label="Enter Password", type="password")
    user_name_input = gr.Textbox(label="Enter Your Name")
    view_results_btn = gr.Button("🔍 View Results")

    results_message = gr.Textbox(label="Message", interactive=False)
    results_table = gr.Dataframe()
    results_chart = gr.Plot()

    view_results_btn.click(show_results, inputs=[password_input, user_name_input], outputs=[results_message, results_table, results_chart])

app.launch()

