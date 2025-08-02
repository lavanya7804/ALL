from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


app = Flask(__name__)

diabetes_model = joblib.load("./models/diabetes_model.pkl")

bp_model = joblib.load("./models/bp_model.pkl")
bp_scaler = joblib.load("./models/bp_scaler.pkl")

heart_model = joblib.load("./models/heart_attack_model.pkl")
heart_scaler = joblib.load("./models/heart_scaler.pkl")

@app.route("/hello")
def chatbot():
    return render_template("hello.html")

@app.route("/food")
def food():
    return render_template("food.html")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/diabetes")
def diabetes():
    return render_template("diabetes.html")

@app.route("/heart")
def heart():
    return render_template("heart.html")

@app.route("/bp")
def bp():
    return render_template("bp.html")


@app.route("/predict/diabetes", methods=["POST"])
def predict_diabetes():
    data = request.get_json()

    input_data = np.array([[
        1 if data["gender"] == "Male" else 0,
        float(data["age"]),
        int(data["hypertension"]),
        int(data["heart_disease"]),
        0 if data["smoking_history"].lower() == "never" else
        1 if data["smoking_history"].lower() == "former" else
        2 if data["smoking_history"].lower() == "current" else 3,
        float(data["bmi"]),
        float(data["hba1c_level"]),
        float(data["blood_glucose_level"])
    ]])

    prediction = diabetes_model.predict(input_data)[0]

    return jsonify({"prediction": "High Risk of Diabetes" if prediction == 1 else "Low Risk"})




@app.route("/predict/bp", methods=["POST"])
def predict_bp():
    data = request.get_json()
    sex = 1 if data["sex"] == "Male" else 0
    smoking = 1 if data["smoking_status"] == "Smoker" else 0
    pregnant = 1 if data["pregnancy"] == "Pregnant" else 0
    activity_map = {"Low": 0, "Moderate": 1, "High": 2}
    activity = activity_map.get(data["activity_level"], 0)

    input_data = [
        float(data["age"]),
        float(data["hemoglobin"]),
        float(data["family_score"]),
        smoking,
        float(data["bmi"]),
        sex,
        pregnant,
        activity
    ]
    while len(input_data) < 13:
        input_data.append(0.0)

    input_scaled = bp_scaler.transform([input_data])
    prediction = bp_model.predict(input_scaled)[0]
    
    return jsonify({"prediction": "High BP" if prediction == 1 else "Normal BP"})



@app.route("/predict/heart", methods=["POST"])
def predict_heart():
    data = request.get_json()

    sex = 1 if data["sex"].lower() == "male" else 0

    cp_map = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }
    cp = cp_map.get(data["cp"], 0)

    fbs = 1 if data["fbs"].lower() == "true" else 0

    restecg_map = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }
    restecg = restecg_map.get(data["restecg"], 0)

    exang = 1 if data["exang"].lower() == "yes" else 0

    # Create 9 inputs as normal
    input_data = [
        float(data["age"]),
        sex,
        cp,
        float(data["trestbps"]),
        float(data["chol"]),
        fbs,
        restecg,
        float(data["thalach"]),
        exang
    ]
    while len(input_data) < 22:
        input_data.append(0.0)

    input_scaled = heart_scaler.transform([input_data])
    prediction = heart_model.predict(input_scaled)[0]
    
    return jsonify({
        "prediction": "⚠️ High Risk of Heart Attack" if prediction == 1 else "✅ Low Risk"
    })

@app.route("/food", methods=["GET", "POST"])
def food_plan():
    result = None

    if request.method == "POST":
        # Retrieve form data
        age = int(request.form["Age"])
        height = float(request.form["Height"])
        weight = float(request.form["Weight"])
        gender = int(request.form["Gender"])
        health = int(request.form["HealthCondition"])
        goal = int(request.form["FitnessGoal"])
        dietary = int(request.form["Dietary"])
        activity = int(request.form["ActivityStatus"])
        allergy = int(request.form["Allergy"])

        # Combine input for prediction model (example)
        input_data = [age, height, weight, gender, health, goal, dietary, activity, allergy]

        
        result = [
            "Oats with berries and almonds",
            "Grilled chicken with quinoa",
            "Mixed nuts and yogurt",
            "Vegetable soup with brown rice"
        ]

    return render_template("food.html", result=result)

if __name__=="__main__":
    app.run(debug=True)

