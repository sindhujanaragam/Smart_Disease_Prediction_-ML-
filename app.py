from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# ==========================================================
# LOAD MODELS
# ==========================================================

# HEART DISEASE
heart_voting_model = joblib.load("disease_model.pkl")
heart_stacking_model = joblib.load("stacking_disease_model.pkl")

# DIABETES
diabetes_voting_model = joblib.load("diabetes_model.pkl")
diabetes_stacking_model = joblib.load("diabetes_stacking_model.pkl")


# ==========================================================
# COMMON FUNCTIONS
# ==========================================================

def get_risk_level(probability):
    if probability < 30:
        return "Low Risk"
    elif probability < 70:
        return "Moderate Risk"
    return "High Risk"


def calculate_prediction(model, features):
    probability = model.predict_proba(features)[0][1] * 100
    prediction = model.predict(features)[0]
    return probability, prediction


def render_error(message):
    return render_template(
        "index.html",
        disease="heart",
        error=message
    )


# ==========================================================
# HOME
# ==========================================================

@app.route("/")
def home():
    return render_template("index.html", disease="heart")


# ==========================================================
# MAIN PREDICTION
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:
        disease = request.form.get("disease", "heart")
        action = request.form.get("action", "predict")

        # ==================================================
        # HEART DISEASE
        # ==================================================

        if disease == "heart":

            age = float(request.form["age"])
            sex = float(request.form["sex"])
            cp = float(request.form["cp"])
            trestbps = float(request.form["trestbps"])
            chol = float(request.form["chol"])
            fbs = float(request.form["fbs"])
            restecg = float(request.form["restecg"])
            thalach = float(request.form["thalach"])
            exang = float(request.form["exang"])
            oldpeak = float(request.form["oldpeak"])
            slope = float(request.form["slope"])
            ca = float(request.form["ca"])
            thal = float(request.form["thal"])

            features = np.array([[
                age, sex, cp, trestbps, chol, fbs, restecg,
                thalach, exang, oldpeak, slope, ca, thal
            ]])

            voting_probability, voting_prediction = calculate_prediction(
                heart_voting_model, features
            )
            stacking_probability, stacking_prediction = calculate_prediction(
                heart_stacking_model, features
            )

            voting_risk = get_risk_level(voting_probability)
            stacking_risk = get_risk_level(stacking_probability)

            result = (
                "Heart Disease Risk Detected"
                if voting_prediction == 1
                else "No Heart Disease Risk Detected"
            )

            common = dict(
                disease="heart",
                disease_title="Heart Disease",
                prediction=result,
                probability=round(voting_probability, 2),
                risk_level=voting_risk,
                stacking_probability=round(stacking_probability, 2),
                stacking_risk_level=stacking_risk,
                age=age,
                sex=sex,
                cp=cp,
                trestbps=trestbps,
                chol=chol,
                fbs=fbs,
                restecg=restecg,
                thalach=thalach,
                exang=exang,
                oldpeak=oldpeak,
                slope=slope,
                ca=ca,
                thal=thal
            )

            # --------------------------------------------------
            # HEART WHAT-IF
            # --------------------------------------------------

            if action == "simulate":

                what_trestbps = float(request.form["what_trestbps"])
                what_chol = float(request.form["what_chol"])
                what_thalach = float(request.form["what_thalach"])
                what_oldpeak = float(request.form["what_oldpeak"])
                what_ca = float(request.form["what_ca"])

                what_if_features = np.array([[
                    age, sex, cp, what_trestbps, what_chol, fbs,
                    restecg, what_thalach, exang, what_oldpeak,
                    slope, what_ca, thal
                ]])

                what_if_probability, _ = calculate_prediction(
                    heart_voting_model, what_if_features
                )
                stacking_what_if_probability, _ = calculate_prediction(
                    heart_stacking_model, what_if_features
                )

                difference = what_if_probability - voting_probability
                stacking_difference = (
                    stacking_what_if_probability - stacking_probability
                )

                common.update(
                    what_if_probability=round(what_if_probability, 2),
                    what_if_risk=get_risk_level(what_if_probability),
                    risk_difference=round(abs(difference), 2),
                    change_message=(
                        "Risk Decreased" if difference < 0
                        else "Risk Increased" if difference > 0
                        else "Risk Unchanged"
                    ),
                    change_symbol=(
                        "↓" if difference < 0
                        else "↑" if difference > 0
                        else "→"
                    ),
                    stacking_what_if_probability=round(
                        stacking_what_if_probability, 2
                    ),
                    stacking_what_if_risk=get_risk_level(
                        stacking_what_if_probability
                    ),
                    stacking_risk_difference=round(
                        abs(stacking_difference), 2
                    ),
                    stacking_change_message=(
                        "Risk Decreased" if stacking_difference < 0
                        else "Risk Increased" if stacking_difference > 0
                        else "Risk Unchanged"
                    ),
                    stacking_change_symbol=(
                        "↓" if stacking_difference < 0
                        else "↑" if stacking_difference > 0
                        else "→"
                    ),
                    what_trestbps=what_trestbps,
                    what_chol=what_chol,
                    what_thalach=what_thalach,
                    what_oldpeak=what_oldpeak,
                    what_ca=what_ca
                )

                return render_template("index.html", **common)

            # --------------------------------------------------
            # HEART RISK REDUCTION
            # --------------------------------------------------

            if action == "risk_reduction":

                scenarios = []

                scenario_chol = max(chol - 50, 150)
                chol_features = np.array([[
                    age, sex, cp, trestbps, scenario_chol, fbs,
                    restecg, thalach, exang, oldpeak, slope, ca, thal
                ]])
                chol_probability, _ = calculate_prediction(
                    heart_voting_model, chol_features
                )
                scenarios.append({
                    "name": "Improve Cholesterol",
                    "value": f"{chol:.0f} → {scenario_chol:.0f}",
                    "probability": round(chol_probability, 2),
                    "change": round(
                        max(voting_probability - chol_probability, 0), 2
                    )
                })

                scenario_bp = max(trestbps - 20, 90)
                bp_features = np.array([[
                    age, sex, cp, scenario_bp, chol, fbs,
                    restecg, thalach, exang, oldpeak, slope, ca, thal
                ]])
                bp_probability, _ = calculate_prediction(
                    heart_voting_model, bp_features
                )
                scenarios.append({
                    "name": "Improve Blood Pressure",
                    "value": f"{trestbps:.0f} → {scenario_bp:.0f}",
                    "probability": round(bp_probability, 2),
                    "change": round(
                        max(voting_probability - bp_probability, 0), 2
                    )
                })

                scenario_oldpeak = max(oldpeak - 1.0, 0)
                oldpeak_features = np.array([[
                    age, sex, cp, trestbps, chol, fbs,
                    restecg, thalach, exang, scenario_oldpeak,
                    slope, ca, thal
                ]])
                oldpeak_probability, _ = calculate_prediction(
                    heart_voting_model, oldpeak_features
                )
                scenarios.append({
                    "name": "Improve Exercise Indicator",
                    "value": f"{oldpeak:.1f} → {scenario_oldpeak:.1f}",
                    "probability": round(oldpeak_probability, 2),
                    "change": round(
                        max(voting_probability - oldpeak_probability, 0), 2
                    )
                })

                combined_features = np.array([[
                    age, sex, cp, scenario_bp, scenario_chol, fbs,
                    restecg, thalach, exang, scenario_oldpeak,
                    slope, ca, thal
                ]])
                combined_probability, _ = calculate_prediction(
                    heart_voting_model, combined_features
                )
                scenarios.append({
                    "name": "Combined Scenario",
                    "value": "Multiple factors improved",
                    "probability": round(combined_probability, 2),
                    "change": round(
                        max(voting_probability - combined_probability, 0), 2
                    )
                })

                common["scenarios"] = scenarios
                return render_template("index.html", **common)

            return render_template("index.html", **common)

        # ==================================================
        # DIABETES
        # ==================================================

        if disease == "diabetes":

            pregnancies = float(request.form["pregnancies"])
            glucose = float(request.form["glucose"])
            blood_pressure = float(request.form["blood_pressure"])
            skin_thickness = float(request.form["skin_thickness"])
            insulin = float(request.form["insulin"])
            bmi = float(request.form["bmi"])
            diabetes_pedigree = float(request.form["diabetes_pedigree"])
            diabetes_age = float(request.form["diabetes_age"])

            features = np.array([[
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree,
                diabetes_age
            ]])

            voting_probability, voting_prediction = calculate_prediction(
                diabetes_voting_model, features
            )
            stacking_probability, stacking_prediction = calculate_prediction(
                diabetes_stacking_model, features
            )

            voting_risk = get_risk_level(voting_probability)
            stacking_risk = get_risk_level(stacking_probability)

            result = (
                "Diabetes Risk Detected"
                if voting_prediction == 1
                else "No Diabetes Risk Detected"
            )

            common = dict(
                disease="diabetes",
                disease_title="Diabetes",
                prediction=result,
                probability=round(voting_probability, 2),
                risk_level=voting_risk,
                stacking_probability=round(stacking_probability, 2),
                stacking_risk_level=stacking_risk,
                pregnancies=pregnancies,
                glucose=glucose,
                blood_pressure=blood_pressure,
                skin_thickness=skin_thickness,
                insulin=insulin,
                bmi=bmi,
                diabetes_pedigree=diabetes_pedigree,
                diabetes_age=diabetes_age
            )

            # --------------------------------------------------
            # DIABETES WHAT-IF
            # --------------------------------------------------

            if action == "simulate":

                what_glucose = float(request.form["what_glucose"])
                what_blood_pressure = float(
                    request.form["what_blood_pressure"]
                )
                what_bmi = float(request.form["what_bmi"])
                what_insulin = float(request.form["what_insulin"])

                what_if_features = np.array([[
                    pregnancies,
                    what_glucose,
                    what_blood_pressure,
                    skin_thickness,
                    what_insulin,
                    what_bmi,
                    diabetes_pedigree,
                    diabetes_age
                ]])

                what_if_probability, _ = calculate_prediction(
                    diabetes_voting_model, what_if_features
                )
                stacking_what_if_probability, _ = calculate_prediction(
                    diabetes_stacking_model, what_if_features
                )

                difference = what_if_probability - voting_probability
                stacking_difference = (
                    stacking_what_if_probability - stacking_probability
                )

                common.update(
                    what_if_probability=round(what_if_probability, 2),
                    what_if_risk=get_risk_level(what_if_probability),
                    risk_difference=round(abs(difference), 2),
                    change_message=(
                        "Risk Decreased" if difference < 0
                        else "Risk Increased" if difference > 0
                        else "Risk Unchanged"
                    ),
                    change_symbol=(
                        "↓" if difference < 0
                        else "↑" if difference > 0
                        else "→"
                    ),
                    stacking_what_if_probability=round(
                        stacking_what_if_probability, 2
                    ),
                    stacking_what_if_risk=get_risk_level(
                        stacking_what_if_probability
                    ),
                    stacking_risk_difference=round(
                        abs(stacking_difference), 2
                    ),
                    stacking_change_message=(
                        "Risk Decreased" if stacking_difference < 0
                        else "Risk Increased" if stacking_difference > 0
                        else "Risk Unchanged"
                    ),
                    stacking_change_symbol=(
                        "↓" if stacking_difference < 0
                        else "↑" if stacking_difference > 0
                        else "→"
                    ),
                    what_glucose=what_glucose,
                    what_blood_pressure=what_blood_pressure,
                    what_bmi=what_bmi,
                    what_insulin=what_insulin
                )

                return render_template("index.html", **common)

            # --------------------------------------------------
            # DIABETES RISK REDUCTION
            # --------------------------------------------------

            if action == "risk_reduction":

                scenarios = []

                scenario_glucose = max(glucose - 20, 70)
                glucose_features = np.array([[
                    pregnancies,
                    scenario_glucose,
                    blood_pressure,
                    skin_thickness,
                    insulin,
                    bmi,
                    diabetes_pedigree,
                    diabetes_age
                ]])
                glucose_probability, _ = calculate_prediction(
                    diabetes_voting_model, glucose_features
                )
                scenarios.append({
                    "name": "Improve Glucose",
                    "value": f"{glucose:.0f} → {scenario_glucose:.0f}",
                    "probability": round(glucose_probability, 2),
                    "change": round(
                        max(voting_probability - glucose_probability, 0), 2
                    )
                })

                scenario_bp = max(blood_pressure - 10, 40)
                bp_features = np.array([[
                    pregnancies,
                    glucose,
                    scenario_bp,
                    skin_thickness,
                    insulin,
                    bmi,
                    diabetes_pedigree,
                    diabetes_age
                ]])
                bp_probability, _ = calculate_prediction(
                    diabetes_voting_model, bp_features
                )
                scenarios.append({
                    "name": "Improve Blood Pressure",
                    "value": (
                        f"{blood_pressure:.0f} → {scenario_bp:.0f}"
                    ),
                    "probability": round(bp_probability, 2),
                    "change": round(
                        max(voting_probability - bp_probability, 0), 2
                    )
                })

                scenario_bmi = max(bmi - 5, 18)
                bmi_features = np.array([[
                    pregnancies,
                    glucose,
                    blood_pressure,
                    skin_thickness,
                    insulin,
                    scenario_bmi,
                    diabetes_pedigree,
                    diabetes_age
                ]])
                bmi_probability, _ = calculate_prediction(
                    diabetes_voting_model, bmi_features
                )
                scenarios.append({
                    "name": "Improve BMI",
                    "value": f"{bmi:.1f} → {scenario_bmi:.1f}",
                    "probability": round(bmi_probability, 2),
                    "change": round(
                        max(voting_probability - bmi_probability, 0), 2
                    )
                })

                combined_features = np.array([[
                    pregnancies,
                    scenario_glucose,
                    scenario_bp,
                    skin_thickness,
                    insulin,
                    scenario_bmi,
                    diabetes_pedigree,
                    diabetes_age
                ]])
                combined_probability, _ = calculate_prediction(
                    diabetes_voting_model, combined_features
                )
                scenarios.append({
                    "name": "Combined Scenario",
                    "value": "Multiple factors improved",
                    "probability": round(combined_probability, 2),
                    "change": round(
                        max(voting_probability - combined_probability, 0), 2
                    )
                })

                common["scenarios"] = scenarios
                return render_template("index.html", **common)

            return render_template("index.html", **common)

        return render_error("Please select a valid disease.")

    except Exception as e:
        print("Error:", e)
        return render_template(
            "index.html",
            disease=request.form.get("disease", "heart"),
            error="Please enter valid values for all fields."
        )


# ==========================================================
# RUN FLASK
# ==========================================================

if __name__ == "__main__":
    app.run(debug=True)
