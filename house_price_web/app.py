
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("house_price_model.pkl")
feature_names = joblib.load("model_features.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None

    if request.method == "POST":

        try:
            area = float(request.form["area"])
            bedrooms = int(request.form["bedrooms"])
            location = request.form["location"].strip()
            resale = int(request.form["resale"])

            # Create empty input with all model features
            new_house = pd.DataFrame(
                0,
                index=[0],
                columns=feature_names
            )

            new_house["Area"] = area
            new_house["No. of Bedrooms"] = bedrooms
            new_house["Resale"] = resale

            # Location matching
            location_columns = [
                col for col in feature_names
                if col.startswith("Location_")
            ]

            location_map = {
                col.replace("Location_", "").strip().lower(): col
                for col in location_columns
            }

            location_key = location.lower()

            if location_key not in location_map:
                error = "Location not found in dataset."
            else:
                actual_location = location_map[location_key]
                new_house[actual_location] = 1

                predicted_price = model.predict(new_house)[0]

                prediction = predicted_price / 100000

        except Exception as e:
            error = "Please enter valid details."


    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
