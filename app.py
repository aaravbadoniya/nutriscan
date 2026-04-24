from flask import Flask, request, render_template, session
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "healthcare_app_secret_key"

app.config['UPLOAD_FOLDER'] = 'uploads'


# =====================================================
# CREATE UPLOAD FOLDER
# =====================================================

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


# =====================================================
# LOAD MODELS
# =====================================================

models = {
    "fruits": tf.keras.models.load_model("Models/Fruits/model.savedmodel"),
    "snacks": tf.keras.models.load_model("Models/Snacks/snack_model.h5"),
    "drinks": tf.keras.models.load_model("Models/Drinks/drink_model.h5"),
    "sweets": tf.keras.models.load_model("Models/Sweets/sweet_model.h5"),
}


# =====================================================
# LOAD LABELS
# =====================================================

labels = {}

# Mapping specific label files to categories
label_files = {
    "fruits": "Models/Fruits/labels.txt",
    "snacks": "Models/Snacks/snack_labels.txt.txt",
    "drinks": "Models/Drinks/drink_label.txt.txt",
    "sweets": "Models/Sweets/sweet_label.txt.txt"
}

for category, label_path in label_files.items():
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            labels[category] = [line.strip() for line in f.readlines()]


# =====================================================
# GOOGLE SHEETS CSV
# =====================================================

sheet_url = "https://docs.google.com/spreadsheets/d/1397xmzJuacgHt7TH6EDXxUpMfeA952Cmq1fS-MJSQVg/export?format=csv"

data = pd.read_csv(sheet_url)


# =====================================================
# AI PREDICTION
# =====================================================

def predict_image(image_path, category):

    model = models[category]

    label_list = labels[category]


    img = Image.open(image_path)

    img = img.resize((224, 224))


    img_array = np.array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)


    prediction = model.predict(img_array)


    index = np.argmax(prediction)

    confidence = prediction[0][index]


    return label_list[index], float(confidence)


# =====================================================
# GET PRODUCT DATA
# =====================================================

def get_product_data(product_name, option):

    row = data[
        (data["product"].str.lower() == product_name.lower()) &
        (data["option"].str.lower() == option.lower())
    ]


    if not row.empty:

        row = row.iloc[0]

        return {
            "calories": row.get("calories", 0),
            "protein": row.get("protein", 0),
            "sugar": row.get("sugar", 0),
            "fat": row.get("fat", 0)
        }


    return {
        "calories": 0,
        "protein": 0,
        "sugar": 0,
        "fat": 0
    }


# =====================================================
# HARDWARE IMAGE UPLOAD ROUTE
# =====================================================

@app.route("/upload_hardware", methods=["POST"])
def upload_hardware():

    if 'file' not in request.files:
        return "ERROR|NO_IMAGE", 400


    file = request.files['file']

    filename = secure_filename(file.filename)

    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        filename
    )

    file.save(filepath)


    final_label = None
    detected_category = None


    all_categories = [
        "fruits",
        "snacks",
        "drinks",
        "sweets"
    ]


    for category in all_categories:

        label, confidence = predict_image(
            filepath,
            category
        )


        if confidence > 0.70:

            final_label = label

            detected_category = category

            break


    os.remove(filepath)


    if not final_label:
        return "ERROR|NOT_RECOGNIZED", 200


    # =========================================
    # GET ALL OPTIONS FOR PRODUCT
    # =========================================

    product_rows = data[
        data["product"].str.lower()
        == final_label.lower()
    ]


    options = (
        product_rows["option"]
        .dropna()
        .astype(str)
        .unique()
    )


    options_string = ",".join(options)


    return (
        f"{final_label}|"
        f"{detected_category}|"
        f"{options_string}"
    )


# =====================================================
# HARDWARE NUTRITION ROUTE
# =====================================================

@app.route("/get_nutrition", methods=["POST"])
def get_nutrition_hardware():

    product = request.form.get("product")

    option = request.form.get("option")


    row = data[
        (data["product"].str.lower() == product.lower()) &
        (data["option"].astype(str).str.lower() == option.lower())
    ]


    if row.empty:
        return "0|0|0|0"


    row = row.iloc[0]


    calories = row.get("calories", 0)
    protein = row.get("protein", 0)
    sugar = row.get("sugar", 0)
    fat = row.get("fat", 0)


    return (
        f"{calories}|"
        f"{protein}|"
        f"{sugar}|"
        f"{fat}"
    )


# =====================================================
# WEBSITE ROUTE
# =====================================================

@app.route("/", methods=["GET", "POST"])
def index():

    if 'total_calories' not in session:
        session['total_calories'] = 0


    label = None
    confidence = None
    meal_cal = 0
    product_info = {}

    daily_goal = 2000


    if request.method == "POST":

        category = request.form["category"]

        quantity = float(
            request.form.get("quantity", 1)
        )

        option = request.form.get("option")


        file = request.files["file"]


        if file and file.filename != '':

            filename = secure_filename(file.filename)

            filepath = os.path.join(
                app.config['UPLOAD_FOLDER'],
                filename
            )

            file.save(filepath)


            label, confidence = predict_image(
                filepath,
                category
            )


            if option:

                product_info = get_product_data(
                    label,
                    option
                )

                meal_cal = (
                    product_info["calories"]
                    * quantity
                )

                session['total_calories'] += meal_cal


            os.remove(filepath)


    progress = (
        session['total_calories']
        / daily_goal
    ) * 100


    return render_template(
        "index.html",
        label=label,
        confidence=confidence,
        meal_cal=meal_cal,
        daily_total=session['total_calories'],
        progress=progress,
        goal=daily_goal,
        protein=product_info.get("protein", 0),
        sugar=product_info.get("sugar", 0),
        fat=product_info.get("fat", 0),
        fibre=product_info.get("fibre", 0)
    )


if __name__ == "__main__":
    app.run(debug=True)