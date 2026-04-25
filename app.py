from flask import Flask, request, render_template, session
import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "healthcare_app_secret_key"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8 MB limit


# =====================================================
# CREATE UPLOAD FOLDER
# =====================================================

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


# =====================================================
# LOAD MODELS
# =====================================================

def safe_load_model(path):
    try:
        return ort.InferenceSession(path)
    except Exception as e:
        print(f"[!] Error loading model at {path}: {e}")
        return None

models = {
    "fruits": safe_load_model("Models/Fruits/model.onnx"),
    "snacks": safe_load_model("Models/Snacks/snack_model.onnx"),
    "drinks": safe_load_model("Models/Drinks/drink_model.onnx"),
    "sweets": safe_load_model("Models/Sweets/sweet_model.onnx"),
}

# ONNX input node names differ slightly per model
model_input_names = {
    "fruits": "sequential_1_input",
    "snacks": "sequential_1_input:0",
    "drinks": "sequential_1_input:0",
    "sweets": "sequential_1_input:0",
}


# =====================================================
# LOAD LABELS
# =====================================================

labels = {}

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

try:
    data = pd.read_csv(sheet_url)
except Exception as e:
    print(f"[!] Could not load nutrition data: {e}")
    data = pd.DataFrame(columns=["product", "option", "calories", "protein", "sugar", "fat", "fibre"])


# =====================================================
# ALLOWED FILE TYPES
# =====================================================

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'webp'}

def allowed_file(file):
    filename = file.filename or ""
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext in ALLOWED_EXTENSIONS and (file.content_type or "").startswith('image/')


# =====================================================
# AI PREDICTION
# =====================================================

def predict_image(image_path, category):

    session  = models[category]
    input_name = model_input_names[category]
    label_list = labels[category]

    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))

    # ONNX runtime requires float32
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = session.run(None, {input_name: img_array})[0]

    index      = int(np.argmax(prediction))
    confidence = float(prediction[0][index])

    return label_list[index], confidence


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
            "protein":  row.get("protein", 0),
            "sugar":    row.get("sugar", 0),
            "fat":      row.get("fat", 0),
            "fibre":    row.get("fibre", row.get("fiber", 0)),
        }

    return {"calories": 0, "protein": 0, "sugar": 0, "fat": 0, "fibre": 0}


# =====================================================
# HARDWARE IMAGE UPLOAD ROUTE
# =====================================================

@app.route("/upload_hardware", methods=["POST"])
def upload_hardware():

    if 'file' not in request.files:
        return "ERROR|NO_IMAGE", 400

    file = request.files['file']

    if not allowed_file(file):
        return "ERROR|INVALID_FILE", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    final_label      = None
    detected_category = None

    try:
        for category in ["fruits", "snacks", "drinks", "sweets"]:
            if models.get(category) is None:
                continue
            label, confidence = predict_image(filepath, category)
            if confidence > 0.70:
                final_label       = label
                detected_category = category
                break
    except Exception as e:
        print(f"[!] Prediction error: {e}")
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    if not final_label:
        return "ERROR|NOT_RECOGNIZED", 200

    product_rows  = data[data["product"].str.lower() == final_label.lower()]
    options       = product_rows["option"].dropna().astype(str).unique()
    options_string = ",".join(options)

    return f"{final_label}|{detected_category}|{options_string}"


# =====================================================
# HARDWARE NUTRITION ROUTE
# =====================================================

@app.route("/get_nutrition", methods=["POST"])
def get_nutrition_hardware():

    product = request.form.get("product", "").strip()
    option  = request.form.get("option",  "").strip()

    if not product or not option:
        return "0|0|0|0"

    row = data[
        (data["product"].str.lower() == product.lower()) &
        (data["option"].astype(str).str.lower() == option.lower())
    ]

    if row.empty:
        return "0|0|0|0"

    row      = row.iloc[0]
    calories = row.get("calories", 0)
    protein  = row.get("protein",  0)
    sugar    = row.get("sugar",    0)
    fat      = row.get("fat",      0)

    return f"{calories}|{protein}|{sugar}|{fat}"


# =====================================================
# WEBSITE ROUTE
# =====================================================

@app.route("/", methods=["GET", "POST"])
def index():

    if 'total_calories' not in session:
        session['total_calories'] = 0

    label        = None
    confidence   = None
    meal_cal     = 0
    product_info = {}
    error        = None
    daily_goal   = 2000

    if request.method == "POST":

        category = request.form.get("category", "")
        option   = request.form.get("option", "").strip()
        file     = request.files.get("file")

        try:
            quantity = float(request.form.get("quantity", 1))
            if quantity <= 0:
                quantity = 1.0
        except ValueError:
            quantity = 1.0

        if category not in models:
            error = "Invalid category selected."

        elif file and file.filename:

            if not allowed_file(file):
                error = "Please upload a valid image file (JPG, PNG, etc.)"
            else:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                try:
                    label, confidence = predict_image(filepath, category)

                    if option:
                        product_info = get_product_data(label, option)
                        meal_cal     = product_info["calories"] * quantity
                        session['total_calories'] += meal_cal
                        session.modified = True

                except Exception as e:
                    print(f"[!] Prediction error: {e}")
                    error = "Could not process image. Please try again with a clearer photo."

                finally:
                    if os.path.exists(filepath):
                        os.remove(filepath)

    progress = min((session['total_calories'] / daily_goal) * 100, 100)

    return render_template(
        "index.html",
        label=label,
        confidence=confidence,
        meal_cal=meal_cal,
        daily_total=session['total_calories'],
        progress=progress,
        goal=daily_goal,
        protein=product_info.get("protein", 0),
        sugar=product_info.get("sugar",    0),
        fat=product_info.get("fat",        0),
        fibre=product_info.get("fibre",    0),
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=False)
