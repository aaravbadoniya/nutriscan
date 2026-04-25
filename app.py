from flask import Flask, request, render_template, session, abort
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import pandas as pd
from werkzeug.utils import secure_filename
import logging
from functools import wraps

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# =====================================================
# APP CONFIG
# =====================================================

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-me-set-SECRET_KEY-env-var")
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8 MB upload limit

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'webp'}
ALLOWED_CATEGORIES = {'fruits', 'snacks', 'drinks', 'sweets'}

# Hardware endpoint API key — set HARDWARE_API_KEY env var to enable auth.
# If left empty, hardware endpoints are unauthenticated (dev/local only).
HARDWARE_API_KEY = os.environ.get("HARDWARE_API_KEY", "")


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
        model = tf.keras.models.load_model(path)
        logger.info(f"Loaded model: {path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model at {path}: {e}")
        return None

models = {
    "fruits": safe_load_model("Models/Fruits/model.savedmodel"),
    "snacks": safe_load_model("Models/Snacks/snack_model.h5"),
    "drinks": safe_load_model("Models/Drinks/drink_model.h5"),
    "sweets": safe_load_model("Models/Sweets/sweet_model.h5"),
}

failed_models = [k for k, v in models.items() if v is None]
if failed_models:
    logger.warning(f"Some models failed to load: {failed_models}")


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
        logger.info(f"Loaded {len(labels[category])} labels for '{category}'")
    else:
        logger.warning(f"Label file not found: {label_path}")


# =====================================================
# GOOGLE SHEETS CSV
# =====================================================

sheet_url = os.environ.get(
    "GOOGLE_SHEET_URL",
    "https://docs.google.com/spreadsheets/d/1397xmzJuacgHt7TH6EDXxUpMfeA952Cmq1fS-MJSQVg/export?format=csv"
)

try:
    data = pd.read_csv(sheet_url)
    logger.info(f"Loaded {len(data)} product records from nutrition spreadsheet")
except Exception as e:
    logger.error(f"Failed to load nutrition data from spreadsheet: {e}")
    data = pd.DataFrame(columns=["product", "option", "calories", "protein", "sugar", "fat", "fibre"])


# =====================================================
# HELPERS
# =====================================================

def allowed_file(file):
    filename = file.filename or ""
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if ext not in ALLOWED_EXTENSIONS:
        return False
    mime = (file.content_type or "").lower()
    return mime.startswith('image/')


def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if HARDWARE_API_KEY:
            provided = request.headers.get("X-API-Key", "")
            if provided != HARDWARE_API_KEY:
                logger.warning("Hardware endpoint rejected: invalid API key")
                abort(401)
        return f(*args, **kwargs)
    return decorated


# =====================================================
# AI PREDICTION
# =====================================================

def predict_image(image_path, category):
    model = models.get(category)
    if model is None:
        raise ValueError(f"Model for '{category}' is not loaded")

    label_list = labels.get(category, [])
    if not label_list:
        raise ValueError(f"Labels for '{category}' not found")

    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)
    index = int(np.argmax(prediction))
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
            "protein":  row.get("protein",  0),
            "sugar":    row.get("sugar",     0),
            "fat":      row.get("fat",       0),
            "fibre":    row.get("fibre", row.get("fiber", 0)),
        }

    return {"calories": 0, "protein": 0, "sugar": 0, "fat": 0, "fibre": 0}


# =====================================================
# HARDWARE IMAGE UPLOAD ROUTE
# =====================================================

@app.route("/upload_hardware", methods=["POST"])
@require_api_key
def upload_hardware():
    if 'file' not in request.files:
        return "ERROR|NO_IMAGE", 400

    file = request.files['file']

    if not allowed_file(file):
        logger.warning(f"Hardware upload rejected: invalid file type '{file.content_type}'")
        return "ERROR|INVALID_FILE", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    final_label = None
    detected_category = None

    try:
        for category in ALLOWED_CATEGORIES:
            if models.get(category) is None:
                continue
            try:
                label, confidence = predict_image(filepath, category)
                logger.info(f"Hardware scan: {label} ({confidence:.2%}) [{category}]")
                if confidence > 0.70:
                    final_label = label
                    detected_category = category
                    break
            except Exception as e:
                logger.error(f"Prediction error for category '{category}': {e}")
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    if not final_label:
        return "ERROR|NOT_RECOGNIZED", 200

    product_rows = data[data["product"].str.lower() == final_label.lower()]
    options = product_rows["option"].dropna().astype(str).unique()
    options_string = ",".join(options)

    return f"{final_label}|{detected_category}|{options_string}"


# =====================================================
# HARDWARE NUTRITION ROUTE
# =====================================================

@app.route("/get_nutrition", methods=["POST"])
@require_api_key
def get_nutrition_hardware():
    product = (request.form.get("product") or "").strip()
    option  = (request.form.get("option")  or "").strip()

    if not product or not option:
        return "0|0|0|0"

    row = data[
        (data["product"].str.lower() == product.lower()) &
        (data["option"].astype(str).str.lower() == option.lower())
    ]

    if row.empty:
        logger.info(f"Nutrition not found: product='{product}' option='{option}'")
        return "0|0|0|0"

    row = row.iloc[0]
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

        if category not in ALLOWED_CATEGORIES:
            error = "Invalid category selected."
        else:
            try:
                quantity = float(request.form.get("quantity", 1))
                quantity = max(0.1, min(quantity, 100.0))
            except (ValueError, TypeError):
                quantity = 1.0

            option = (request.form.get("option") or "").strip()
            file   = request.files.get("file")

            if file and file.filename:
                if not allowed_file(file):
                    error = "Invalid file type. Please upload a JPG, PNG, or WebP image."
                else:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    try:
                        label, confidence = predict_image(filepath, category)
                        logger.info(f"Web scan: {label} ({confidence:.2%}) category={category}")

                        if option:
                            product_info = get_product_data(label, option)
                            meal_cal = product_info["calories"] * quantity
                            session['total_calories'] = session.get('total_calories', 0) + meal_cal
                            session.modified = True
                    except Exception as e:
                        logger.error(f"Prediction error: {e}")
                        error = "Could not process the image. Please try a clearer photo."
                    finally:
                        if os.path.exists(filepath):
                            os.remove(filepath)

    progress = (session['total_calories'] / daily_goal) * 100

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
        fat=product_info.get("fat",       0),
        fibre=product_info.get("fibre",   0),
        error=error,
    )


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug)
