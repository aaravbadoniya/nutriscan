# NutriScan

NutriScan is an AI-powered food recognition and nutrition tracking application. Point your camera at any food item — the system identifies it and logs the calories, protein, sugar, fat, and fibre in real time. It works both as a web dashboard and as a backend for an ESP32-CAM IoT device with an OLED display.

---

## How It Works

1. User uploads a food image (web) or the ESP32-CAM captures one (hardware)
2. The Flask server runs the image through an ONNX model to identify the food
3. The server looks up the nutritional facts from a Google Sheets database
4. Results are shown on the web dashboard or sent back to the OLED display

---

## Architecture

| Layer | Technology |
|---|---|
| Web backend | Python, Flask |
| AI inference | ONNX Runtime (converted from Keras/TensorFlow) |
| Image processing | Pillow, NumPy |
| Nutrition database | Pandas reading a live Google Sheets CSV |
| Web frontend | HTML, Bootstrap 5 |
| Hardware | ESP32-CAM, OLED (SH1106), 4 push buttons |

### Food Categories & Models

| Category | Classes | Model file |
|---|---|---|
| Fruits | 9 (Apple, Banana, Grapes, Kiwi, Mango, Orange, Plum, Strawberry, Watermelon) | `Models/Fruits/model.onnx` |
| Snacks | 13 (Lays variants, Kurkure, Oreo, Samosa, Vada Pav, Kachori, Bhujia) | `Models/Snacks/snack_model.onnx` |
| Drinks | 6 (Coca-Cola, Pepsi, Sprite, Fanta, Frooti, Apple Juice) | `Models/Drinks/drink_model.onnx` |
| Sweets | 6 (Kit-Kat, 5 Star, Perk, Snickers, Dairy Milk, Reese's) | `Models/Sweets/sweet_model.onnx` |

All models expect a `224×224` RGB image normalized to `[0, 1]`.

---

## Project Structure

```
NutriScan/
├── app.py                      # Flask server — routes, inference, nutrition lookup
├── requirements.txt            # Python dependencies
├── templates/
│   └── index.html              # Web dashboard (Bootstrap 5)
├── Models/
│   ├── Fruits/
│   │   ├── model.onnx          # ONNX inference model
│   │   └── labels.txt          # Class names
│   ├── Snacks/
│   │   ├── snack_model.onnx
│   │   └── snack_labels.txt.txt
│   ├── Drinks/
│   │   ├── drink_model.onnx
│   │   └── drink_label.txt.txt
│   └── Sweets/
│       ├── sweet_model.onnx
│       └── sweet_label.txt.txt
└── uploads/                    # Temp folder for images during prediction (auto-created)
```

---

## Setup on a New Device

### Requirements
- Python 3.9 or higher
- pip
- Internet connection (to fetch nutrition data from Google Sheets on startup)

### Steps

**1. Clone the repository**
```bash
git clone https://github.com/aaravbadoniya/nutriscan.git
cd nutriscan
```

**2. Create a virtual environment**
```bash
python3 -m venv venv
```

**3. Activate it**

macOS / Linux:
```bash
source venv/bin/activate
```

Windows:
```cmd
venv\Scripts\activate
```

**4. Install dependencies**
```bash
pip install -r requirements.txt
```

**5. Run the server**
```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

---

## API Endpoints

### Web
| Route | Method | Description |
|---|---|---|
| `/` | GET | Load the dashboard |
| `/` | POST | Submit an image + category + option + quantity to scan a meal |

### Hardware (ESP32)
| Route | Method | Description |
|---|---|---|
| `/upload_hardware` | POST | Send a raw JPEG; returns `FOOD\|CATEGORY\|option1,option2,...` |
| `/get_nutrition` | POST | Send `product` + `option`; returns `calories\|protein\|sugar\|fat` |

---

## Hardware Setup (ESP32-CAM)

The firmware lives in the companion PlatformIO project `NutriScan_Hardware`.

1. Open the project in PlatformIO (VS Code extension or CLI)
2. In `src/main.cpp`, update these two lines:
   ```cpp
   const char* ssid     = "YOUR_WIFI_NAME";
   const char* password = "YOUR_WIFI_PASSWORD";
   const char* serverUrl = "http://YOUR_COMPUTER_IP:5000/upload_hardware";
   ```
   Replace `YOUR_COMPUTER_IP` with the local IP of the machine running `app.py`
   (find it with `ipconfig` on Windows or `ifconfig` on macOS/Linux)
3. Flash to the ESP32-CAM via PlatformIO: `pio run --target upload`
4. Make sure the Flask server is running on the same WiFi network

### Hardware Controls
| Button | GPIO | Action |
|---|---|---|
| SCAN | 14 | Capture image and send to server |
| UP | 12 | Navigate menu up / increase quantity |
| DOWN | 13 | Navigate menu down / decrease quantity |
| SELECT | 15 | Confirm option → then confirm quantity |

---

## Dependencies

```
Flask==3.0.3
onnxruntime==1.19.2
numpy==1.26.4
Pillow==10.3.0
pandas==2.2.2
Werkzeug==3.0.1
```

> **Note:** The project previously used TensorFlow. All models have been converted to ONNX format (`tf2onnx`), which is much lighter (~10 MB vs ~500 MB) and starts faster. No GPU or TensorFlow installation needed.
