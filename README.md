# NutriScan

NutriScan is an AI-powered food recognition and nutrition tracking application. Point a camera at any food item — the system identifies it using ONNX models and logs the calories, protein, sugar, fat, and fibre in real time. It works as a web dashboard and as the backend for an ESP32-CAM IoT device with an OLED display and 4 physical buttons.

---

## How It Works

1. User uploads a food image (web) or the ESP32-CAM captures one (hardware)
2. The Flask server runs the image through an ONNX model to identify the food
3. The server looks up nutritional facts from a live Google Sheets database
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
| Hardware | ESP32-CAM, SH1106 OLED, 4 push buttons |

### Food Categories & Models

| Category | Classes | Model file |
|---|---|---|
| Fruits | 9 (Apple, Banana, Grapes, Kiwi, Mango, Orange, Plum, Strawberry, Watermelon) | `Models/Fruits/model.onnx` |
| Snacks | 13 (Lays variants, Kurkure, Oreo, Samosa, Vada Pav, Kachori, Bhujia) | `Models/Snacks/snack_model.onnx` |
| Drinks | 6 (Coca-Cola, Pepsi, Sprite, Fanta, Frooti, Apple Juice) | `Models/Drinks/drink_model.onnx` |
| Sweets | 6 (Kit-Kat, 5 Star, Perk, Snickers, Dairy Milk, Reese's) | `Models/Sweets/sweet_model.onnx` |

All models expect a `224×224` RGB image normalised to `[0, 1]`.

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
│   │   ├── model.onnx
│   │   └── labels.txt
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

## Setup on a New Machine

### Requirements
- Python 3.9 or higher
- pip
- Internet connection (nutrition data is fetched from Google Sheets on startup)

### Steps

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd NutriScan
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
python3 app.py
```

The server starts on **http://0.0.0.0:5001** — accessible from your browser at `http://127.0.0.1:5001` and from the ESP32 at `http://<your-laptop-ip>:5001`.

> **macOS note:** Port 5000 is occupied by AirPlay Receiver. The server runs on **5001** to avoid this conflict.

**6. Find your local IP (for ESP32 config)**

macOS:
```bash
ipconfig getifaddr en0
```
Windows:
```cmd
ipconfig
```
Look for the IPv4 address on your WiFi adapter (e.g. `192.168.0.104`).

---

## API Endpoints

### Web
| Route | Method | Description |
|---|---|---|
| `/` | GET | Load the dashboard |
| `/` | POST | Submit image + category + option + quantity |

### Hardware (ESP32)
| Route | Method | Description |
|---|---|---|
| `/upload_hardware` | POST | Raw JPEG body → returns `FOOD\|CATEGORY\|option1,option2,...` |
| `/get_nutrition` | POST | Form: `product` + `option` + `quantity` → returns `calories\|protein\|sugar\|fat` |

> The ESP32 sends images as a raw JPEG byte stream (`Content-Type: image/jpeg`), not as a multipart file upload.

---

## Hardware Setup

The firmware lives in the companion PlatformIO project `NutriScan_Hardware`. See its README for full wiring and flashing instructions.

**Quick steps:**
1. Open `NutriScan_Hardware` in VS Code with the PlatformIO extension
2. In `src/main.cpp` update your WiFi credentials and server IP:
   ```cpp
   const char* ssid     = "YOUR_WIFI_NAME";
   const char* password = "YOUR_WIFI_PASSWORD";
   const char* serverUrl = "http://192.168.X.X:5001/upload_hardware";
   ```
3. Flash: `pio run -e esp32cam --target upload`
4. Make sure the Flask server is running on the same WiFi network

---

## Dependencies

```
Flask
onnxruntime
numpy
Pillow
pandas
Werkzeug
```

Full pinned versions are in `requirements.txt`.

> **Note:** The project previously used TensorFlow. All models are now ONNX (~10 MB vs ~500 MB), so no GPU or TensorFlow installation is needed.
