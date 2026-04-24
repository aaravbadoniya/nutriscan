# NutriScan

NutriScan is an AI-powered healthcare application designed to classify images of food and output real-time nutritional information (calories, protein, sugar, fat, fibre). It features a Python (Flask) backend handling AI model inferences using TensorFlow and a dynamic web interface. It is also designed to interface tightly with IoT hardware devices (such as ESP32/ESP8266), providing minimal text responses for seamless integration.

## Core Features

- **AI Image Classification**: Uses pre-trained TensorFlow/Keras models to classify images into four core categories: `Fruits`, `Snacks`, `Drinks`, and `Sweets`.
- **Nutritional Database Integration**: Pulls live nutritional facts straight from a Google Sheets CSV.
- **Hardware Endpoints**: Contains specialized hardware endpoints (`/upload_hardware` and `/get_nutrition`) optimized for lower-bandwidth microcontrollers.
- **Dashboard & Daily Tracking**: A web UI calculates your total running calorie count against a fixed daily goal (2000 kcal).

## Architecture
- **Web Backend**: Flask
- **AI / ML**: TensorFlow, NumPy, Pillow
- **Data Source**: Pandas pulling from a live Google Sheet URL

---

## Installation Guide (Local Device Setup)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aaravbadoniya/nutriscan.git
   cd nutriscan
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment:**
   - **MacOS / Linux**:
     ```bash
     source venv/bin/activate
     ```
   - **Windows**:
     ```cmd
     venv\Scripts\activate
     ```

4. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Start the application:**
   ```bash
   python app.py
   ```
   *The server will start running at `http://127.0.0.1:5000/`. You can navigate to this in your web browser interface.*

--- 

## Project Structure

- `app.py`: The main Flask server holding web views, hardware views, and ML prediction functions.
- `index.html` & `/templates/`: The front-end view handling file uploads and displaying the daily goals summary dashboard.
- `Models/`: Directory containing labeled datasets (`labels.txt`) and `.savedmodel` configurations for the ML categories.
- `Nutriscan.ino`: A stub/placeholder for IoT microcontrollers making POST requests against the Flask server endpoints.
- `uploads/`: Transitory folder created on boot used to temporarily store predicted images before they are deleted.
