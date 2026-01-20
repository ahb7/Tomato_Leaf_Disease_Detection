# Tomato Leaf Disease Detection 🍅🍃

An AI-powered web application that identifies diseases in tomato leaves using Deep Learning. This project is optimized for high-performance inference in resource-constrained environments.

## 🚀 Live Demo
[Check out the app on Render](https://tomato-leaf-disease-detection.onrender.com/)

## ✨ Features
* Instant Diagnosis: Upload a photo of a tomato leaf to detect 10 common conditions (Blight, Mold, Virus, etc.).
* Mobile Optimized: Lightweight UI built with Flask and Bootstrap.
* Edge Optimization: Uses **TensorFlow Lite (TFLite)** to ensure fast predictions under 512MB of RAM.

## 🛠️ Technical Stack
* Frontend: HTML5, CSS3, Bootstrap
* Backend: Flask (Python)
* Machine Learning: TensorFlow, Keras
* Deployment: Render (with Gunicorn)
* Optimization: Model quantized and converted to `.tflite` for low-memory footprint.

Local Setup  
Clone the repository:
git clone https://github.com/ahb7/Tomato_Leaf_Disease_Detection.git

Install dependencies:
pip install -r requirements.txt

Run the app:
python app.py  

