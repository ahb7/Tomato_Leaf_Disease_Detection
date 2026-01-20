import os
import numpy as np
from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
from tensorflow.keras import models
from PIL import Image
import tensorflow as tf


try:
    import markupsafe
    import flask
    flask.Markup = markupsafe.Markup
except ImportError:
    pass

app = Flask(__name__)
app.secret_key = 'super-secret-key-for-local-dev'

# Configuration
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load Model
# MODEL = models.load_model("./model")
CLASS_NAMES = [
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Target_Spot',
    'Tomato_Two-spotted_spider_mite', 'Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato_healthy', 'Tomato_mosaic_virus'
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_class = "None"
    confidence = 0
    filename = request.form.get('current_filename')

    if request.method == 'POST':
        file = request.files.get('file')
        
        # Scenario A: User uploaded a new file
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # We don't predict yet! Just save and show the preview.

        # Scenario B: User clicked the PREDICT button
        if 'predict_btn' in request.form and filename:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(filepath):
                img = Image.open(filepath).convert('RGB')
                img = img.resize((256, 256)) 
                image = np.asarray(img)
                img_batch = np.expand_dims(image, 0).astype(np.float32)

                interpreter.set_tensor(input_details[0]['index'], img_batch)
                interpreter.invoke()
                prediction = interpreter.get_tensor(output_details[0]['index'])
                predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
                confidence = round(100 * (np.max(prediction[0])), 2)

    return render_template("index.html", clas=predicted_class, confd=confidence, img_name=filename)

@app.route('/clear')
def clear():
    return redirect('/')

if __name__ == "__main__":
    app.run(debug=True, port=5000)
    
