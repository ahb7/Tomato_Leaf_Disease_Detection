# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 12:33:15 2022

@author: Abdullah
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from flask_session import Session
from flask_dropzone import Dropzone

from tensorflow.keras import models
from PIL import Image

MODEL = models.load_model("./model")

CLASS_NAMES = ['Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
               'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Target_Spot',
               'Tomato_Two-spotted_spider_mite', 'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_healthy',
               'Tomato_mosaic_virus']

basedir = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
sess = Session()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config.update(
    UPLOADED_PATH = os.path.join(basedir, 'uploads'),
    DROPZONE_MAX_FILE_SIZE = 1024,
    DROPZONE_TIMEOUT = 5*60*1000,
    DROPZONE_UPLOAD_MULTIPLE = False,
    DROPZONE_MAX_FILE_EXCEED = 1,
    DROPZONE_ALLOWED_FILE_TYPE = 'image'    
)

dropzone = Dropzone(app)

confidence = 0
predicted_class = "None"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global confidence
    global predicted_class
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filename)
            img = Image.open(filename)
            image = np.asarray(img, dtype="int32")
            img_batch = np.expand_dims(image, 0)
            prediction = MODEL.predict(img_batch)
            predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
            #print(predicted_class, file=sys.stderr)
            confidence = round(100 * (np.max(prediction[0])), 2)
    return render_template("index.html", clas=predicted_class, confd=confidence)

@app.route("/clear/", methods=['POST'])
def clear():
    global confidence
    global predicted_class
    return render_template('index.html', clas="None", confd=0)

@app.route("/predict/", methods=['POST'])
def predict():
    global confidence
    global predicted_class
    return render_template('index.html', clas=predicted_class, confd=confidence)

if __name__ == "__main__":
    # Secret key!
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    sess.init_app(app)
    app.run(debug=True)
    