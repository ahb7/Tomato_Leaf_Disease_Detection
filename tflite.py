# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 18:56:14 2026

@author: Abdullah
"""

import tensorflow as tf
# Load your existing model
model = tf.keras.models.load_model('./model')
# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
# Save it
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
    
