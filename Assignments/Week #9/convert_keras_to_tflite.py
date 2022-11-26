## Import Libraries
import tensorflow as tf
from tensorflow import keras


## Convert downloaded Keras model to TFlite model

model = keras.models.load_model('./dino_dragon_10_0.899.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open("dino-dragon-model.tflite", "wb") as f_out:
    f_out.write(tflite_model)