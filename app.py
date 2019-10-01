from PIL import Image
from io import BytesIO
import base64
import numpy as np
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.preprocessing.image import img_to_array
import re
import io
import sys
import os
import tensorflow as tf

from tensorflow.keras.models import load_model

# global model

# model = load_model("model.h5")

# sys.path.append(os.path.abspath("./model"))
# from tensorflow.keras.models import model_from_json

# def init():
#   json_file = open('./model/model.json','r')
#   loaded_model_json = json_file.read()
#   json_file.close()
#   print(f'test: {loaded_model_json}')
#   loaded_model = model_from_json(loaded_model_json)
#   #load weights into new model
#   loaded_model.load_weights("./model/model.h5")
#   print("Loaded Model from disk")
#   #compile and evaluate loaded model
#   loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#   graph = tf.get_default_graph()
#   return loaded_model,graph



app = Flask(__name__)
# model, graph = init()
# model = load_model("model.h5")

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # NEED IF THEN OR KNOWS POST OR GET
    image_data = request.get_data()
    # print(image_data)

    # Convert base64 to image
    imgstr = re.search(r'base64,(.*)', str(image_data)).group(1)
    image_bytes = io.BytesIO(base64.b64decode(imgstr))
    image = Image.open(image_bytes).convert("L")
    image_size = 28, 28
    image = image.resize(image_size, Image.ANTIALIAS)
    image_array = img_to_array(image)
    image_array = image_array.flatten()
    image_array = image_array.reshape(1,28,28,1)
    # print(image_array)

    model = load_model("digit.h5")
    prediction = model.predict(image_array)
    print(prediction)
    result = np.argmax(prediction, axis=1)
    print(result)

    return jsonify(str(result[0]))

    # return render_template("index.html", result = jsonify(str(result[0])))

 

if __name__ == "__main__":
    app.run(debug=True)