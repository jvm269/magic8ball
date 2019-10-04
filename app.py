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


app = Flask(__name__)


def init():
   global model, graph
   # load the pre-trained Keras model
   model = load_model("modellabeledfinal.h5")
   graph = tf.compat.v1.get_default_graph()


init()


@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # NEED IF THEN OR KNOWS POST OR GET
    image_data = request.get_data()

    # Convert base64 to image
    imgstr = re.search(r'base64,(.*)', str(image_data)).group(1)
    image_bytes = io.BytesIO(base64.b64decode(imgstr))
    image = Image.open(image_bytes).convert("L")
    image_size = 28, 28
    image = image.resize(image_size, Image.ANTIALIAS)
    image_array = img_to_array(image)
    image_array /= 255
    image_array = image_array.flatten()
    image_array = image_array.reshape(1,28,28,1)
    # print(image_array)

    # model = load_model("modellabeledfinal.h5")
    prediction = model.predict(image_array)
    print(prediction)
    result = np.argmax(prediction, axis=1)
    print(result)

    mapping =  {"0" : "0","1": "1","2": "2","3": "3","4": "4","5": "5","6": "6","7": "7","8": "8","9": "9","10": 'A',"11": 'B',"12": 'C',"13": 'D',"14": 'E',"15": 'F',"16": 'G',"17": 'H',"18": 'I',"19": 'J',"20": 'K',"21": 'L',"22": 'M',"23": 'N',"24": 'O',"25": 'P',"26": 'Q',"27": 'R',"28": 'S',"29": 'T',"30": 'U',"31": 'V',"32": 'W',"33": 'X',"34": 'Y',"35": 'Z'}
    print(mapping[str(result[0])])

    return jsonify(mapping[str(result[0])])



# if __name__ == "__main__":
#     app.run(debug=False)
if __name__ == "__main__":
   print(("* Loading Keras model and Flask starting server..."
       "please wait until server has fully started"))
   app.run(debug=False)