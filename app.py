
from flask import Flask, render_template, request
import PIL
from PIL import Image
import numpy as np
import keras.models
import re
import io
from io import BytesIO
import base64
import sys
import os

#sys.path.append(os.path.abspath("./model"))
#from load import *


# Initalize our flask app
app = Flask(__name__)
#global model, graph
#model, graph = init()

# Route to render home page
@app.route('/')
def index():
    return render_template("index.html")

# Prediction route
@app.route('/predict/', methods=['POST'])
def predict():
    # NEED IF THEN OR KNOWS POST OR GET
    image_data = request.get_data()

    # Convert base64 to image
    imgstr = re.search(r'base64,(.*)', str(image_data)).group(1)
    image_bytes = io.BytesIO(base64.b64decode(imgstr))
    
    # Convert image to pixels
    image = Image.open(image_bytes)
    image_size = 28, 28
    image = image.resize(image_size, Image.ANTIALIAS)
    image = image.convert('1')
    image_array = np.asarray(image)
   
    # Perform prediction and return response
    with graph.as_default():
        prediction = model.predict(image_array)
        print(prediction)
        print(np.argmax(prediction, axis=1))
        result = np.argmax(prediction, axis=1)
        # DO I NEED TO JSONIFY STRING?
        # return jsonify(str(response[0]))
        return jsonify(str(result[0]))

# Run locally on port 5000
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

# app.run(debug=True)