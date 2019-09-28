# requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template, request
# scientific computing library for saving, reading, and resizing images
import PIL
from PIL import Image
# for matrix math
import numpy as np
import keras.models
# for regular expressions, saves time dealing with string data
import re
import io
from io import BytesIO
import base64

# system level operations (like loading files)
import sys
# for reading operating system data
import os
# tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))
from load import *
#from keras.models import model_from_json

# initalize our flask app
app = Flask(__name__)
# global vars for easy reusability
global model, graph
# initialize these variables
model, graph = init()

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    # whenever the predict method is called, we're going
    # to input the user drawn character as an image into the model
    # perform inference, and return the classification
    # get the raw data format of the image
    # NEED IF THEN OR KNOWS POST OR GET
    imgData = request.get_data()
    # decoding an image from base64 into raw representation
    imgstr = re.search(r'base64,(.*)', str(imgData)).group(1)
    image_bytes = io.BytesIO(base64.b64decode(imgstr))
    
    # read the image into memory
    image = Image.open(image_bytes)
    # resize width & height to 28 x 28
    img_size = 28, 28
    image = image.resize(img_size, Image.ANTIALIAS)
    image = image.convert('1')
    img_array = np.asarray(image)
   
    # in our computation graph
    with graph.as_default():
        # perform the prediction
        prediction = model.predict(img_array)
        print(prediction)
        # use np.argmax to find prediction value
        print(np.argmax(prediction, axis=1))
        # convert the response to a string
        response = np.argmax(prediction, axis=1)
        # DO I NEED TO JSONIFY STRING?
        # return jsonify(str(response[0]))
        return str(response[0])

if __name__ == "__main__":
    # run the app locally on the given port
    app.run(host='0.0.0.0', port=5000)
# optional if we want to run in debugging mode
# app.run(debug=True)