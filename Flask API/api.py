from flask import Flask, request
from PIL import Image
from io import BytesIO
import numpy
import base64

import cnn_loader

app = Flask(__name__)

@app.route("/")
def home():
    return "This is a flask API!"

@app.route("/e3autologin", methods=['POST', 'GET'])
def e3autologin():
	re = "None"
	try:
		file = request.files['file'].read();
		np_arr = numpy.fromstring(file, numpy.uint8)
		#print(np_arr)
		re = cnn_loader.predict(np_arr)
	except Exception as e:
		print("error: " + str(e))
	return re


@app.route("/e3autologin_base64", methods=['POST', 'GET'])
def e3autologin_base64():
	re = "None"
	try:
		b64 = request.form['file'].replace(' ','+');

		imgdata = b64.split(',')[1]
		decoded = base64.b64decode(imgdata)
		np_arr = numpy.array(Image.open(BytesIO(decoded)).convert('RGB'))
		np_arr = np_arr[:, :, ::-1].copy() # Convert RGB to BGR
		#print(np_arr)
		re = cnn_loader.predict_bgr(np_arr)
	except Exception as e:
		print("error: " + str(e))
	return re

app.run(port=5000)
