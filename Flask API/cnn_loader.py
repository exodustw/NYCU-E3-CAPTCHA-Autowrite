from tensorflow.keras import models
import tensorflow as tf
import numpy as np
import string

import denoise_spliter_v3

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def predict(arr):
	print('Loading Model')
	model_name = 'model_split_v1'
	model = models.load_model(model_name)

	imgs = denoise_spliter_v3.image_spliter(arr)
	imgs = np.array(imgs)

	pred = np.argmax(model.predict(imgs), axis=-1).tolist()

	pval = str(pred[0]) + str(pred[1]) + str(pred[2]) + str(pred[3])

	print('Get Prediction')
	return pval

def predict_bgr(arr):
	print('Loading Model')
	model_name = 'model_split_v1'
	model = models.load_model(model_name)

	imgs = denoise_spliter_v3.image_spliter_bgr(arr)
	imgs = np.array(imgs)

	pred = np.argmax(model.predict(imgs), axis=-1).tolist()

	pval = str(pred[0]) + str(pred[1]) + str(pred[2]) + str(pred[3])

	print('Get Prediction')
	return pval
