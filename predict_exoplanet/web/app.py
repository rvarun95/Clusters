#!/usr/bin/env python3

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import model_from_json
from keras import backend as K
from keras.models import load_model
from keras.models import model_from_yaml
import pickle
from keras import backend as K
from scipy.ndimage.filters import uniform_filter1d
import pickle, sys, json
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World!"

@app.route('/hithere')
def hithere():
    return "Hi there!"

@app.route('/add',methods=['POST'])
def add():
    #If I am here, then the resouce Add was requested using the method POST
    #Step 1: Get posted data:
    app.logger.info("add enter")
    postedData = request.get_json()
    x = postedData["x"]
    y = postedData["y"]
    x = int(x)
    y = int(y)
    #Step 2: Add the posted data
    ret = x+y
    retMap = {
        "Message" : ret,
        "Status Code" : 200
    }
    app.logger.info("add exit")
    return jsonify(retMap)


def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def load_model():

    app.logger.debug("load_model enter")

    '''
    #Approach1
    app.logger.debug("load_model approach1")
    json_file = open('final_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    app.logger.info("cheers5")
    model.load_weights("final_model_weights.h5") '''

    #Approach2
    app.logger.debug("load_model approach2")
    global model
    model = tf.keras.models.load_model('final_model.h5',
        custom_objects={
            'recall_m' : recall_m,
            'precision_m' : precision_m,
            'f1_m' : f1_m
            })

    '''
    #Approach3
    app.logger.debug("load_model approach3")
    yaml_file = open('final_model.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    model.load_weights("final_model_weights.h5") '''

    app.logger.debug("load_model exit")

    return model

def preprocess_data(request):

    app.logger.debug("preprocess_data enter")
    app.logger.debug(sys.getfilesystemencoding())
    #sys.getfilesystemencoding = lambda: 'UTF-8'
    #app.logger.info(sys.getfilesystemencoding())
    #app.logger.info(request.headers)
    #request_str = request.get_data(as_text=True)
    #app.logger.info(request_str)

    request_json = request.get_json(force=True)
    #app.logger.info(request_json)
    app.logger.info(type(request_json))
    request_array = request_json["request_array"]
    #app.logger.info(request_array)
    app.logger.info(type(request_array))

    #request_json2 = request.json
    #app.logger.info(request_json2)
    #app.logger.info(type(request_json2))
    #request_json3 = json.loads(request_json1)
    #app.logger.info(request_json3)
    #app.logger.info(type(request_json3))'''
    #request_json4 = json.loads(request_json2)
    #app.logger.info(request_json4)
    #app.logger.info(type(request_json4))
    #request_arr4 = request_json2["request_array"]
    #app.logger.info(request_arr4)
    #app.logger.info(type(request_arr4))
    #request_array_np = np.fromstring(request_array)

    request_array_np = np.array(request_array, dtype=float)
    #app.logger.info(request_array_np)
    app.logger.info(type(request_array_np))
    app.logger.info(request_array_np.shape)
    app.logger.info(request_array_np.shape[0])
    #app.logger.info(request_array_np.shape[1])
    app.logger.info(request_array_np.reshape(1,request_array_np.shape[0]).shape)
    request_array_np = request_array_np.reshape(1,request_array_np.shape[0])

    x_test = request_array_np[:, 1:]
    #app.logger.info(x_test)
    y_test = request_array_np[:, 0, np.newaxis] - 1.
    x_test_mean = np.mean(x_test, axis=1).reshape(-1,1)
    x_test_std = np.std(x_test, axis=1).reshape(-1,1)
    x_test = ((x_test - x_test_mean) / x_test_std)
    #app.logger.info(x_test.shape)
    x_test = np.stack([x_test, uniform_filter1d(x_test, axis=1, size=200)], axis=2)
    #app.logger.info(x_test)
    app.logger.info(x_test.shape)
    app.logger.debug("preprocess_data exit")
    return x_test, y_test.astype(int)


@app.route('/exoplanet_predict',methods=['POST'])
def exoplanet_predict():
    app.logger.info("exoplanet_predict enter")
    K.clear_session()
    with tf.device('/cpu:0'):
        model = load_model()
        #app.logger.info(model.summary())
        x_test, y_test = preprocess_data(request)
        prediction = model.predict_classes(x_test)
        app.logger.debug(prediction)
        app.logger.debug(type(prediction))
        app.logger.debug(prediction.shape)
        #app.logger.debug(prediction[0].shape)
        #app.logger.debug(prediction[0][0].shape)
        #app.logger.debug(prediction[0][0])
        #app.logger.debug(type(prediction[0][0]))
        #app.logger.debug(y_test)
        #app.logger.debug(type(y_test))
        #app.logger.debug(y_test.shape)
        #app.logger.debug(y_test[0][0])
        if prediction[0][0] == y_test[0][0] :
            status_code = 200
            status_message = "Model prediction correct"
        else :
            status_code = 222
            status_message = "Model prediction wrong"
        output = {
            "Prediction": prediction[0][0],
            "Actual": y_test[0][0],
            "Status Code": status_code,
            "Message" : status_message
        }
        app.logger.info(output)
        app.logger.info(type(output))
        output_string=str(output)
        app.logger.info(output_string)
        app.logger.info(type(output_string))
        #app.logger.info(jsonify(output_string))
        #app.logger.info(type(jsonify(output_string)))

        #output_encode=str(output).encode('utf-8')
        #app.logger.info(output_encode)
        #app.logger.info(type(output_encode))
        #app.logger.info(jsonify(output_encode))


    K.clear_session()
    app.logger.info("exoplanet_predict exit")
    return (jsonify(output_string))


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5205)
