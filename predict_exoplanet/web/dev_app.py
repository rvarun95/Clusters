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

#@app.route('/')
#def home():
#    return render_template('index.html')

@app.route('/')
def home():
    return "Hello World!"

@app.route('/hithere')
def hithere():
    return "Hi there!"

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))




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

    app.logger.info("cheers1")
    json_file = open('final_model.json', 'r')
    app.logger.info("cheers2")
    loaded_model_json = json_file.read()
    app.logger.info("cheers3")
    json_file.close()
    app.logger.info("cheers4")

    #Try1
    #model = model_from_json(loaded_model_json)
    # load weights into new model
    #app.logger.info("cheers5")
    #model.load_weights("final_model_weights.h5")

    #Try2
    app.logger.info("cheers4.5")
    global model
    model = tf.keras.models.load_model('final_model.h5',
        custom_objects={
            'recall_m' : recall_m,
            'precision_m' : precision_m,
            'f1_m' : f1_m
            })
    #model = load_model('final_model.h5')

    #Try3
    #yaml_file = open('final_model.yaml', 'r')
    #loaded_model_yaml = yaml_file.read()
    #yaml_file.close()
    #loaded_model = model_from_yaml(loaded_model_yaml)
    #model.load_weights("final_model_weights.h5")

    app.logger.info("cheers6")

    return model

def preprocess_data(request):

    app.logger.info("logger hey0")
    app.logger.info(sys.getfilesystemencoding())
    #sys.getfilesystemencoding = lambda: 'UTF-8'
    app.logger.info("logger hey0.1")
    #app.logger.info(sys.getfilesystemencoding())
    app.logger.info("logger hey0.2")
    app.logger.info(request.headers)
    app.logger.info("logger hey1.0")
    #request_str = request.get_data(as_text=True)

    app.logger.info("logger hey1.1")
    #app.logger.info(request_str)

    request_json1 = request.get_json(force=True)
    app.logger.info("logger hey1.2")
    #app.logger.info(request_json1)
    app.logger.info(type(request_json1))

    #request_json2 = request.json
    #app.logger.info("logger hey1.3")
    #app.logger.info(request_json2)
    #app.logger.info(type(request_json2))
    #request_json3 = json.loads(request_json1)
    #app.logger.info("logger hey1.4")
    #app.logger.info(request_json3)
    #app.logger.info(type(request_json3))'''
    #request_json4 = json.loads(request_json2)
    #app.logger.info("logger hey1.5")
    #app.logger.info(request_json4)
    #app.logger.info(type(request_json4))
    app.logger.info("logger hey1.35")
    request_arr3 = request_json1["request_array"]
    app.logger.info("logger hey1.6")
    #app.logger.info(request_arr3)
    app.logger.info(type(request_arr3))

    #request_arr4 = request_json2["request_array"]
    app.logger.info("logger hey1.7")
    #app.logger.info(request_arr4)
    #app.logger.info(type(request_arr4))

    #request_arr5 = np.fromstring(request_arr3)
    request_arr5 = np.array(request_arr3, dtype=float)
    app.logger.info("logger hey1.8")
    #app.logger.info(request_arr5)
    app.logger.info(type(request_arr5))
    app.logger.info(request_arr5.shape)
    app.logger.info(request_arr5.shape[0])
    #app.logger.info(request_arr5.shape[1])
    app.logger.info(request_arr5.reshape(1,request_arr5.shape[0]).shape)
    request_arr5 = request_arr5.reshape(1,request_arr5.shape[0])

    x_test = request_arr5[:, 1:]
    app.logger.info("logger hey4")
    #app.logger.info(x_test)
    y_test = request_arr5[:, 0, np.newaxis] - 1.
    app.logger.info("logger hey5")
    x_test_mean = np.mean(x_test, axis=1).reshape(-1,1)
    app.logger.info("logger hey6")
    x_test_std = np.std(x_test, axis=1).reshape(-1,1)
    app.logger.info("logger hey7")
    x_test = ((x_test - x_test_mean) / x_test_std)
    app.logger.info("logger hey8")
    #app.logger.info(x_test)
    x_test = np.stack([x_test, uniform_filter1d(x_test, axis=1, size=200)], axis=2)
    app.logger.info("logger hey9")

    #app.logger.info(x_test)
    app.logger.info(x_test.shape)
    return x_test, y_test.astype(int)


@app.route('/exoplanet_predict',methods=['POST'])
def exoplanet_predict():
    #If I am here, then the resouce Add was requested using the method POST
    #Step 1: Get posted data:
    #sys.getfilesystemencoding = lambda: 'UTF-8'
    app.logger.info("logger hippy1")
    K.clear_session()
    with tf.device('/CPU:0'):


        app.logger.info("logger hippy2")
        model = load_model()
        app.logger.info(model.summary())
        app.logger.info("logger hippy3")
        #postedData = request.get_json()
        x_test, y_test = preprocess_data(request)
        app.logger.info("logger hippy4")
        prediction = model.predict_classes(x_test)
        print("hey5")
        app.logger.info("logger hippy4.5")
        app.logger.info(prediction)
        app.logger.info(type(prediction))
        app.logger.info(prediction.shape)
        app.logger.info(prediction[0].shape)
        app.logger.info(prediction[0][0].shape)
        app.logger.info(prediction[0][0])
        app.logger.info(type(prediction[0][0]))
        #prediction_df = pd.Dataframe(prediction[0])
        #prediction_df.to_csv('file1.csv')
        app.logger.info("logger hippy4.8")
        app.logger.info(y_test)
        app.logger.info(type(y_test))
        app.logger.info(y_test.shape)
        app.logger.info(y_test[0][0])
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
        app.logger.info("logger hippy4.9")
        app.logger.info(output)
        #output = prediction[0]
        #x = postedData["x"]
        #y = postedData["y"]
        #x = int(x)
        #y = int(y)
        #Step 2: Add the posted data
        #ret = 400
        #retMap = {
        #    "Message" : ret,
        #    "Status Code" : 200
        #}
        #output2 = {
        #    'Prediction': 1,
        #    'Status Code': 200
        #}
        #app.logger.info(output2)
        #app.logger.info(type(retMap))
        #app.logger.info(type(output2))
        #app.logger.info("logger hippy5.0")


        app.logger.info("logger hippy5")
        app.logger.info(type(output))
        app.logger.info("logger hippy5.1")
        #output1 = json.loads(output)
        #output1 = json.loads(output2)
        #app.logger.info("logger hippy5.2")
    #    app.logger.info(type(output1))
    #    app.logger.info(output1)
        #app.logger.info("logger hippy5.5")
        output_string=str(output)
        app.logger.info("logger hippy5.6")
        app.logger.info(output_string)
        app.logger.info(type(output_string))

        #output_encode=str(output).encode('utf-8')

        #app.logger.info("logger hippy5.7")
        #app.logger.info(output_encode)
        #app.logger.info(type(output_encode))

        app.logger.info("logger hippy5.8")
        app.logger.info(jsonify(output_string))
        app.logger.info(type(jsonify(output_string)))

        #app.logger.info("logger hippy5.9")
        #app.logger.info(jsonify(output_encode))
        #app.logger.info("logger hippy5.10")

    K.clear_session()
    return (jsonify(output_string))


@app.route('/add',methods=['POST'])
def add():
    #If I am here, then the resouce Add was requested using the method POST
    #Step 1: Get posted data:

    app.logger.info("logger hippy1")
    postedData = request.get_json()
    x = postedData["x"]
    y = postedData["y"]
    x = int(x)
    y = int(y)
    #Step 2: Add the posted data
    ret = x+y

    app.logger.info("logger hippy2")
    retMap = {
        "Message" : ret,
        "Status Code" : 200
    }
    return jsonify(retMap)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5205)
