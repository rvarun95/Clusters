import pandas as pd
import numpy as np
import tensorflow as tf

#from tensorflow.python.client import device_lib
#from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.preprocessing import OneHotEncoder, StandardScaler
#from sklearn.impute import SimpleImputer
#from sklearn.pipeline import FeatureUnion, Pipeline
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose, DecomposeResult
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics import recall_score, precision_score, make_scorer
import matplotlib.lines as mlines
import seaborn as sns; sns.set()
from sklearn.preprocessing import normalize, StandardScaler
from sklearn import linear_model
from imblearn.over_sampling import SMOTE
from scipy import ndimage, fft

from keras.models import Sequential, Model
from scipy.ndimage.filters import uniform_filter1d
from keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten
from keras.layers import BatchNormalization, Input, concatenate, Activation
from keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from keras import backend as K

from keras.models import model_from_json
from keras.models import model_from_yaml
import pickle
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import ParameterGrid
from keras import backend as K



#%matplotlib inline
#%load_ext autoreload
#%autoreload 2


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

def batch_generator(x_train, y_train, batch_size=32):
    """
    Gives equal number of positive and negative samples, and rotates them randomly in time
    """
    #print(x_train.shape, y_train.shape, "one")
    half_batch = batch_size // 2
    x_batch = np.empty((batch_size, x_train.shape[1], x_train.shape[2]), dtype='float32')
    y_batch = np.empty((batch_size, y_train.shape[1]), dtype='float32')
    #print(x_batch.shape, y_batch.shape, "two")
    yes_idx = np.where(y_train[:,0] == 1.)[0]
    non_idx = np.where(y_train[:,0] == 0.)[0]

    while True:
        np.random.shuffle(yes_idx)
        np.random.shuffle(non_idx)

        x_batch[:half_batch] = x_train[yes_idx[:half_batch]]
        x_batch[half_batch:] = x_train[non_idx[half_batch:batch_size]]
        y_batch[:half_batch] = y_train[yes_idx[:half_batch]]
        y_batch[half_batch:] = y_train[non_idx[half_batch:batch_size]]

        for i in range(batch_size):
            sz = np.random.randint(x_batch.shape[1])
            x_batch[i] = np.roll(x_batch[i], sz, axis = 0)
        #print(x_batch.shape, y_batch.shape, "three")
        yield x_batch, y_batch



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


INPUT_LIB = '../web/'
raw_data = np.loadtxt(INPUT_LIB + 'exoTrain.csv', skiprows=1, delimiter=',')
x_train = raw_data[:, 1:]
y_train = raw_data[:, 0, np.newaxis] - 1.
raw_data = np.loadtxt(INPUT_LIB + 'exoTest.csv', skiprows=1, delimiter=',')
x_test = raw_data[:, 1:]
y_test = raw_data[:, 0, np.newaxis] - 1.
del raw_data

print(x_train.shape, x_test.shape)
x_train = ((x_train - np.mean(x_train, axis=1).reshape(-1,1)) / np.std(x_train, axis=1).reshape(-1,1))
x_test = ((x_test - np.mean(x_test, axis=1).reshape(-1,1)) / np.std(x_test, axis=1).reshape(-1,1))
print(x_train.shape, x_test.shape)

seed = 7
np.random.seed(seed)
sm = SMOTE(ratio = 1.0)
print(x_train.shape, y_train.shape)
x_train_sm, y_train_sm = sm.fit_sample(x_train, y_train)


print(len(x_train_sm))
print(x_train_sm.shape, y_train_sm.shape)
x_train_sm = np.stack([x_train_sm, uniform_filter1d(x_train_sm, axis=1, size=200)], axis=2)
x_test = np.stack([x_test, uniform_filter1d(x_test, axis=1, size=200)], axis=2)


def create_model(init_mode='glorot_uniform', activation='relu', dropout_rate=0.5, neurons =64,
                 optimizer='sgd', filters=8):

    #print(init_mode, activation, dropout_rate, neurons, optimizer, filters )
    seed = 7
    np.random.seed(seed)
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=11, kernel_initializer=init_mode,
                     activation=activation, input_shape=x_train_sm.shape[1:]))
    model.add(MaxPool1D(strides=4))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=(2 * filters), kernel_size=11, kernel_initializer=init_mode, activation=activation))
    model.add(MaxPool1D(strides=4))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=(4 * filters), kernel_size=11, kernel_initializer=init_mode, activation=activation))
    model.add(MaxPool1D(strides=4))
    model.add(BatchNormalization())
    model.add(Conv1D(filters=(8 * filters), kernel_size=11, kernel_initializer=init_mode, activation=activation))
    model.add(MaxPool1D(strides=4))
    model.add(Flatten())
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=neurons, activation=activation, kernel_initializer=init_mode))
    model.add(Dropout(dropout_rate/2))
    model.add(Dense(units=neurons, activation=activation, kernel_initializer=init_mode))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=init_mode))
    #model.compile(optimizer="sgd", loss = 'binary_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=optimizer, loss = 'binary_crossentropy', metrics=['accuracy', f1_m,precision_m, recall_m])
    model.compile(optimizer=optimizer, loss = 'binary_crossentropy')
    return model


#batch_size = 128
epochs = 30

K.clear_session()
seed = 7
np.random.seed(seed)
#nb_epoch=3
#with tf.device('/cpu:0'):
if True:
    seed = 7
    np.random.seed(seed)
    model_CV = KerasClassifier(build_fn=create_model, epochs = epochs, verbose=1)

    # define the grid search parameters
    #init_mode = ['glorot_normal', 'glorot_uniform']
    #activation = ['relu', 'sigmoid']
    #weight_constraint = [1, 2, 3, 4, 5]
    #optimizer = ['Adam', 'RMSprop'] #, 'sgd', 'Nadam']
    #epochs = [10, 20]

    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    dropout_rate = [0.2, 0.4, 0.6]
    neurons = [32, 64, 128]
    batch_size = [32, 64]
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    filters = [8, 16]

    '''
    init_mode = ['glorot_normal']
    activation = ['sigmoid']
    dropout_rate = [0.6]
    neurons = [32]
    batch_size = [32]
    optimizer = ['Adam']
    filters = [16]
    '''

    #f1 = {'f1' : f1_m}
    f1 = make_scorer(f1_score)
    f1_scorer = make_scorer(f1_m)
    scoring = {'f1' : f1_m}

    param_grid = dict( init_mode = init_mode, activation = activation, dropout_rate = dropout_rate,
                          neurons = neurons, batch_size = batch_size, optimizer = optimizer, filters = filters)

    #grid = GridSearchCV(estimator=model_CV, param_grid=param_grid, scoring = scoring, cv=3, n_iter=2,
    #                    pre_dispatch=3, n_jobs=3)

    grid = RandomizedSearchCV(estimator=model_CV, param_distributions=param_grid, scoring = f1, cv=2,
                              n_iter=300, verbose=1) #, random_state = np.random.seed(seed)) #, pre_dispatch=3, n_jobs=3)
    grid_result = grid.fit(x_train_sm, y_train_sm)
    print(f'Best Accuracy for {grid_result.best_score_} using {grid_result.best_params_}')
    result_df = pd.DataFrame(grid_result.cv_results_)
    print(result_df)

    result_df.to_csv('RandomizedSearchCV_result_df.csv', index=False, encoding='utf-8')
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    #means_train = grid_result.cv_results_['mean_train_score']
    #stds_train = grid_result.cv_results_['std_train_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f' mean={mean:.4}, std={stdev:.4} using {param}')



randon_search_conv_model = grid_result.best_estimator_
seed = 7
np.random.seed(seed)
x_train_input = np.stack([x_train, uniform_filter1d(x_train, axis=1, size=200)], axis=2)

y_train_hat = randon_search_conv_model.predict(x_train_input)[:,0]
y_train_pred = np.where(y_train_hat > 0.9,1.,0.)
print('Accuracy:', accuracy_score(y_train, y_train_pred))
print('F1 score:', f1_score(y_train, y_train_pred))
print('Recall:', recall_score(y_train, y_train_pred))
print('Precision:', precision_score(y_train, y_train_pred))
print('Cohens Kappa :', cohen_kappa_score(y_train, y_train_pred))
print('ROC AUC Score:', roc_auc_score(y_train, y_train_pred))
print('\n clasification report:\n', classification_report(y_train, y_train_pred))
print('\n confussion matrix:\n',confusion_matrix(y_train, y_train_pred))

y_hat = randon_search_conv_model.predict(x_test)[:,0]
y_pred = np.where(y_hat > 0.9,1.,0.)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('F1 score:', f1_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Cohens Kappa :', cohen_kappa_score(y_test, y_pred))
print('ROC AUC Score:', roc_auc_score(y_test, y_pred))
print('\n clasification report:\n', classification_report(y_test,y_pred))
print('\n confussion matrix:\n',confusion_matrix(y_test, y_pred))


final_model_json = randon_search_conv_model.model.to_json()
with open("final_model_rs_conv.json", "w") as json_file:
    json_file.write(final_model_json)


# Save model 1 - serialize model to YAML
model_yaml = randon_search_conv_model.model.to_yaml()
with open("final_model_rs_conv.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)

# Serialize weights to HDF5
randon_search_conv_model.model.save_weights("final_model_rs_conv_weights.h5")

pickle.dump(randon_search_conv_model.model, open('final_model_rs_conv.pkl', 'wb'))

randon_search_conv_model.model.save("final_model_rs_conv.h5")

print("Saved model to disk")

seed = 7
np.random.seed(seed)
model = tf.keras.models.load_model('final_model_rs_conv.h5',
        custom_objects={
            #'recall_m' : recall_m,
            #'precision_m' : precision_m,
            #'f1_m' : f1_m
            'f1' : f1
            })

seed = 7
np.random.seed(seed)
y_hat = model.predict(x_test)[:,0]
y_pred = np.where(y_hat > 0.9,1.,0.)
print('Accuracy:', accuracy_score(y_test, y_pred))
print('F1 score:', f1_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Cohens Kappa :', cohen_kappa_score(y_test, y_pred))
print('ROC AUC Score:', roc_auc_score(y_test, y_pred))
print('\n clasification report:\n', classification_report(y_test,y_pred))
print('\n confussion matrix:\n',confusion_matrix(y_test, y_pred))
