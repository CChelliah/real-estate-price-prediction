import numpy as np
from flask import Flask, request, jsonify
import pickle
import os
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn import linear_model
from sklearn import neighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectPercentile, chi2, SelectFromModel
from sklearn.impute import KNNImputer
from scipy import stats
import seaborn as sns
import pickle
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)

def createPrediction(json): 
    test = {
        "postalCode": "86",
        "bed": "1",
        "bath": "2",
        "car": "1", 
        "propType": "1", 
        "Year": "2019", 
        "Months": "10",
        "latitude": "-33.643809", 
        "longitude": "151.315858"
    }

    temp = pd.DataFrame([test])
    pred = model.predict(temp)
    return str(pred[0])

@app.route('/', methods=['POST'])
def home():
    json = request.json
    prediction = createPrediction(json)
    print("Prediction")
    print(prediction)
    return jsonify({'prediction' : prediction })

if __name__ == "__main__":
    f = open('../models/model.pkl', 'rb')
    model = pickle.load(f)
    app.run(host='localhost', port=4003, debug=True)


'''
  test = {
        "postalCode": "86",
        "bed": "1",
        "bath": "2",
        "car": "1", 
        "propType": "1", 
        "Year": "2019", 
        "Months": "10",
        "latitude": "-33.643809", 
        "longitude": "151.315858"
    }

    temp = pd.DataFrame([test])
    print(model.predict(temp))
    '''

