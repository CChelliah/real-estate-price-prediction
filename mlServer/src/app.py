import numpy as np
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
import json
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app)

maps = {
    "house": 0,
    "duplex/semi-detached": 1,
    "terrace": 2,
    "townhouse": 3,
    "villa": 4
}

def mapPostCodes(): 
    df = pd.read_csv('../data/SydneyHousePrices.csv')
    suburb_df = pd.read_csv('../data/sydney_suburbs.csv')
    df = pd.merge(df,suburb_df, on='suburb')
    df = df.drop(['Date', 'suburb', 'Id', 'bed', 'bath', 'car', 'propType', 'sellPrice'], axis=1) 
    df = df.drop_duplicates('postalCode', keep='last')
    postCodes = df.set_index('postalCode').T.to_dict('dict')
    return postCodes

def createRequest(req): 
    req['latitude'] = postCodes[int(req['postalCode'])]['latitude']
    req['longitude'] = postCodes[int(req['postalCode'])]['longitude']
    for postcode in postCodes: 
        req["postalCode_" + str(postcode)] = 0
    req["postalCode_" + str(req['postalCode'])] = 1
    for i in range(0, 5):
        req["propType_" + str(i)] = 0
    propType = req['propType']
    req["propType_" + str(maps[propType])] = 1
    req['Years'] = 2019
    req['Months'] = 1
    del req['propType']
    del req['postalCode']
    del req['prediction']
    return req

def createPrediction(req):
    pred = model.predict(pd.DataFrame([req]))
    return str(round(pred[0]))

@app.route('/', methods=['POST'])
def home():
    json = request.json
    req = createRequest(json)
    prediction = createPrediction(req)
    return jsonify({'prediction' : prediction })

if __name__ == "__main__":
    f = open('../models/model.pkl', 'rb')
    model = pickle.load(f)
    postCodes = mapPostCodes()
    app.run(host='localhost', port=4005, debug=True)