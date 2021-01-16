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
import json

pd.options.display.max_columns = None

df = pd.read_csv('../data/SydneyHousePrices.csv')
temp_date = pd.to_datetime(df['Date'])
df['Year'] = temp_date.dt.year
df['Months'] = temp_date.dt.month
labelencoder= LabelEncoder()
df['propType'] = labelencoder.fit_transform(df['propType'])
df['postalCode'] = labelencoder.fit_transform(df['postalCode'])
suburb_df = pd.read_csv('../data/sydney_suburbs.csv')
df = pd.merge(df,suburb_df, on='suburb')
df = df.drop(['Date', 'suburb', 'Id'], axis=1)
knn_imputer=KNNImputer()
df["bed"]=knn_imputer.fit_transform(df[["bed"]])
df["car"]=knn_imputer.fit_transform(df[["car"]])
#df['car'] = df['car'].fillna(df['car'].mean())
#df['bed'] = df['bed'].fillna(df['bed'].mean())
scaler = MinMaxScaler()
scaler.fit(df)
df = df[(np.abs(stats.zscore(df)) < 2).all(axis=1)]
print(df)
print(df.describe())
y = df[["sellPrice"]]
X = df.drop("sellPrice", axis=1)
params = {
    'n_estimators': [10]
}

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, test_size=0.20)
rfr = RandomForestRegressor()
model = GridSearchCV(rfr, params, cv=5)
model.fit(X_train, y_train.values.ravel())
y_pred = model.predict(X_test)
print(df.describe())
print("Model R^2 Score: ", model.score(X_train, y_train))
print("Model Test R^2 Score: ", model.score(X_test, y_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("Cross Validation Score: ", cross_val_score(model, X_test, y_test.values.ravel(), cv=5))

print('/n')

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
print("11111111")
print(df.iloc[:3])
print("")
print(model.predict(temp))
print("")
pickle.dump(model, open('model.pkl', 'wb'))
