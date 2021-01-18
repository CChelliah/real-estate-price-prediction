import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn import metrics
from scipy import stats
import pickle

def removeOutliers(percentage, df, target):
    q = df[target].quantile(percentage)
    df = df[(df[target] < q)]
    return df 

def cleanData(): 
    df = pd.read_csv('../data/SydneyHousePrices.csv')
    suburb_df = pd.read_csv('../data/sydney_suburbs.csv')
    temp_date = pd.to_datetime(df['Date'])
    df['Year'] = temp_date.dt.year
    df['Months'] = temp_date.dt.month
    df = pd.merge(df,suburb_df, on='suburb')
    df = df.drop(['Date', 'suburb', 'Id'], axis=1)
    le = LabelEncoder() 
    df['propType'] = le.fit_transform(df['propType'])
    knn = KNNImputer(n_neighbors=10)
    df["bed"]=knn.fit_transform(df[["bed"]])
    df["car"]=knn.fit_transform(df[["car"]])
    df = removeOutliers(0.975, df, "bed")
    df = removeOutliers(0.975, df, "bath")
    df = removeOutliers(0.975, df, "car")
    df = removeOutliers(0.975, df, "sellPrice")
    df = pd.get_dummies(df, columns=['postalCode', 'propType'])
    return df

def mlModel(df):
    y = df["sellPrice"]
    X = df.drop("sellPrice", axis=1)
    params = {
        'n_estimators': [500],
        'max_depth': [None, 2, 4, 6, 8, 10],
        'n_jobs': [10]
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
    return model 

def main(): 
    data = cleanData()
    model = mlModel(data)
    pickle.dump(model, open('model.pkl', 'wb'))

if __name__ == "__main__":
    main()
   
