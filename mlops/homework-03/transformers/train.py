from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import pandas as pd

@transformer
def vectorize(df):

    #Setting up the train_dicts
    train_dicts = df[['PULocationID','DOLocationID']].to_dict(orient='records')

    #Vectorizing using the dict vectorizer
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)

    target = 'duration'
    y_train = df[target].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    print(lr.intercept_)
    return dv,lr