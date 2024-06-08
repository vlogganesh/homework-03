import mlflow
import pickle

@data_exporter
def logging(params):
    dv=params[0]
    lr=params[1]

    # Start an MLflow run
    mlflow.set_tracking_uri("http://mlflow:5000")
    with mlflow.start_run():

        # Log the model
        mlflow.sklearn.log_model(lr, "linear_regression_model")

        # Save and log the DictVectorizer as an artifact
        with open("dict_vectorizer.pkl", "wb") as f:
            pickle.dump(dv, f)
        mlflow.log_artifact("dict_vectorizer.pkl")
