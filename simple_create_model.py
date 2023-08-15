from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from mlflow.models.signature import infer_signature
import mlflow

import numpy as np
import pandas as pd
import sys
import warnings


# Evaluate metrics
def eval_metrics(actual, pred):
    rmse_metric = np.sqrt(mean_squared_error(actual, pred))
    mae_metric = mean_absolute_error(actual, pred)
    r2_metric = r2_score(actual, pred)
    return rmse_metric, mae_metric, r2_metric


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    cancer = load_breast_cancer()

    training_df = pd.DataFrame(cancer.data, columns=[cancer.feature_names])
    training_df['Target'] = pd.Series(data=cancer.target, index=training_df.index)

    # Separating the features and labels
    labels = training_df['Target']
    features = training_df.sort_index(axis=1).drop(
        labels=['Target'],
        axis=1)

    # Splitting the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        labels,
                                                        stratify=labels)

    # Creating and training LogisticRegression
    model = LogisticRegression(max_iter=100)
    model.fit(X=X_train[sorted(X_train)], y=y_train.values.ravel())

    predicted_qualities = model.predict(X_test)
    (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

    # Infer model signature
    predictions = model.predict(X_train)
    signature = infer_signature(X_train, predictions)

    # mlflow.set_experiment('Breast cancer')
    # with mlflow.start_run(run_name="breast_simple_train") as run:
    #
    #     mlflow.log_metric("rmse", rmse)
    #     mlflow.log_metric("r2", r2)
    #     mlflow.log_metric("mae", mae)
    #     mlflow.sklearn.log_model(model, "model", signature=signature)
    #
    #     # Register model in MLFlow
    #     mlflow.register_model(
    #         "runs:/{}/model".format(mlflow.active_run().info.run_id),
    #         "breast_cancer"
    #     )
    #
    #     stdout = sys.stdout
