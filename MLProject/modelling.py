import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

X_train = pd.read_csv('online_retail_preprocessing/X_train.csv')
X_test = pd.read_csv('online_retail_preprocessing/X_test.csv')
y_train = pd.read_csv('online_retail_preprocessing/y_train.csv').squeeze()
y_test = pd.read_csv('online_retail_preprocessing/y_test.csv').squeeze()

mlflow.set_experiment("Online Retail Churn Prediction - CI")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_leaf': [1, 5, 10]
}

with mlflow.start_run(run_name="ci_retrain"):
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_test_pred))
    mlflow.log_metric("test_f1", f1_score(y_test, y_test_pred))
    mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_test_proba))

    mlflow.sklearn.log_model(best_model, "model")

    print(f"Best params: {grid_search.best_params_}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
