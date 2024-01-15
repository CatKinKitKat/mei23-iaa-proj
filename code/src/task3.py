from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR


def load_data(file_path):
    return pd.read_csv(file_path)


def save_model(model, filename):
    model_path = Path.home() / "Desktop" / "56870" / "code" / "models" / filename
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)


def main():
    data_path = Path.home() / "Desktop" / "56870" / "code" / "data" / "parsed" / "merged_wine_processed.csv"
    data = load_data(data_path)

    data = pd.get_dummies(data, columns=['type'])

    x = data.drop('quality', axis=1)
    y = data['quality']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pred_lr = lr.predict(x_test)
    print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_lr))
    print("Linear Regression R2 Score:", r2_score(y_test, y_pred_lr))
    print("Linear Regression MAE:", mean_absolute_error(y_test, y_pred_lr))
    save_model(lr, 'task3_linear_regression_quality.pk1')

    gbr = GradientBoostingRegressor()
    gbr.fit(x_train, y_train)
    y_pred_gbr = gbr.predict(x_test)
    print("Gradient Boosting Regressor MSE:", mean_squared_error(y_test, y_pred_gbr))
    print("Gradient Boosting Regressor R2 Score:", r2_score(y_test, y_pred_gbr))
    print("Gradient Boosting Regressor MAE:", mean_absolute_error(y_test, y_pred_gbr))
    save_model(gbr, 'task3_gradient_boosting_quality.pk1')

    knn = KNeighborsRegressor()
    knn.fit(x_train, y_train)
    y_pred_knn = knn.predict(x_test)
    print("KNN Regressor MSE:", mean_squared_error(y_test, y_pred_knn))
    print("KNN Regressor R2 Score:", r2_score(y_test, y_pred_knn))
    print("KNN Regressor MAE:", mean_absolute_error(y_test, y_pred_knn))
    save_model(knn, 'task3_knn_quality.pk1')

    svr = SVR()
    svr.fit(x_train, y_train)
    y_pred_svr = svr.predict(x_test)
    print("SVR MSE:", mean_squared_error(y_test, y_pred_svr))
    print("SVR R2 Score:", r2_score(y_test, y_pred_svr))
    print("SVR MAE:", mean_absolute_error(y_test, y_pred_svr))
    save_model(svr, 'task3_svr_quality.pk1')

    rf = RandomForestRegressor()
    rf.fit(x_train, y_train)
    y_pred_rf = rf.predict(x_test)
    print("Random Forest Regressor MSE:", mean_squared_error(y_test, y_pred_rf))
    print("Random Forest Regressor R2 Score:", r2_score(y_test, y_pred_rf))
    print("Random Forest Regressor MAE:", mean_absolute_error(y_test, y_pred_rf))
    save_model(rf, 'task3_random_forest_quality.pk1')


if __name__ == "__main__":
    main()
