import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from joblib import dump


def load_data(file_path):
    return pd.read_csv(file_path)


def save_model(model, filename):
    model_path = Path.home() / "Desktop" / "56870" / "code" / "models" / filename
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)


def main():
    data_path = Path.home() / "Desktop" / "56870" / "code" / "data" / \
        "parsed" / "merged_wine_processed.csv"
    data = load_data(data_path)

    data = pd.get_dummies(data, columns=['type'])

    X = data.drop('alcohol', axis=1)
    y = data['alcohol']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    print("Linear Regression MSE:", mean_squared_error(y_test, y_pred))
    save_model(lr, 'task1_linear_regression_alcohol.pk1')

    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    print("Random Forest Regressor MSE:",
          mean_squared_error(y_test, y_pred_rf))
    save_model(rf, 'task1_random_forest_alcohol.pk1')


if __name__ == "__main__":
    main()
