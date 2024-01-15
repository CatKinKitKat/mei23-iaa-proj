from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split


def load_data(file_path):
    return pd.read_csv(file_path)


def save_model(model, filename):
    model_path = Path.home() / "Desktop" / "56870" / "code" / "models" / filename
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)


def feature_importance_plot(model, x):
    feature_importance = pd.Series(
        model.feature_importances_, index=x.columns).sort_values(ascending=False)
    sns.barplot(x=feature_importance, y=feature_importance.index)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")

    fig_save_path = Path.home() / "Desktop" / "56870" / "code" / "data" / "out" / "graphs" / f"feature_importance.jpg"
    fig_save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_save_path)


def main():
    data_path = Path.home() / "Desktop" / "56870" / "code" / "data" / "parsed" / "merged_wine_processed.csv"
    data = load_data(data_path)

    data = pd.get_dummies(data, columns=['type'])

    x = data.drop('alcohol', axis=1)
    y = data['alcohol']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pred = lr.predict(x_test)
    print("Linear Regression MSE:", mean_squared_error(y_test, y_pred))
    print("Linear Regression R2 Score:", r2_score(y_test, y_pred))
    print("Linear Regression MAE:", mean_absolute_error(y_test, y_pred))
    save_model(lr, 'task1_linear_regression_alcohol.pk1')

    rf = RandomForestRegressor()
    rf.fit(x_train, y_train)
    y_pred_rf = rf.predict(x_test)
    print("Random Forest Regressor MSE:", mean_squared_error(y_test, y_pred_rf))
    print("Random Forest Regressor R2 Score:", r2_score(y_test, y_pred_rf))
    print("Random Forest Regressor MAE:", mean_absolute_error(y_test, y_pred_rf))
    save_model(rf, 'task1_random_forest_alcohol.pk1')

    feature_importance_plot(rf, x)


if __name__ == "__main__":
    main()
