from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import RobustScaler


def load_data():
    home = Path.home()
    red_raw_path = home / "Desktop" / "56870" / "code" / "data" / "raw" / "red_wine.csv"
    white_raw_path = home / "Desktop" / "56870" / "code" / "data" / "raw" / "white_wine.csv"

    red_wine = pd.read_csv(red_raw_path)
    white_wine = pd.read_csv(white_raw_path)
    return red_wine, white_wine


def parse_and_clean(data):
    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].astype("str")
        elif data[col].dtype == "int":
            data[col] = data[col].astype("int32")
        else:
            data[col] = data[col].astype("float64")

    data.replace([None, "", "NaN"], pd.NA, inplace=True)
    return data


def eda(data, wine_type, fig_size=(10, 6)):
    print(data.describe())

    for col in data.columns:
        plt.figure(figsize=fig_size)
        sns.histplot(data[col], kde=True)
        plt.title(f'Histogram of {col}')
        col_name = col.replace(" ", "_")

        fig_save_path = Path.home() / "Desktop" / "56870" / "code" / "data" / "out" / "graphs" / "eda" / f"{wine_type}_{col_name}_histogram.png"
        fig_save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_save_path)
        plt.close()

def correlation(data, wine_type, heatmap_size=(10, 8)):
    corr = data.corr()
    plt.figure(figsize=heatmap_size)
    sns.heatmap(corr, annot=True)
    plt.title(f'Correlation Heatmap of {wine_type} Wine')

    fig_save_path = Path.home() / "Desktop" / "56870" / "code" / "data" / "out" / "graphs" / f"{wine_type}_correlation.png"
    fig_save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_save_path)


def handle_outliers(data):
    for col in data.select_dtypes(include=["float64", "int32"]):
        data[col] = winsorize(data[col], limits=[0.05, 0.05])
    return data


def normalize_data(data):
    scaler = RobustScaler()
    scaled_data = pd.DataFrame(
        scaler.fit_transform(data), columns=data.columns)
    return scaled_data


def merge_datasets(red_wine, white_wine):
    red_wine["type"] = "red"
    white_wine["type"] = "white"
    merged_wine = pd.concat([red_wine, white_wine], ignore_index=True)
    return merged_wine


def main():
    home = Path.home()
    script_dir = home / "Desktop" / "56870" / "code" / "src"
    Path.cwd().joinpath(script_dir).resolve()

    red_wine, white_wine = load_data()

    red_wine = parse_and_clean(red_wine)
    white_wine = parse_and_clean(white_wine)

    print("Red Wine Data Analysis:")
    eda(red_wine, "red")
    print("\nWhite Wine Data Analysis:")
    eda(white_wine, "white")

    print("\nRed Wine Correlation Heatmap:")
    correlation(red_wine, "red")
    print("\nWhite Wine Correlation Heatmap:")
    correlation(white_wine, "white")

    red_wine = handle_outliers(red_wine)
    white_wine = handle_outliers(white_wine)

    red_wine = normalize_data(red_wine)
    white_wine = normalize_data(white_wine)

    merged_wine = merge_datasets(red_wine, white_wine)

    red_export_path = home / "Desktop" / "56870" / "code" / "data" / "parsed" / "red_wine_processed.csv"
    white_export_path = home / "Desktop" / "56870" / "code" / "data" / "parsed" / "white_wine_processed.csv"
    merged_export_path = home / "Desktop" / "56870" / "code" / "data" / "parsed" / "merged_wine_processed.csv"

    red_wine.to_csv(red_export_path, index=False)
    white_wine.to_csv(white_export_path, index=False)
    merged_wine.to_csv(merged_export_path, index=False)

    print("Datasets exported successfully.")


if __name__ == "__main__":
    main()
