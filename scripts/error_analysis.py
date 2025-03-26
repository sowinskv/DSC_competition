import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.train_model import ModelTrainer
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_model(model, X_test, y_test, X_raw):
    # Make predictions
    y_pred = model.predict(X_test)

    # Convert target variable back from log scale
    y_test_actual = np.expm1(y_test)

    # Handle currency conversion
    currencies = X_test["Waluta"].copy()
    test_pred = np.expm1(y_pred)
    mask_eur = currencies == 'EUR'
    test_pred[mask_eur] = test_pred[mask_eur] / 4.18

    # Calculate residuals (errors)
    residuals = y_test_actual - test_pred
    percentage_error = (residuals / y_test_actual) * 100

    # Select important features to inspect
    selected_features = X_raw.columns
    print(X_test.columns)
    print(X_raw.columns)
    features_df = X_raw[selected_features].reset_index(drop=True)

    # Create a DataFrame to store results
    results_df = pd.DataFrame({
        'Actual': y_test_actual,
        'Predicted': test_pred,
        'Residual': residuals,
        'Percentage Error': percentage_error,
    })
    #RMSE
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
    print(f"RMSE: {rmse}")

    # plot distribution of price predicted and actual
    plt.figure(figsize=(12, 6))
    sns.histplot(results_df["Actual"], bins=50, kde=True, color='blue', label='Actual')
    sns.histplot(results_df["Predicted"], bins=50, kde=True, color='red', label='Predicted', alpha=0.5)
    plt.title("Distribution of Prices")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("data\\errors\\price_distribution_predicted_actual.png")

    # Combine predictions and feature information
    results_df = pd.concat([results_df, features_df], axis=1)

    # Print summary statistics
    print(results_df[['Actual', 'Predicted', 'Residual']].describe())

    # Save results for further inspection
    results_df.to_csv("error_analysis.csv", index=False)

    return results_df

def inspect_worst_predictions(results_df, quantile=0.95):
    # Find the threshold for the largest errors
    error_threshold = results_df['Percentage Error'].abs().quantile(quantile)

    # Select rows where the error exceeds the threshold
    worst_predictions = results_df[results_df['Percentage Error'].abs() >= error_threshold]

    print(worst_predictions.head(10))
    worst_predictions.to_csv("data\\errors\\worst_predictions.csv", index=False)

    return worst_predictions


def visualize_errors(results_df):
    plt.figure(figsize=(12, 6))
    sns.histplot(results_df["Percentage Error"], bins=50, kde=True)
    plt.title("Distribution of Prediction Errors (%)")
    plt.xlabel("Percentage Error")
    plt.ylabel("Frequency")
    #save figure
    plt.savefig("data\\errors\\error_distribution.png")

    plt.figure(figsize=(12, 6))
    plt.scatter(results_df['Actual'], results_df['Residual'], alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residuals vs Actual")
    plt.xlabel("Actual Price")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.savefig("data\\errors\\residuals_vs_actual.png")


def analyze_errors_by_category(results_df, X_test, category_column):
    # Merge test features with errors for detailed analysis
    merged_df = X_test.copy()
    merged_df['Percentage Error'] = results_df['Percentage Error']

    # Analyze mean error by category
    print(results_df)
    category_analysis = results_df.groupby(category_column)['Percentage Error'].mean().sort_values()
    print("cat analysis")
    print(category_analysis)

    # Plot the error by category
    plt.figure(figsize=(12, 6))
    category_analysis.plot(kind='bar')
    plt.title(f"Mean Percentage Error by {category_column}")
    plt.ylabel("Mean % Error")
    plt.savefig(f"data\\errors\\error_by_{category_column}.png")

def error_vs_features(results_df, X_test):
    merged_df = X_test.copy()
    merged_df['Percentage Error'] = results_df['Percentage Error']

    # Check correlation of errors with numeric features
    corr = merged_df.corr()['Percentage Error'].drop('Percentage Error').sort_values()
    print(corr)

    # Visualize error against a key numeric feature
    feature_to_plot = 'Wiek_samochodu_lata'
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=merged_df[feature_to_plot], y=merged_df['Percentage Error'])
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f"Percentage Error vs {feature_to_plot}")
    plt.savefig(f"data\\errors\\error_vs_{feature_to_plot}.png")

def main():
    model_trainer = ModelTrainer('../data/results/', ["data/raw/sales_ads_train.csv"], encoder=pickle.load(open("data/results/2025-03-24_16-07-56/encoder.pkl", "rb")))
    data = pd.concat([pd.read_csv(file) for file in model_trainer.data_file_paths], ignore_index=True)
    data_raw = data.copy()
    X_raw, y_raw = data_raw.drop(columns=['Cena']), data_raw['Cena']
    X, y = model_trainer.prepare_data(data, False, False)
    # y = np.log1p(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    loaded_model = pickle.load(open("data/results/2025-03-24_16-07-56/lgbm_model_2025-03-24_16-07-56.pkl", 'rb'))
    print(len(loaded_model.feature_name()), len(X_test.columns))
    print(set(loaded_model.feature_name()) - set(X_test.columns))
    print(set(X_test.columns) - set(loaded_model.feature_name()))
    results_df = evaluate_model(loaded_model, X, y, X_raw)

    # Inspect the worst predictions
    worst_predictions = inspect_worst_predictions(results_df)
    print(worst_predictions['Percentage Error'].describe())

    # Visualize error patterns
    visualize_errors(results_df)

    # Analyze errors by specific category (e.g., 'Naped')
    feature_importance = pd.DataFrame({
        'Feature': loaded_model.feature_name(),
        'Importance': loaded_model.feature_importance(importance_type='gain')
    })

    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    print(feature_importance)
    for c in feature_importance.Feature.tolist()[:10]:
        analyze_errors_by_category(results_df, X_test, c)

    # Correlate errors with numeric features
    error_vs_features(results_df, X_test)

if __name__ == "__main__":
    main()