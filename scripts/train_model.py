from __future__ import annotations

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, train_test_split
from matplotlib import pyplot as plt
from datetime import datetime

from sklearn.impute import SimpleImputer
import pickle
import optuna
import re
from sklearn.ensemble import IsolationForest


class ModelTrainer:
    def __init__(self, results_path: str | None = None, data_file_paths: list[str] | None = None):
        self.results_path = results_path or "data\\results"
        self.data_file_paths = data_file_paths or ["data\\raw\\sales_ads_train.csv"]
        self.target_variable = 'Cena'

    def train(self):
        data = pd.concat([pd.read_csv(file) for file in self.data_file_paths], ignore_index=True)
        X, y = self.prepare_data(data, drop_rows=False)
        print(set(X.columns))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train, y_train = self.remove_outliers_isolation_forest(X_train, y_train, contamination=0.05)

        self.train_lightgbm_regressor(X_train, y_train, X_test, y_test)

    def remove_outliers_isolation_forest(self, X: pd.DataFrame, y: pd.Series, contamination: float = 0.01) -> tuple[pd.DataFrame, pd.Series]:
        iso = IsolationForest(contamination=contamination, random_state=42)
        iso.fit(X.select_dtypes(include=[np.number]))
        preds = iso.predict(X.select_dtypes(include=[np.number]))
        mask = preds != -1
        return X[mask], y[mask]

    def prepare_data(self, data: pd.DataFrame, drop_rows: bool = True) -> tuple[pd.DataFrame, pd.Series | None]:
        if drop_rows:
            data = data.dropna()

        for col in ["Data_pierwszej_rejestracji", "Data_publikacji_oferty"]:
            data[col] = pd.to_datetime(data[col], errors='coerce', format='%d/%m/%Y')
        data["Wiek_samochodu_lata"] = (data["Data_publikacji_oferty"] - data["Data_pierwszej_rejestracji"]).dt.days // 365

        conversion_rates = {'EUR': 4.18, 'PLN': 1.0}
        if 'Cena' in data.columns:
            data['Cena'] = data.apply(lambda row: row['Cena'] * conversion_rates.get(row['Waluta'], 1), axis=1)

        data = data.drop(columns=["Data_pierwszej_rejestracji", "Data_publikacji_oferty"], errors='ignore')

        if self.target_variable in data.columns:
            y = np.log1p(data[self.target_variable])
            X = data.drop(columns=[self.target_variable, "ID"], errors='ignore')
        else:
            X = data
            y = None

        data["mileage_per_year"] = data["Przebieg_km"] / (data["Wiek_samochodu_lata"] + 1)
        data["power_weight_ratio"] = data["Moc_KM"] / (data["Pojemnosc_cm3"] + 1)

        # change so that 4x4 is consistent
        X["Napęd"] = X["Napęd"].str.replace("4x4", "4WD")

        # Ensure categorical columns are properly typed
        categorical_cols = X.select_dtypes(include='object').columns.tolist()
        for col in categorical_cols:
            X[col] = X[col].astype('category')
        print(categorical_cols)
        return X, y

    def train_lightgbm_regressor(self, X_train, y_train, X_test, y_test):
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.9, random_state=42)
        train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature='auto')
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature='auto')

        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'verbosity': -1,
                'seed': 42,
                'feature_pre_filter' : False,
                'n_estimators': trial.suggest_int('n_estimators', 5, 100),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'max_cat_threshold': trial.suggest_int('max_cat_threshold', 10, 100),
                'min_data_per_group': trial.suggest_int('min_data_per_group', 5, 100),
                'cat_l2': trial.suggest_float('cat_l2', 1.0, 100.0),
                'cat_smooth': trial.suggest_float('cat_smooth', 1.0, 100.0),
                # "max_bin": trial.suggest_int("max_bin", 255, 1023),
                # "min_data_in_bin": trial.suggest_int("min_data_in_bin", 1, 100),
            }

            cv_results = lgb.cv(params, train_data, nfold=5, num_boost_round=1000, seed=42, stratified=False)
            return cv_results['valid rmse-mean'][-1]

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=2)
        print("Best Parameters:", study.best_params)

        best_params = study.best_params
        best_params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'seed': 42
        })

        final_model = lgb.train(best_params, train_data, num_boost_round=1000, valid_sets=[valid_data], callbacks=[lgb.early_stopping(stopping_rounds=50)])
        print("Best iteration:", final_model.best_iteration)

        test_cv_scores = cross_val_score(lgb.LGBMRegressor(**best_params), X_test, y_test, cv=5, scoring="neg_root_mean_squared_error")
        print("%0.2f RMSE with a standard deviation of %0.2f test" % (test_cv_scores.mean(), test_cv_scores.std()))

        output_folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(f"{self.results_path}/{output_folder_name}", exist_ok=True)
        study.trials_dataframe().to_csv(f"{self.results_path}\\{output_folder_name}\\optuna_trials.csv", index=False)


        with open(f"{self.results_path}/{output_folder_name}/lgbm_model.pkl", 'wb') as f:
            pickle.dump(final_model, f)

        lgb.plot_importance(final_model, importance_type='gain', max_num_features=30, figsize=(10, 10))
        plt.savefig(f"{self.results_path}/{output_folder_name}/feature_importance_gain.png")

        lgb.plot_importance(final_model, importance_type='split', max_num_features=30, figsize=(10, 10))
        plt.savefig(f"{self.results_path}/{output_folder_name}/feature_importance_split.png")

        # prediction on test data
        X_test_raw = pd.read_csv("../data/raw/sales_ads_test.csv")
        currencies = X_test_raw["Waluta"].copy()
        X_test_prepared, _ = self.prepare_data(X_test_raw, drop_rows=False)
        print(set(X_test_prepared.columns))
        test_pred = final_model.predict(X_test_prepared.drop(columns=["ID"]))
        test_pred = np.expm1(test_pred)  # reverse log transform
        mask_eur = currencies == 'EUR'
        test_pred[mask_eur] = test_pred[mask_eur] / 4.18

        test_prediction_df = pd.DataFrame({"ID": X_test_prepared.index + 1, "Cena": test_pred})
        print(test_prediction_df.head())
        test_prediction_df.to_csv(f"{self.results_path}\\{output_folder_name}\\kaggle_upload_prediction.csv",
                                  index=False)


if __name__ == "__main__":
    model_trainer = ModelTrainer('../data/results/', ["../data/raw/sales_ads_train.csv"])
    model_trainer.train()
