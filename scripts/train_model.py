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
import category_encoders as ce


class ModelTrainer:
    def __init__(self, results_path: str | None = None, data_file_paths: list[str] | None = None):
        self.results_path = results_path or "data\\results"
        self.data_file_paths = data_file_paths or ["data\\raw\\sales_ads_train.csv"]
        self.target_variable = 'Cena'
        self.encoder = None

    def train(self):
        data = pd.concat([pd.read_csv(file) for file in self.data_file_paths], ignore_index=True)
        X, y = self.prepare_data(data, drop_rows=False, fit_encoder=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train, y_train = self.remove_outliers_isolation_forest(X_train, y_train, contamination=0.05)

        self.train_lightgbm_regressor(X_train, y_train, X_test, y_test)

    def remove_outliers_isolation_forest(self, X: pd.DataFrame, y: pd.Series, contamination: float = 0.01) -> tuple[pd.DataFrame, pd.Series]:
        iso = IsolationForest(contamination=contamination, random_state=42)
        iso.fit(X)
        preds = iso.predict(X)  # 1 for inliners, -1 for outliers
        mask = preds != -1
        X_clean = X[mask]
        y_clean = y[mask]
        return X_clean, y_clean

    def prepare_data(
        self,
        data: pd.DataFrame,
        drop_rows: bool = True,
        fit_encoder: bool = False
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        # cleaning, imputation:
        if drop_rows:
            data = data.dropna()
        else:
            missing_threshold = 0.05  # imputing only columns with more than 5% missing values
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
            numeric_to_impute = [col for col in numeric_cols if data[col].isnull().mean() > missing_threshold]
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            categorical_to_impute = [col for col in categorical_cols if data[col].isnull().mean() > missing_threshold]

            if numeric_to_impute:
                num_imputer = SimpleImputer(strategy='mean')
                data[numeric_to_impute] = num_imputer.fit_transform(data[numeric_to_impute])
            if categorical_to_impute:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                data[categorical_to_impute] = cat_imputer.fit_transform(data[categorical_to_impute])

        # date conversion
        for col in ["Data_pierwszej_rejestracji", "Data_publikacji_oferty"]:
            data[col] = pd.to_datetime(data[col], errors='coerce', format='%d/%m/%Y')
        data["Wiek_samochodu_lata"] = (data["Data_publikacji_oferty"] - data["Data_pierwszej_rejestracji"]).dt.days // 365

        # price conversion
        conversion_rates = {'EUR': 4.18, 'PLN': 1.0}
        if 'Cena' in data.columns:
            data['Cena'] = data.apply(
                lambda row: row['Cena'] * conversion_rates.get(row['Waluta'], 1),
                axis=1
            )

        # category encoding
        categorical_columns = [
            "Waluta", "Stan", "Marka_pojazdu", "Model_pojazdu", "Wersja_pojazdu",
            "Generacja_pojazdu", "Rodzaj_paliwa", "Naped", "Skrzynia_biegow",
            "Typ_nadwozia", "Kolor", "Kraj_pochodzenia", "Lokalizacja_oferty"
        ]

        # optional: feature extraction from 'Wyposazenie'
        if 'Wyposazenie' in data.columns:
            import ast
            from sklearn.preprocessing import MultiLabelBinarizer
            def parse_equipment(x):
                try:
                    return ast.literal_eval(x)
                except Exception:
                    return []
            data['wyposazenie_list'] = data['Wyposazenie'].apply(parse_equipment)
            mlb = MultiLabelBinarizer(sparse_output=True)
            equipment_matrix = mlb.fit_transform(data['wyposazenie_list'])
            equipment_df = pd.DataFrame.sparse.from_spmatrix(equipment_matrix, columns=mlb.classes_)
            common_features = equipment_df.columns[equipment_df.sum() > 50]  # opcjonalnie
            equipment_df = equipment_df[common_features]
            data = pd.concat([data.drop(columns=["Wyposazenie", "wyposazenie_list"]), equipment_df], axis=1)

        # mapping binary features
        data["Pierwszy_wlasciciel"] = data["Pierwszy_wlasciciel"].replace({"Yes": 1, "No": 0})

        if self.target_variable in data.columns:
            y = np.log1p(data[self.target_variable])  # log transform
        else:
            y = None

        # target encoding
        if fit_encoder:
            self.encoder = ce.TargetEncoder(cols=categorical_columns)
            data[categorical_columns] = self.encoder.fit_transform(data[categorical_columns], y)
        else:
            data[categorical_columns] = self.encoder.transform(data[categorical_columns])

        # delete columns
        data = data.drop(columns=["Data_pierwszej_rejestracji", "Data_publikacji_oferty"])
        if self.target_variable in data.columns:
            X = data.drop(columns=[self.target_variable, "ID"])
        else:
            X = data

        return X, y

    def train_lightgbm_regressor(self, X_train, y_train, X_test, y_test):
        # splitting 90/10
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
        train_data = lgb.Dataset(X_tr, label=y_tr)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        def objective(trial):
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'verbosity': -1,
                'seed': 42,
                'feature_pre_filter': False,
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50)
            }
            cv_results = lgb.cv(
                params,
                train_data,
                nfold=5,
                num_boost_round=1000,
                seed=42,
                stratified=False
            )
            return cv_results['valid rmse-mean'][-1]

        # optimize hyperparameters
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=50)
        print("Best Parameters:", study.best_params)

        best_params = study.best_params
        best_params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'seed': 42
        })

        # training the final model (callbacks for early stopping)
        final_model = lgb.train(
            best_params,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        print("Best iteration:", final_model.best_iteration)

        # cross validation
        lgbm_model = lgb.LGBMRegressor()
        lgbm_model._Booster = final_model

        test_cv_scores = cross_val_score(lgbm_model, X_test, y_test, cv=5, scoring="neg_root_mean_squared_error")
        train_cv_scores = cross_val_score(lgbm_model, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")
        print("%0.2f accuracy with a standard deviation of %0.2f test" % (test_cv_scores.mean(), test_cv_scores.std()))
        print("%0.2f accuracy with a standard deviation of %0.2f train" % (train_cv_scores.mean(), train_cv_scores.std()))

        output_folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.mkdir(f"{self.results_path}\\{output_folder_name}")
        with open(f"{self.results_path}\\{output_folder_name}\\scores.txt", "a") as f:
            f.write("%0.2f accuracy with a standard deviation of %0.2f test\n" % (test_cv_scores.mean(), test_cv_scores.std()))
            f.write("%0.2f accuracy with a standard deviation of %0.2f train" % (train_cv_scores.mean(), train_cv_scores.std()))
        study.trials_dataframe().to_csv(f"{self.results_path}\\{output_folder_name}\\optuna_trials.csv", index=False)

        # model saving
        with open(f'{self.results_path}\\{output_folder_name}\\lgbm_model_{output_folder_name}.pkl', 'wb') as f:
            pickle.dump(final_model, f)

        # feature importance
        lgb.plot_importance(final_model, importance_type='gain', max_num_features=30, figsize=(10, 10))
        plt.savefig(f"{self.results_path}\\{output_folder_name}\\feature_importance_{output_folder_name}.png")

        # prediction on test data
        X_test_raw = pd.read_csv("../data/raw/sales_ads_test.csv")
        currencies = X_test_raw["Waluta"].copy()
        X_test_prepared, _ = self.prepare_data(X_test_raw, drop_rows=False, fit_encoder=False)
        test_pred = final_model.predict(X_test_prepared.drop(columns=["ID"]))
        test_pred = np.expm1(test_pred)  # reverse log transform
        mask_eur = currencies == 'EUR'
        test_pred[mask_eur] = test_pred[mask_eur] / 4.18

        test_prediction_df = pd.DataFrame({"ID": X_test_prepared.index + 1, "Cena": test_pred})
        print(test_prediction_df.head())
        test_prediction_df.to_csv(f"{self.results_path}\\{output_folder_name}\\kaggle_upload_prediction.csv", index=False)


if __name__ == "__main__":
    model_trainer = ModelTrainer('../data/results/', ["../data/raw/sales_ads_train.csv"])
    model_trainer.train()
