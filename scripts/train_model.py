from __future__ import annotations

import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from datetime import datetime

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import pickle
import optuna
import optuna.integration
import re
from lightgbm import cv


class ModelTrainer:
    def __init__(self, results_path: str | None = None, data_file_paths: list[str] | None = None):
        self.results_path = results_path or "data\\results"
        self.data_file_paths = data_file_paths or ["data\\raw\\sales_ads_train.csv"]
        self.target_variable = 'Cena'
        self.ohe = None # 1 one hot encoder for all data

    def train(self):
        data = pd.concat([pd.read_csv(file) for file in self.data_file_paths], ignore_index=True)
        X, y = self.prepare_data(data, False, True) # todo change to False when making final predictions
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.train_lightgbm_regressor(X_train, y_train, X_test, y_test)


    def prepare_data(self, data: pd.DataFrame, drop_rows: bool = True, fit_encoder: bool = False) -> tuple[pd.DataFrame, pd.Series | None]:
        if drop_rows:
            data = data.dropna()
        else:
            # impute missing values
            missing_threshold = 0.05  # only impute columns with more than 5% missing values
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


        for col in ["Data_pierwszej_rejestracji", "Data_publikacji_oferty"]:
            data[col] = pd.to_datetime(data[col], errors='coerce', format='%d/%m/%Y')

        data["Wiek_samochodu_lata"] = (data["Data_publikacji_oferty"] - data["Data_pierwszej_rejestracji"]).dt.days // 365

        conversion_rates = {'EUR': 4.18, 'PLN': 1.0}
        if 'Cena' in data.columns:
            data['Cena'] = data.apply(
                lambda row: row['Cena'] * conversion_rates.get(row['Waluta'], 1),
                axis=1
            )

        categorical_columns = [
            "Waluta", "Stan", "Marka_pojazdu", "Model_pojazdu", "Wersja_pojazdu",
            "Generacja_pojazdu", "Rodzaj_paliwa", "Naped", "Skrzynia_biegow",
            "Typ_nadwozia", "Kolor", "Kraj_pochodzenia", "Lokalizacja_oferty"
        ]

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
            common_features = equipment_df.columns[equipment_df.sum() > 50] # optional
            equipment_df = equipment_df[common_features]
            data = pd.concat([data.drop(columns=["Wyposazenie", "wyposazenie_list"]), equipment_df], axis=1)

        def make_unique(columns):
            seen = {}
            new_cols = []
            for col in columns:
                if col in seen:
                    seen[col] += 1
                    new_col = f"{col}_{seen[col]}"
                else:
                    seen[col] = 0
                    new_col = col
                new_cols.append(new_col)
            return new_cols


        # encode
        if fit_encoder or self.ohe is None:
            self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
            ohe_array = self.ohe.fit_transform(data[categorical_columns])
        else:
            ohe_array = self.ohe.transform(data[categorical_columns])
        ohe_df = pd.DataFrame.sparse.from_spmatrix(
            ohe_array,
            columns=self.ohe.get_feature_names_out(categorical_columns)
        )
        ohe_df.columns = [re.sub(r'\W+', '_', col).strip('_') for col in ohe_df.columns]
        ohe_df.columns = make_unique(ohe_df.columns)

        data = pd.concat([data.drop(columns=categorical_columns), ohe_df], axis=1)
        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        data[numeric_cols] = data[numeric_cols].fillna(0)

        data["Pierwszy_wlasciciel"] = data["Pierwszy_wlasciciel"].replace({"Yes": 1, "No": 0})

        data = data.drop(columns=["Data_pierwszej_rejestracji", "Data_publikacji_oferty"])

        if self.target_variable in data.columns:
            X = data.drop(columns=[self.target_variable, "ID"])
            y = np.log1p(data[self.target_variable]) # log scaling

        else:
            # when we prepare test data there is no target variable
            X = data
            y = None

        return X, y


    def train_lightgbm_regressor(self, X_train, y_train, X_test, y_test):

        train_data = lgb.Dataset(X_train, label=y_train)

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
                stratified = False # basic kfold for continous (log) target
            )

            return cv_results['valid rmse-mean'][-1]

        # optimize hyperparameters
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=1)
        print("Best Parameters:", study.best_params)

        best_params = study.best_params
        best_params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'seed': 42
        })

        cv_results = lgb.cv(
            best_params,
            train_data,
            nfold=5,
            num_boost_round=1000,
            seed=42,
            stratified=False
        )
        best_iteration = len(cv_results['valid rmse-mean'])

        final_model = lgb.train(best_params, train_data, num_boost_round=best_iteration)
        # we need this because cross validation requires estimator to be passed but train returns booster
        lgbm_model = lgb.LGBMRegressor()
        lgbm_model._Booster = final_model

        # eval on the held-out test set
        test_cv_scores = cross_val_score(lgbm_model, X_test, y_test, cv=5, scoring="neg_root_mean_squared_error")
        train_cv_scores = cross_val_score(lgbm_model, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")

        print("%0.2f accuracy with a standard deviation of %0.2f test" % (test_cv_scores.mean(), test_cv_scores.std()))
        print("%0.2f accuracy with a standard deviation of %0.2f train" % (train_cv_scores.mean(), train_cv_scores.std()))


        output_folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        os.mkdir(f"{self.results_path}\\{output_folder_name}")

        with open(f"{self.results_path}\\{output_folder_name}\\scores.txt", "a") as f:
            f.write("%0.2f accuracy with a standard deviation of %0.2f test" % (test_cv_scores.mean(), test_cv_scores.std()))
            f.write("\n")
            f.write("%0.2f accuracy with a standard deviation of %0.2f train" % (train_cv_scores.mean(), train_cv_scores.std()))


        study.trials_dataframe().to_csv(f"{self.results_path}\\{output_folder_name}\\optuna_trials.csv", index=False)

        # save model
        with open(f'{self.results_path}\\{output_folder_name}\\lgbm_model_{output_folder_name}.pkl', 'wb') as f:
            pickle.dump(final_model, f)

        # save feature importance plot
        lgb.plot_importance(final_model, importance_type='gain', figsize=(10, 10))
        plt.savefig(f"{self.results_path}\\{output_folder_name}\\feature_importance_{output_folder_name}.png")

        #todo add eval on our test set

        X_test = pd.read_csv("../data/raw/sales_ads_test.csv")
        currencies = X_test["Waluta"].copy()
        X_test, _ = self.prepare_data(X_test, drop_rows=False)
        test_pred = final_model.predict(X_test.drop(columns=["ID"]))
        test_pred = np.expm1(test_pred)  # reverse log scaling

        mask_eur = currencies == 'EUR'
        test_pred[mask_eur] = test_pred[mask_eur] / 4.18

        # save csv with columns ID and Cena - final data for upload
        test_prediction_df = pd.DataFrame({"ID": X_test.index + 1, "Cena": test_pred})  # start ID from 1
        print(test_prediction_df.head())
        test_prediction_df.to_csv(f"{self.results_path}\\{output_folder_name}\\kaggle_upload_prediction.csv", index=False)




model_trainer = ModelTrainer('../data/results/', ["../data/raw/sales_ads_train.csv"])
model_trainer.train()