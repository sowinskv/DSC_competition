import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from datetime import datetime

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import pickle
import optuna
import optuna.integration
import re
import ast
from sklearn.preprocessing import MultiLabelBinarizer

from scripts.constants import premium_equipment, naped_mapping, common_price_to_car_model
from scripts.utils import parse_equipment, make_unique, assign_voivodeship

class ModelTrainer:
    def __init__(self, results_path: str | None = None, data_file_paths: list[str] | None = None, encoder=None):
        self.results_path = results_path or "data\\results"
        self.data_file_paths = data_file_paths or ["data\\raw\\sales_ads_train.csv"]
        self.target_variable = 'Cena'
        self.ohe = None  # 1 one hot encoder for all data
        if encoder is not None:
            self.ohe = encoder

    def train(self):
        data = pd.concat([pd.read_csv(file) for file in self.data_file_paths], ignore_index=True)
        X, y = self.prepare_data(data, False, True)  # False when making final predictions
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.train_lightgbm_regressor(X_train, y_train, X_test, y_test)

    def prepare_data(self, data: pd.DataFrame, drop_rows: bool = True, fit_encoder: bool = False) -> tuple[
        pd.DataFrame, pd.Series | None]:
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

        data["Wiek_samochodu_lata"] = (data["Data_publikacji_oferty"] - data[
            "Data_pierwszej_rejestracji"]).dt.days // 365

        conversion_rates = {'EUR': 4.18, 'PLN': 1.0}
        if 'Cena' in data.columns:
            data['Cena'] = data.apply(
                lambda row: row['Cena'] * conversion_rates.get(row['Waluta'], 1),
                axis=1
            )

        data['Marka_pojazdu_freq'] = data['Marka_pojazdu'].map(data['Marka_pojazdu'].value_counts() / len(data))
        data['Model_pojazdu_freq'] = data['Model_pojazdu'].map(data['Model_pojazdu'].value_counts() / len(data))
        data['Common_market_price_per_brand'] = data['Marka_pojazdu'].apply(
            lambda x: common_price_to_car_model.get(x, np.nan))
        data['Common_market_price_per_brand'] = data['Common_market_price_per_brand'].fillna(7600) # default for dataset

        categorical_columns = [
            "Waluta", "Stan", "Marka_pojazdu", "Model_pojazdu", "Wersja_pojazdu",
            "Generacja_pojazdu", "Rodzaj_paliwa", "Naped", "Skrzynia_biegow",
            "Typ_nadwozia", "Kolor", "Kraj_pochodzenia", "Lokalizacja_oferty"
        ]

        if 'Wyposazenie' in data.columns:
            data['wyposazenie_list'] = data['Wyposazenie'].apply(parse_equipment)
            mlb = MultiLabelBinarizer(sparse_output=True)
            equipment_matrix = mlb.fit_transform(data['wyposazenie_list'])
            equipment_df = pd.DataFrame.sparse.from_spmatrix(equipment_matrix, columns=mlb.classes_)
            common_features = equipment_df.columns[equipment_df.sum() > 2000]  # optional
            equipment_df = equipment_df[common_features]

            data["contains_premium_equipment"] = data['wyposazenie_list'].apply(
                lambda x: any(e in x for e in premium_equipment))

            data = pd.concat([data.drop(columns=["Wyposazenie", "wyposazenie_list"]), equipment_df], axis=1)

        # encode
        if fit_encoder or self.ohe is None:
            self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
            ohe_array = self.ohe.fit_transform(data[categorical_columns])
            # save encoder
            encoder_folder = f"{self.results_path}{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            os.mkdir(encoder_folder)
            with open(f"{encoder_folder}\\ohe.pkl", 'wb') as f:
                pickle.dump(self.ohe, f)
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

        # ---- feature engineering ----
        data["Naped"] = data["Naped"].map(naped_mapping)
        data["Pierwszy_wlasciciel"] = data["Pierwszy_wlasciciel"].replace({"Yes": 1, "No": 0})
        data['Wojewodztwo'] = data['Lokalizacja_oferty'].apply(lambda x: assign_voivodeship(x))

        data = data.drop(columns=["Data_pierwszej_rejestracji", "Data_publikacji_oferty"])
        if "Przebieg_km" in data.columns and "Wiek_samochodu_lata" in data.columns:
            data["Wiek_samochodu_lata"] = data["Wiek_samochodu_lata"].replace({0: np.nan})
            data["Sredni_roczny_przebieg"] = data["Przebieg_km"] / data["Wiek_samochodu_lata"]

        if "Moc_KM" in data.columns and "Pojemnosc_cm3" in data.columns:
            data["Moc_na_pojemnosc"] = data["Moc_KM"] / data["Pojemnosc_cm3"]

        if "Marka_pojazdu" in data.columns and "Cena" in data.columns:
            brand_avg_price = data.groupby("Marka_pojazdu")["Cena"].mean()
            premium_brands = brand_avg_price[brand_avg_price > 100000].index
            data["Premium_marka"] = data["Marka_pojazdu"].isin(premium_brands).astype(int)

        if self.target_variable in data.columns:
            X = data.drop(columns=[self.target_variable, "ID"])
            y = np.log1p(data[self.target_variable])  # log scaling

        else:
            # when we prepare test data there is no target variable
            X = data
            y = None

        return X, y

    def train_lightgbm_regressor(self, X_train, y_train, X_test, y_test):
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
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            }
            pruning_callback = optuna.integration.LightGBMPruningCallback(trial, 'rmse')

            gbm = lgb.train(
                params,
                train_data,
                num_boost_round=2000,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), pruning_callback]
            )
            # eval on the valid set
            y_pred_val = gbm.predict(X_val)
            rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
            return rmse_val

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

        final_model = lgb.train(
            best_params,
            train_data,
            num_boost_round=2000,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)]
        )

        # eval on the held-out test set
        y_pred = final_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f'Final RMSE on test set: {rmse:.4f}')

        output_folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.mkdir(f"{self.results_path}\\{output_folder_name}")
        study.trials_dataframe().to_csv(f"{self.results_path}\\{output_folder_name}\\optuna_trials.csv", index=False)

        # save model
        with open(f'{self.results_path}\\{output_folder_name}\\lgbm_model_{output_folder_name}.pkl', 'wb') as f:
            pickle.dump(final_model, f)

        # save feature importance plot
        lgb.plot_importance(final_model, importance_type='gain', max_num_features=30, figsize=(10, 10))
        plt.savefig(f"{self.results_path}\\{output_folder_name}\\feature_importance_{output_folder_name}-gain.png")
        lgb.plot_importance(final_model, importance_type='split', max_num_features=30, figsize=(10, 10))
        plt.savefig(f"{self.results_path}\\{output_folder_name}\\feature_importance_{output_folder_name}-split.png")

        # cross validation
        lgbm_model = lgb.LGBMRegressor()
        lgbm_model._Booster = final_model

        test_cv_scores = cross_val_score(lgbm_model, X_test, y_test, cv=5, scoring="neg_root_mean_squared_error")
        train_cv_scores = cross_val_score(lgbm_model, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error")
        print("%0.2f accuracy with a standard deviation of %0.2f test" % (test_cv_scores.mean(), test_cv_scores.std()))
        print(
            "%0.2f accuracy with a standard deviation of %0.2f train" % (train_cv_scores.mean(), train_cv_scores.std()))

        with open(f"{self.results_path}\\{output_folder_name}\\scores.txt", "a") as f:
            f.write(
                "%0.2f accuracy with a standard deviation of %0.2f test\n" % (
                test_cv_scores.mean(), test_cv_scores.std()))
            f.write(
                "%0.2f accuracy with a standard deviation of %0.2f train" % (
                train_cv_scores.mean(), train_cv_scores.std()))

        X_test = pd.read_csv("../data/raw/sales_ads_test.csv")
        currencies = X_test["Waluta"].copy()
        X_test, _ = self.prepare_data(X_test, drop_rows=False)
        test_pred = final_model.predict(X_test.drop(columns=["ID"]))
        test_pred = np.expm1(test_pred)  # reverse log scaling

        mask_eur = currencies == 'EUR'
        test_pred[mask_eur] = test_pred[mask_eur] / 4.18

        # save csv with columns ID and Cena - final data for upload
        test_prediction_df = pd.DataFrame({"ID": X_test.index + 1, "Cena": test_pred})  # start ID from 1
        test_prediction_df.to_csv(f"{self.results_path}\\{output_folder_name}\\kaggle_upload_prediction.csv",
                                  index=False)
        assert test_prediction_df.shape[0] == 72907, "Wrong number of predictions"


model_trainer = ModelTrainer('../data/results/', ["../data/raw/sales_ads_train.csv"])
model_trainer.train()
