import os

import numpy as np
import pandas as pd
import lightgbm as lgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
import pickle
import optuna
import optuna.integration


class ModelTrainer:
    def __init__(self, results_path: str | None = None, data_file_paths: list[str] | None = None):
        self.results_path = results_path or "data\\results"
        self.data_file_paths = data_file_paths or ["data\\raw\\sales_ads_train.csv"]
        self.target_variable = 'Cena'

    def train(self):
        data = pd.concat([pd.read_csv(file) for file in self.data_file_paths], ignore_index=True)
        X,y = self.prepare_data(data, True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        self.train_lightgbm_regressor(X_train, X_valid, y_train, y_valid)


    def prepare_data(self, data: pd.DataFrame, drop_rows: bool = True) -> tuple[pd.DataFrame, pd.Series | None]:
        if drop_rows:
            #todo dont drop rows when making final preds
            data = data.dropna()  # todo interpolate or drop missing values
        else:
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns

            # numeric cols: impute with linear interpolation/mean
            for col in numeric_cols:
                data[col] = data[col].interpolate(method='linear', limit_direction='both')
                # data[col].fillna(data[col].mean(), inplace=True)

            # categorical cols: impute with mode
            for col in categorical_cols:
                data[col].fillna(data[col].mode()[0], inplace=True)

        for col in ["Data_pierwszej_rejestracji", "Data_publikacji_oferty"]:
            data[col] = pd.to_datetime(data[col], errors='coerce')

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

        for col in categorical_columns:
            le = LabelEncoder() #todo change encoders
            data[col] = le.fit_transform(data[col].astype(str))

        data["Pierwszy_wlasciciel"] = data["Pierwszy_wlasciciel"].replace({"Yes": 1, "No": 0})

        #todo handle this
        data = data.drop(columns=["Wyposazenie", "Data_pierwszej_rejestracji", "Data_publikacji_oferty"])

        if self.target_variable in data.columns:
            # todo handle different currencies - DONE
            X = data.drop(columns=[self.target_variable, "ID", "Waluta"])
            y = data[self.target_variable]

        else:
            # when we prepare test data there is no target variable
            X = data
            y = None

        return X, y


    def train_lightgbm_regressor(self, X_train, X_valid, y_train, y_valid):
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
        def objective(trial):
            #todo add pruning - DONE
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'verbosity': -1,
                'seed': 42,
                'feature_pre_filter': False,
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50)
            }

            model = lgb.train(
                params,
                train_data,
                valid_sets=[valid_data],
                num_boost_round=1000,
                callbacks=[optuna.integration.LightGBMPruningCallback(trial, "rmse")]
            )
            y_pred = model.predict(X_valid)
            return np.sqrt(mean_squared_error(y_valid, y_pred))

        # Optimize hyperparameters
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100) #todo use more
        print("Best Parameters:", study.best_params)

        best_params = study.best_params
        best_params.update(
            {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'verbosity': -1, 'seed': 42})

        final_model = lgb.train(best_params, train_data, valid_sets=[valid_data],num_boost_round=1000)

        y_pred = final_model.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
        print(f'Final RMSE: {rmse:.4f}')

        output_folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        os.mkdir(f"{self.results_path}\\{output_folder_name}")

        study.trials_dataframe().to_csv(f"{self.results_path}\\{output_folder_name}\\optuna_trials.csv", index=False)

        # save model
        with open(f'{self.results_path}\\{output_folder_name}\\lgbm_model_{output_folder_name}.pkl', 'wb') as f:
            pickle.dump(final_model, f)

        # save feature importance plot
        lgb.plot_importance(final_model, importance_type='gain', figsize=(10, 10))
        plt.savefig(f"{self.results_path}\\{output_folder_name}\\feature_importance_{output_folder_name}.png")

        #todo add eval on our test set

        X_test = pd.read_csv("../data/raw/sales_ads_test.csv")
        X_test, _ = self.prepare_data(X_test, drop_rows=True)
        test_pred = final_model.predict(X_test.drop(columns=["ID"]))

        # save csv with columns ID and Cena - final data for upload
        test_prediction_df = pd.DataFrame({"ID": X_test.index, "Cena": test_pred})
        print(test_prediction_df.head())
        test_prediction_df.to_csv(f"{self.results_path}\\{output_folder_name}\\kaggle_upload_prediction.csv", index=False)




model_trainer = ModelTrainer('../data/results', ["../data/raw/sales_ads_train.csv"])
model_trainer.train()