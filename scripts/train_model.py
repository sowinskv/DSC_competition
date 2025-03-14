import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from datetime import datetime

from sklearn.preprocessing import LabelEncoder


class ModelTrainer:
    def __init__(self, results_path: str | None = None, data_file_paths: list[str] | None = None):
        self.results_path = results_path or "data\\results"
        self.data_file_paths = data_file_paths or ["data\\raw\\sales_ads_train.csv"]
        self.target_variable = 'Cena'

    def train(self):
        X_train, X_valid, y_train, y_valid, X_test, y_test = self.prepare_data()
        self.train_lightgbm_regressor(X_train, X_valid, y_train, y_valid)
    def prepare_data(self):
        data = pd.concat([pd.read_csv(file) for file in self.data_file_paths], ignore_index=True)
        data = data.dropna() #todo interpolate or drop missing values

        for col in ["Data_pierwszej_rejestracji", "Data_publikacji_oferty"]:
            data[col] = pd.to_datetime(data[col], errors='coerce')

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
        data = data.drop(columns=["Wyposazenie", "Data_pierwszej_rejestracji", "Data_publikacji_oferty", "Pierwszy_wlasciciel"])

        X = data.drop(columns=[self.target_variable])
        y = data[self.target_variable]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        return X_train, X_valid, y_train, y_valid, X_test, y_test


    def train_lightgbm_regressor(self, X_train, X_valid, y_train, y_valid):
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'seed': 42
        }

        model = lgb.train(params, train_data, valid_sets=[valid_data], num_boost_round=1000)

        y_pred = model.predict(X_valid)

        rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
        print(f'RMSE: {rmse:.4f}')

        #todo replace with optuna
        param_grid = {
            'num_leaves': [31, 50],
            'learning_rate': [0.05, 0.1],
            'n_estimators': [100, 500]
        }

        gbm = lgb.LGBMRegressor(**params)
        grid_search = GridSearchCV(gbm, param_grid, scoring='neg_root_mean_squared_error', cv=5, verbose=1)
        grid_search.fit(X_train, y_train)

        print("Best Parameters:", grid_search.best_params_)

        y_pred_best = grid_search.best_estimator_.predict(X_valid)
        rmse_best = np.sqrt(mean_squared_error(y_valid, y_pred_best))
        print(f'Best RMSE: {rmse_best:.4f}')

        datetime_hour = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results = pd.DataFrame(grid_search.cv_results_)
        results.to_csv( f"{self.results_path}\\results-{datetime_hour}.csv", index=False)


model_trainer = ModelTrainer()
model_trainer.train()