import ast

import pandas as pd


def parse_equipment(x):
    try:
        return ast.literal_eval(x)
    except Exception:
        return []
def assign_voivodeship(location):
    if pd.isnull(location):
        return "inne"
    poland_voivodeships = [
        'dolnośląskie', 'kujawsko-pomorskie', 'lubelskie', 'lubuskie', 'łódzkie', 'małopolskie', 'mazowieckie',
        'opolskie', 'podkarpackie', 'podlaskie', 'pomorskie', 'śląskie', 'świętokrzyskie', 'warmińsko-mazurskie',
        'wielkopolskie', 'zachodniopomorskie'
    ]

    location_to_voivodeship = {
        "Warszawa": "mazowieckie",
        "Kraków": "małopolskie",
        "Wrocław": "dolnośląskie",
        "Poznań": "wielkopolskie",
        "Gdańsk": "pomorskie",
        "Sopot": "pomorskie",
        "Szczecin": "zachodniopomorskie",
        "Białystok": "podlaskie",
        "Bydgoszcz": "kujawsko-pomorskie",
        "Częstochowa": "śląskie",
        "Łódź": "łódzkie",
        "Gliwice": "śląskie",
        "Katowice": "śląskie",
        "Bieruń": "śląskie",
        "Bielsko-Biała": "śląskie",
        "Imielin": "śląskie",
        "Gdynia": "pomorskie",
        "Będzin": "śląskie",
        "Zawiercie": "śląskie",
        "Mikołów": "śląskie",
        "Orzesze": "śląskie",
        "Józefów": "mazowieckie",

    }

    for voivodeship in poland_voivodeships:
        if voivodeship in location.lower():
            return voivodeship
    for key, value in location_to_voivodeship.items():
        if key.lower() in location.lower():
            return value
    return None

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