import os
import gdown

'''
downloading remaining large files from google drive using gdown
'''

DATA_DIR = os.path.join('../data', 'raw')
os.makedirs(DATA_DIR, exist_ok=True)


file_ids = {
    'sales_ads_train.csv': '1VjRYys7vhV1JrXBK9k06qnkDUE1FYWEP',
    'synthetic_training_data_mostlyai_pl.csv': '1rSYbwXt2fGpif_0dR5q1m10WGrpP-baz',
    'synthetic_training_data_sdv_pl.csv': '1MrAJ9J-CEp3E2DAg3OKYOvj3n2M4szeJ'
}

for filename, file_id in file_ids.items():
    destination = os.path.join(DATA_DIR, filename)
    if os.path.exists(destination):
        print(f"{filename} already exists, skipping download.")
    else:
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {filename}...")
        gdown.download(url, destination, quiet=False)
        print(f"Downloaded {filename} to {destination}")
