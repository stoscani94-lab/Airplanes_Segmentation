import os
import zipfile

# Esegue il comando da terminale per scaricare il dataset da Kaggle
# (Richiede di aver configurato le API key di Kaggle sul proprio PC)
# Link del dataset: https://www.kaggle.com/datasets/samueltoscani94/airplane-segmentation-dataset
print("Scaricamento da Kaggle in corso...")
os.system('kaggle datasets download -d samueltoscani94/airplane-segmentation-dataset')

nome_zip_scaricato = 'airplane-segmentation-dataset.zip'

print("Estrazione in corso...")
# Estrae lo zip
with zipfile.ZipFile(nome_zip_scaricato, 'r') as zip_ref:
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
        zip_ref.extractall('dataset')
    else:
        print("Dataset già presente")

# Pulisce
os.remove(nome_zip_scaricato)
print("Dataset pronto!")