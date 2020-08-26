import pandas as pd
import urllib.request
import json


url = 'https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv'
filename = '../data/Italy.csv'
print(f'Downloading {url}')
filename, headers = urllib.request.urlretrieve(url, filename=filename)
print("download complete!")
print("download file location: ", filename)
print("Processing")
