import pandas as pd
import urllib.request
import json


url = 'https://raw.githubusercontent.com/M3IT/COVID-19_Data/master/Data/COVID19_Data_Hub.csv'
filename = '../data/AU.csv'
print(f'Downloading {url}')
filename, headers = urllib.request.urlretrieve(url, filename=filename)
print("download complete!")
print("download file location: ", filename)
print("Processing")

df = pd.read_csv(filename, delimiter=',')
df_victoria = df[df['administrative_area_level_2']=='Victoria']
df_victoria["Currently Positive Number"] = df_victoria["confirmed"] - (df_victoria["deaths"]+df_victoria["recovered"])

df_victoria["Total Cases"] = df_victoria["confirmed"]/df_victoria["population"]
df_victoria["Total Deaths"] = df_victoria["deaths"]/df_victoria["population"]
df_victoria["Total Recovered"] = df_victoria["recovered"]/df_victoria["population"]
df_victoria["Currently Positive"] = df_victoria["Currently Positive Number"]/df_victoria["population"]
df_victoria["Currently Hospitalized"] = df_victoria["hosp"]/df_victoria["population"]
df_victoria["Currently Critical"] = (df_victoria["icu"]+df_victoria["vent"])/df_victoria["population"]

df_victoria = df_victoria.drop(columns=["confirmed","deaths","tests","positives","recovered","hosp",
                                "icu", "vent", "population", "administrative_area_level","administrative_area_level_1",
                                "administrative_area_level_2","administrative_area_level_3", "id","state_abbrev",
                                "Currently Positive Number"])
df_victoria.to_csv(f'../data/Victoria.csv', index=False)

