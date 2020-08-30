import pandas as pd
import urllib.request
import json

def get_resultant_DF(countryList, df):
    dfList =[]
    for (country, province) in countryList:
        if province:
            tdf = df[(df['Country/Region'] == country) & (df['Province/State']==province)]
        else:
            tdf = df[(df['Country/Region']==country) & (df['Province/State'].isnull())]
        dfList.append(tdf)
    df = pd.concat(dfList)
    df = df.drop(columns=['Province/State', 'Lat', 'Long'])
    columns = df.columns.tolist()
    dates = columns[1:]
    start = dates[0]
    end = dates[-1]
    date_range = pd.date_range(start=start, end=end)
    df.columns = ['Country'] + [i for i in range(0,len(date_range))]
    return [df, start, end, dates]


def getConfirmed(filename,countryList):
    df = pd.read_csv(filename, delimiter=',')
    return get_resultant_DF(countryList, df)

def getDeath(filename,countryList):
    df = pd.read_csv(filename, delimiter=',')
    return get_resultant_DF(countryList, df)

def getRecovered(filename,countryList):
    df = pd.read_csv(filename, delimiter=',')
    return get_resultant_DF(countryList, df)


def update():
    url_confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    filename_confirmed = '../data/time_series_covid19_confirmed_global.csv'
    url_deaths = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    filename_deaths = '../data/time_series_covid19_deaths_global.csv'
    url_recovered = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
    filename_recovered = '../data/time_series_covid19_recovered_global.csv'

    print(f'Downloading {url_confirmed}')
    filename_confirmed, headers_confirmed = urllib.request.urlretrieve(url_confirmed, filename=filename_confirmed)
    print("download complete!")
    print("download file location: ", filename_confirmed)
    print("Processing")

    print(f'Downloading {url_deaths}')
    filename_deaths, headers_deaths = urllib.request.urlretrieve(url_deaths, filename=filename_deaths)
    print("download complete!")
    print("download file location: ", filename_deaths)
    print("Processing")

    print(f'Downloading {url_recovered}')
    filename_recovered, headers_deaths = urllib.request.urlretrieve(url_recovered, filename=filename_recovered)
    print("download complete!")
    print("download file location: ", filename_recovered)
    print("Processing")

def prepare():
    filename_confirmed = '../data/time_series_covid19_confirmed_global.csv'
    filename_deaths = '../data/time_series_covid19_deaths_global.csv'
    filename_recovered = '../data/time_series_covid19_recovered_global.csv'

    countryList = [("India", None), ("Australia", "Victoria")]

    # Load population data from The World Bank Group
    # https://data.worldbank.org/indicator/SP.POP.TOTL?end=2019&start=1960&view=chart&year=2019
    population_df = pd.read_csv('../data/API_SP.POP.TOTL_DS2_en_csv_v2_1308146.csv', delimiter=',')
    population_df = population_df.loc[:, ['Country Name', '2019']]

    [confirmed_df, start, end, dates] = getConfirmed(filename_confirmed, countryList)
    [death_df, start, end, dates] = getDeath(filename_deaths, countryList)
    [recovered_df, start, end, dates] = getRecovered(filename_recovered, countryList)

    ## Assumption - dates match
    with open('../data/dates.json', 'w', encoding='utf-8') as f:
        json.dump({'start':start, 'end': end, 'dates': dates}, f, ensure_ascii=False, indent=4)

    for (c,p) in countryList:
        population = population_df[population_df["Country Name"]==c].values.tolist()[0][1]
        if p=='Victoria':
            population = 6651100
        confirmed = [a/population for a in confirmed_df[confirmed_df["Country"]==c].values.tolist()[0][1:]]
        confirmed_n = [a for a in confirmed_df[confirmed_df["Country"]==c].values.tolist()[0][1:]]
        death = [a/population for a in death_df[death_df["Country"] == c].values.tolist()[0][1:]]
        death_n = [a for a in death_df[death_df["Country"] == c].values.tolist()[0][1:]]
        recovered = [a/population for a in recovered_df[recovered_df["Country"] == c].values.tolist()[0][1:]]
        recovered_n = [a for a in recovered_df[recovered_df["Country"] == c].values.tolist()[0][1:]]
        country_df = pd.DataFrame({
            "Total Cases": confirmed,
            "Deaths" : death,
            "Recovered": recovered,
            "Total Cases Number": confirmed_n,
            "Deaths Number": death_n,
            "Recovered Number": recovered_n,
            "Date" : dates
        })
        country_df["Currently Positive"] = country_df["Total Cases"] - (country_df["Deaths"]+country_df["Recovered"])
        country_df["Currently Positive Number"] = country_df["Total Cases Number"] - (country_df["Deaths Number"] + country_df["Recovered Number"])
        if p:
            country_df.to_csv(f'../data/{c}_{p}.csv', index=False)
        else:
            country_df.to_csv(f'../data/{c}.csv', index=False)
update()
prepare()