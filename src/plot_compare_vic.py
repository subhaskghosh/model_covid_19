import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import lmfit
import argparse
from src.plots import plot_evolution_vic, plot_compare_vic

#Load
df_1 = pd.read_csv(f'../data/victoria_plot_compare_1.csv')
df_2 = pd.read_csv(f'../data/victoria_plot_compare_2.csv')
df_3 = pd.read_csv(f'../data/victoria_plot_compare_3.csv')

#Adjust the cumulatives df_2
l = len(df_1['t'])
DR_1 = df_1['DR'][l-1]
DR_2 = [(a+DR_1) for a in df_2['DR']]
recovered_1 = df_1['recovered'][l-1]
recovered_2 = [(a+recovered_1) for a in df_2['recovered']]
D_1 = df_1['D'][l-1]
D_2 = [(a+D_1) for a in df_2['D']]
deaths_1 = df_1['deaths'][l-1]
deaths_2 = [(a+deaths_1) for a in df_2['deaths']]
tot_1 = df_1['TI+DR+D'][l-1]
tot_2 = [(a+tot_1) for a in df_2['TI+DR+D']]
total_cases_1 = df_1['total_cases'][l-1]
total_cases_2 = [(a+total_cases_1) for a in df_2['total_cases']]

data = {'t': df_2['t'],
        'TI': df_2['TI'],
        'currently_infected': df_2['currently_infected'],
        'DR': DR_2,
        'recovered': recovered_2,
        'C': df_2['C'],
        'critical': df_2['critical'],
        'D': D_2,
        'deaths': deaths_2,
        'TI+DR+D': tot_2,
        'total_cases': total_cases_2,
        'H': df_2['H'],
        'isolated': df_2['isolated'],
        }
df_2 = pd.DataFrame(data)

#Adjust the cumulatives df_3
l = len(df_2['t'])
DR_2 = df_2['DR'][l-1]
DR_3 = [(a+DR_2) for a in df_3['DR']]
recovered_2 = df_2['recovered'][l-1]
recovered_3 = [(a+recovered_2) for a in df_3['recovered']]
D_2 = df_2['D'][l-1]
D_3 = [(a+D_2) for a in df_3['D']]
deaths_2 = df_2['deaths'][l-1]
deaths_3 = [(a+deaths_2) for a in df_3['deaths']]
tot_2 = df_2['TI+DR+D'][l-1]
tot_3 = [(a+tot_2) for a in df_3['TI+DR+D']]
total_cases_2 = df_2['total_cases'][l-1]
total_cases_3 = [(a+total_cases_2) for a in df_3['total_cases']]

data = {'t': df_3['t'],
        'TI': df_3['TI'],
        'currently_infected': df_3['currently_infected'],
        'DR': DR_3,
        'recovered': recovered_3,
        'C': df_3['C'],
        'critical': df_3['critical'],
        'D': D_3,
        'deaths': deaths_3,
        'TI+DR+D': tot_3,
        'total_cases': total_cases_3,
        'H': df_3['H'],
        'isolated': df_3['isolated'],
        }
df_3 = pd.DataFrame(data)

plot_compare_df = pd.concat([df_1, df_2, df_3])

t = plot_compare_df['t']
TI = plot_compare_df['TI']
currently_infected = plot_compare_df['currently_infected']
DR = plot_compare_df['DR']
recovered = plot_compare_df['recovered']
C = plot_compare_df['C']
critical = plot_compare_df['critical']
D = plot_compare_df['D']
deaths = plot_compare_df['deaths']
tot = plot_compare_df['TI+DR+D']
total_cases = plot_compare_df['total_cases']
H = plot_compare_df['H']
isolated = plot_compare_df['isolated']

plot_compare_vic(t,
             TI, currently_infected,
             DR, recovered,
             C, critical,
             D, deaths,
             tot,total_cases,
             H, isolated,
             0, len(t))
