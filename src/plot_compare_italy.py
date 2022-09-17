import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import lmfit
import argparse
from src.plots import plot_evolution_vic, plot_compare

#Load
df_1 = pd.read_csv(f'../data/italy_plot_compare_1.csv')
df_2 = pd.read_csv(f'../data/italy_plot_compare_2.csv')
df_3 = pd.read_csv(f'../data/italy_plot_compare_3.csv')
df_4 = pd.read_csv(f'../data/italy_plot_compare_4.csv')
df_5 = pd.read_csv(f'../data/italy_plot_compare_5.csv')

#Adjust the cumulatives df_2
l = len(df_1['t'])

DR_1 = df_1['DR'][l-1]
DR_2 = [(a+DR_1) for a in df_2['DR']]
recovered_1 = df_1['recovered'][l-1]
recovered_2 = [(a+recovered_1) for a in df_2['recovered']]

C_1 = df_1['C'][l-1]
C_2 = [(a+C_1) for a in df_2['C']]
critical_1 = df_1['critical'][l-1]
critical_2 = [(a+critical_1) for a in df_2['critical']]

D_1 = df_1['D'][l-1]
D_2 = [(a+D_1) for a in df_2['D']]
deaths_1 = df_1['deaths'][l-1]
deaths_2 = [(a+deaths_1) for a in df_2['deaths']]

tot_1 = df_1['TI+DR+D'][l-1]
tot_2 = [(a+tot_1) for a in df_2['TI+DR+D']]
total_cases_1 = df_1['total_cases'][l-1]
total_cases_2 = [(a+total_cases_1) for a in df_2['total_cases']]

H_1 = df_1['H'][l-1]
H_2 = [(a+H_1) for a in df_2['H']]
isolated_1 = df_1['isolated'][l-1]
isolated_2 = [(a+isolated_1) for a in df_2['isolated']]

data = {'t': df_2['t'],
        'TI': df_2['TI'],
        'currently_infected': df_2['currently_infected'],
        'DR': DR_2,
        'recovered': recovered_2,
        'C': C_2,
        'critical': critical_2,
        'D': D_2,
        'deaths': deaths_2,
        'TI+DR+D': tot_2,
        'total_cases': total_cases_2,
        'H': H_2,
        'isolated': isolated_2,
        }
df_2 = pd.DataFrame(data)

#Adjust the cumulatives df_3
l = len(df_2['t'])

DR_2 = df_2['DR'][l-1]
DR_3 = [(a+DR_2) for a in df_3['DR']]
recovered_2 = df_2['recovered'][l-1]
recovered_3 = [(a+recovered_2) for a in df_3['recovered']]

C_2 = df_2['C'][l-1]
C_3 = [(a+C_2) for a in df_3['C']]
critical_2 = df_2['critical'][l-1]
critical_3 = [(a+critical_2) for a in df_3['critical']]

D_2 = df_2['D'][l-1]
D_3 = [(a+D_2) for a in df_3['D']]
deaths_2 = df_2['deaths'][l-1]
deaths_3 = [(a+deaths_2) for a in df_3['deaths']]

tot_2 = df_2['TI+DR+D'][l-1]
tot_3 = [(a+tot_2) for a in df_3['TI+DR+D']]
total_cases_2 = df_2['total_cases'][l-1]
total_cases_3 = [(a+total_cases_2) for a in df_3['total_cases']]

H_2 = df_2['H'][l-1]
H_3 = [(a+H_2) for a in df_3['H']]
isolated_2 = df_2['isolated'][l-1]
isolated_3 = [(a+isolated_2) for a in df_3['isolated']]

data = {'t': df_3['t'],
        'TI': df_3['TI'],
        'currently_infected': df_3['currently_infected'],
        'DR': DR_3,
        'recovered': recovered_3,
        'C': C_3,
        'critical': critical_3,
        'D': D_3,
        'deaths': deaths_3,
        'TI+DR+D': tot_3,
        'total_cases': total_cases_3,
        'H': H_3,
        'isolated': isolated_3,
        }
df_3 = pd.DataFrame(data)

#Adjust the cumulatives df_4
l = len(df_3['t'])

DR_3 = df_3['DR'][l-1]
DR_4 = [(a+DR_3) for a in df_4['DR']]
recovered_3 = df_3['recovered'][l-1]
recovered_4 = [(a+recovered_3) for a in df_4['recovered']]

C_3 = df_3['C'][l-1]
C_4 = [(a+C_3) for a in df_4['C']]
critical_3 = df_3['critical'][l-1]
critical_4 = [(a+critical_3) for a in df_4['critical']]

D_3 = df_3['D'][l-1]
D_4 = [(a+D_3) for a in df_4['D']]
deaths_3 = df_3['deaths'][l-1]
deaths_4 = [(a+deaths_3) for a in df_4['deaths']]

tot_3 = df_3['TI+DR+D'][l-1]
tot_4 = [(a+tot_3) for a in df_4['TI+DR+D']]
total_cases_3 = df_3['total_cases'][l-1]
total_cases_4 = [(a+total_cases_3) for a in df_4['total_cases']]

H_3 = df_3['H'][l-1]
H_4 = [(a+H_3) for a in df_4['H']]
isolated_3 = df_3['isolated'][l-1]
isolated_4 = [(a+isolated_3) for a in df_4['isolated']]

data = {'t': df_4['t'],
        'TI': df_4['TI'],
        'currently_infected': df_4['currently_infected'],
        'DR': DR_4,
        'recovered': recovered_3,
        'C': C_4,
        'critical': critical_4,
        'D': D_4,
        'deaths': deaths_4,
        'TI+DR+D': tot_4,
        'total_cases': total_cases_4,
        'H': H_4,
        'isolated': isolated_4,
        }
df_4 = pd.DataFrame(data)

#Adjust the cumulatives df_5
l = len(df_4['t'])

DR_4 = df_4['DR'][l-1]
DR_5 = [(a+DR_4) for a in df_5['DR']]
recovered_4 = df_4['recovered'][l-1]
recovered_5 = [(a+recovered_4) for a in df_5['recovered']]

C_4 = df_4['C'][l-1]
C_5 = [(a+C_4) for a in df_5['C']]
critical_4 = df_4['critical'][l-1]
critical_5 = [(a+critical_4) for a in df_5['critical']]

D_4 = df_4['D'][l-1]
D_5 = [(a+D_4) for a in df_5['D']]
deaths_4 = df_4['deaths'][l-1]
deaths_5 = [(a+deaths_4) for a in df_5['deaths']]

tot_4 = df_4['TI+DR+D'][l-1]
tot_5 = [(a+tot_4) for a in df_5['TI+DR+D']]
total_cases_4 = df_4['total_cases'][l-1]
total_cases_5 = [(a+total_cases_4) for a in df_5['total_cases']]

H_4 = df_4['H'][l-1]
H_5 = [(a+H_4) for a in df_5['H']]
isolated_4 = df_4['isolated'][l-1]
isolated_5 = [(a+isolated_4) for a in df_5['isolated']]

data = {'t': df_5['t'],
        'TI': df_5['TI'],
        'currently_infected': df_5['currently_infected'],
        'DR': DR_5,
        'recovered': recovered_5,
        'C': C_5,
        'critical': critical_5,
        'D': D_5,
        'deaths': deaths_5,
        'TI+DR+D': tot_5,
        'total_cases': total_cases_5,
        'H': H_5,
        'isolated': isolated_5,
        }
df_5 = pd.DataFrame(data)

plot_compare_df = pd.concat([df_1, df_2, df_3, df_4, df_5])

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

plot_compare(t,
             TI, currently_infected,
             DR, recovered,
             C, critical,
             D, deaths,
             tot,total_cases,
             H, isolated,
             0, len(t))
