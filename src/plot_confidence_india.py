import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import lmfit
import argparse
from src.plots import plot_evolution_vic, plot_compare_vic

#Load
df_1 = pd.read_csv(f'../data/india_confidence_plot_1.csv')
df_2 = pd.read_csv(f'../data/india_confidence_plot_2.csv')
df_3 = pd.read_csv(f'../data/india_confidence_plot_3.csv')
confidence_plot_df = pd.concat([df_1, df_2, df_3])
x_data = confidence_plot_df['x_data']
y_data = confidence_plot_df['y_data']
best_fit = confidence_plot_df['best_fit']
dely = confidence_plot_df['dely']

plt.figure(figsize=(12,8))
plt.plot(x_data, y_data, marker='o', markersize=1, label='data')
plt.plot(x_data, best_fit, '--', label='best-fit')

#result.plot_fit(datafmt="-")
plt.fill_between(x_data, best_fit-dely, best_fit+dely, color="#E8E9EA",
                 label='2-$\sigma$ uncertainty band')
plt.legend()
plt.title("95\% confidence bands for the model (India)")
plt.xlabel('Time (days)')
plt.ylabel('Cases (fraction of the population)')
plt.ylim(0, 5e-3)
#plt.show()
plt.savefig(f'../doc/India_model_confidence.pdf', dpi=600)
plt.clf()
