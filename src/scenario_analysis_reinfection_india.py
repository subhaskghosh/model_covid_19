import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import lmfit
import argparse
from src.plots import plot_scenario_reinfection_india

parser = argparse.ArgumentParser(description='The Baseline Epidemic Model')

### Simulation time related arguments
parser.add_argument('--horizon', type=int, default=208, help='Simulation horizon')
parser.add_argument('--step', type=float, default=0.01, help='Time-step - discretisation of the continuous-time system')

### Population
parser.add_argument('--country', type=str, default='India', help='Total population')

### Model parameters
parser.add_argument('--kappa', type=float, default=0.6, help='infectiousness factor asymptomatic')
parser.add_argument('--omega', type=float, default=0.0114, help='infectiousness factor quarantined')
parser.add_argument('--rho', type=float, default=0.0114, help='infectiousness factor isolated')
parser.add_argument('--sigma', type=float, default=0.1923, help='transition rate exposed to infectious')
parser.add_argument('--alpha', type=float, default=0.3, help='fraction of infections that become symptomatic')
parser.add_argument('--nu', type=float, default=0.1254, help='transition rate  asymptomatic to symptomatic')
parser.add_argument('--varphi', type=float, default=0.1254, help='rate of quarantined to isolation')
parser.add_argument('--theta', type=float, default=0.371, help='rate of detection of symptomatic')
parser.add_argument('--tau', type=float, default=0.0274, help='rate of developing life-threatening symptoms in isolation')
parser.add_argument('--lamda', type=float, default=0.0171, help='rate of developing life-threatening symptoms for symptomatic')
parser.add_argument('--gamma', type=float, default=0.06, help='recovery rate of asymptomatic')
parser.add_argument('--eta', type=float, default=0.14, help='recovery rate of symptomatic')
parser.add_argument('--mu', type=float, default=0.0742, help='recovery rate of quarantined')
parser.add_argument('--psi', type=float, default=0.073, help='recovery rate of isolated')
parser.add_argument('--zeta', type=float, default=0.078, help='recovery rate of critical')
parser.add_argument('--delta', type=float, default=0.01, help='mortality rate')



args = parser.parse_args()

kappa = args.kappa
omega = args.omega
rho = args.rho
sigma = args.sigma
alpha = args.alpha
nu = args.nu
varphi = args.varphi
theta = args.theta
tau = args.tau
lamda = args.lamda
gamma = args.gamma
eta = args.eta
mu = args.mu
psi = args.psi
zeta = args.zeta
delta = args.delta

## Time horizon
T = np.arange(1.0,args.horizon,args.step)
LENGTH = T.shape[0]

## Population
# Load population data from The World Bank Group
# https://data.worldbank.org/indicator/SP.POP.TOTL?end=2019&start=1960&view=chart&year=2019
population_df = pd.read_csv('../data/API_SP.POP.TOTL_DS2_en_csv_v2_1308146.csv', delimiter=',')
population_df = population_df.loc[:, ['Country Name', '2019']]
population = population_df[population_df["Country Name"]==args.country].values.tolist()[0][1]

## Load actual data
actuals_df = pd.read_csv(f'../data/{args.country}.csv', delimiter=',')

## Total Cases
total_cases = actuals_df["Total Cases"].values.tolist()
## Deaths : deceduti
deaths = actuals_df["Deaths"].values.tolist()
## Recovered : dimessi_guariti
recovered = actuals_df["Recovered"].values.tolist()
## Currently Infected
currently_infected = actuals_df["Currently Positive"].values.tolist()



def deriv(y, t, beta, kappa, omega, rho, sigma, alpha, nu, epsilon, varphi, theta, tau, lamda, gamma, eta, mu, psi, zeta, delta, chi):
    E, I, A, Q, H, C, D, R, S, DR, TR = y

    dEdt = beta(t) * (I + kappa * A + omega * Q + rho * H) * S - sigma * E
    dIdt = alpha * sigma * E + nu * A - ( eta + theta + lamda ) * I
    dAdt= (1 - alpha) * sigma * E - ( epsilon(t) + nu + gamma ) * A
    dQdt = epsilon(t) * A - (varphi + mu) * Q
    dHdt = theta * I + varphi * Q - (tau + psi) * H
    dCdt = tau * H + lamda * I - (delta + zeta) * C
    dDdt = delta * C
    dRdt = (eta * I + gamma * A + mu * Q + psi * H + zeta * C) - chi(t) * R
    dSdt = -beta(t) * (I + kappa * A + omega * Q + rho * H) * S + chi(t) * R
    dDRdt = ( mu * Q + psi * H + zeta * C )
    dTRdt = (eta * I + gamma * A + mu * Q + psi * H + zeta * C)

    return dEdt, dIdt, dAdt, dQdt, dHdt, dCdt, dDdt, dRdt, dSdt, dDRdt, dTRdt

def Model(days, beta_0, t_0, beta_min, r, epsilon_0, s, epsilon_max, et_0, chi_val):
    # Contact
    def beta(t):
        if t < t_0:
            return beta_0
        else:
            beta_0_now = beta_min + (beta_0 - beta_min) * np.exp(-r * (t - t_0))
            return beta_0_now

    # Testing
    def epsilon(t):
        if t < et_0:
            return epsilon_0
        else:
            return epsilon_max - (epsilon_max - epsilon_0) * np.exp(-s * (t - et_0))

    def chi(t):
        if t < 239-41:
            return chi_val
        else:
            return chi_val

    E_0 = 0.0
    I_0 = 1.0 / population
    A_0 = 200.0 / population
    Q_0 = 20.0 / population
    H_0 = 2.0 / population
    C_0 = 0.0
    D_0 = 0.0
    R_0 = 0.0
    S_0 = 1.0 - (E_0 + I_0 + A_0 + Q_0 + H_0 + C_0 + D_0 + R_0)
    DR_0 = 0.0
    TR_0 = 0.0
    y0 = E_0, I_0, A_0, Q_0, H_0, C_0, D_0, R_0, S_0, DR_0, TR_0

    t = np.linspace(0, days - 1, days)
    ret = odeint(deriv, y0, t, args=(beta, kappa, omega, rho, sigma, alpha, nu, epsilon, varphi, theta, tau, lamda, gamma, eta, mu, psi, zeta, delta, chi))
    E, I, A, Q, H, C, D, R, S, DR, TR = ret.T
    TI = Q + H + C

    beta_over_time = [beta(i) for i in range(len(t))]
    epsilon_over_time = [epsilon(i) for i in range(len(t))]

    return t, E, I, A, Q, H, C, D, R, S, DR, TI, beta_over_time, epsilon_over_time, TR

outbreak_shift = 41
till_day = 239-41

best_values = {'beta_0': 0.8090335200647616,
               't_0': 8.08386394436345,
               'beta_min': 0.43441975778861147,
               'r': 0.023768650534936653,
               'epsilon_0': 0.07734723080740596,
               's': 0.0010758136180701358,
               'epsilon_max': 0.5022871556667635,
               'et_0': 46.11587710399446}



t, E, I, A, Q, H, C, D, R, S, DR, TI, beta_over_time, epsilon_over_time, TR = Model(700,
                                                                                best_values['beta_0'],
                                                                                best_values['t_0'],
                                                                                best_values['beta_min'],
                                                                                best_values['r'],
                                                                                best_values['epsilon_0'],
                                                                                best_values['s'],
                                                                                best_values['epsilon_max'],
                                                                                best_values['et_0'],1/250)
[TI1, DR1, R1, D1, CI1, CR1] = [TI, DR, TR, D, (I + A + Q + H + C + TR), C]

t, E, I, A, Q, H, C, D, R, S, DR, TI, beta_over_time, epsilon_over_time, TR = Model(700,
                                                                                best_values['beta_0'],
                                                                                best_values['t_0'],
                                                                                best_values['beta_min'],
                                                                                best_values['r'],
                                                                                best_values['epsilon_0'],
                                                                                best_values['s'],
                                                                                best_values['epsilon_max'],
                                                                                best_values['et_0'],1/200)
[TI2, DR2, R2, D2, CI2, CR2] = [TI, DR, TR, D, (I + A + Q + H + C + TR), C]

t, E, I, A, Q, H, C, D, R, S, DR, TI, beta_over_time, epsilon_over_time, TR = Model(700,
                                                                                best_values['beta_0'],
                                                                                best_values['t_0'],
                                                                                best_values['beta_min'],
                                                                                best_values['r'],
                                                                                best_values['epsilon_0'],
                                                                                best_values['s'],
                                                                                best_values['epsilon_max'],
                                                                                best_values['et_0'],1/150)
[TI3, DR3, R3, D3, CI3, CR3] = [TI, DR, TR, D, (I + A + Q + H + C + TR), C]

t, E, I, A, Q, H, C, D, R, S, DR, TI, beta_over_time, epsilon_over_time, TR = Model(700,
                                                                                best_values['beta_0'],
                                                                                best_values['t_0'],
                                                                                best_values['beta_min'],
                                                                                best_values['r'],
                                                                                best_values['epsilon_0'],
                                                                                best_values['s'],
                                                                                best_values['epsilon_max'],
                                                                                best_values['et_0'],1/100)
[TI4, DR4, R4, D4, CI4, CR4] = [TI, DR, TR, D, (I + A + Q + H + C + TR), C]

t, E, I, A, Q, H, C, D, R, S, DR, TI, beta_over_time, epsilon_over_time, TR = Model(700,
                                                                                best_values['beta_0'],
                                                                                best_values['t_0'],
                                                                                best_values['beta_min'],
                                                                                best_values['r'],
                                                                                best_values['epsilon_0'],
                                                                                best_values['s'],
                                                                                best_values['epsilon_max'],
                                                                                best_values['et_0'],1/50)
[TI5, DR5, R5, D5, CI5, CR5] = [TI, DR, TR, D, (I + A + Q + H + C + TR), C]


plot_scenario_reinfection_india(t,
                            TI1, DR1, R1, D1, CI1, CR1,
                            TI2, DR2, R2, D2, CI2, CR2,
                            TI3, DR3, R3, D3, CI3, CR3,
                            TI4, DR4, R4, D4, CI4, CR4,
                            TI5, DR5, R5, D5, CI5, CR5,
                            currently_infected)