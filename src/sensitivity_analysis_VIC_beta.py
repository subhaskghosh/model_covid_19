import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import lmfit
import argparse
from src.plots import plot_sensitivity_beta_vic

parser = argparse.ArgumentParser(description='The Baseline Epidemic Model')

### Simulation time related arguments
parser.add_argument('--horizon', type=int, default=208, help='Simulation horizon')
parser.add_argument('--step', type=float, default=0.01, help='Time-step - discretisation of the continuous-time system')

### Population
parser.add_argument('--country', type=str, default='Italy', help='Total population')

### Model parameters
parser.add_argument('--kappa', type=float, default=0.52, help='infectiousness factor asymptomatic')
parser.add_argument('--omega', type=float, default=0.0114, help='infectiousness factor quarantined')
parser.add_argument('--rho', type=float, default=0.0114, help='infectiousness factor isolated')
parser.add_argument('--sigma', type=float, default=0.1923, help='transition rate exposed to infectious')
parser.add_argument('--alpha', type=float, default=0.30, help='fraction of infections that become symptomatic')
parser.add_argument('--nu', type=float, default=0.0104, help='transition rate  asymptomatic to symptomatic')
parser.add_argument('--varphi', type=float, default=0.0114, help='rate of quarantined to isolation')
parser.add_argument('--theta', type=float, default=0.01, help='rate of detection of symptomatic')
parser.add_argument('--tau', type=float, default=0.0009, help='rate of developing life-threatening symptoms in isolation')
parser.add_argument('--lamda', type=float, default=0.009, help='rate of developing life-threatening symptoms for symptomatic')
parser.add_argument('--gamma', type=float, default=0.0447, help='recovery rate of asymptomatic')
parser.add_argument('--eta', type=float, default=0.049, help='recovery rate of symptomatic')
parser.add_argument('--mu', type=float, default=0.0442, help='recovery rate of quarantined')
parser.add_argument('--psi', type=float, default=0.043, help='recovery rate of isolated')
parser.add_argument('--zeta', type=float, default=0.048, help='recovery rate of critical')
parser.add_argument('--delta', type=float, default=0.065, help='mortality rate')


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

population = 6629870

## Load actual data
actuals_df = pd.read_csv(f'../data/Victoria.csv', delimiter=',')

## Total Cases
total_cases = actuals_df["Total Cases"].values.tolist()

## Deaths
deaths = actuals_df["Total Deaths"].values.tolist()

## Recovered
recovered = actuals_df["Total Recovered"].values.tolist()

## Currently Infected
currently_infected = actuals_df["Currently Positive"].values.tolist()

## Currently positive: hospitalised
isolated = actuals_df["Currently Hospitalized"].values.tolist()

## Currently positive: ICU
critical = actuals_df["Currently Critical"].values.tolist()


def deriv(y, t, beta, kappa, omega, rho, sigma, alpha, nu, epsilon, varphi, theta, tau, lamda, gamma, eta, mu, psi, zeta, delta):
    E, I, A, Q, H, C, D, R, S, DR = y

    dEdt = beta(t) * (I + kappa * A + omega * Q + rho * H) * S - sigma * E
    dIdt = alpha * sigma * E + nu * A - ( eta + theta + lamda ) * I
    dAdt= (1 - alpha) * sigma * E - ( epsilon(t) + nu + gamma ) * A
    dQdt = epsilon(t) * A - (varphi + mu) * Q
    dHdt = theta * I + varphi * Q - (tau + psi) * H
    dCdt = tau * H + lamda * I - (delta + zeta) * C
    dDdt = delta * C
    dRdt = (eta * I + gamma * A + mu * Q + psi * H + zeta * C)
    dSdt = -beta(t) * (I + kappa * A + omega * Q + rho * H) * S
    dDRdt = ( mu * Q + psi * H + zeta * C )

    return dEdt, dIdt, dAdt, dQdt, dHdt, dCdt, dDdt, dRdt, dSdt, dDRdt

def Model(days, beta_0, t_0, beta_min, r, epsilon_0, s, epsilon_max, et_0, beta_new, u, t_1, beta_min_1, t_2, v, m):

    # Contact
    def beta(t):
        if t < t_0:
            return beta_0
        else:
            if t < t_1:
                beta_0_now = beta_min + (beta_0 - beta_min) * np.exp(-r * (t - t_0))
                return beta_0_now
            else:
                if t < t_2:
                    curr = beta_min + (beta_0 - beta_min) * np.exp(-r * (t - t_0))
                    beta_0_now = beta_new - (beta_new - curr) * np.exp(-u * (t - t_1))
                    return beta_0_now
                else:
                    if t < 195:
                        curr = beta_min + (beta_0 - beta_min) * np.exp(-r * (t - t_0))
                        beta_0_now = beta_new - (beta_new - curr) * np.exp(-u * (t - t_1))
                        beta_1_now = beta_min_1 + (beta_0_now - beta_min_1) * np.exp(-v * (t - t_2))
                        return beta_1_now
                    else:
                        curr = beta_min + (beta_0 - beta_min) * np.exp(-r * (t - t_0))
                        beta_0_now = beta_new - (beta_new - curr) * np.exp(-u * (t - t_1))
                        beta_1_now = beta_min_1 + (beta_0_now - beta_min_1) * np.exp(-v * (t - t_2))
                        return beta_1_now*m

    # Testing
    def epsilon(t):
        if t < et_0:
            return epsilon_0
        else:
            return epsilon_max - (epsilon_max - epsilon_0) * np.exp(-s * (t-et_0))

    def r_0(t):
        r_1 = (eta + theta + lamda)
        r_2 = (tau + psi)
        r_3 = (epsilon(t) + nu + gamma)
        r_4 = (varphi + mu)

        p1 = beta(t) * (((alpha) / (r_1)) + ((nu * (1 - alpha)) / (r_1 * r_3)))
        p2 = kappa * beta(t) * ((1 - alpha) / (r_3))
        p3 = omega * beta(t) * ((epsilon(t) * (1 - alpha)) / (r_3 * r_4))
        p4 = rho * beta(t) * (((alpha * theta) / (r_1 * r_2)) + (((1 - alpha) * epsilon(t) * varphi) / (r_2 * r_3 * r_4)) + (((1 - alpha) * nu * theta) / (r_1 * r_2 * r_3)))

        res = p1 + p2 + p3 + p4
        return res

    E_0 = 0.0
    I_0 = 100.0 / population
    A_0 = 300.0 / population
    Q_0 = 101.0 / population
    H_0 = 200.0 / population
    C_0 = 30.0 / population
    D_0 = 7.0 / population
    R_0 = 1.0 / population
    S_0 = 1.0 - (E_0 + I_0 + A_0 + Q_0 + H_0 + C_0 + D_0 + R_0)
    DR_0 = 0.0
    y0 = E_0, I_0, A_0, Q_0, H_0, C_0, D_0, R_0, S_0, DR_0

    t = np.linspace(0, days - 1, days)
    ret = odeint(deriv, y0, t, args=(beta, kappa, omega, rho, sigma, alpha, nu, epsilon, varphi, theta, tau, lamda, gamma, eta, mu, psi, zeta, delta))
    E, I, A, Q, H, C, D, R, S, DR = ret.T
    TI = Q + H + C
    beta_over_time = [beta(i) for i in range(len(t))]
    epsilon_over_time = [epsilon(i) for i in range(len(t))]
    r_not_over_time = [r_0(i) for i in range(len(t))]

    return t, E, I, A, Q, H, C, D, R, S, DR, TI, beta_over_time, epsilon_over_time, r_not_over_time

outbreak_shift = 34
till_day = 195

best_values = {'beta_0': 0.9000325863474273,
                       't_0': 1.6120659630068177,
                       'beta_min': 0.021026213826081688,
                       'r': 0.11770395332098955,
                       'epsilon_0': 0.33462372037335647,
                       's': 0.01961364585462267,
                       'epsilon_max': 0.930054476535794,
                       'et_0': 31.930766946335567,
                       'beta_new': 0.7461336926939366,
                       'u': 0.020366625695670458,
                       't_1': 64.46255630871755,
                       'beta_min_1': 0.0012911206978629684,
                       't_2': 147.59568045911854,
                       'v': 0.19178052213885108}


t, E, I, A, Q, H, C, D, R, S, DR, TI, beta_over_time, epsilon_over_time, r_not_over_time = Model(350,
                                                                                best_values['beta_0'],
                                                                                best_values['t_0'],
                                                                                best_values['beta_min'],
                                                                                best_values['r'],
                                                                                best_values['epsilon_0'],
                                                                                best_values['s'],
                                                                                best_values['epsilon_max'],
                                                                                best_values['et_0'],
                                                                                best_values['beta_new'],
                                                                                best_values['u'],
                                                                                best_values['t_1'],
                                                                                best_values['beta_min_1'],
                                                                                best_values['t_2'],
                                                                                best_values['v'],0.5)
[TI1, DR1, R1, D1, CI1, CR1] = [TI, DR, R, D, (I + A + Q + H + C + R), C]

t, E, I, A, Q, H, C, D, R, S, DR, TI, beta_over_time, epsilon_over_time, r_not_over_time = Model(350,
                                                                                best_values['beta_0'],
                                                                                best_values['t_0'],
                                                                                best_values['beta_min'],
                                                                                best_values['r'],
                                                                                best_values['epsilon_0'],
                                                                                best_values['s'],
                                                                                best_values['epsilon_max'],
                                                                                best_values['et_0'],
                                                                                best_values['beta_new'],
                                                                                best_values['u'],
                                                                                best_values['t_1'],
                                                                                best_values['beta_min_1'],
                                                                                best_values['t_2'],
                                                                                best_values['v'],0.6)
[TI2, DR2, R2, D2, CI2, CR2] = [TI, DR, R, D, (I + A + Q + H + C + R), C]

t, E, I, A, Q, H, C, D, R, S, DR, TI, beta_over_time, epsilon_over_time, r_not_over_time = Model(350,
                                                                                best_values['beta_0'],
                                                                                best_values['t_0'],
                                                                                best_values['beta_min'],
                                                                                best_values['r'],
                                                                                best_values['epsilon_0'],
                                                                                best_values['s'],
                                                                                best_values['epsilon_max'],
                                                                                best_values['et_0'],
                                                                                best_values['beta_new'],
                                                                                best_values['u'],
                                                                                best_values['t_1'],
                                                                                best_values['beta_min_1'],
                                                                                best_values['t_2'],
                                                                                best_values['v'],0.7)
[TI3, DR3, R3, D3, CI3, CR3] = [TI, DR, R, D, (I + A + Q + H + C + R), C]

t, E, I, A, Q, H, C, D, R, S, DR, TI, beta_over_time, epsilon_over_time, r_not_over_time = Model(350,
                                                                                best_values['beta_0'],
                                                                                best_values['t_0'],
                                                                                best_values['beta_min'],
                                                                                best_values['r'],
                                                                                best_values['epsilon_0'],
                                                                                best_values['s'],
                                                                                best_values['epsilon_max'],
                                                                                best_values['et_0'],
                                                                                best_values['beta_new'],
                                                                                best_values['u'],
                                                                                best_values['t_1'],
                                                                                best_values['beta_min_1'],
                                                                                best_values['t_2'],
                                                                                best_values['v'],0.8)
[TI4, DR4, R4, D4, CI4, CR4] = [TI, DR, R, D, (I + A + Q + H + C + R), C]

t, E, I, A, Q, H, C, D, R, S, DR, TI, beta_over_time, epsilon_over_time, r_not_over_time = Model(350,
                                                                                best_values['beta_0'],
                                                                                best_values['t_0'],
                                                                                best_values['beta_min'],
                                                                                best_values['r'],
                                                                                best_values['epsilon_0'],
                                                                                best_values['s'],
                                                                                best_values['epsilon_max'],
                                                                                best_values['et_0'],
                                                                                best_values['beta_new'],
                                                                                best_values['u'],
                                                                                best_values['t_1'],
                                                                                best_values['beta_min_1'],
                                                                                best_values['t_2'],
                                                                                best_values['v'],0.9)
[TI5, DR5, R5, D5, CI5, CR5] = [TI, DR, R, D, (I + A + Q + H + C + R), C]


plot_sensitivity_beta_vic(t,
                            TI1, DR1, R1, D1, CI1, CR1,
                            TI2, DR2, R2, D2, CI2, CR2,
                            TI3, DR3, R3, D3, CI3, CR3,
                            TI4, DR4, R4, D4, CI4, CR4,
                            TI5, DR5, R5, D5, CI5, CR5,
                            currently_infected)