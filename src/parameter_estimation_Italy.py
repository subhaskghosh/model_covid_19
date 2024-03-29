import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import lmfit
import argparse
from src.plots import plot_compare

parser = argparse.ArgumentParser(description='The Baseline Epidemic Model')

### Simulation time related arguments
parser.add_argument('--horizon', type=int, default=208, help='Simulation horizon')
parser.add_argument('--step', type=float, default=0.01, help='Time-step - discretisation of the continuous-time system')

### Population
parser.add_argument('--country', type=str, default='Italy', help='Total population')

### Model parameters
parser.add_argument('--kappa', type=float, default=0.5, help='infectiousness factor asymptomatic')
parser.add_argument('--omega', type=float, default=0.0114, help='infectiousness factor quarantined')
parser.add_argument('--rho', type=float, default=0.0114, help='infectiousness factor isolated')
parser.add_argument('--sigma', type=float, default=0.223, help='transition rate exposed to infectious')
parser.add_argument('--alpha', type=float, default=0.85, help='fraction of infections that become symptomatic')
parser.add_argument('--nu', type=float, default=0.154, help='transition rate  asymptomatic to symptomatic')
parser.add_argument('--varphi', type=float, default=0.0527, help='rate of quarantined to isolation')
parser.add_argument('--theta', type=float, default=0.171, help='rate of detection of symptomatic')
parser.add_argument('--tau', type=float, default=0.0007, help='rate of developing life-threatening symptoms in isolation')
parser.add_argument('--lamda', type=float, default=0.0007, help='rate of developing life-threatening symptoms for symptomatic')
parser.add_argument('--gamma', type=float, default=0.0742, help='recovery rate of asymptomatic')
parser.add_argument('--eta', type=float, default=0.0171, help='recovery rate of symptomatic')
parser.add_argument('--mu', type=float, default=0.0342, help='recovery rate of quarantined')
parser.add_argument('--psi', type=float, default=0.0291, help='recovery rate of isolated')
parser.add_argument('--zeta', type=float, default=0.0291, help='recovery rate of critical')
parser.add_argument('--delta', type=float, default=0.15, help='mortality rate')


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
total_cases = [a/population for a in actuals_df["totale_casi"].values.tolist()]
## Deaths : deceduti
deaths = [a/population for a in actuals_df["deceduti"].values.tolist()]
## Recovered : dimessi_guariti
recovered = [a/population for a in actuals_df["dimessi_guariti"].values.tolist()]
## Currently Infected
currently_infected = [a/population for a in actuals_df["totale_positivi"].values.tolist()]
## Currently positive: isolated at home: isolamento_domiciliare
isolated = [a/population for a in actuals_df["isolamento_domiciliare"].values.tolist()]
## Currently positive: hospitalised : ricoverati_con_sintomi
quarantined = [a/population for a in actuals_df["ricoverati_con_sintomi"].values.tolist()]
## Currently positive: ICU: terapia_intensiva
critical = [a/population for a in actuals_df["terapia_intensiva"].values.tolist()]


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

def Model(days, beta_0, t_0, beta_min, r, epsilon_0, s, epsilon_max, et_0):

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

    E_0 = 100.0 / population
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

outbreak_shift = 190
till_day = 120
currently_infected_start = currently_infected[outbreak_shift-1]
currently_infected = [(a-currently_infected_start) for a in currently_infected]
y_data = currently_infected
y_data = y_data[outbreak_shift:outbreak_shift+till_day]
days = len(y_data)
x_data = np.linspace(0, days - 1, days, dtype=int)

params_init_min_max = {"beta_0": (1.14, 0.9, 1.5),
                       "beta_min": (0.36, 0.03, 1.5),
                       "t_0": (4, 2, 30),
                       "et_0": (4, 2, 20),
                       "r": (0.03, 0.001, 0.5),
                       "epsilon_0": (0.6, 0.54, 0.8),
                       "epsilon_max": (0.6,0.5,0.8),
                       "s":(0.32, 0.3, 0.9)
                       } # {initial, min, max}

def fitter(x, beta_0, t_0, beta_min, r, epsilon_0, s, epsilon_max, et_0):
    ret = Model(days, beta_0, t_0, beta_min, r, epsilon_0, s, epsilon_max, et_0)
    return ret[11]

mod = lmfit.Model(fitter)

for kwarg, (init, mini, maxi) in params_init_min_max.items():
    mod.set_param_hint(str(kwarg), value=init, min=mini, max=maxi, vary=True)

params = mod.make_params()
fit_method = "least_squares"

result = mod.fit(y_data, params, method="least_squares", x=x_data)

print(result.best_values)
print(result.fit_report())
#print(result.ci_report())
result.plot_fit(datafmt="-")
#result.plot_residuals(datafmt="-")
plt.show()

recovered_start = recovered[outbreak_shift-1]
recovered = [(a-recovered_start) for a in recovered]

critical_start = critical[outbreak_shift-1]
critical = [(a-critical_start) for a in critical]

deaths_start = deaths[outbreak_shift-1]
deaths = [(a-deaths_start) for a in deaths]

quarantined_start = quarantined[outbreak_shift-1]
quarantined = [(a-quarantined_start) for a in quarantined]

isolated_start = isolated[outbreak_shift-1]
isolated = [(a-isolated_start) for a in isolated]

t, E, I, A, Q, H, C, D, R, S, DR, TI, beta_over_time, epsilon_over_time, r_not_over_time = Model(days,
                                                                                result.best_values['beta_0'],
                                                                                result.best_values['t_0'],
                                                                                result.best_values['beta_min'],
                                                                                result.best_values['r'],
                                                                                result.best_values['epsilon_0'],
                                                                                result.best_values['s'],
                                                                                result.best_values['epsilon_max'],
                                                                                result.best_values['et_0'])



plot_compare(t,
             TI, currently_infected,
             DR, recovered,
             C, critical,
             D, deaths,
             Q+A, quarantined,
             H, isolated,
             outbreak_shift, till_day)
