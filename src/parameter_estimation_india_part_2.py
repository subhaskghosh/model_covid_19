import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import lmfit
import argparse
from src.plots import plot_compare_india, plot_evolution_India

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
parser.add_argument('--alpha', type=float, default=0.23, help='fraction of infections that become symptomatic')
parser.add_argument('--nu', type=float, default=0.1254, help='transition rate  asymptomatic to symptomatic')
parser.add_argument('--varphi', type=float, default=0.1254, help='rate of quarantined to isolation')
parser.add_argument('--theta', type=float, default=0.371, help='rate of detection of symptomatic')
parser.add_argument('--tau', type=float, default=0.0274, help='rate of developing life-threatening symptoms in isolation')
parser.add_argument('--lamda', type=float, default=0.0171, help='rate of developing life-threatening symptoms for symptomatic')
parser.add_argument('--gamma', type=float, default=0.1, help='recovery rate of asymptomatic')
parser.add_argument('--eta', type=float, default=0.09, help='recovery rate of symptomatic')
parser.add_argument('--mu', type=float, default=0.1, help='recovery rate of quarantined')
parser.add_argument('--psi', type=float, default=0.1, help='recovery rate of isolated')
parser.add_argument('--zeta', type=float, default=0.1, help='recovery rate of critical')
parser.add_argument('--delta', type=float, default=0.008, help='mortality rate')



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
## Deaths
deaths = actuals_df["Deaths"].values.tolist()
## Recovered
recovered = actuals_df["Recovered"].values.tolist()
## Currently Infected
currently_infected = actuals_df["Currently Positive"].values.tolist()



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
            return epsilon_max - (epsilon_max - epsilon_0) * np.exp(-s * (t - et_0))

    def r_0(t):
        r_1 = (eta + theta + lamda)
        r_2 = (tau + psi)
        r_3 = (epsilon(t) + nu + gamma)
        r_4 = (varphi + mu)

        p1 = beta(t) * (((alpha) / (r_1)) + ((nu * (1 - alpha)) / (r_1 * r_3)))
        p2 = kappa * beta(t) * ((1 - alpha) / (r_3))
        p3 = omega * beta(t) * ((epsilon(t) * (1 - alpha)) / (r_3 * r_4))
        p4 = rho * beta(t) * (
                    ((alpha * theta) / (r_1 * r_2)) + (((1 - alpha) * epsilon(t) * varphi) / (r_2 * r_3 * r_4)) + (
                        ((1 - alpha) * nu * theta) / (r_1 * r_2 * r_3)))

        res = p1 + p2 + p3 + p4
        return res

    E_0 = 2.6749410756174385e-05
    I_0 = 5.277971883086385e-06
    A_0 = 6.41112627360641e-06
    Q_0 = 1.7280961313165736e-05
    H_0 = 5.120001398618197e-05
    C_0 = 2.164423916129957e-05
    D_0 = 0.00
    R_0 = 0.00
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

outbreak_shift = 396
till_day = 120
y_data = currently_infected
y_data = y_data[outbreak_shift:outbreak_shift+till_day]
days = len(y_data)
x_data = np.linspace(0, days - 1, days, dtype=int)

params_init_min_max = {"beta_0": (1.14, 0.7, 1.5),
                       "beta_min": (0.03, 0.01, 0.54),
                       "t_0": (12, 8, 40),
                       "et_0": (40, 20, 80),
                       "r": (0.03, 0.001, 0.1),
                       "epsilon_0": (0.171, 0.017, 0.4),
                       "epsilon_max": (0.6,0.5,0.8),
                       "s":(0.03, 0.001, 0.1)} # {initial, min, max}

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
# print(result.ci_report())

dely = result.eval_uncertainty(params, sigma=2)

# Plot
plt.figure(figsize=(6,8))
plt.plot(x_data,y_data,label='data')
plt.plot(x_data,result.best_fit,label='best-fit')
plt.fill_between(x_data, result.best_fit-dely, result.best_fit+dely, color="#E8E9EA",
                 label='2-$\sigma$ uncertainty band')
plt.legend()
plt.title("95\% confidence bands for the model (Italy)")
plt.xlabel('Time (days)')
plt.ylabel('Cases (fraction of the population)')
plt.ylim(-1e-2, 1e-2)
plt.show()

# Save plot data
x_mod_data = [(a+396) for a in x_data]
data = {'x_data': x_mod_data,
        'y_data': y_data,
        'best_fit': result.best_fit,
        'dely': dely}
confidence_plot_df = pd.DataFrame(data)
confidence_plot_df.to_csv(f'../data/india_confidence_plot_2.csv', index=False)
#
t, E, I, A, Q, H, C, D, R, S, DR, TI, beta_over_time, epsilon_over_time, r_not_over_time = Model(days, # change to 700 for plot_evolution_India, otherwise use days
                                                                                result.best_values['beta_0'],
                                                                                result.best_values['t_0'],
                                                                                result.best_values['beta_min'],
                                                                                result.best_values['r'],
                                                                                result.best_values['epsilon_0'],
                                                                                result.best_values['s'],
                                                                                result.best_values['epsilon_max'],
                                                                                result.best_values['et_0'])

# currently_infected_start = currently_infected[outbreak_shift-1]
# currently_infected = [(a-currently_infected_start) for a in currently_infected]

recovered_start = recovered[outbreak_shift-1]
recovered = [(a-recovered_start) for a in recovered]

deaths_start = deaths[outbreak_shift-1]
deaths = [(a-deaths_start) for a in deaths]

total_cases_start = total_cases[outbreak_shift-1]
total_cases = [(a-total_cases_start) for a in total_cases]

#Save plot data
t_mod = [(a+348) for a in t]
data = {'t': t_mod,
        'TI': TI,
        'currently_infected': currently_infected[outbreak_shift:outbreak_shift + till_day],
        'DR': DR,
        'recovered': recovered[outbreak_shift:outbreak_shift + till_day],
        'D': D,
        'deaths': deaths[outbreak_shift:outbreak_shift + till_day],
        'TI+DR+D': TI+DR+D,
        'total_cases': total_cases[outbreak_shift:outbreak_shift + till_day],
        }
plot_compare_df = pd.DataFrame(data)
plot_compare_df.to_csv(f'../data/india_plot_compare_2.csv', index=False)

plot_compare_india(t,
             TI, currently_infected,
             DR, recovered,
             D, deaths,
             TI+DR+D, total_cases,
             outbreak_shift, days)

#plot_evolution_India(t, I, A, Q, H, C, D, DR, TI, R, currently_infected, beta_over_time, epsilon_over_time)
print(f'E_0 = {E[-1]}\nI_0 = {I[-1]}\nA_0 = {A[-1]}\nQ_0 = {Q[-1]}\nH_0 = {H[-1]}\nC_0 = {C[-1]}\nD_0 = {D[-1]}\nR_0 = {R[-1]}')

