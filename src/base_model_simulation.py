import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='The Baseline Epidemic Model')

### Simulation time related arguments
parser.add_argument('--horizon', type=int, default=365, help='Simulation horizon')
parser.add_argument('--step', type=float, default=0.01, help='Time-step - discretisation of the continuous-time system')

### Population
parser.add_argument('--country', type=str, default='Italy', help='Total population')

### Model parameters
parser.add_argument('--beta_0', type=float, default=1.5, help='infectious contact rate')
parser.add_argument('--beta_min', type=float, default=0.35, help='infectious contact rate minimum')
parser.add_argument('--t_0', type=int, default=22, help='social distanceing and lockdown start day')
parser.add_argument('--r', type=float, default=0.03, help='beta_0 decay rate')
parser.add_argument('--kappa', type=float, default=0.5, help='infectiousness factor asymptomatic')
parser.add_argument('--omega', type=float, default=0.0114, help='infectiousness factor quarantined')
parser.add_argument('--rho', type=float, default=0.0114, help='infectiousness factor isolated')
parser.add_argument('--sigma', type=float, default=0.1923, help='transition rate exposed to infectious')
parser.add_argument('--alpha', type=float, default=0.95, help='fraction of infections that become symptomatic')
parser.add_argument('--nu', type=float, default=0.1254, help='transition rate  asymptomatic to symptomatic')
parser.add_argument('--epsilon_0', type=float, default=0.171, help='detection rate asymptomatic')
parser.add_argument('--varphi', type=float, default=0.1254, help='rate of quarantined to isolation')
parser.add_argument('--theta', type=float, default=0.371, help='rate of detection of symptomatic')
parser.add_argument('--tau', type=float, default=0.0274, help='rate of developing life-threatening symptoms in isolation')
parser.add_argument('--lamda', type=float, default=0.0171, help='rate of developing life-threatening symptoms for symptomatic')
parser.add_argument('--gamma', type=float, default=0.0342, help='recovery rate of asymptomatic')
parser.add_argument('--eta', type=float, default=0.0171, help='recovery rate of symptomatic')
parser.add_argument('--mu', type=float, default=0.0342, help='recovery rate of quarantined')
parser.add_argument('--psi', type=float, default=0.0171, help='recovery rate of isolated')
parser.add_argument('--zeta', type=float, default=0.0171, help='recovery rate of critical')
parser.add_argument('--delta', type=float, default=0.01, help='mortality rate')

args = parser.parse_args()

beta_0 = args.beta_0
beta_min = args.beta_min
t_0 = args.t_0
r = args.r
kappa = args.kappa
omega = args.omega
rho = args.rho
sigma = args.sigma
alpha = args.alpha
nu = args.nu
epsilon_0 = args.epsilon_0
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
actuals_df = actuals_df.drop(actuals_df[actuals_df["Total Cases"] <= 1.0282367749346921e-06].index)

total_cases = actuals_df["Total Cases"].values.tolist()
deaths = actuals_df["Deaths"].values.tolist()
recovered = actuals_df["Recovered"].values.tolist()
currently_infected = actuals_df["Currently Positive"].values.tolist()



## States over time
E = np.zeros(LENGTH)
I = np.zeros(LENGTH)
A = np.zeros(LENGTH)
Q = np.zeros(LENGTH)
H = np.zeros(LENGTH)
C = np.zeros(LENGTH)
D = np.zeros(LENGTH)
R = np.zeros(LENGTH)
S = np.zeros(LENGTH)

## Extra states

# Diagnosed and Recovered
DR = np.zeros(LENGTH)

# Total Infected
TI = np.zeros(LENGTH)


## INITIAL CONDITIONS
E[0] = 0.0
I[0] = 1.0/population
A[0] = 200.0/population
Q[0] = 20.0/population
H[0] = 2.0/population
C[0] = 0.0
D[0] = 0.0
R[0] = 0.0
S[0] = 1.0 - (E[0] + I[0] + A[0] + Q[0] + H[0] + C[0] + D[0] + R[0])
DR[0] = 0.0
TI[0] = Q[0] + H[0] + C[0]

## State vector
X = np.array([[E[0]],
               [I[0]],
               [A[0]],
               [Q[0]],
               [H[0]],
               [C[0]],
               [D[0]],
               [R[0]],
               [S[0]],
               [DR[0]]
               ])

## SIMULATION

def update_beta(beta_0, i):
    if i < t_0/args.step:
        return beta_0
    else:
        t = i*args.step
        beta_0 = beta_min + (beta_0 - beta_min) * np.exp(-r * (t - t_0))
        return beta_0

for i in range(1,LENGTH):
    #beta_0 = update_beta(beta_0,i)

    f = np.array([[-sigma, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, beta_0 * (X[1] + X[2] * kappa + X[3] * omega + X[4] * rho)[0], 0.0],
                   [alpha * sigma, -(eta + theta + lamda), nu, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [(1 - alpha) * sigma, 0.0, -(epsilon_0 + nu + gamma), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, epsilon_0, -(varphi + mu), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, theta, 0.0, varphi, -(tau + psi), 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, lamda, 0.0, 0.0, tau, -(delta + zeta), 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, delta, 0.0, 0.0, 0.0, 0.0],
                   [0.0, eta, gamma, mu, psi, zeta, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -beta_0 * (X[1] + X[2] * kappa + X[3] * omega + X[4] * rho)[0], 0.0],
                   [0.0, 0.0, 0.0, mu, psi, zeta, 0.0, 0.0, 0.0, 0.0]
                   ])
    X = X + f.dot(X) * args.step

    ## Update variables
    E[i] = X[0]
    I[i] = X[1]
    A[i] = X[2]
    Q[i] = X[3]
    H[i] = X[4]
    C[i] = X[5]
    D[i] = X[6]
    R[i] = X[7]
    S[i] = X[8]
    DR[i] = X[9]
    TI[i] = Q[i] + H[i] + C[i]


import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt

dpi=300
#plt.style.use('fivethirtyeight')
pd.plotting.register_matplotlib_converters()
plt.style.use("seaborn")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"] = 11.0
plt.rcParams["figure.figsize"] = (12, 6)

plt.plot(T,TI,label='Currently Infected')
#plt.plot(T,C,label='Critical')
plt.plot(T,D,label='Deaths')
#plt.plot(T,R,label='Recovered')
#plt.plot(T,DR,label='Diagnosed and Recovered')
#plt.plot(recovered,label='Actual Recovered Cases')
plt.plot(deaths,label='Actual Deaths')
plt.plot(currently_infected,label='Actual Currently Infected')

plt.xlabel('Time (days)')
plt.ylabel('Cases (fraction of the population)')
plt.legend()
plt.show()


