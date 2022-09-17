"""
Latin Hypercube Sampling & Partial Rank Correlation Coefficients
"""
import argparse
import numpy as np
from scipy import special
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib import rc
from numpy.random import RandomState
import matplotlib.cm as mplcm
import matplotlib.colors as colors

rs = RandomState(0)

NUM_COLORS = 20

cm = plt.get_cmap('tab20')
cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)
scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)

dpi=300
rc('text', usetex=True)
plt.style.use('seaborn-whitegrid')
pd.plotting.register_matplotlib_converters()
plt.style.use("seaborn-ticks")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"] = 11.0
plt.rcParams["figure.figsize"] = (21, 6)

# Models
parser = argparse.ArgumentParser(description='The Baseline Epidemic Model')

### Simulation time related arguments
parser.add_argument('--horizon', type=int, default=180, help='Simulation horizon')
parser.add_argument('--step', type=float, default=0.1, help='Time-step - discretisation of the continuous-time system')

### Population
parser.add_argument('--country', type=str, default='Italy', help='Total population')

### Model parameters
parser.add_argument('--kappa', type=float, default=0.50, help='infectiousness factor asymptomatic')
parser.add_argument('--omega', type=float, default=0.0114, help='infectiousness factor quarantined')
parser.add_argument('--rho', type=float, default=0.0114, help='infectiousness factor isolated')
parser.add_argument('--sigma', type=float, default=0.223, help='transition rate exposed to infectious')
parser.add_argument('--alpha', type=float, default=0.30, help='fraction of infections that become symptomatic')
parser.add_argument('--nu', type=float, default=0.0104, help='transition rate  asymptomatic to symptomatic')
parser.add_argument('--varphi', type=float, default=0.0114, help='rate of quarantined to isolation')
parser.add_argument('--theta', type=float, default=0.01, help='rate of detection of symptomatic')
parser.add_argument('--tau', type=float, default=0.0009, help='rate of developing life-threatening symptoms in isolation')
parser.add_argument('--lamda', type=float, default=0.009, help='rate of developing life-threatening symptoms for symptomatic')
parser.add_argument('--gamma', type=float, default=0.0447, help='recovery rate of asymptomatic')
parser.add_argument('--eta', type=float, default=0.049, help='recovery rate of symptomatic')
parser.add_argument('--mu', type=float, default=0.0445, help='recovery rate of quarantined')
parser.add_argument('--psi', type=float, default=0.089, help='recovery rate of isolated')
parser.add_argument('--zeta', type=float, default=0.048, help='recovery rate of critical')
parser.add_argument('--delta', type=float, default=0.095, help='mortality rate')
args = parser.parse_args()

## Population
population = 1e5

def deriv(y, t, sampledParams, unsampledParams):
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

    E, I, A, Q, H, C, D, R, S, DR = y
    kappa, omega, rho, sigma, alpha, nu, varphi, theta, tau, lamda, gamma, eta, mu, psi, zeta, delta = unsampledParams
    beta_0, beta_min, t_0, et_0, r, epsilon_0, epsilon_max, s = sampledParams

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

# Number of parameters to sample
parameterCount = 8
# Number of samples to draw for each parameter
sampleCount = 120

def sampleDistrib(modelParamName, distrib, distribSpecs):
    samples = None
    if distrib == 'uniform':
        mmin = distribSpecs['min']['value']
        mmax = distribSpecs['max']['value']
        intervalwidth = (mmax - mmin) / sampleCount  # width of each
        # sampling interval
        samples = []
        for sample in range(sampleCount):
            lower = mmin + intervalwidth * (sample - 1)  # lb of interval
            upper = mmin + intervalwidth * (sample)  # ub of interval
            sampleVal = rs.uniform(lower, upper)  # draw a random sample
            # within the interval
            samples.append(sampleVal)
    elif distrib == 'normal':
        mmean = distribSpecs['mean']['value']
        mvar = distribSpecs['var']['value']
        lower = mvar * np.sqrt(2) * special.erfinv(-0.9999) + mmean  # set lb of 1st
        # sample interval
        samples = []
        for sample in range(sampleCount):
            n = sample + 1
            if n != sampleCount:
                upper = (np.sqrt(2 * mvar) * special.erfinv(2 * n / sampleCount - 1)
                         + mmean)  # ub of sample interval
            else:
                upper = np.sqrt(2 * mvar) * special.erfinv(0.9999) + mmean

            sampleVal = np.random.uniform(lower, upper)  # draw a random sample
            # within the interval
            samples.append(sampleVal)
            lower = upper  # set current ub as the lb for next interval
    elif distrib == 'triangle':
        mmin = distribSpecs['min']['value']
        mmax = distribSpecs['max']['value']
        mmode = distribSpecs['mode']['value']
        samples = []
        for sample in range(sampleCount):
            n = sample + 1
            intervalarea = 1 / sampleCount
            ylower = intervalarea * (n - 1)  # use cdf to read off area as y's &
            yupper = intervalarea * (n)  # get corresponding x's for the pdf
            # Check to see if y values = cdf(x <= mmode)
            # for calculating correxponding x values:
            if ylower <= ((mmode - mmin) / (mmax - mmin)):
                lower = np.sqrt(ylower * (mmax - mmin) * (mmode - mmin)) + mmin
            else:
                lower = mmax - np.sqrt((1 - ylower) * (mmax - mmin) * (mmax - mmode))
            if yupper <= ((mmode - mmin) / (mmax - mmin)):
                upper = np.sqrt(yupper * (mmax - mmin) * (mmode - mmin)) + mmin
            else:
                upper = mmax - np.sqrt((1 - yupper) * (mmax - mmin) * (mmax - mmode))
            sampleVal = np.random.uniform(lower, upper)
            samples.append(sampleVal)
    # b = int(np.ceil(sampleCount / 10))
    # plt.hist(samples, density=1, bins=b)
    #
    # B = str(b)
    #
    # plt.title('Histogram of ' + modelParamName
    #           + ' parameter samples for ' + B + ' bins')
    #
    # plt.ylabel('proportion of samples')
    # plt.xlabel(modelParamName + ' value')
    #
    # plt.show()
    return samples

params_init_min_max = {"beta_0": (1.14, 0.9, 1.5),
                       "beta_min": (0.36, 0.03, 1.5),
                       "t_0": (4, 2, 30),
                       "t_2": (4, 2, 20),
                       "r": (0.03, 0.001, 0.5),
                       "epsilon_0": (0.6, 0.54, 0.8),
                       "epsilon_max": (0.6,0.5,0.8),
                       "s":(0.32, 0.3, 0.9)
                       } # {initial, min, max}

params = [
{'result': {'Name': '$\\beta_0$', 'Dist': 'uniform'}},
{'result': {'Name': '$\\beta_{\min}$', 'Dist': 'uniform'}},
{'result': {'Name': '$t_0$', 'Dist': 'uniform'}},
{'result': {'Name': '$t_2$', 'Dist': 'uniform'}},
{'result': {'Name': '$r$', 'Dist': 'uniform'}},
{'result': {'Name': '$\\epsilon_0$', 'Dist': 'uniform'}},
{'result': {'Name': '$\\epsilon_{\max}$', 'Dist': 'uniform'}},
{'result': {'Name': '$s$', 'Dist': 'uniform'}}
]

distribSpecs={
'$\\beta_0$': {'min': {'value':0.5}, 'max': {'value':1.5}},
'$\\beta_{\min}$': {'min':{'value':0.01}, 'max':{'value':1.5}},
'$t_0$': {'min': {'value':1.0}, 'max': {'value':10.0}},
'$t_2$': {'min':{'value':1.0}, 'max':{'value':10.0}},
'$r$': {'min': {'value':0.001}, 'max': {'value':0.5}},
'$\\epsilon_0$': {'min':{'value':0.1}, 'max':{'value':0.8}},
'$\\epsilon_{\max}$': {'min': {'value':0.1}, 'max': {'value':0.8}},
'$s$': {'min':{'value':0.3}, 'max':{'value':0.9}},
}

parameters = {}
for j in range(parameterCount):
    parameters[params[j]['result']['Name']] = sampleDistrib(params[j]['result']['Name'],
                                                         params[j]['result']['Dist'],
                                                         distribSpecs[params[j]['result']['Name']])
LHSparams = []
for p in parameters:
    temp = parameters[p]
    rs.shuffle(temp)
    LHSparams.append(temp)

## Time horizon
t = np.arange(1.0,args.horizon,args.step)
LENGTH = t.shape[0]

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

odesic = [E_0, I_0, A_0, Q_0, H_0, C_0, D_0, R_0, S_0, DR_0]

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

unsampledParams = [kappa, omega, rho, sigma, alpha, nu, varphi, theta, tau, lamda, gamma, eta, mu, psi, zeta, delta]

Simdata = {}
Output = []

for i in range(sampleCount):
    Simdata[i] = {}
    Simdata[i]['TI'] = []

for j in range(sampleCount):
    sampledParams = [i[j] for i in LHSparams]
    ret = odeint(deriv, odesic, t, args=(sampledParams, unsampledParams))
    E, I, A, Q, H, C, D, R, S, DR = ret.T
    TI = Q + H + C
    Simdata[j]['TI'] = TI
    Output.append(TI)

labelstring = 'Cases (fraction of the population) '

yavg = np.mean(Output, axis=0)
yerr = np.std(Output, axis=0)
# plt.errorbar(t,yavg,yerr)
# plt.xlabel('t')
# plt.ylabel(labelstring)
# plt.title('Error bar plot of ' + labelstring + ' from LHS simulations')
# plt.show()

SampleResult = []

resultIdx = parameterCount + 1
prcc = np.zeros((resultIdx, len(t)))

LHS = [*zip(*LHSparams)]
LHSarray = np.array(LHS)
Outputarray = np.array(Output)

for xi in range(len(t)):  # loop through time or location of sim results
    xi2 = xi + 1  # to compare w/ parameter sample vals

    subOut = Outputarray[0:, xi:xi2]

    LHSout = np.hstack((LHSarray, subOut))
    SampleResult = LHSout.tolist()

    Ranks = []

    for s in range(sampleCount):

        indices = list(range(len(SampleResult[s])))
        indices.sort(key=lambda k: SampleResult[s][k])
        r = [0] * len(indices)
        for i, k in enumerate(indices):
            r[k] = i

        Ranks.append(r)

    C = np.corrcoef(Ranks)

    if np.linalg.det(C) < 1e-16:  # determine if singular
        Cinv = np.linalg.pinv(C)  # may need to use pseudo inverse
    else:
        Cinv = np.linalg.inv(C)

    for w in range(parameterCount):  # compute PRCC btwn each param & sim result
        prcc[w, xi] = -Cinv[w, resultIdx] / np.sqrt(Cinv[w, w] * Cinv[resultIdx, resultIdx])

fig, ax = plt.subplots(1)
ax.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
values = []
labels=list(parameters.keys())
for p in range(parameterCount):
    values.append(np.mean(prcc[p,]))

bars = plt.bar(labels,values,color=[scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
for bar in bars:
    yval = bar.get_height()
    if yval > 0:
        plt.text(bar.get_x(), yval + .005, str(round(yval, 5)))
    else:
        plt.text(bar.get_x(), (yval - .05), str(round(yval, 5)))
plt.ylabel('PRCC')
plt.xlabel('Parameters')
N=str(sampleCount)
plt.title('Partial rank correlation of parameters with ' + labelstring
          + ' from ' + N + ' LHS samples')
#plt.show()
plt.savefig(f'../doc/model_sensitivity_b.pdf', dpi=600)
plt.clf()