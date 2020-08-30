from matplotlib import rc
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as mtick

dpi=300
rc('text', usetex=True)
plt.style.use('seaborn-whitegrid')
pd.plotting.register_matplotlib_converters()
plt.style.use("seaborn-ticks")
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["font.size"] = 11.0
plt.rcParams["figure.figsize"] = (8, 6)

# Model simulation compared to real data.
def plot_compare(t,
                 TI, currently_infected,
                 R, recovered,
                 C, critical,
                 D, deaths,
                 Q, quarantined,
                 H, isolated,
                 outbreak_shift, till_day):

    fig, axes = plt.subplots(3,2, figsize = (12,12))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    # 'Currently Infected: Model vs. Data'
    (markers, stemlines, baseline) = axes[0, 0].stem(t, currently_infected,
                                              label='Actual Currently Infected', linefmt='orange', basefmt=" ",
                                              use_line_collection=True)
    plt.setp(markers, marker='o', markersize=6, markeredgecolor="orange", markeredgewidth=0.5, markerfacecolor=(0, 0, 0, 0.0))
    axes[0, 0].plot(t, TI, label='Currently Infected', color='#11B0DB')
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Cases (fraction of the population)')
    axes[0, 0].set_title('Currently Infected: Model vs. Data')
    axes[0, 0].set_xlim(outbreak_shift,till_day)
    axes[0, 0].set_ylim(0, 5e-3)
    axes[0, 0].text(-0.09, 1.15, 'a', transform=axes[0, 0].transAxes, size=16, weight='bold')

    # 'Recovered: Model vs. Data'
    (markers, stemlines, baseline) = axes[0, 1].stem(t, recovered,
                                                     label='Actual Recovered', linefmt='orange', basefmt=" ",
                                                     use_line_collection=True)
    plt.setp(markers, marker='o', markersize=6, markeredgecolor="orange", markeredgewidth=0.5,
             markerfacecolor=(0, 0, 0, 0.0))
    axes[0, 1].plot(t, R, label='Recovered', color='#11B0DB')
    axes[0, 1].set_xlabel('Time (days)')
    axes[0, 1].set_ylabel('Cases (fraction of the population)')
    axes[0, 1].set_title('Recovered: Model vs. Data')
    axes[0, 1].set_xlim(outbreak_shift, till_day)
    axes[0, 1].set_ylim(0, 5e-3)
    axes[0, 1].text(-0.1, 1.15, 'b', transform=axes[0, 1].transAxes, size=16, weight='bold')

    # 'Infected, Life-Threatening Symptoms: Model vs. Data'
    (markers, stemlines, baseline) = axes[1, 0].stem(t, critical,
                                                     label='Actual Recovered', linefmt='orange', basefmt=" ",
                                                     use_line_collection=True)
    plt.setp(markers, marker='o', markersize=6, markeredgecolor="orange", markeredgewidth=0.5,
             markerfacecolor=(0, 0, 0, 0.0))
    axes[1, 0].plot(t, C, label='Recovered', color='#11B0DB')
    axes[1, 0].set_xlabel('Time (days)')
    axes[1, 0].set_ylabel('Cases (fraction of the population)')
    axes[1, 0].set_title('Infected, Life-Threatening Symptoms: Model vs. Data')
    axes[1, 0].set_xlim(outbreak_shift, till_day)
    axes[1, 0].set_ylim(0, 2.5e-3)
    axes[1, 0].text(-0.1, 1.15, 'c', transform=axes[1, 0].transAxes, size=16, weight='bold')

    # 'Deaths: Model vs. Data - NOTE: EXCLUDED FROM FITTING'
    (markers, stemlines, baseline) = axes[1, 1].stem(t, deaths,
                                                     label='Actual Recovered', linefmt='orange', basefmt=" ",
                                                     use_line_collection=True)
    plt.setp(markers, marker='o', markersize=6, markeredgecolor="orange", markeredgewidth=0.5,
             markerfacecolor=(0, 0, 0, 0.0))
    axes[1, 1].plot(t, D, label='Recovered', color='#11B0DB')
    axes[1, 1].set_xlabel('Time (days)')
    axes[1, 1].set_ylabel('Cases (fraction of the population)')
    axes[1, 1].set_title('Deaths: Model vs. Data')
    axes[1, 1].set_xlim(outbreak_shift, till_day)
    axes[1, 1].set_ylim(0, 5e-3)
    axes[1, 1].text(-0.1, 1.15, 'd', transform=axes[1, 1].transAxes, size=16, weight='bold')

    # 'Infected, No Symptoms: Model vs. Data'
    (markers, stemlines, baseline) = axes[2, 0].stem(t, quarantined,
                                                     label='Actual Recovered', linefmt='orange', basefmt=" ",
                                                     use_line_collection=True)
    plt.setp(markers, marker='o', markersize=6, markeredgecolor="orange", markeredgewidth=0.5,
             markerfacecolor=(0, 0, 0, 0.0))
    axes[2, 0].plot(t, Q, label='Recovered', color='#11B0DB')
    axes[2, 0].set_xlabel('Time (days)')
    axes[2, 0].set_ylabel('Cases (fraction of the population)')
    axes[2, 0].set_title('Infected, No Symptoms: Model vs. Data')
    axes[2, 0].set_xlim(outbreak_shift, till_day)
    axes[2, 0].set_ylim(0, 5e-3)
    axes[2, 0].text(-0.1, 1.15, 'e', transform=axes[2, 0].transAxes, size=16, weight='bold')

    # 'Infected, Symptoms: Model vs. Data'
    (markers, stemlines, baseline) = axes[2, 1].stem(t, isolated,
                                                     label='Actual Recovered', linefmt='orange', basefmt=" ",
                                                     use_line_collection=True)
    plt.setp(markers, marker='o', markersize=6, markeredgecolor="orange", markeredgewidth=0.5,
             markerfacecolor=(0, 0, 0, 0.0))
    axes[2, 1].plot(t, H, label='Recovered', color='#11B0DB')
    axes[2, 1].set_xlabel('Time (days)')
    axes[2, 1].set_ylabel('Cases (fraction of the population)')
    axes[2, 1].set_title('Infected, Symptoms: Model vs. Data')
    axes[2, 1].set_xlim(outbreak_shift, till_day)
    axes[2, 1].set_ylim(0, 5e-3)
    axes[2, 1].text(-0.1, 1.15, 'f', transform=axes[2, 1].transAxes, size=16, weight='bold')

    fig.tight_layout()
    # plt.show()

    plt.savefig(f'../doc/Italy_plot_model_data.pdf', dpi=600)
    plt.clf()

def plot_evolution(t, I, A, Q, H, C, D, DR, TI, R, currently_infected, beta_over_time, epsilon_over_time):
    short = 219
    long = 700
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    axes[0, 0].plot(t, R, label='Recovered', color='#46B39D')
    axes[0, 0].plot(t, D, label='Deaths', color='#334752')
    axes[0, 0].plot(t, DR, label='Diagnosed recovered', color='#F0CA4D')
    axes[0, 0].plot(t, I + A + Q + H + C + R, label='Cumulative infected', color='#E37332')
    axes[0, 0].plot(t, TI, label='Current total infected', color='#829FD9')
    axes[0, 0].plot(t[:186], currently_infected, label='Current total infected: Data', color='#000000')

    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Cases (fraction of the population)')
    axes[0, 0].set_title('Short-term evolution: actual vs diagnosed cases of infection')
    axes[0, 0].set_xlim(0, short)
    axes[0, 0].set_ylim(0, 5e-3)
    axes[0, 0].text(-0.1, 1.15, 'a', transform=axes[0, 0].transAxes, size=16, weight='bold')
    axes[0, 0].legend(loc='upper left')

    axes[0, 1].plot(t, R, label='Recovered', color='#46B39D')
    axes[0, 1].plot(t, D, label='Deaths', color='#334752')
    axes[0, 1].plot(t, DR, label='Diagnosed recovered', color='#F0CA4D')
    axes[0, 1].plot(t, I+A+Q+H+C+R, label='Cumulative infected', color='#E37332')
    axes[0, 1].plot(t, TI, label='Current total infected', color='#829FD9')
    axes[0, 1].plot(t[:186], currently_infected, label='Current total infected: Data', color='#000000')

    axes[0, 1].set_xlabel('Time (days)')
    axes[0, 1].set_ylabel('Cases (fraction of the population)')
    axes[0, 1].set_title('Long-term evolution: actual vs diagnosed cases of infection')
    axes[0, 1].set_xlim(0, long)
    axes[0, 1].set_ylim(0, 6e-1)
    axes[0, 1].text(-0.1, 1.15, 'b', transform=axes[0, 1].transAxes, size=16, weight='bold')
    axes[0, 1].legend(loc='upper left')

    axes[1, 0].plot(t, I, label='Symptomatic, Undected', color='#46B39D')
    axes[1, 0].plot(t, H, label='Symptomatic, Diagnosed', color='#334752')
    axes[1, 0].plot(t, A, label='Asymptomatic, Undected', color='#F0CA4D')
    axes[1, 0].plot(t, Q, label='Isolated', color='#E37332')
    axes[1, 0].plot(t, C, label='Critical', color='#829FD9')

    axes[1, 0].set_xlabel('Time (days)')
    axes[1, 0].set_ylabel('Cases (fraction of the population)')
    axes[1, 0].set_title('Short-term evolution: Infected sub-population')
    axes[1, 0].set_xlim(0, short)
    axes[1, 0].set_ylim(0, 5e-3)
    axes[1, 0].text(-0.1, 1.15, 'c', transform=axes[1, 0].transAxes, size=16, weight='bold')
    axes[1, 0].legend(loc='upper left')

    axes[1, 1].plot(t, I, label='Symptomatic, Undected', color='#46B39D')
    axes[1, 1].plot(t, H, label='Symptomatic, Diagnosed', color='#334752')
    axes[1, 1].plot(t, A, label='Asymptomatic, Undected', color='#F0CA4D')
    axes[1, 1].plot(t, Q, label='Isolated', color='#E37332')
    axes[1, 1].plot(t, C, label='Critical', color='#829FD9')

    axes[1, 1].set_xlabel('Time (days)')
    axes[1, 1].set_ylabel('Cases (fraction of the population)')
    axes[1, 1].set_title('Long-term evolution: Infected sub-population')
    axes[1, 1].set_xlim(0, long)
    axes[1, 1].set_ylim(0, 2.5e-1)
    axes[1, 1].text(-0.1, 1.15, 'd', transform=axes[1, 1].transAxes, size=16, weight='bold')
    axes[1, 1].legend(loc='upper left')

    fig.tight_layout()
    #plt.show()

    plt.savefig(f'../doc/Italy_evolution.pdf', dpi=600)
    plt.clf()

def plot_compare_india(t,
                 TI, currently_infected,
                 R, recovered,
                 D, deaths,
                 outbreak_shift, till_day):

    fig, axes = plt.subplots(3,1, figsize = (5,12))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    # 'Currently Infected: Model vs. Data'
    (markers, stemlines, baseline) = axes[0].stem(t, currently_infected[outbreak_shift:outbreak_shift + till_day],
                                              label='Actual Currently Infected', linefmt='orange', basefmt=" ",
                                              use_line_collection=True)
    plt.setp(markers, marker='o', markersize=6, markeredgecolor="orange", markeredgewidth=0.5, markerfacecolor=(0, 0, 0, 0.0))
    axes[0].plot(t, TI, label='Currently Infected', color='#11B0DB')
    axes[0].set_xlabel('Time (days)')
    axes[0].set_ylabel('Cases (fraction of the population)')
    axes[0].set_title('Currently Infected: Model vs. Data')
    axes[0].set_xlim(outbreak_shift,till_day)
    axes[0].set_ylim(0, 1e-3)
    axes[0].text(-0.09, 1.15, 'd', transform=axes[0].transAxes, size=16, weight='bold')

    # 'Recovered: Model vs. Data'
    (markers, stemlines, baseline) = axes[1].stem(t, recovered[outbreak_shift:outbreak_shift + till_day],
                                                     label='Actual Recovered', linefmt='orange', basefmt=" ",
                                                     use_line_collection=True)
    plt.setp(markers, marker='o', markersize=6, markeredgecolor="orange", markeredgewidth=0.5,
             markerfacecolor=(0, 0, 0, 0.0))
    axes[1].plot(t, R, label='Recovered', color='#11B0DB')
    axes[1].set_xlabel('Time (days)')
    axes[1].set_ylabel('Cases (fraction of the population)')
    axes[1].set_title('Recovered: Model vs. Data')
    axes[1].set_xlim(outbreak_shift, till_day)
    axes[1].set_ylim(0, 2e-3)
    axes[1].text(-0.1, 1.15, 'e', transform=axes[1].transAxes, size=16, weight='bold')

    # 'Deaths: Model vs. Data - NOTE: EXCLUDED FROM FITTING'
    (markers, stemlines, baseline) = axes[2].stem(t, deaths[outbreak_shift:outbreak_shift + till_day],
                                                     label='Actual Recovered', linefmt='orange', basefmt=" ",
                                                     use_line_collection=True)
    plt.setp(markers, marker='o', markersize=6, markeredgecolor="orange", markeredgewidth=0.5,
             markerfacecolor=(0, 0, 0, 0.0))
    axes[2].plot(t, D, label='Recovered', color='#11B0DB')
    axes[2].set_xlabel('Time (days)')
    axes[2].set_ylabel('Cases (fraction of the population)')
    axes[2].set_title('Deaths: Model vs. Data')
    axes[2].set_xlim(outbreak_shift, till_day)
    axes[2].set_ylim(0, 2.5e-4)
    axes[2].text(-0.1, 1.15, 'f', transform=axes[2].transAxes, size=16, weight='bold')

    fig.tight_layout()
    plt.show()

    #plt.savefig(f'../doc/India_plot_model_data.pdf', dpi=600)
    #plt.clf()

def plot_evolution_India(t, I, A, Q, H, C, D, DR, TI, R, currently_infected, beta_over_time, epsilon_over_time):
    short = 90
    long = 700
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    axes[0, 0].plot(t, R, label='Recovered', color='#46B39D')
    axes[0, 0].plot(t, D, label='Deaths', color='#334752')
    axes[0, 0].plot(t, DR, label='Diagnosed recovered', color='#F0CA4D')
    axes[0, 0].plot(t, I + A + Q + H + C + R, label='Cumulative infected', color='#E37332')
    axes[0, 0].plot(t, TI, label='Current total infected', color='#829FD9')

    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Cases (fraction of the population)')
    axes[0, 0].set_title('Short-term evolution: actual vs diagnosed cases of infection')
    axes[0, 0].set_xlim(0, short)
    axes[0, 0].set_ylim(0, 2.5e-4)
    axes[0, 0].text(-0.1, 1.15, 'a', transform=axes[0, 0].transAxes, size=16, weight='bold')
    axes[0, 0].legend(loc='upper left')

    axes[0, 1].plot(t, R, label='Recovered', color='#46B39D')
    axes[0, 1].plot(t, D, label='Deaths', color='#334752')
    axes[0, 1].plot(t, DR, label='Diagnosed recovered', color='#F0CA4D')
    axes[0, 1].plot(t, I+A+Q+H+C+R, label='Cumulative infected', color='#E37332')
    axes[0, 1].plot(t, TI, label='Current total infected', color='#829FD9')

    axes[0, 1].set_xlabel('Time (days)')
    axes[0, 1].set_ylabel('Cases (fraction of the population)')
    axes[0, 1].set_title('Long-term evolution: actual vs diagnosed cases of infection')
    axes[0, 1].set_xlim(0, long)
    axes[0, 1].set_ylim(0, 5e-1)
    axes[0, 1].text(-0.1, 1.15, 'b', transform=axes[0, 1].transAxes, size=16, weight='bold')
    axes[0, 1].legend(loc='upper left')

    axes[1, 0].plot(t, I, label='Symptomatic, Undected', color='#46B39D')
    axes[1, 0].plot(t, H, label='Symptomatic, Diagnosed', color='#334752')
    axes[1, 0].plot(t, A, label='Asymptomatic, Undected', color='#F0CA4D')
    axes[1, 0].plot(t, Q, label='Isolated', color='#E37332')
    axes[1, 0].plot(t, C, label='Critical', color='#829FD9')

    axes[1, 0].set_xlabel('Time (days)')
    axes[1, 0].set_ylabel('Cases (fraction of the population)')
    axes[1, 0].set_title('Short-term evolution: Infected sub-population')
    axes[1, 0].set_xlim(0, short)
    axes[1, 0].set_ylim(0, 2.5e-4)
    axes[1, 0].text(-0.1, 1.15, 'c', transform=axes[1, 0].transAxes, size=16, weight='bold')
    axes[1, 0].legend(loc='upper left')

    axes[1, 1].plot(t, I, label='Symptomatic, Undected', color='#46B39D')
    axes[1, 1].plot(t, H, label='Symptomatic, Diagnosed', color='#334752')
    axes[1, 1].plot(t, A, label='Asymptomatic, Undected', color='#F0CA4D')
    axes[1, 1].plot(t, Q, label='Isolated', color='#E37332')
    axes[1, 1].plot(t, C, label='Critical', color='#829FD9')

    axes[1, 1].set_xlabel('Time (days)')
    axes[1, 1].set_ylabel('Cases (fraction of the population)')
    axes[1, 1].set_title('Long-term evolution: Infected sub-population')
    axes[1, 1].set_xlim(0, long)
    axes[1, 1].set_ylim(0, 2.5e-2)
    axes[1, 1].text(-0.1, 1.15, 'd', transform=axes[1, 1].transAxes, size=16, weight='bold')
    axes[1, 1].legend(loc='upper left')

    fig.tight_layout()
    plt.show()

    #plt.savefig(f'../doc/India_evolution.pdf', dpi=600)
    #plt.clf()

def plot_sensitivity_beta_india(t,
                            TI1, DR1, R1, D1, CI1, CR1,
                            TI2, DR2, R2, D2, CI2, CR2,
                            TI3, DR3, R3, D3, CI3, CR3,
                            TI4, DR4, R4, D4, CI4, CR4,
                            TI5, DR5, R5, D5, CI5, CR5,
                            currently_infected):
    long = 700
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    axes[0, 0].plot(t, TI1, label='Current total infected', color='#D0E9F2')
    axes[0, 0].plot(t, TI2, label='Current total infected', color='#829FD9')
    axes[0, 0].plot(t, TI3, label='Current total infected', color='#F2A516')
    axes[0, 0].plot(t, TI4, label='Current total infected', color='#D98218')
    axes[0, 0].plot(t, TI5, label='Current total infected', color='#8C5511')
    axes[0, 0].plot(t[:160], currently_infected[45:208-3], label='Current total infected: Data', color='#000000')
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Cases (fraction of the population)')
    axes[0, 0].set_title('Sensitivity w. r. t. $\\beta(t)$: Currently infected')
    axes[0, 0].set_xlim(0, long)
    axes[0, 0].set_ylim(0, 1e-3)
    axes[0, 0].text(-0.1, 1.15, 'a', transform=axes[0, 0].transAxes, size=16, weight='bold')

    axes[0, 1].plot(t, DR1, label='Current total infected', color='#D0E9F2')
    axes[0, 1].plot(t, DR2, label='Current total infected', color='#829FD9')
    axes[0, 1].plot(t, DR3, label='Current total infected', color='#F2A516')
    axes[0, 1].plot(t, DR4, label='Current total infected', color='#D98218')
    axes[0, 1].plot(t, DR5, label='Current total infected', color='#8C5511')
    axes[0, 1].set_xlabel('Time (days)')
    axes[0, 1].set_ylabel('Cases (fraction of the population)')
    axes[0, 1].set_title('Sensitivity w. r. t. $\\beta(t)$: Detected and Recovered')
    axes[0, 1].set_xlim(0, long)
    axes[0, 1].set_ylim(0, 1e-2)
    axes[0, 1].text(-0.1, 1.15, 'b', transform=axes[0, 1].transAxes, size=16, weight='bold')

    axes[0, 2].plot(t, R1, label='Current total infected', color='#D0E9F2')
    axes[0, 2].plot(t, R2, label='Current total infected', color='#829FD9')
    axes[0, 2].plot(t, R3, label='Current total infected', color='#F2A516')
    axes[0, 2].plot(t, R4, label='Current total infected', color='#D98218')
    axes[0, 2].plot(t, R5, label='Current total infected', color='#8C5511')
    axes[0, 2].set_xlabel('Time (days)')
    axes[0, 2].set_ylabel('Cases (fraction of the population)')
    axes[0, 2].set_title('Sensitivity w. r. t. $\\beta(t)$: Cumulative Recovered')
    axes[0, 2].set_xlim(0, long)
    axes[0, 2].set_ylim(0, 1e-2)
    axes[0, 2].text(-0.1, 1.15, 'c', transform=axes[0, 2].transAxes, size=16, weight='bold')

    axes[1, 0].plot(t, D1, label='Current total infected', color='#D0E9F2')
    axes[1, 0].plot(t, D2, label='Current total infected', color='#829FD9')
    axes[1, 0].plot(t, D3, label='Current total infected', color='#F2A516')
    axes[1, 0].plot(t, D4, label='Current total infected', color='#D98218')
    axes[1, 0].plot(t, D5, label='Current total infected', color='#8C5511')
    axes[1, 0].set_xlabel('Time (days)')
    axes[1, 0].set_ylabel('Cases (fraction of the population)')
    axes[1, 0].set_title('Sensitivity w. r. t. $\\beta(t)$: Deaths')
    axes[1, 0].set_xlim(0, long)
    axes[1, 0].set_ylim(0, 1e-3)
    axes[1, 0].text(-0.1, 1.15, 'd', transform=axes[1, 0].transAxes, size=16, weight='bold')

    axes[1, 1].plot(t, CI1, label='Current total infected', color='#D0E9F2')
    axes[1, 1].plot(t, CI2, label='Current total infected', color='#829FD9')
    axes[1, 1].plot(t, CI3, label='Current total infected', color='#F2A516')
    axes[1, 1].plot(t, CI4, label='Current total infected', color='#D98218')
    axes[1, 1].plot(t, CI5, label='Current total infected', color='#8C5511')
    axes[1, 1].set_xlabel('Time (days)')
    axes[1, 1].set_ylabel('Cases (fraction of the population)')
    axes[1, 1].set_title('Sensitivity w. r. t. $\\beta(t)$: Cumulative Infected')
    axes[1, 1].set_xlim(0, long)
    axes[1, 1].set_ylim(0, 1e-2)
    axes[1, 1].text(-0.1, 1.15, 'e', transform=axes[1, 1].transAxes, size=16, weight='bold')

    axes[1, 2].plot(t, CR1, label='Current total infected', color='#D0E9F2')
    axes[1, 2].plot(t, CR2, label='Current total infected', color='#829FD9')
    axes[1, 2].plot(t, CR3, label='Current total infected', color='#F2A516')
    axes[1, 2].plot(t, CR4, label='Current total infected', color='#D98218')
    axes[1, 2].plot(t, CR5, label='Current total infected', color='#8C5511')
    axes[1, 2].set_xlabel('Time (days)')
    axes[1, 2].set_ylabel('Cases (fraction of the population)')
    axes[1, 2].set_title('Sensitivity w. r. t. $\\beta(t)$: Critical')
    axes[1, 2].set_xlim(0, long)
    axes[1, 2].set_ylim(0, 1e-3)
    axes[1, 2].text(-0.1, 1.15, 'f', transform=axes[1, 2].transAxes, size=16, weight='bold')

    fig.tight_layout()
    plt.show()

    #plt.savefig(f'../doc/India_sensitivity_beta.pdf', dpi=600)
    #plt.clf()

def plot_sensitivity_epsilon_india(t,
                            TI1, DR1, R1, D1, CI1, CR1,
                            TI2, DR2, R2, D2, CI2, CR2,
                            TI3, DR3, R3, D3, CI3, CR3,
                            TI4, DR4, R4, D4, CI4, CR4,
                            TI5, DR5, R5, D5, CI5, CR5,
                            currently_infected):
    long = 800
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    axes[0, 0].plot(t, TI1, label='Current total infected', color='#D0E9F2')
    axes[0, 0].plot(t, TI2, label='Current total infected', color='#829FD9')
    axes[0, 0].plot(t, TI3, label='Current total infected', color='#F2A516')
    axes[0, 0].plot(t, TI4, label='Current total infected', color='#D98218')
    axes[0, 0].plot(t, TI5, label='Current total infected', color='#8C5511')
    axes[0, 0].plot(t[:160], currently_infected[45:208-3], label='Current total infected: Data', color='#000000')
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Cases (fraction of the population)')
    axes[0, 0].set_title('Sensitivity w. r. t. $\epsilon(t)$: Currently infected')
    axes[0, 0].set_xlim(0, long)
    axes[0, 0].set_ylim(0, 2.5e-2)
    axes[0, 0].text(-0.1, 1.15, 'a', transform=axes[0, 0].transAxes, size=16, weight='bold')

    axes[0, 1].plot(t, DR1, label='Current total infected', color='#D0E9F2')
    axes[0, 1].plot(t, DR2, label='Current total infected', color='#829FD9')
    axes[0, 1].plot(t, DR3, label='Current total infected', color='#F2A516')
    axes[0, 1].plot(t, DR4, label='Current total infected', color='#D98218')
    axes[0, 1].plot(t, DR5, label='Current total infected', color='#8C5511')
    axes[0, 1].set_xlabel('Time (days)')
    axes[0, 1].set_ylabel('Cases (fraction of the population)')
    axes[0, 1].set_title('Sensitivity w. r. t. $\epsilon(t)$: Detected and Recovered')
    axes[0, 1].set_xlim(0, long)
    axes[0, 1].set_ylim(0, 5e-1)
    axes[0, 1].text(-0.1, 1.15, 'b', transform=axes[0, 1].transAxes, size=16, weight='bold')

    axes[0, 2].plot(t, R1, label='Current total infected', color='#D0E9F2')
    axes[0, 2].plot(t, R2, label='Current total infected', color='#829FD9')
    axes[0, 2].plot(t, R3, label='Current total infected', color='#F2A516')
    axes[0, 2].plot(t, R4, label='Current total infected', color='#D98218')
    axes[0, 2].plot(t, R5, label='Current total infected', color='#8C5511')
    axes[0, 2].set_xlabel('Time (days)')
    axes[0, 2].set_ylabel('Cases (fraction of the population)')
    axes[0, 2].set_title('Sensitivity w. r. t. $\epsilon(t)$: Cumulative Recovered')
    axes[0, 2].set_xlim(0, long)
    axes[0, 2].set_ylim(0, 5e-1)
    axes[0, 2].text(-0.1, 1.15, 'c', transform=axes[0, 2].transAxes, size=16, weight='bold')

    axes[1, 0].plot(t, D1, label='Current total infected', color='#D0E9F2')
    axes[1, 0].plot(t, D2, label='Current total infected', color='#829FD9')
    axes[1, 0].plot(t, D3, label='Current total infected', color='#F2A516')
    axes[1, 0].plot(t, D4, label='Current total infected', color='#D98218')
    axes[1, 0].plot(t, D5, label='Current total infected', color='#8C5511')
    axes[1, 0].set_xlabel('Time (days)')
    axes[1, 0].set_ylabel('Cases (fraction of the population)')
    axes[1, 0].set_title('Sensitivity w. r. t. $\epsilon(t)$: Deaths')
    axes[1, 0].set_xlim(0, long)
    axes[1, 0].set_ylim(0, 2.5e-2)
    axes[1, 0].text(-0.1, 1.15, 'd', transform=axes[1, 0].transAxes, size=16, weight='bold')

    axes[1, 1].plot(t, CI1, label='Current total infected', color='#D0E9F2')
    axes[1, 1].plot(t, CI2, label='Current total infected', color='#829FD9')
    axes[1, 1].plot(t, CI3, label='Current total infected', color='#F2A516')
    axes[1, 1].plot(t, CI4, label='Current total infected', color='#D98218')
    axes[1, 1].plot(t, CI5, label='Current total infected', color='#8C5511')
    axes[1, 1].set_xlabel('Time (days)')
    axes[1, 1].set_ylabel('Cases (fraction of the population)')
    axes[1, 1].set_title('Sensitivity w. r. t. $\epsilon(t)$: Cumulative Infected')
    axes[1, 1].set_xlim(0, long)
    axes[1, 1].set_ylim(0, 5e-1)
    axes[1, 1].text(-0.1, 1.15, 'e', transform=axes[1, 1].transAxes, size=16, weight='bold')

    axes[1, 2].plot(t, CR1, label='Current total infected', color='#D0E9F2')
    axes[1, 2].plot(t, CR2, label='Current total infected', color='#829FD9')
    axes[1, 2].plot(t, CR3, label='Current total infected', color='#F2A516')
    axes[1, 2].plot(t, CR4, label='Current total infected', color='#D98218')
    axes[1, 2].plot(t, CR5, label='Current total infected', color='#8C5511')
    axes[1, 2].set_xlabel('Time (days)')
    axes[1, 2].set_ylabel('Cases (fraction of the population)')
    axes[1, 2].set_title('Sensitivity w. r. t. $\epsilon(t)$: Critical')
    axes[1, 2].set_xlim(0, long)
    axes[1, 2].set_ylim(0, 2.5e-2)
    axes[1, 2].text(-0.1, 1.15, 'f', transform=axes[1, 2].transAxes, size=16, weight='bold')

    fig.tight_layout()
    plt.show()

    #plt.savefig(f'../doc/India_sensitivity_epsilon.pdf', dpi=600)
    #plt.clf()

# Italy
def plot_sensitivity_beta_italy(t,
                            TI1, DR1, R1, D1, CI1, CR1,
                            TI2, DR2, R2, D2, CI2, CR2,
                            TI3, DR3, R3, D3, CI3, CR3,
                            TI4, DR4, R4, D4, CI4, CR4,
                            TI5, DR5, R5, D5, CI5, CR5,
                            currently_infected):
    long = 700
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    axes[0, 0].plot(t, TI1, label='Current total infected', color='#D0E9F2')
    axes[0, 0].plot(t, TI2, label='Current total infected', color='#829FD9')
    axes[0, 0].plot(t, TI3, label='Current total infected', color='#F2A516')
    axes[0, 0].plot(t, TI4, label='Current total infected', color='#D98218')
    axes[0, 0].plot(t, TI5, label='Current total infected', color='#8C5511')
    axes[0, 0].plot(t[:186], currently_infected, label='Current total infected: Data', color='#000000')
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Cases (fraction of the population)')
    axes[0, 0].set_title('Sensitivity w. r. t. $\\beta(t)$: Currently infected')
    axes[0, 0].set_xlim(0, long)
    axes[0, 0].set_ylim(0, 2.5e-1)
    axes[0, 0].text(-0.1, 1.15, 'a', transform=axes[0, 0].transAxes, size=16, weight='bold')

    axes[0, 1].plot(t, DR1, label='Current total infected', color='#D0E9F2')
    axes[0, 1].plot(t, DR2, label='Current total infected', color='#829FD9')
    axes[0, 1].plot(t, DR3, label='Current total infected', color='#F2A516')
    axes[0, 1].plot(t, DR4, label='Current total infected', color='#D98218')
    axes[0, 1].plot(t, DR5, label='Current total infected', color='#8C5511')
    axes[0, 1].set_xlabel('Time (days)')
    axes[0, 1].set_ylabel('Cases (fraction of the population)')
    axes[0, 1].set_title('Sensitivity w. r. t. $\\beta(t)$: Detected and Recovered')
    axes[0, 1].set_xlim(0, long)
    axes[0, 1].set_ylim(0, 2.5e-1)
    axes[0, 1].text(-0.1, 1.15, 'b', transform=axes[0, 1].transAxes, size=16, weight='bold')

    axes[0, 2].plot(t, R1, label='Current total infected', color='#D0E9F2')
    axes[0, 2].plot(t, R2, label='Current total infected', color='#829FD9')
    axes[0, 2].plot(t, R3, label='Current total infected', color='#F2A516')
    axes[0, 2].plot(t, R4, label='Current total infected', color='#D98218')
    axes[0, 2].plot(t, R5, label='Current total infected', color='#8C5511')
    axes[0, 2].set_xlabel('Time (days)')
    axes[0, 2].set_ylabel('Cases (fraction of the population)')
    axes[0, 2].set_title('Sensitivity w. r. t. $\\beta(t)$: Cumulative Recovered')
    axes[0, 2].set_xlim(0, long)
    axes[0, 2].set_ylim(0, 2.5e-1)
    axes[0, 2].text(-0.1, 1.15, 'c', transform=axes[0, 2].transAxes, size=16, weight='bold')

    axes[1, 0].plot(t, D1, label='Current total infected', color='#D0E9F2')
    axes[1, 0].plot(t, D2, label='Current total infected', color='#829FD9')
    axes[1, 0].plot(t, D3, label='Current total infected', color='#F2A516')
    axes[1, 0].plot(t, D4, label='Current total infected', color='#D98218')
    axes[1, 0].plot(t, D5, label='Current total infected', color='#8C5511')
    axes[1, 0].set_xlabel('Time (days)')
    axes[1, 0].set_ylabel('Cases (fraction of the population)')
    axes[1, 0].set_title('Sensitivity w. r. t. $\\beta(t)$: Deaths')
    axes[1, 0].set_xlim(0, long)
    axes[1, 0].set_ylim(0, 2.5e-1)
    axes[1, 0].text(-0.1, 1.15, 'd', transform=axes[1, 0].transAxes, size=16, weight='bold')

    axes[1, 1].plot(t, CI1, label='Current total infected', color='#D0E9F2')
    axes[1, 1].plot(t, CI2, label='Current total infected', color='#829FD9')
    axes[1, 1].plot(t, CI3, label='Current total infected', color='#F2A516')
    axes[1, 1].plot(t, CI4, label='Current total infected', color='#D98218')
    axes[1, 1].plot(t, CI5, label='Current total infected', color='#8C5511')
    axes[1, 1].set_xlabel('Time (days)')
    axes[1, 1].set_ylabel('Cases (fraction of the population)')
    axes[1, 1].set_title('Sensitivity w. r. t. $\\beta(t)$: Cumulative Infected')
    axes[1, 1].set_xlim(0, long)
    axes[1, 1].set_ylim(0, 2.5e-1)
    axes[1, 1].text(-0.1, 1.15, 'e', transform=axes[1, 1].transAxes, size=16, weight='bold')

    axes[1, 2].plot(t, CR1, label='Current total infected', color='#D0E9F2')
    axes[1, 2].plot(t, CR2, label='Current total infected', color='#829FD9')
    axes[1, 2].plot(t, CR3, label='Current total infected', color='#F2A516')
    axes[1, 2].plot(t, CR4, label='Current total infected', color='#D98218')
    axes[1, 2].plot(t, CR5, label='Current total infected', color='#8C5511')
    axes[1, 2].set_xlabel('Time (days)')
    axes[1, 2].set_ylabel('Cases (fraction of the population)')
    axes[1, 2].set_title('Sensitivity w. r. t. $\\beta(t)$: Critical')
    axes[1, 2].set_xlim(0, long)
    axes[1, 2].set_ylim(0, 2.5e-2)
    axes[1, 2].text(-0.1, 1.15, 'f', transform=axes[1, 2].transAxes, size=16, weight='bold')

    fig.tight_layout()
    #plt.show()

    plt.savefig(f'../doc/Italy_sensitivity_beta.pdf', dpi=600)
    plt.clf()

def plot_sensitivity_epsilon_italy(t,
                            TI1, DR1, R1, D1, CI1, CR1,
                            TI2, DR2, R2, D2, CI2, CR2,
                            TI3, DR3, R3, D3, CI3, CR3,
                            TI4, DR4, R4, D4, CI4, CR4,
                            TI5, DR5, R5, D5, CI5, CR5,
                            currently_infected):
    long = 700
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    axes[0, 0].plot(t, TI1, label='Current total infected', color='#D0E9F2')
    axes[0, 0].plot(t, TI2, label='Current total infected', color='#829FD9')
    axes[0, 0].plot(t, TI3, label='Current total infected', color='#F2A516')
    axes[0, 0].plot(t, TI4, label='Current total infected', color='#D98218')
    axes[0, 0].plot(t, TI5, label='Current total infected', color='#8C5511')
    axes[0, 0].plot(t[:186], currently_infected, label='Current total infected: Data', color='#000000')
    axes[0, 0].set_xlabel('Time (days)')
    axes[0, 0].set_ylabel('Cases (fraction of the population)')
    axes[0, 0].set_title('Sensitivity w. r. t. $\epsilon(t)$: Currently infected')
    axes[0, 0].set_xlim(0, long)
    axes[0, 0].set_ylim(0, 2.5e-1)
    axes[0, 0].text(-0.1, 1.15, 'a', transform=axes[0, 0].transAxes, size=16, weight='bold')

    axes[0, 1].plot(t, DR1, label='Current total infected', color='#D0E9F2')
    axes[0, 1].plot(t, DR2, label='Current total infected', color='#829FD9')
    axes[0, 1].plot(t, DR3, label='Current total infected', color='#F2A516')
    axes[0, 1].plot(t, DR4, label='Current total infected', color='#D98218')
    axes[0, 1].plot(t, DR5, label='Current total infected', color='#8C5511')
    axes[0, 1].set_xlabel('Time (days)')
    axes[0, 1].set_ylabel('Cases (fraction of the population)')
    axes[0, 1].set_title('Sensitivity w. r. t. $\epsilon(t)$: Detected and Recovered')
    axes[0, 1].set_xlim(0, long)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].text(-0.1, 1.15, 'b', transform=axes[0, 1].transAxes, size=16, weight='bold')

    axes[0, 2].plot(t, R1, label='Current total infected', color='#D0E9F2')
    axes[0, 2].plot(t, R2, label='Current total infected', color='#829FD9')
    axes[0, 2].plot(t, R3, label='Current total infected', color='#F2A516')
    axes[0, 2].plot(t, R4, label='Current total infected', color='#D98218')
    axes[0, 2].plot(t, R5, label='Current total infected', color='#8C5511')
    axes[0, 2].set_xlabel('Time (days)')
    axes[0, 2].set_ylabel('Cases (fraction of the population)')
    axes[0, 2].set_title('Sensitivity w. r. t. $\epsilon(t)$: Cumulative Recovered')
    axes[0, 2].set_xlim(0, long)
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].text(-0.1, 1.15, 'c', transform=axes[0, 2].transAxes, size=16, weight='bold')

    axes[1, 0].plot(t, D1, label='Current total infected', color='#D0E9F2')
    axes[1, 0].plot(t, D2, label='Current total infected', color='#829FD9')
    axes[1, 0].plot(t, D3, label='Current total infected', color='#F2A516')
    axes[1, 0].plot(t, D4, label='Current total infected', color='#D98218')
    axes[1, 0].plot(t, D5, label='Current total infected', color='#8C5511')
    axes[1, 0].set_xlabel('Time (days)')
    axes[1, 0].set_ylabel('Cases (fraction of the population)')
    axes[1, 0].set_title('Sensitivity w. r. t. $\epsilon(t)$: Deaths')
    axes[1, 0].set_xlim(0, long)
    axes[1, 0].set_ylim(0, 2.5e-1)
    axes[1, 0].text(-0.1, 1.15, 'd', transform=axes[1, 0].transAxes, size=16, weight='bold')

    axes[1, 1].plot(t, CI1, label='Current total infected', color='#D0E9F2')
    axes[1, 1].plot(t, CI2, label='Current total infected', color='#829FD9')
    axes[1, 1].plot(t, CI3, label='Current total infected', color='#F2A516')
    axes[1, 1].plot(t, CI4, label='Current total infected', color='#D98218')
    axes[1, 1].plot(t, CI5, label='Current total infected', color='#8C5511')
    axes[1, 1].set_xlabel('Time (days)')
    axes[1, 1].set_ylabel('Cases (fraction of the population)')
    axes[1, 1].set_title('Sensitivity w. r. t. $\epsilon(t)$: Cumulative Infected')
    axes[1, 1].set_xlim(0, long)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].text(-0.1, 1.15, 'e', transform=axes[1, 1].transAxes, size=16, weight='bold')

    axes[1, 2].plot(t, CR1, label='Current total infected', color='#D0E9F2')
    axes[1, 2].plot(t, CR2, label='Current total infected', color='#829FD9')
    axes[1, 2].plot(t, CR3, label='Current total infected', color='#F2A516')
    axes[1, 2].plot(t, CR4, label='Current total infected', color='#D98218')
    axes[1, 2].plot(t, CR5, label='Current total infected', color='#8C5511')
    axes[1, 2].set_xlabel('Time (days)')
    axes[1, 2].set_ylabel('Cases (fraction of the population)')
    axes[1, 2].set_title('Sensitivity w. r. t. $\epsilon(t)$: Critical')
    axes[1, 2].set_xlim(0, long)
    axes[1, 2].set_ylim(0, 2.5e-2)
    axes[1, 2].text(-0.1, 1.15, 'f', transform=axes[1, 2].transAxes, size=16, weight='bold')

    fig.tight_layout()
    #plt.show()

    plt.savefig(f'../doc/Italy_sensitivity_epsilon.pdf', dpi=600)
    plt.clf()