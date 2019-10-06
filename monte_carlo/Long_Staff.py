import numpy as np

random_seed = 150000
T = 1
no_of_slices = 252
mue = 0.1
simulation_rounds = 500
S0 = 100
K = 100
sigma = 0.1


def american_call():
    np.random.seed(random_seed)  # fix the seed for every valuation
    dt = T / no_of_slices  # time interval
    discount_factor = np.exp(- mue * dt)  # discount factor per time time interval

    # Simulation of Index Levels
    S = np.zeros((no_of_slices + 1, simulation_rounds), dtype=np.float64)  # stock price matrix
    S[0, :] = S0  # initial values for stock price
    print(S)

    for t in range(1, int(T) * no_of_slices + 1):
        ran = np.random.standard_normal(simulation_rounds / 2)
        ran = np.concatenate((ran, -ran))  # antithetic variates
        ran = ran - np.mean(ran)  # correct first moment
        ran = ran / np.std(ran)  # correct second moment
        S[t, :] = S[t - 1, :] * np.exp((mue - sigma ** 2 / 2) * dt
                                       + sigma * ran * np.sqrt(dt))
    h = np.maximum(S - K, 0)  # inner values for call option
    V = np.zeros_like(h)  # value matrix
    V[-1] = h[-1]

    # Valuation by LSM
    for t in range(int(T) * no_of_slices - 1, 0, -1):
        rg = np.polyfit(S[t, :], V[t + 1, :] * discount_factor, 5)  # regression
        C = np.polyval(rg, S[t, :])  # evaluation of regression
        V[t, :] = np.where(h[t, :] > C, h[t, :],
                           V[t + 1, :] * discount_factor)  # exercise decision/optimization
    V0 = np.sum(V[1, :] * discount_factor) / simulation_rounds  # LSM estimator
    standard_error = np.std(V[1, :] * discount_factor) / np.sqrt(simulation_rounds)

    print("S0 %4.1f | vol %4.2f | T %2.1f | Call Option Value %8.3f | Standard Error %4.2f " % (
        S0, sigma, T, V0, standard_error))

    return V0


american_call()
