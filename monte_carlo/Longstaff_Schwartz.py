import numpy as np

# Parameters
np.random.seed(150000)

S0 = 100.0
K = 100.0
r = 0.05  # mue
sigma = 0.25
div_yield = 0.0

T = 1
no_of_slices = 252
simulation_rounds = 5000

dt = T / float(no_of_slices)

# Position value
row = 0
column = 0

# Regression
x = []
y = []

# Define discount rate (one period)
df = np.exp(-r * dt)

# price simulation factor
drift = (r - div_yield - 0.5 * (sigma ** 2)) * dt
diffusion = sigma * np.sqrt(dt)


def longstaff_schwartz():
    # Stock price simulation
    stock_price_table = np.zeros([simulation_rounds, no_of_slices], dtype=np.float64)
    stock_price_table[:, 0] = S0
    for i in range(0, simulation_rounds):
        for j in range(1, int(T) * no_of_slices):
            stock_price_table[i, j] = stock_price_table[i, j - 1] * np.exp(
                drift + diffusion * np.random.standard_normal())

    # Get intrinsic value table = profit if exercise immediately
    intrinsic_value_table = stock_price_table - K
    intrinsic_value_table[intrinsic_value_table < 0] = 0  # set all loss = 0 i.e. not exercise

    # initialize stopping rule table
    stopping_rule_table = np.zeros([simulation_rounds, no_of_slices])

    # initialize cash flow matrix
    cashflow_matrix = np.zeros([simulation_rounds, no_of_slices])

    # check t-1 intrinsic value > 0, if > 0, price_{t-1} = X, IV_{t} = Y
    for t in range(no_of_slices - 1, 1, -1):  # from 252 to 1
        stock_price_slice = stock_price_table[:, t - 1]  # price @ t-1
        intrinsic_value_slice = intrinsic_value_table[:, t]  # intrinsic value @ t

        # initialize x and y for regression
        x = []
        y = []
        exercise_value_slice = []  # exercise at t

        if t == no_of_slices - 1:
            stopping_rule_table[:, t][intrinsic_value_table[:, t] > 0] = 1  # last round, set stopping rule if IV > 0

        for l in intrinsic_value_table[:, t - 1]:  # select only price_{t-1} > K
            if l > 0:  # if > 0
                position = np.where(intrinsic_value_table[:, t - 1] == l)[0][
                    0]  # get the position and create new list, essentially get rid of non positive
                x.append(stock_price_slice[position])  # get regression x
                y.append(intrinsic_value_slice[position] * df)  # get regression y
                exercise_value_slice.append(intrinsic_value_table[:, t - 1][position])

        x, y = np.array(x), np.array(y)
        rg = np.polyfit(x, y, 2)  # regression
        continuation_value = np.polyval(rg, x)

        # check if exercise_value > fitted continuation value, if yes, stopping rule = 1 (we will exercise)
        for m in range(len(exercise_value_slice)):
            if exercise_value_slice[m] > continuation_value[m]:
                # get where I should update the stopping rule
                stopping_rule_position = np.where(intrinsic_value_table[:, t - 1] == exercise_value_slice[m])[0][0]

                # update stopping rule position
                stopping_rule_table[:, t - 1][stopping_rule_position] = 1
                stopping_rule_table[:, t][
                stopping_rule_position:] = 0  # once we exercise at t-1, future no value any more

                # update cashflow matrix
                cashflow_matrix[:, t - 1][stopping_rule_position] = intrinsic_value_table[:, t - 1][
                    stopping_rule_position]
                cashflow_matrix[:, t - 1][stopping_rule_position] *= np.exp(-r * dt * t)  # discount the cashflow
                cashflow_matrix[:, t][stopping_rule_position:] = 0

    expectation = np.mean(cashflow_matrix[np.nonzero(cashflow_matrix)])
    std_error = np.mean(cashflow_matrix) / np.sqrt(simulation_rounds)
    print(expectation, std_error)


if __name__ == '__main__':
    longstaff_schwartz()
