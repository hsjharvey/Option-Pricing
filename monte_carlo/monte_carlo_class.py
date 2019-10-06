'''
Valuation of European, Asian, American call option
- class-based implementation
monte_carlo.py

By: Shijie Huang (Harvey)
'''

import numpy as np


class monte_carlo_option_pricing():
    """
        Class for european call options in BSM model (incl. dividend)

        Attributes
        ==========
        S0: float
            initial stock/index level
        K: float
            strike price
        T: float
            maturity (in year fractions)
        r: float
            constant risk-free short rate
            assume flat term structure
        sigma: float
            volatility factor in diffusion term
        div_yield: float
            dividend_yield, default = 0.0%
    """

    def __init__(self, S0, K, T, mue, sigma, div_yield=0.0, simulation_rounds=10000, no_of_slices=1,
                 fix_random_seed=False):

        assert sigma >= 0, 'volatility cannot be less than zero'
        assert S0 >= 0, 'initial stock price cannot be less than zero'
        assert T >= 0, 'time to maturity cannot be less than zero'
        assert div_yield >= 0, 'dividend yield cannot be less than zero'
        assert no_of_slices >= 0, 'no of slices must be greater than zero'
        assert simulation_rounds >= 0, 'simulation rounds must be greater than zero'

        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.mue = float(mue)
        self.sigma = float(sigma)
        self.div_yield = float(div_yield)

        self.no_of_slices = int(no_of_slices)
        self.simulation_rounds = int(simulation_rounds)

        if fix_random_seed:
            np.random.seed(150000)

    def stock_price_simulation(self):
        self.expectation = 0.0
        self.standard_error = 0.0
        self.z_t = 0.0
        self.diffusion_factor = 0.0
        self.price_t = 0.0

        self.price_list = []

        self.exp_mean = (self.mue - self.div_yield - 0.5 * (self.sigma ** 2)) * self.T
        self.exp_diffusion = self.sigma * np.sqrt(self.T)

        for i in range(self.simulation_rounds):
            self.sum_z_t = []

            for j in range(self.no_of_slices * int(self.T)):
                self.z_t = np.random.standard_normal()
                self.sum_z_t.append(self.z_t)

            self.diffusion_factor = np.sum(self.sum_z_t) / np.sqrt(self.no_of_slices)
            self.price_t = self.S0 * np.exp(self.exp_mean + self.exp_diffusion * self.diffusion_factor)
            self.price_list.append(self.price_t)

        self.stock_price_expectation = np.mean(self.price_list)
        self.stock_price_standard_error = np.std(self.price_list) / np.sqrt(len(self.price_list))

        print(
            "S0 %4.1f | vol %4.2f | T %2.1f | Maximum Stock price %4.2f | Minimum Stock price %4.2f | Average stock price %8.3f | Standard Error %4.2f " % (
                self.S0, self.sigma, self.T, np.max(self.price_list), np.min(self.price_list),
                self.stock_price_expectation, self.stock_price_standard_error))

        return self.stock_price_expectation, self.stock_price_standard_error

    def european_call(self):
        np.random.seed(self.random_seed)
        self.terminal_profit = []
        self.expectation = 0.0
        self.standard_error = 0.0
        self.z_t = 0.0
        self.diffusion_factor = 0.0
        self.price_t = 0.0

        self.exp_mean = (self.mue - self.div_yield - 0.5 * (self.sigma ** 2)) * self.T
        self.exp_diffusion = self.sigma * np.sqrt(self.T)

        for i in range(self.simulation_rounds):
            self.sum_z_t = []
            self.price_list = []

            for j in range(self.no_of_slices * int(self.T)):
                self.z_t = np.random.standard_normal()
                self.sum_z_t.append(self.z_t)

            self.diffusion_factor = np.sum(self.sum_z_t) / np.sqrt(self.no_of_slices)
            self.price_t = self.S0 * np.exp(self.exp_mean + self.exp_diffusion * self.diffusion_factor)
            self.price_list.append(self.price_t)
            self.terminal_profit.append(max((self.price_t - self.K), 0.0))

        self.expectation = np.mean(self.terminal_profit) * np.exp(- self.mue * self.T)
        self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit))

        print(
            "S0 %4.1f | vol %4.2f | T %2.1f | Call Option Value %8.3f | Standard Error %4.2f " % (
                self.S0, self.sigma, self.T, self.expectation, self.standard_error))

        return self.expectation, self.standard_error

    def european_put(self):
        """
        Use put call parity (incl. continuous dividend) to calculate the put option value
        :param call_value: can be calculated or observed call option value
        :return: put option value
        """
        self.european_call_value = monte_carlo_option_pricing(self.S0, self.K, self.T, self.mue,
                                                              self.sigma, self.div_yield, self.simulation_rounds,
                                                              self.no_of_slices).european_call()[0]
        self.put_value = self.european_call_value + np.exp(-self.mue * self.T) * self.K - np.exp(
            -self.div_yield * self.T) * self.S0

        return self.put_value

    def asian_avg_price_call(self):
        """
        Asian call using average price method
        Arithmetic average
        :return: asian call value
        """
        np.random.seed(self.random_seed)
        self.terminal_profit = []
        self.expectation = 0.0
        self.standard_error = 0.0
        self.sum_z_t = []
        self.z_t = 0.0
        self.price_t = 0.0

        self.h = self.T / self.no_of_slices

        self.exp_mean = (self.mue - self.div_yield - (self.sigma ** 2.0) * 0.5) * self.h
        self.exp_diffusion = self.sigma * np.sqrt(self.h)

        for i in range(self.simulation_rounds):
            self.price_t = self.S0
            self.price_list = []

            for j in range(self.no_of_slices * int(self.T)):
                self.z_t = np.random.standard_normal()
                self.price_t = self.price_t * np.exp(self.exp_mean + self.exp_diffusion * self.z_t)
                self.price_list.append(self.price_t)

            self.terminal_profit.append(max((np.mean(self.price_list) - self.K), 0))

        self.expectation = np.mean(self.terminal_profit) * np.exp(- self.mue * self.T)
        self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit))

        print(
            "S0 %4.1f | vol %4.2f | T %2.1f | Call Option Value %8.3f | Standard Error %4.2f " % (
                self.S0, self.sigma, self.T, self.expectation, self.standard_error))

        return self.expectation, self.standard_error

    def asian_avg_price_put(self):
        """
        Asian put using average price method
        Arithmetic average
        :return: asian put value
        """
        np.random.seed(self.random_seed)
        self.terminal_profit = []
        self.expectation = 0.0
        self.standard_error = 0.0
        self.sum_z_t = []
        self.z_t = 0.0
        self.price_t = 0.0

        self.h = self.T / self.no_of_slices

        self.exp_mean = (self.mue - self.div_yield - (self.sigma ** 2.0) * 0.5) * self.h
        self.exp_diffusion = self.sigma * np.sqrt(self.h)

        for i in range(self.simulation_rounds):
            self.price_t = self.S0
            self.price_list = []

            for j in range(self.no_of_slices * int(self.T)):
                self.z_t = np.random.standard_normal()
                self.price_t = self.price_t * np.exp(self.exp_mean + self.exp_diffusion * self.z_t)
                self.price_list.append(self.price_t)

            self.terminal_profit.append(max((self.K - np.mean(self.price_list)), 0))

        self.expectation = np.mean(self.terminal_profit) * np.exp(- self.mue * self.T)
        self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit))

        print(
            "S0 %4.1f | vol %4.2f | T %2.1f | Call Option Value %8.3f | Standard Error %4.2f " % (
                self.S0, self.sigma, self.T, self.expectation, self.standard_error))

        return self.expectation, self.standard_error

    def american_call(self):
        np.random.seed(self.random_seed)  # fix the seed for every valuation
        self.dt = self.T / self.no_of_slices  # time interval
        self.discount_factor = np.exp(- self.mue * self.dt)  # discount factor per time time interval

        # Simulation of Index Levels
        self.S = np.zeros((self.no_of_slices + 1, self.simulation_rounds), dtype=np.float64)  # stock price matrix
        self.S[0, :] = self.S0  # initial values for stock price

        for t in range(1, int(self.T) * self.no_of_slices + 1):
            self.ran = np.random.standard_normal(self.simulation_rounds / 2)
            self.ran = np.concatenate((self.ran, -self.ran))  # antithetic variates
            self.ran = self.ran - np.mean(self.ran)  # correct first moment
            self.ran = self.ran / np.std(self.ran)  # correct second moment
            self.S[t, :] = self.S[t - 1, :] * np.exp((self.mue - self.sigma ** 2 / 2) * self.dt
                                                     + self.sigma * self.ran * np.sqrt(self.dt))
        self.h = np.maximum(self.S - self.K, 0)  # inner values for call option
        self.V = np.zeros_like(self.h)  # value matrix
        self.V[-1] = self.h[-1]

        # Valuation by LSM
        for t in range(int(self.T) * self.no_of_slices - 1, 0, -1):
            self.rg = np.polyfit(self.S[t, :], self.V[t + 1, :] * self.discount_factor, 5)  # regression
            self.C = np.polyval(self.rg, self.S[t, :])  # evaluation of regression
            self.V[t, :] = np.where(self.h[t, :] > self.C, self.h[t, :],
                                    self.V[t + 1, :] * self.discount_factor)  # exercise decision/optimization
        self.V0 = np.sum(self.V[1, :] * self.discount_factor) / self.simulation_rounds  # LSM estimator
        self.standard_error = np.std(self.V[1, :] * self.discount_factor) / np.sqrt(self.simulation_rounds)

        print(
            "S0 %4.1f | vol %4.2f | T %2.1f | Call Option Value %8.3f | Standard Error %4.2f " % (
                self.S0, self.sigma, self.T, self.V0, self.standard_error))

        return self.V0

    def american_put(self):
        np.random.seed(self.random_seed)  # fix the seed for every valuation
        self.dt = self.T / self.no_of_slices  # time interval
        self.discount_factor = np.exp(- self.mue * self.dt)  # discount factor per time time interval

        # Simulation of Index Levels
        self.S = np.zeros((self.no_of_slices + 1, self.simulation_rounds), dtype=np.float64)  # stock price matrix
        self.S[0, :] = self.S0  # initial values for stock price

        for t in range(1, int(self.T) * self.no_of_slices + 1):
            self.ran = np.random.standard_normal(self.simulation_rounds / 2)
            self.ran = np.concatenate((self.ran, -self.ran))  # antithetic variates
            self.ran = self.ran - np.mean(self.ran)  # correct first moment
            self.ran = self.ran / np.std(self.ran)  # correct second moment
            self.S[t, :] = self.S[t - 1, :] * np.exp((self.mue - self.sigma ** 2 / 2) * self.dt
                                                     + self.sigma * self.ran * np.sqrt(self.dt))
        self.h = np.maximum(self.K - self.S, 0)  # inner values for put option
        self.V = np.zeros_like(self.h)  # value matrix
        self.V[-1] = self.h[-1]

        # Valuation by LSM
        for t in range(int(self.T) * self.no_of_slices - 1, 0, -1):
            self.rg = np.polyfit(self.S[t, :], self.V[t + 1, :] * self.discount_factor, 5)  # regression
            self.C = np.polyval(self.rg, self.S[t, :])  # evaluation of regression
            self.V[t, :] = np.where(self.h[t, :] > self.C, self.h[t, :],
                                    self.V[t + 1, :] * self.discount_factor)  # exercise decision/optimization
        self.V0 = np.sum(self.V[1, :] * self.discount_factor) / self.simulation_rounds  # LSM estimator
        self.standard_error = np.std(self.V[1, :] * self.discount_factor) / np.sqrt(self.simulation_rounds)

        print(
            "S0 %4.1f | vol %4.2f | T %2.1f | Put Option Value %8.3f | Standard Error %4.4f " % (
                self.S0, self.sigma, self.T, self.V0, self.standard_error))

        return self.V0

    def down_and_in_parisian_monte_carlo(self, barrier_price, barrier_condition, option_type):
        # Stock price simulation
        np.random.seed(self.random_seed)
        self.expectation = 0.0
        self.standard_error = 0.0
        self.z_t = 0.0
        self.diffusion_factor = 0.0
        self.terminal_profit = []
        self.count = 0

        self.h = self.T / self.no_of_slices

        self.price_list = []

        self.exp_mean = (self.mue - self.div_yield - 0.5 * (self.sigma ** 2)) * self.h
        self.exp_diffusion = self.sigma * np.sqrt(self.h)

        for i in range(self.simulation_rounds):
            self.check = bool(False)
            self.count = 0
            self.price_t = self.S0

            for j in range(self.no_of_slices * int(self.T)):
                self.z_t = np.random.standard_normal()
                self.exp_factor = self.exp_mean + self.exp_diffusion * self.z_t
                self.price_t = self.price_t * np.exp(self.exp_factor)

                # check
                if self.check == False:
                    if self.price_t <= barrier_price:
                        self.count += 1
                    else:
                        self.count = 0

                if self.count >= barrier_condition:
                    self.check = bool(True)

            if self.check == True:
                if option_type == "call":
                    self.terminal_profit.append(max((self.price_t - self.K), 0.0))
                elif option_type == "put":
                    self.terminal_profit.append(max((self.K - self.price_t), 0.0))
            else:
                self.terminal_profit.append(0.0)

        self.expectation = np.mean(self.terminal_profit) * np.exp(- self.mue * self.T)
        self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit))

        return self.expectation, self.standard_error

    def down_and_out_parisian_monte_carlo(self, barrier_price, barrier_condition, option_type):
        # Stock price simulation
        np.random.seed(self.random_seed)
        self.expectation = 0.0
        self.standard_error = 0.0
        self.z_t = 0.0
        self.diffusion_factor = 0.0
        self.terminal_profit = []
        self.count = 0

        self.h = self.T / self.no_of_slices

        self.price_list = []

        self.exp_mean = (self.mue - self.div_yield - 0.5 * (self.sigma ** 2)) * self.h
        self.exp_diffusion = self.sigma * np.sqrt(self.h)

        for i in range(self.simulation_rounds):
            self.check = bool(False)
            self.count = 0
            self.price_t = self.S0

            for j in range(self.no_of_slices * int(self.T)):
                self.z_t = np.random.standard_normal()
                self.exp_factor = self.exp_mean + self.exp_diffusion * self.z_t
                self.price_t = self.price_t * np.exp(self.exp_factor)

                # check
                if self.check == False:
                    if self.price_t <= barrier_price:
                        self.count += 1
                    else:
                        self.count = 0

                if self.count >= barrier_condition:
                    self.check = bool(True)

            if self.check == False:
                if option_type == "call":
                    self.terminal_profit.append(max((self.price_t - self.K), 0.0))
                elif option_type == "put":
                    self.terminal_profit.append(max((self.K - self.price_t), 0.0))
            else:
                self.terminal_profit.append(0.0)

        self.expectation = np.mean(self.terminal_profit) * np.exp(- self.mue * self.T)
        self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit))

        return self.expectation, self.standard_error

    def up_and_in_parisian_monte_carlo(self, barrier_price, barrier_condition, option_type):
        # Stock price simulation
        np.random.seed(self.random_seed)
        self.expectation = 0.0
        self.standard_error = 0.0
        self.z_t = 0.0
        self.diffusion_factor = 0.0
        self.terminal_profit = []
        self.count = 0

        self.h = self.T / self.no_of_slices

        self.price_list = []

        self.exp_mean = (self.mue - self.div_yield - 0.5 * (self.sigma ** 2)) * self.h
        self.exp_diffusion = self.sigma * np.sqrt(self.h)

        for i in range(self.simulation_rounds):
            self.check = bool(False)
            self.count = 0
            self.price_t = self.S0

            for j in range(self.no_of_slices * int(self.T)):
                self.z_t = np.random.standard_normal()
                self.exp_factor = self.exp_mean + self.exp_diffusion * self.z_t
                self.price_t = self.price_t * np.exp(self.exp_factor)

                # check
                if self.check == False:
                    if self.price_t >= barrier_price:
                        self.count += 1
                    else:
                        self.count = 0

                if self.count >= barrier_condition:
                    self.check = bool(True)

            if self.check == True:
                if option_type == "call":
                    self.terminal_profit.append(max((self.price_t - self.K), 0.0))
                elif option_type == "put":
                    self.terminal_profit.append(max((self.K - self.price_t), 0.0))
            else:
                self.terminal_profit.append(0.0)

        self.expectation = np.mean(self.terminal_profit) * np.exp(- self.mue * self.T)
        self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit))

        return self.expectation, self.standard_error

    def up_and_out_parisian_monte_carlo(self, barrier_price, barrier_condition, option_type):
        # Stock price simulation
        np.random.seed(self.random_seed)
        self.expectation = 0.0
        self.standard_error = 0.0
        self.z_t = 0.0
        self.diffusion_factor = 0.0
        self.terminal_profit = []
        self.count = 0

        self.h = self.T / self.no_of_slices

        self.price_list = []

        self.exp_mean = (self.mue - self.div_yield - 0.5 * (self.sigma ** 2)) * self.h
        self.exp_diffusion = self.sigma * np.sqrt(self.h)

        for i in range(self.simulation_rounds):
            self.check = bool(False)
            self.count = 0
            self.price_t = self.S0

            for j in range(self.no_of_slices * int(self.T)):
                self.z_t = np.random.standard_normal()
                self.exp_factor = self.exp_mean + self.exp_diffusion * self.z_t
                self.price_t = self.price_t * np.exp(self.exp_factor)

                # check
                if self.check == False:
                    if self.price_t >= barrier_price:
                        self.count += 1
                    else:
                        self.count = 0

                if self.count >= barrier_condition:
                    self.check = bool(True)

            if self.check == False:
                if option_type == "call":
                    self.terminal_profit.append(max((self.price_t - self.K), 0.0))
                elif option_type == "put":
                    self.terminal_profit.append(max((self.K - self.price_t), 0.0))
            else:
                self.terminal_profit.append(0.0)

        self.expectation = np.mean(self.terminal_profit) * np.exp(- self.mue * self.T)
        self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit))

        return self.expectation, self.standard_error

    def down_and_in_option(self, barrier_price, option_type):
        '''
        :param option_type: call, put
        :return: Option price
        '''
        # Stock price simulation
        np.random.seed(self.random_seed)
        self.expectation = 0.0
        self.standard_error = 0.0
        self.z_t = 0.0
        self.diffusion_factor = 0.0
        self.terminal_profit = []

        self.h = self.T / self.no_of_slices

        self.price_list = []

        self.exp_mean = (self.mue - self.div_yield - 0.5 * (self.sigma ** 2)) * self.h
        self.exp_diffusion = self.sigma * np.sqrt(self.h)

        for i in range(self.simulation_rounds):
            self.check = bool(False)
            self.price_t = self.S0

            for j in range(self.no_of_slices * int(self.T)):
                self.z_t = np.random.standard_normal()
                self.exp_factor = self.exp_mean + self.exp_diffusion * self.z_t
                self.price_t = self.price_t * np.exp(self.exp_factor)

                # check
                if self.price_t <= barrier_price:
                    self.check = bool(True)

            if self.check == True:
                if option_type == "call":
                    self.terminal_profit.append(max((self.price_t - self.K), 0.0))
                elif option_type == "put":
                    self.terminal_profit.append(max((self.K - self.price_t), 0.0))
            else:
                self.terminal_profit.append(0.0)

        self.expectation = np.mean(self.terminal_profit) * np.exp(- self.mue * self.T)
        self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit))

        return self.expectation, self.standard_error

    def down_and_out_option(self, barrier_price, option_type):
        """
        :param option_type: call, put
        :return: Option price
        """
        # Stock price simulation
        np.random.seed(self.random_seed)
        self.expectation = 0.0
        self.standard_error = 0.0
        self.z_t = 0.0
        self.diffusion_factor = 0.0
        self.terminal_profit = []

        self.h = self.T / self.no_of_slices

        self.price_list = []

        self.exp_mean = (self.mue - self.div_yield - 0.5 * (self.sigma ** 2)) * self.h
        self.exp_diffusion = self.sigma * np.sqrt(self.h)

        for i in range(self.simulation_rounds):
            self.check = bool(False)
            self.price_t = self.S0

            for j in range(self.no_of_slices * int(self.T)):
                self.z_t = np.random.standard_normal()
                self.exp_factor = self.exp_mean + self.exp_diffusion * self.z_t
                self.price_t = self.price_t * np.exp(self.exp_factor)

                # check
                if self.price_t <= barrier_price:
                    self.check = bool(True)

            if self.check == False:
                if option_type == "call":
                    self.terminal_profit.append(max((self.price_t - self.K), 0.0))
                elif option_type == "put":
                    self.terminal_profit.append(max((self.K - self.price_t), 0.0))
            else:
                self.terminal_profit.append(0.0)

        self.expectation = np.mean(self.terminal_profit) * np.exp(- self.mue * self.T)
        self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit))

        return self.expectation, self.standard_error

    def up_and_in_option(self, barrier_price, option_type):
        """
        :param option_type: call, put
        :return: Option price
        """
        # Stock price simulation
        np.random.seed(self.random_seed)
        self.expectation = 0.0
        self.standard_error = 0.0
        self.z_t = 0.0
        self.diffusion_factor = 0.0
        self.terminal_profit = []

        self.h = self.T / self.no_of_slices

        self.price_list = []

        self.exp_mean = (self.mue - self.div_yield - 0.5 * (self.sigma ** 2)) * self.h
        self.exp_diffusion = self.sigma * np.sqrt(self.h)

        for i in range(self.simulation_rounds):
            self.check = bool(False)
            self.price_t = self.S0

            for j in range(self.no_of_slices * int(self.T)):
                self.z_t = np.random.standard_normal()
                self.exp_factor = self.exp_mean + self.exp_diffusion * self.z_t
                self.price_t = self.price_t * np.exp(self.exp_factor)

                # check
                if self.price_t <= barrier_price:
                    self.check = bool(True)

            if self.check == True:
                if option_type == "call":
                    self.terminal_profit.append(max((self.price_t - self.K), 0.0))
                elif option_type == "put":
                    self.terminal_profit.append(max((self.K - self.price_t), 0.0))
            else:
                self.terminal_profit.append(0.0)

        self.expectation = np.mean(self.terminal_profit) * np.exp(- self.mue * self.T)
        self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit))

        return self.expectation, self.standard_error

    def up_and_out_option(self, barrier_price, option_type):
        """
        :param option_type: call, put
        :return: Option price
        """
        # Stock price simulation
        np.random.seed(self.random_seed)
        self.expectation = 0.0
        self.standard_error = 0.0
        self.z_t = 0.0
        self.diffusion_factor = 0.0
        self.terminal_profit = []

        self.h = self.T / self.no_of_slices

        self.price_list = []

        self.exp_mean = (self.mue - self.div_yield - 0.5 * (self.sigma ** 2)) * self.h
        self.exp_diffusion = self.sigma * np.sqrt(self.h)

        for i in range(self.simulation_rounds):
            self.check = bool(False)
            self.price_t = self.S0

            for j in range(self.no_of_slices * int(self.T)):
                self.z_t = np.random.standard_normal()
                self.exp_factor = self.exp_mean + self.exp_diffusion * self.z_t
                self.price_t = self.price_t * np.exp(self.exp_factor)

                # check
                if self.price_t >= barrier_price:
                    self.check = bool(True)

            if self.check == False:
                if option_type == "call":
                    self.terminal_profit.append(max((self.price_t - self.K), 0.0))
                elif option_type == "put":
                    self.terminal_profit.append(max((self.K - self.price_t), 0.0))
            else:
                self.terminal_profit.append(0.0)

        self.expectation = np.mean(self.terminal_profit) * np.exp(- self.mue * self.T)
        self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit))

        return self.expectation, self.standard_error

    def lookback_monte_carlo(self, option_type):
        """
        :param option_type: call, put
        :return: Option price
        """
        # Stock price simulation
        np.random.seed(self.random_seed)
        self.expectation = 0.0
        self.standard_error = 0.0
        self.z_t = 0.0
        self.diffusion_factor = 0.0
        self.max_price = 0.0
        self.min_price = 0.0
        self.terminal_profit = []

        self.h = self.T / self.no_of_slices

        self.exp_mean = (self.mue - self.div_yield - 0.5 * (self.sigma ** 2)) * self.h
        self.exp_diffusion = self.sigma * np.sqrt(self.h)

        for i in range(self.simulation_rounds):
            self.price_t = self.S0  # reset the stock price for simulation
            self.price_list = []

            for j in range(self.no_of_slices * int(self.T)):
                self.z_t = np.random.standard_normal()
                self.exp_factor = self.exp_mean + self.exp_diffusion * self.z_t
                self.price_t = self.price_t * np.exp(self.exp_factor)
                self.price_list.append(self.price_t)

            self.max_price = max(self.price_list)
            self.min_price = min(self.price_list)

            if option_type == "call":
                self.terminal_profit.append(max((self.max_price - self.K), 0.0))
            elif option_type == "put":
                self.terminal_profit.append(max((self.K - self.min_price), 0.0))

        self.expectation = np.mean(self.terminal_profit) * np.exp(- self.mue * self.T)
        self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit))

        return self.expectation, self.standard_error

    def vasicek_model(self, r0, alpha, b, interest_vol):
        """
        if interest rate is stochastic
        :parameter: alpha: speed of mean-reversion
        b: risk-free rate is mean-reverting to b
        incorporate term structure to model risk free rate (u)
        Interest rate in vasicek model can be negative, which may cause negative bond yield in the long term
        :return:
        """
        # np.random.seed(self.random_seed) # here we cannot fix seed, must let interest dynamics independent from the stock price dynamics
        self.terminal_profit = []
        self.expectation = 0.0
        self.standard_error = 0.0
        self.z_t = 0.0
        self.diffusion_factor = 0.0
        self.price_t = 0.0

        self.h = self.T / self.no_of_slices
        self.diffusion_factor = self.sigma * np.sqrt(self.h)

        self.discount_factor_list = []

        for i in range(self.simulation_rounds):
            self.sum_z_t = []
            self.price_list = []
            self.discount_factor_list = []
            self.discount_factor = r0
            self.price_t = self.S0

            for j in range(self.no_of_slices * int(self.T)):
                # for interest rate simulation
                self.discount_factor = b + np.exp(-alpha / self.no_of_slices) * (self.discount_factor - b) + np.sqrt(
                    interest_vol ** 2 / (2 * alpha) * (
                            1 - np.exp(-2 * alpha / self.no_of_slices))) * np.random.standard_normal()
                self.discount_factor_list.append(self.discount_factor)

                # for drift simulation
                self.exp_mean = (self.discount_factor - self.div_yield - 0.5 * self.sigma ** 2) * self.h

                # for price simulation
                self.price_t = self.price_t * np.exp(
                    self.exp_mean + self.diffusion_factor * np.random.standard_normal())

            self.stochastic_df = np.exp(-np.sum(self.discount_factor_list) / float(self.no_of_slices))
            self.price_list.append(self.price_t)
            self.terminal_profit.append(max((self.price_t - self.K) * self.stochastic_df, 0.0))

        self.expectation = np.mean(self.terminal_profit)

        self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit))

        print(
            "S0 %4.1f | vol %4.2f | T %2.1f | Call Option Value %8.3f | Standard Error %4.2f " % (
                self.S0, self.sigma, self.T, self.expectation, self.standard_error))
        return self.expectation, self.standard_error

    def Cox_Ingersoll_Ross_model(self, r0, alpha, b, interest_vol):
        """
        if asset volatility is stochastic
        incorporate term structure to model risk free rate (u)
        non central chi-square distribution
        Interest rate in CIR model cannot be negative
        :return:
        """
        # np.random.seed(self.random_seed) # here we cannot fix seed, must let interest dynamics independent from the stock price dynamics
        self.terminal_profit = []
        self.expectation = 0.0
        self.standard_error = 0.0
        self.z_t = 0.0
        self.diffusion_factor = 0.0

        self.h = self.T / self.no_of_slices

        self.diffusion_factor = self.sigma * np.sqrt(self.h)

        self.degree_freedom = 4 * b * alpha / float(
            interest_vol ** 2)  # CIR noncentral chi-square distribution degree of freedom

        for i in range(self.simulation_rounds):
            self.sum_z_t = []
            self.price_list = []
            self.discount_factor_list = []
            self.discount_factor = r0
            self.price_t = self.S0

            for j in range(self.no_of_slices * int(self.T)):
                # for interest rate simulation
                self.Lambda = (4 * alpha * np.exp(-alpha / self.no_of_slices) * self.discount_factor / (
                        interest_vol ** 2 * (1 - np.exp(-alpha / self.no_of_slices))))
                self.chi_square_factor = np.random.noncentral_chisquare(df=self.degree_freedom,
                                                                        nonc=self.Lambda)  # Lambda = noncentrality factor
                self.discount_factor = interest_vol ** 2 * (1 - np.exp(-alpha / self.no_of_slices)) / (
                        4 * alpha) * self.chi_square_factor
                self.discount_factor_list.append(self.discount_factor)

                # for drift simulation
                self.exp_mean = (self.discount_factor - self.div_yield - 0.5 * self.sigma ** 2) * self.h

                # for price simulation
                self.price_t = self.price_t * np.exp(
                    self.exp_mean + self.diffusion_factor * np.random.standard_normal())

            self.stochastic_df = np.exp(-np.sum(self.discount_factor_list) / float(self.no_of_slices))
            self.price_list.append(self.price_t)
            self.terminal_profit.append(max((self.price_t - self.K) * self.stochastic_df, 0.0))

        self.expectation = np.mean(self.terminal_profit)

        self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit))

        print(
            "S0 %4.1f | vol %4.2f | T %2.1f | Call Option Value %8.3f | Standard Error %4.2f " % (
                self.S0, self.sigma, self.T, self.expectation, self.standard_error))
        return self.expectation, self.standard_error

    def CIR_Heston(self, r0, alpha_r, b_r, interest_vol, v0, alpha_v, b_v, asset_vol):
        """
        if asset volatility is stochastic
        incorporate term structure to model risk free rate (u)
        non central chi-square distribution
        Interest rate in CIR model cannot be negative
        :return:
        """
        # np.random.seed(self.random_seed) # here we cannot fix seed, must let interest dynamics independent from the stock price dynamics
        self.price_list = []
        self.expectation = 0.0
        self.standard_error = 0.0
        self.z_t = 0.0
        self.diffusion_factor = 0.0
        self.price_t = 0.0

        self.h = self.T / self.no_of_slices

        self.degree_freedom_r = 4 * b_r * alpha_r / float(
            interest_vol ** 2)  # CIR noncentral chi-square distribution degree of freedom
        self.degree_freedom_v = 4 * b_v * alpha_v / float(
            asset_vol ** 2)  # CIR noncentral chi-square distribution degree of freedom

        for i in range(self.simulation_rounds):
            self.price_t = self.S0
            self.discount_factor_list = []
            self.discount_factor = r0
            self.v_t = v0

            for j in range(self.no_of_slices * int(self.T)):
                # for diffusion simulation
                self.Lambda_v = (4 * alpha_v * np.exp(-alpha_v / self.no_of_slices) * self.v_t / (
                        asset_vol ** 2 * (1 - np.exp(-alpha_v / self.no_of_slices))))
                self.chi_square_factor_v = np.random.noncentral_chisquare(df=self.degree_freedom_v,
                                                                          nonc=self.Lambda_v)  # Lambda = noncentrality factor
                self.v_t = asset_vol ** 2 * (1 - np.exp(-alpha_v / self.no_of_slices)) / (
                        4 * alpha_v) * self.chi_square_factor_v
                self.diffusion_factor = np.sqrt(self.v_t * self.h)

                # for interest rate
                self.Lambda_r = (4 * alpha_r * np.exp(-alpha_r / self.no_of_slices) * self.discount_factor / (
                        interest_vol ** 2 * (1 - np.exp(-alpha_r / self.no_of_slices))))
                self.chi_square_factor_r = np.random.noncentral_chisquare(df=self.degree_freedom_r,
                                                                          nonc=self.Lambda_r)  # Lambda = noncentrality factor
                self.discount_factor = interest_vol ** 2 * (1 - np.exp(-alpha_r / self.no_of_slices)) / (
                        4 * alpha_r) * self.chi_square_factor_r
                self.discount_factor_list.append(self.discount_factor)

                # for drift simulation
                self.exp_mean = (self.discount_factor - self.div_yield - 0.5 * self.v_t) * self.h

                # price simulation
                self.price_t = self.price_t * np.exp(
                    self.exp_mean + self.diffusion_factor * np.random.standard_normal())

            self.stochastic_df = np.exp(-np.sum(self.discount_factor_list) / float(self.no_of_slices))
            self.price_list.append(max((self.price_t - self.K) * self.stochastic_df, 0.0))  # for call

        self.expectation = np.mean(self.price_list)

        self.standard_error = np.std(self.price_list) / np.sqrt(len(self.price_list))

        print(
            "S0 %4.1f | vol %4.2f | T %2.1f | Call Option Value %8.3f | Standard Error %4.2f " % (
                self.S0, self.sigma, self.T, self.expectation, self.standard_error))
        return self.expectation, self.standard_error
