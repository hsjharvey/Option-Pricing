# -*- coding:utf-8 -*-
#######################################################################
# Copyright (C) 2016 Shijie Huang (harveyh@student.unimelb.edu.au)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################


import numpy as np
import scipy.stats as sts


class MonteCarloOptionPricing:
    def __init__(self, S0, K, T, mue, sigma, div_yield=0.0, simulation_rounds=10000, no_of_slices=4,
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

        self.h = self.T / self.no_of_slices

        self.exp_mean = (self.mue - self.div_yield - (self.sigma ** 2.0) * 0.5) * self.h
        self.exp_diffusion = self.sigma * np.sqrt(self.h)

        self.terminal_prices = []

        if fix_random_seed:
            np.random.seed(12000)

    def stock_price_simulation(self):
        """
        Based on Geometric Brownian Motion
        :return:
        """
        self.expectation = 0.0
        self.standard_error = 0.0
        self.z_t = 0.0

        self.z_t = np.random.standard_normal((self.simulation_rounds, self.no_of_slices))

        self.price_array = np.zeros((self.simulation_rounds, self.no_of_slices))

        self.price_array[:, 0] = self.S0

        for i in range(1, self.no_of_slices):
            self.price_array[:, i] = self.price_array[:, i - 1] * np.exp(
                self.exp_mean + self.exp_diffusion * self.z_t[:, i]
            )

        self.terminal_prices = self.price_array[:, -1]
        self.stock_price_expectation = np.mean(self.terminal_prices)
        self.stock_price_standard_error = np.std(self.terminal_prices) / np.sqrt(len(self.terminal_prices))

        print('-' * 64)
        print(
            " number of simulations %4.1i \n S0 %4.1f \n vol %4.2f \n T %2.1f \n Maximum Stock price %4.2f \n"
            " Minimum Stock price %4.2f \n Average stock price %84.3f \n Standard Error %4.5f " % (
                self.simulation_rounds, self.S0, self.sigma, self.T, np.max(self.terminal_prices),
                np.min(self.terminal_prices), self.stock_price_expectation, self.stock_price_standard_error
            )
        )
        print('-' * 64)

        return self.stock_price_expectation, self.stock_price_standard_error

    def Cox_Ingersoll_Ross_stock_price_simulation(self, r0, alpha, b, interest_vol):
        pass

    def CIR_Heston_stock_price_simulation(self, r0, alpha_r, b_r, interest_vol, v0, alpha_v, b_v, asset_vol):
        pass

    def vasicek_stock_price_simulation(self, r0, alpha, b, interest_vol):
        pass

    def european_call(self):
        assert len(self.terminal_prices) != 0, 'Please simulate the stock price first'

        self.terminal_profit = np.maximum((self.terminal_prices - self.K), 0.0)

        self.expectation = np.mean(self.terminal_profit) * np.exp(- self.mue * self.T)
        self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit))

        print('-' * 64)
        print(
            " European call monte carlo \n S0 %4.1f \n vol %4.2f \n T %2.1f \n "
            "Call Option Value %4.3f \n Standard Error %4.5f " % (
                self.S0, self.sigma, self.T, self.expectation, self.standard_error
            )
        )
        print('-' * 64)

        return self.expectation, self.standard_error

    def european_put(self):
        """
        Use put call parity (incl. continuous dividend) to calculate the put option value
        :param call_value: can be calculated or observed call option value
        :return: put option value
        """
        self.european_call_value = self.european_call()
        self.put_value = self.european_call_value + np.exp(-self.mue * self.T) * self.K - np.exp(
            -self.div_yield * self.T) * self.S0

        return self.put_value

    def asian_avg_price_call(self, avg_method='arithmetic'):
        """
        Asian call using average price method
        Arithmetic average
        :return: asian call value
        """
        assert len(self.terminal_prices) != 0, 'Please simulate the stock price first'
        assert avg_method == 'arithmetic' or avg_method == 'geometric', 'arithmetic or geometric average?'

        average_prices = np.average(self.price_array, axis=1)

        self.terminal_profit = np.maximum((average_prices - self.K), 0.0)

        if avg_method == 'arithmetic':
            self.expectation = np.mean(self.terminal_profit) * np.exp(- self.mue * self.T)
        elif avg_method == 'geometric':
            self.expectation = sts.gmean(self.terminal_profit) * np.exp(- self.mue * self.T)

        self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit))

        print('-' * 64)
        print(
            " Asian call monte carlo arithmetic average \n S0 %4.1f \n vol %4.2f \n T %2.1f \n "
            "Call Option Value %4.3f \n Standard Error %4.5f " % (
                self.S0, self.sigma, self.T, self.expectation, self.standard_error
            )
        )
        print('-' * 64)

    def asian_avg_price_put(self, avg_method='arithmetic'):
        """
        Asian put using average price method
        Arithmetic average
        :return: asian put value
        """
        assert len(self.terminal_prices) != 0, 'Please simulate the stock price first'
        assert avg_method == 'arithmetic' or avg_method == 'geometric', 'arithmetic or geometric average?'

        average_prices = np.average(self.price_array, axis=1)

        self.terminal_profit = np.maximum((self.K - average_prices), 0.0)

        if avg_method == 'arithmetic':
            self.expectation = np.mean(self.terminal_profit) * np.exp(- self.mue * self.T)
        elif avg_method == 'geometric':
            self.expectation = sts.gmean(self.terminal_profit) * np.exp(- self.mue * self.T)

        self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit))

        print('-' * 64)
        print(
            " Asian call monte carlo arithmetic average \n S0 %4.1f \n vol %4.2f \n T %2.1f \n "
            "Call Option Value %4.3f \n Standard Error %4.5f " % (
                self.S0, self.sigma, self.T, self.expectation, self.standard_error
            )
        )
        print('-' * 64)

    def american_call(self, poly_degree=2):
        """
        American call option
        Longstaff and Schwartz method
        :param poly_degree: x^n, default = 2
        :return:
        """
        assert len(self.terminal_prices) != 0, 'Please simulate the stock price first'

        self.dt = self.T / self.no_of_slices  # time interval
        self.dis_factor = np.exp(- self.mue * self.dt)  # discount factor per time time interval

        self.intrinsic_val = np.maximum((self.price_array - self.K), 0.0)
        self.value_matrix = np.zeros_like(self.intrinsic_val)  # sample shape
        self.value_matrix[:, -1] = self.intrinsic_val[:, -1]  # last day american option value = intrinsic value

        # Longstaff and Schwartz
        for t in range(self.no_of_slices - 2, 0, -1):  # fill out the value table from backwards
            self.rg = np.polyfit(x=self.price_array[:, t], y=self.value_matrix[:, t + 1] * self.dis_factor,
                                 deg=poly_degree)  # regression fitting
            self.hold_val = np.polyval(p=self.rg, x=self.price_array[:, t])  # regression estimated value

            # determine hold or exercise
            self.value_matrix[:, t] = np.where(self.intrinsic_val[:, t] > self.hold_val, self.intrinsic_val[:, t],
                                               self.value_matrix[:, t + 1] * self.dis_factor)

        self.american_call_val = np.average(self.value_matrix[:, 1]) * self.dis_factor
        self.am_std_error = np.std(self.value_matrix[:, 1] * self.dis_factor) / np.sqrt(self.simulation_rounds)

        print('-' * 64)
        print(
            " American call Long Staff method \n polynomial degree = %i \n S0 %4.1f \n vol %4.2f \n T %2.1f \n "
            "Call Option Value %4.3f \n Standard Error %4.5f " % (
                poly_degree, self.S0, self.sigma, self.T, self.american_call_val, self.am_std_error
            )
        )
        print('-' * 64)

        return self.american_call_val, self.am_std_error

    def american_put(self, poly_degree):
        """
        American put option
        Longstaff and Schwartz method
        :param poly_degree: x^n, default = 2
        :return:
        """
        assert len(self.terminal_prices) != 0, 'Please simulate the stock price first'

        self.dt = self.T / self.no_of_slices  # time interval
        self.dis_factor = np.exp(- self.mue * self.dt)  # discount factor per time time interval

        self.intrinsic_val = np.maximum((self.K - self.price_array), 0.0)
        self.value_matrix = np.zeros_like(self.intrinsic_val)  # sample shape
        self.value_matrix[:, -1] = self.intrinsic_val[:, -1]  # last day american option value = intrinsic value

        # Longstaff and Schwartz
        for t in range(self.no_of_slices - 2, 0, -1):  # fill out the value table from backwards
            self.rg = np.polyfit(x=self.price_array[:, t], y=self.value_matrix[:, t + 1] * self.dis_factor,
                                 deg=poly_degree)  # regression fitting
            self.hold_val = np.polyval(p=self.rg, x=self.price_array[:, t])  # regression estimated value

            # determine hold or exercise
            self.value_matrix[:, t] = np.where(self.intrinsic_val[:, t] > self.hold_val, self.intrinsic_val[:, t],
                                               self.value_matrix[:, t + 1] * self.dis_factor)

        self.american_put_val = np.average(self.value_matrix[:, 1]) * self.dis_factor
        self.am_std_error = np.std(self.value_matrix[:, 1] * self.dis_factor) / np.sqrt(self.simulation_rounds)

        print('-' * 64)
        print(
            " American put Long Staff method \n polynomial degree = %i \n S0 %4.1f \n vol %4.2f \n T %2.1f \n "
            "Call Option Value %4.3f \n Standard Error %4.5f " % (
                poly_degree, self.S0, self.sigma, self.T, self.american_put_val, self.am_std_error
            )
        )
        print('-' * 64)

        return self.american_put_val, self.am_std_error

    def down_and_in_parisian_monte_carlo(self, barrier_price, option_type, barrier_condition=1):
        assert option_type == 'call' or option_type == 'put', 'option_type must be either call or put'
        assert type(barrier_condition) is int, 'barrier condition must be integer, i.e. how many consecutive days'

        self.check = np.where(self.price_array <= barrier_price, 1, 0)
        self.terminal_profit = np.zeros(self.simulation_rounds)

        # parisian check
        self.check_final = np.zeros((self.simulation_rounds, self.no_of_slices - barrier_condition))
        for i in range(0, self.no_of_slices - barrier_condition):
            self.check_final[:, i] = np.where(
                np.sum(self.check[:, i:i + barrier_condition], axis=1) >= barrier_condition,
                1, 0)

        if option_type == 'call':
            self.intrinsic_val = np.maximum((self.price_array - self.K), 0.0)
        elif option_type == 'put':
            self.intrinsic_val = np.maximum((self.K - self.price_array), 0.0)

        self.terminal_profit = np.where(np.sum(self.check_final, axis=1) >= 1, self.intrinsic_val[:, -1], 0)

        self.expectation = np.mean(self.terminal_profit) * np.exp(- self.mue * self.T)
        self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit))

        print('-' * 64)
        print(
            " down-and-in %s monte carlo \n S0 %4.1f \n vol %4.2f \n T %2.1f \n "
            "Option Value %4.3f \n Standard Error %4.5f " % (
                option_type, self.S0, self.sigma, self.T, self.expectation, self.standard_error
            )
        )
        print('-' * 64)

        return self.expectation, self.standard_error

    def down_and_out_parisian_monte_carlo(self, barrier_price, option_type, barrier_condition=1):
        assert option_type == 'call' or option_type == 'put', 'option_type must be either call or put'
        assert type(barrier_condition) is int, 'barrier condition must be integer, i.e. how many consecutive days'

        self.check = np.where(self.price_array >= barrier_price, 1, 0)
        self.terminal_profit = np.zeros(self.simulation_rounds)

        # parisian check
        self.check_final = np.zeros((self.simulation_rounds, self.no_of_slices - barrier_condition))
        for i in range(0, self.no_of_slices - barrier_condition):
            self.check_final[:, i] = np.where(
                np.sum(self.check[:, i:i + barrier_condition], axis=1) >= barrier_condition,
                1, 0)

        if option_type == 'call':
            self.intrinsic_val = np.maximum((self.price_array - self.K), 0.0)
        elif option_type == 'put':
            self.intrinsic_val = np.maximum((self.K - self.price_array), 0.0)

        self.terminal_profit = np.where(np.sum(self.check_final, axis=1) >= 1, 0, self.intrinsic_val[:, -1])

        self.expectation = np.mean(self.terminal_profit) * np.exp(- self.mue * self.T)
        self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit))

        print('-' * 64)
        print(
            " down-and-in %s monte carlo \n S0 %4.1f \n vol %4.2f \n T %2.1f \n "
            "Option Value %4.3f \n Standard Error %4.5f " % (
                option_type, self.S0, self.sigma, self.T, self.expectation, self.standard_error
            )
        )
        print('-' * 64)

        return self.expectation, self.standard_error

    def up_and_in_parisian_monte_carlo(self, barrier_price, barrier_condition, option_type):
        pass

    def up_and_out_parisian_monte_carlo(self, barrier_price, barrier_condition, option_type):
        pass

    def LookBack(self):
        pass


if __name__ == '__main__':
    # initialize parameters
    S0 = 100.0  # e.g. spot price = 35
    K = 100.0  # e.g. exercise price = 40
    T = 1.0  # e.g. one year
    r = 0.05  # e.g. risk free rate = 1%
    sigma = 0.25  # e.g. volatility = 5%
    div_yield = 0.0  # e.g. dividend yield = 1%
    no_of_slice = 252  # e.g. quarterly adjusted

    barrier_price = 80.0  # barrier level for barrier options
    barrier_condition = 21  # no.of consecutive trading days required for parisian options

    # optional parameter
    simulation_rounds = 100000  # For monte carlo simulation, a large number of simulations required

    # initialize
    MT = MonteCarloOptionPricing(S0, K, T, r, sigma, div_yield, simulation_rounds=simulation_rounds,
                                 no_of_slices=no_of_slice, fix_random_seed=True)
    MT.stock_price_simulation()
    # MT.european_call()
    # MT.asian_avg_price_call()
    # MT.asian_avg_price_put()
    # MT.american_call(poly_degree=2)
    # MT.american_put(poly_degree=2)
    MT.down_and_in_parisian_monte_carlo(barrier_price=barrier_price, option_type='call',
                                        barrier_condition=barrier_condition)
