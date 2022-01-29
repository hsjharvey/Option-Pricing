# -*- coding:utf-8 -*-
#######################################################################
# Copyright (C) 2016 Shijie Huang (harveyh@student.unimelb.edu.au)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import scipy.stats as sts
from typing import Tuple


class MonteCarloOptionPricing:
    def __init__(self, r, S0: float, K: float, T: float, sigma: float, div_yield: float = 0.0,
                 simulation_rounds: int = 10000, no_of_slices: int = 4, fix_random_seed: bool or int = False):
        """
        An important reminder, by default the implementation assumes constant interest rate and volatility.
        To allow for stochastic interest rate and vol, run Vasicek/CIR for stochastic interest rate and
        run Heston for stochastic volatility.

        :param S0: current price of the underlying asset (e.g. stock)
        :param K: exercise price
        :param T: time to maturity, in years, can be float
        :param r: interest rate, by default we assume constant interest rate model
        :param sigma: volatility (in standard deviation) of the asset annual returns
        :param div_yield: annual dividend yield
        :param simulation_rounds: in general, monte carlo option pricing requires many simulations
        :param no_of_slices: between time 0 and time T, the number of slices PER YEAR, e.g. 252 if trading days are required
        :param fix_random_seed: boolean or integer
        """
        assert sigma >= 0, 'volatility cannot be less than zero'
        assert S0 >= 0, 'initial stock price cannot be less than zero'
        assert T >= 0, 'time to maturity cannot be less than zero'
        assert div_yield >= 0, 'dividend yield cannot be less than zero'
        assert no_of_slices >= 0, 'no of slices per year cannot be less than zero'
        assert simulation_rounds >= 0, 'simulation rounds cannot be less than zero'

        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.div_yield = float(div_yield)

        self.no_of_slices = int(no_of_slices)
        self.simulation_rounds = int(simulation_rounds)

        self._dt = self.T / self.no_of_slices

        self.mue = r  # under risk-neutral measure, asset expected return = risk-free rate
        self.r = np.full((self.simulation_rounds, self.no_of_slices), r * self._dt)
        self.discount_table = np.exp(np.cumsum(-self.r, axis=1))

        self.sigma = np.full((self.simulation_rounds, self.no_of_slices), sigma)

        self.terminal_prices = []

        self.z_t = np.random.standard_normal((self.simulation_rounds, self.no_of_slices))

        if type(fix_random_seed) is bool:
            if fix_random_seed:
                np.random.seed(15000)
        elif type(fix_random_seed) is int:
            np.random.seed(fix_random_seed)

    def vasicek_model(self, a: float, b: float, r_sigma: float) -> np.ndarray:
        """
        When interest rate follows a stochastic process. Vasicek model for interest rate simulation.
        this is the continuous-time analog of the AR(1) process.
        Interest rate in the Vasicek model can be negative. \n

        dr = a(b-r) * dt + r_sigma * dz
        :param a: speed of mean-reversion
        :param b: risk-free rate is mean-reverting to b
        :param r_sigma: interest rate volatility (standard deviation)
        :return:
        """
        _interest_z_t = np.random.standard_normal((self.simulation_rounds, self.no_of_slices))
        _interest_array = np.full((self.simulation_rounds, self.no_of_slices), self.r[0, 0] * self._dt)

        for i in range(1, self.no_of_slices):
            _interest_array[:, i] = b + np.exp(-a / self.no_of_slices) * (_interest_array[:, i - 1] - b) + np.sqrt(
                r_sigma ** 2 / (2 * a) * (1 - np.exp(-2 * a / self.no_of_slices))
            ) * _interest_z_t[:, i]

        # re-define the interest rate array
        self.r = _interest_array

        return _interest_array

    def cox_ingersoll_ross_model(self, a: float, b: float, r_sigma: float) -> np.ndarray:
        """
        When interest rate follows a stochastic process. Incorporate term structure to model risk-free rate (r)
        under non-central chi-square transition probability density. \n
        Note that interest rate in CIR model cannot be negative.

        dr = a(b-4) * dt + r_sigma * sqrt(r) * dz
        """
        _interest_array = np.full((self.simulation_rounds, self.no_of_slices), self.r[0, 0] * self._dt)

        # CIR non-central chi-square distribution degree of freedom
        _dof = 4 * b * a / r_sigma ** 2

        for i in range(1, self.no_of_slices):
            _Lambda = (4 * a * np.exp(-a / self.no_of_slices) * _interest_array[:, i - 1] / (
                    r_sigma ** 2 * (1 - np.exp(-a / self.no_of_slices))))
            _chi_square_factor = np.random.noncentral_chisquare(df=_dof,
                                                                nonc=_Lambda,
                                                                size=self.simulation_rounds)

            _interest_array[:, i] = r_sigma ** 2 * (1 - np.exp(-a / self.no_of_slices)) / (
                    4 * a) * _chi_square_factor

        # re-define the interest rate array
        self.r = _interest_array

        return _interest_array

    def heston(self, kappa: float, theta: float, sigma_v: float, rho: float = 0.0) -> np.ndarray:
        """
        When asset volatility (variance NOT sigma!) follows a stochastic process.

        dv(t) = kappa[theta - v(t)] * dt + sigma_v * sqrt(v(t)) * dZ
        :param: kappa: rate at which vt reverts to theta
        :param: theta: long-term variance
        :param: sigma_v: sigma of the volatility
        :param: rho: correlation between the volatility and the rate of return
        :return: stochastic volatility array
        """
        _variance_v = sigma_v ** 2
        assert (2 * kappa * theta > _variance_v)  # Feller condition

        # step 1: simulate correlated zt
        _mu = np.array([0, 0])
        _cov = np.array([[1, rho], [rho, 1]])

        _zt = sts.multivariate_normal.rvs(mean=_mu, cov=_cov, size=(self.simulation_rounds, self.no_of_slices))
        _variance_array = np.full((self.simulation_rounds, self.no_of_slices),
                                  self.sigma[0, 0] ** 2)
        self.z_t = _zt[:, :, 0]
        _zt_v = _zt[:, :, 1]

        # step 2: simulation
        for i in range(1, self.no_of_slices):
            _drift = kappa * (theta - _variance_array[:, i - 1]) * self._dt
            _diffusion = sigma_v * np.sqrt(_variance_array[:, i - 1]) * _zt_v[:, i - 1] * np.sqrt(self._dt)
            _delta_vt = _drift + _diffusion
            _variance_array[:, i] = _variance_array[:, i - 1] + _delta_vt

        # re-define the interest rate and volatility path
        self.sigma = np.sqrt(_variance_array)

        return self.sigma

    def stock_price_simulation(self) -> Tuple[np.ndarray, float]:
        self.exp_mean = (self.mue - self.div_yield - (self.sigma ** 2.0) * 0.5) * self._dt
        self.exp_diffusion = self.sigma * np.sqrt(self._dt)

        self.price_array = np.zeros((self.simulation_rounds, self.no_of_slices))
        self.price_array[:, 0] = self.S0

        for i in range(1, self.no_of_slices):
            self.price_array[:, i] = self.price_array[:, i - 1] * np.exp(
                self.exp_mean[:, i - 1] + self.exp_diffusion[:, i - 1] * self.z_t[:, i - 1]
            )

        self.terminal_prices = self.price_array[:, -1]
        self.stock_price_expectation = np.mean(self.terminal_prices)
        self.stock_price_standard_error = np.std(self.terminal_prices) / np.sqrt(len(self.terminal_prices))

        print('-' * 64)
        print(
            " Number of simulations %4.1i \n S0 %4.1f \n K %2.1f \n Maximum Stock price %4.2f \n"
            " Minimum Stock price %4.2f \n Average stock price %4.3f \n Standard Error %4.5f " % (
                self.simulation_rounds, self.S0, self.K, np.max(self.terminal_prices),
                np.min(self.terminal_prices), self.stock_price_expectation, self.stock_price_standard_error
            )
        )
        print('-' * 64)

        return self.stock_price_expectation, self.stock_price_standard_error

    def stock_price_simulation_with_poisson_jump(self, jump_alpha: float, jump_std: float, poisson_lambda: float) -> \
            Tuple[np.ndarray, float]:
        """
        jump diffusion model
        Parameters
        ----------
        jump_alpha: jump size mean
        jump_std: jump size standard deviation
        poisson_lambda: average number of jumps per year.

        Returns
        -------

        """
        self.z_t_stock = np.random.standard_normal((self.simulation_rounds, self.no_of_slices))
        self.price_array = np.zeros((self.simulation_rounds, self.no_of_slices))
        self.price_array[:, 0] = self.S0

        self.k = np.exp(jump_alpha) - 1

        self.exp_mean = (self.mue - self.div_yield - poisson_lambda * self.k - (self.sigma ** 2.0) * 0.5) * self._dt
        self.exp_diffusion = self.sigma * np.sqrt(self._dt)

        for i in range(1, self.no_of_slices):
            # poisson jump
            self.sum_w = []
            self.m = np.random.poisson(lam=poisson_lambda, size=self.simulation_rounds)

            for j in self.m:
                self.sum_w.append(np.sum(np.random.standard_normal(j)))

            self.poisson_jump_factor = np.exp(
                self.m * (jump_alpha - 0.5 * jump_std ** 2) + jump_std * np.array(self.sum_w)
            )

            self.price_array[:, i] = self.price_array[:, i - 1] * np.exp(
                self.exp_mean[:, i] + self.exp_diffusion[:, i]
                * self.z_t_stock[:, i]
            ) * self.poisson_jump_factor

        self.terminal_prices = self.price_array[:, -1]
        self.stock_price_expectation = np.mean(self.terminal_prices)
        self.stock_price_standard_error = np.std(self.terminal_prices) / np.sqrt(len(self.terminal_prices))

        print('-' * 64)
        print(
            " Number of simulations %4.1i \n S0 %4.1f \n K %2.1f \n Maximum Stock price %4.2f \n"
            " Minimum Stock price %4.2f \n Average stock price %4.3f \n Standard Error %4.5f " % (
                self.simulation_rounds, self.S0, self.K, np.max(self.terminal_prices),
                np.min(self.terminal_prices), self.stock_price_expectation, self.stock_price_standard_error
            )
        )
        print('-' * 64)

        return self.stock_price_expectation, self.stock_price_standard_error

    def european_call(self) -> Tuple[float, float]:
        assert len(self.terminal_prices) != 0, 'Please simulate the stock price first'

        self.terminal_profit = np.maximum((self.terminal_prices - self.K), 0.0)

        self.expectation = np.mean(self.terminal_profit * np.exp(-np.sum(self.r, axis=1)))
        self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit))

        print('-' * 64)
        print(
            " European call monte carlo \n S0 %4.1f \n K %2.1f \n"
            " Call Option Value %4.3f \n Standard Error %4.5f " % (
                self.S0, self.K, self.expectation, self.standard_error
            )
        )
        print('-' * 64)

        return self.expectation, self.standard_error

    def european_put(self, empirical_call: float or None = None) -> float:
        """
        Use put call parity (incl. continuous dividend) to calculate the put option value
        :param empirical_call: can be calculated or observed call option value
        :return: put option value
        """
        if empirical_call is not None:
            self.european_call_value = self.european_call()
        else:
            self.european_call_value = empirical_call

        self.put_value = self.european_call_value + np.exp(-np.sum(self.r, axis=1)) * self.K - np.exp(
            -self.div_yield * self.T) * self.S0

        return self.put_value

    def asian_avg_price_option(self, avg_method: str = 'arithmetic', option_type: str = 'call') -> Tuple[float, float]:
        assert option_type == 'call' or option_type == 'put', 'option_type must be either call or put'
        assert len(self.terminal_prices) != 0, 'Please simulate the stock price first'
        assert avg_method == 'arithmetic' or avg_method == 'geometric', 'arithmetic or geometric average?'

        average_prices = np.average(self.price_array, axis=1)

        if option_type == 'call':
            self.terminal_profit = np.maximum((average_prices - self.K), 0.0)
        elif option_type == 'put':
            self.terminal_profit = np.maximum((self.K - average_prices), 0.0)

        if avg_method == 'arithmetic':
            self.expectation = np.mean(self.terminal_profit * np.exp(-np.sum(self.r, axis=1)))
        elif avg_method == 'geometric':
            self.expectation = sts.gmean(self.terminal_profit * np.exp(-np.sum(self.r, axis=1)))

        self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit))

        print('-' * 64)
        print(
            " Asian %s monte carlo arithmetic average \n S0 %4.1f \n K %2.1f \n"
            " Option Value %4.3f \n Standard Error %4.5f " % (
                option_type, self.S0, self.K, self.expectation, self.standard_error
            )
        )
        print('-' * 64)

        return self.expectation, self.standard_error

    def american_option_longstaff_schwartz(self, poly_degree: int = 2, option_type: str = 'call') -> \
            Tuple[float, float]:
        """
        American option, Longstaff and Schwartz method

        :param poly_degree: x^n, default = 2
        :param option_type: call or put
        """
        assert option_type == 'call' or option_type == 'put', 'option_type must be either call or put'
        assert len(self.terminal_prices) != 0, 'Please simulate the stock price first'

        if option_type == 'call':
            self.intrinsic_val = np.maximum((self.price_array - self.K), 0.0)
        elif option_type == 'put':
            self.intrinsic_val = np.maximum((self.K - self.price_array), 0.0)

        # last day cashflow == last day intrinsic value
        cf = self.intrinsic_val[:, -1]

        stopping_rule = np.zeros_like(self.price_array)
        stopping_rule[:, -1] = np.where(self.intrinsic_val[:, -1] > 0, 1, 0)

        # Longstaff and Schwartz iteration
        for t in range(self.no_of_slices - 2, 0, -1):  # fill out the value table from backwards
            # find out in-the-money path to better estimate the conditional expectation function
            # where exercise is relevant and significantly improves the efficiency of the algorithm
            itm_path = np.where(self.intrinsic_val[:, t] > 0)  # <==> self.price_array[:, t] vs. self.K

            cf = cf * np.exp(-self.r[:, t + 1])
            Y = cf[itm_path]
            X = self.price_array[itm_path, t]

            # initialize continuation value
            hold_val = np.zeros(shape=self.simulation_rounds)
            # if there is only 5 in-the-money paths (most likely appear in out-of-the-money options
            # then simply assume that value of holding = 0.
            # otherwise, run regression and compute conditional expectation E[Y|X].
            if len(itm_path) > 5:
                rg = np.polyfit(x=X[0], y=Y, deg=poly_degree)  # regression fitting
                hold_val[itm_path] = np.polyval(p=rg, x=X[0])  # conditional expectation E[Y|X]

            # 1 <==> exercise, 0 <==> hold
            stopping_rule[:, t] = np.where(self.intrinsic_val[:, t] > hold_val, 1, 0)
            # if exercise @ t, all future stopping rules = 0 as the option contract is exercised.
            stopping_rule[np.where(self.intrinsic_val[:, t] > hold_val), (t + 1):] = 0

            # cashflow @ t, if hold, cf = 0, if exercise, cf = intrinsic value @ t.
            cf = np.where(self.intrinsic_val[:, t] > 0, self.intrinsic_val[:, t], 0)

        simulation_vals = (self.intrinsic_val * stopping_rule * self.discount_table).sum(axis=1)
        self.expectation = np.average(simulation_vals)
        self.standard_error = np.std(simulation_vals) / np.sqrt(self.simulation_rounds)

        print('-' * 64)
        print(
            " American %s Longstaff-Schwartz method (assume polynomial fit)"
            " \n polynomial degree = %i \n S0 %4.1f \n K %2.1f \n"
            " Option Value %4.3f \n Standard Error %4.5f " % (
                option_type, poly_degree, self.S0, self.K, self.expectation, self.standard_error
            )
        )
        print('-' * 64)

        return self.expectation, self.standard_error

    def barrier_option(self, option_type: str, barrier_price: float, barrier_type: str, barrier_direction: str,
                       parisian_barrier_days: int or None = None) -> Tuple[float, float]:
        assert option_type == "call" or option_type == "put", 'option type must be either call or put'
        assert barrier_type == "knock-in" or barrier_type == "knock-out", \
            'barrier type must be either knock-in or knock-out'
        assert barrier_direction == "up" or barrier_direction == "down", \
            'barrier direction must be either up or down'
        if barrier_direction == "up":
            barrier_check = np.where(self.price_array >= barrier_price, 1, 0)

            if parisian_barrier_days is not None:
                days_to_slices = int(parisian_barrier_days * self.no_of_slices / (self.T * 252))
                parisian_barrier_check = np.zeros((self.simulation_rounds, self.no_of_slices - days_to_slices))
                for i in range(0, self.no_of_slices - days_to_slices):
                    parisian_barrier_check[:, i] = np.where(
                        np.sum(barrier_check[:, i:i + days_to_slices], axis=1) >= parisian_barrier_days, 1, 0
                    )

                barrier_check = parisian_barrier_check

        elif barrier_direction == "down":
            barrier_check = np.where(self.price_array <= barrier_price, 1, 0)

            if parisian_barrier_days is not None:
                days_to_slices = int(parisian_barrier_days * self.no_of_slices / (self.T * 252))
                parisian_barrier_check = np.zeros((self.simulation_rounds, self.no_of_slices - days_to_slices))
                for i in range(0, self.no_of_slices - days_to_slices):
                    parisian_barrier_check[:, i] = np.where(
                        np.sum(barrier_check[:, i:i + days_to_slices], axis=1) >= parisian_barrier_days, 1, 0
                    )

                barrier_check = parisian_barrier_check

        if option_type == 'call':
            self.intrinsic_val = np.maximum((self.price_array - self.K), 0.0)
        elif option_type == 'put':
            self.intrinsic_val = np.maximum((self.K - self.price_array), 0.0)

        if barrier_type == "knock-in":
            self.terminal_profit = np.where(np.sum(barrier_check, axis=1) >= 1, self.intrinsic_val[:, -1], 0)
        elif barrier_type == "knock-out":
            self.terminal_profit = np.where(np.sum(barrier_check, axis=1) >= 1, 0, self.intrinsic_val[:, -1])

        self.expectation = np.mean(self.terminal_profit * np.exp(-np.sum(self.r, axis=1)))
        self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit))

        print('-' * 64)
        print(
            " Barrier european %s \n Type: %s \n Direction: %s @ %s \n S0 %4.1f \n K %2.1f \n"
            " Option Value %4.3f \n Standard Error %4.5f " % (
                option_type, barrier_type, barrier_direction, barrier_price,
                self.S0, self.K, self.expectation, self.standard_error
            )
        )
        print('-' * 64)

        return self.expectation, self.standard_error

    def look_back_european(self, option_type: str = 'call') -> Tuple[float, float]:
        assert len(self.terminal_prices) != 0, 'Please simulate the stock price first'
        assert option_type == 'call' or option_type == 'put', 'option_type must be either call or put'

        self.max_price = np.max(self.price_array, axis=1)
        self.min_price = np.min(self.price_array, axis=1)

        if option_type == "call":
            self.terminal_profit = np.maximum((self.max_price - self.K), 0.0)
        elif option_type == "put":
            self.terminal_profit = np.maximum((self.K - self.min_price), 0.0)

        self.expectation = np.mean(self.terminal_profit * np.exp(-np.sum(self.r, axis=1)))
        self.standard_error = np.std(self.terminal_profit) / np.sqrt(len(self.terminal_profit))

        print('-' * 64)
        print(
            " Lookback european %s monte carlo \n S0 %4.1f \n K %2.1f \n"
            " Option Value %4.3f \n Standard Error %4.5f " % (
                option_type, self.S0, self.K, self.expectation, self.standard_error
            )
        )
        print('-' * 64)
        return self.expectation, self.standard_error
