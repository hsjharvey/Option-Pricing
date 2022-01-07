# -*- coding:utf-8 -*-
#######################################################################
# Copyright (C) 2016 Shijie Huang (harveyh@student.unimelb.edu.au)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
import numpy as np
from numpy import log, exp, sqrt
from scipy import stats
from typing import Tuple


class BSMOptionValuation:
    """
    Valuation of European call options in Black-Scholes-Merton Model (incl. dividend)
    Attributes
    ==========
    S0: float
        initial stock/index level
    K: float
        strike price
    T: float
        time to maturity (in year fractions)
    r: float
        constant risk-free short rate
        assume flat term structure
    sigma: float
        volatility factor in diffusion term
    div_yield: float
        dividend_yield, in percentage %, default = 0.0%
    """

    def __init__(self, S0: float, K: float, T: float, r: float, sigma: float, div_yield: float = 0.0):
        assert sigma >= 0, 'volatility cannot be less than zero'
        assert S0 >= 0, 'initial stock price cannot be less than zero'
        assert T >= 0, 'time to maturity cannot be less than zero'
        assert div_yield >= 0, 'dividend yield cannot be less than zero'

        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.div_yield = float(div_yield)

        self.d1, self.d2 = self._calculate_d1_d2()

    def _calculate_d1_d2(self):
        d1 = ((log(self.S0 / self.K) + (self.r - self.div_yield + 0.5 * self.sigma ** 2) * self.T) / (
                self.sigma * sqrt(self.T)))
        d2 = d1 - self.sigma * sqrt(self.T)

        return d1, d2

    def call_value(self, observed_put_price: float = None) -> float:
        """
        :return: call option value
        """
        if observed_put_price is None:
            call_value = (self.S0 * exp(-self.div_yield * self.T) * stats.norm.cdf(self.d1, 0.0, 1.0) - self.K * exp(
                -self.r * self.T) * stats.norm.cdf(self.d2, 0.0, 1.0))
        else:
            call_value = observed_put_price + exp(-self.div_yield * self.T) * self.S0 - exp(-self.r * self.T) * self.K

        return call_value

    def delta(self) -> Tuple[float, float]:
        """
        Delta measures the change in the option price for a $1 change in the stock price
        :return: delta of the option
        """
        delta_call = exp(- self.div_yield * self.T) * stats.norm.cdf(self.d1, 0.0, 1.0)
        delta_put = -exp(- self.div_yield * self.T) * stats.norm.cdf(-self.d1, 0.0, 1.0)

        return delta_call, delta_put

    def gamma(self) -> float:
        """
        Gamma measures the change in delta when the stock price changes
        :return: gamma of the option
        """
        gamma = exp(-self.div_yield * self.T) * stats.norm.pdf(self.d1) / (self.S0 * self.sigma * sqrt(self.T))

        return gamma

    def theta(self) -> Tuple[float, float]:
        """
        Theta measures the change in the option price with respect to calendar time (t ),
        holding fixed time to expiration (T).

        If time to expiration is measured in years, theta will be the annualized change in the option value.
        To obtain a per-day theta, divide by 252.
        :return: theta of the option
        """
        part1 = self.div_yield * self.S0 * exp(-self.div_yield * self.T) * stats.norm.cdf(self.d1)
        part2 = self.r * self.K * stats.norm.cdf(self.d2)
        part3 = (self.K * exp(-self.r * self.T) * stats.norm.pdf(self.d2) * self.sigma) / (2 * sqrt(self.T))

        theta_call = part1 - part2 - part3
        theta_put = theta_call + self.r * self.K * exp(-self.r * self.T) - self.div_yield * self.S0 * exp(
            -self.div_yield * self.T)

        return theta_call, theta_put

    def vega(self) -> float:
        """
        Vega measures the change in the option price when volatility changes. Some writers also
        use the terms lambda or kappa to refer to this measure:
        It is common to report vega as the change in the option price per percentage point change
        in the volatility. This requires dividing the vega formula above by 100.
        :return: vega of option
        """
        vega = self.S0 * exp(-self.div_yield * self.T) * stats.norm.pdf(self.d1, 0.0, 1.0) * sqrt(self.T)

        return vega

    def rho(self) -> Tuple[float, float]:
        """
        Rho is the partial derivative of the option price with respect to the interest rate.
        These expressions for rho assume a change in r of 1.0. We are typically interested in
        evaluating the effect of a change of 0.01 (100 basis points) or 0.0001 (1 basis point). To
        report rho as a change per percentage point in the interest rate, divide this measure by 100.
        To interpret it as a change per basis point, divide by 10,000.
        :return: call_rho, put_rho
        """
        call_rho = self.T * self.K * exp(-self.r * self.T) * stats.norm.cdf(self.d2)
        put_rho = -self.T * self.K * exp(-self.r * self.T) * stats.norm.cdf(-self.d2)

        return call_rho, put_rho

    def psi(self) -> Tuple[float, float]:
        """
        Psi is the partial derivative of the option price with respect to the continuous dividend yield:
        To interpret psi as a price change per percentage point change in the dividend yield, divide
        by 100.
        :return: call_psi, put_psi
        """
        call_psi = - self.T * self.S0 * exp(-self.div_yield * self.T) * stats.norm.cdf(self.d1)
        put_psi = self.T * self.S0 * exp(-self.div_yield * self.T) * stats.norm.cdf(-self.d1)

        return call_psi, put_psi

    def implied_vol(self, observed_call_price: float, num_iterations: int = 1000, tolerance: float = 1e-4) -> float:
        """
        Newton-Raphson iterative approach, assuming BSM model
        :param observed_call_price: call price from the market
        :param num_iterations: no. of iteration
        :param tolerance: allows to specify the tolerance level
        :return: implied volatility given the observed option price
        """
        implied_vol = self.sigma

        for _ in range(num_iterations):
            new_implied_vol = (self.call_value() - observed_call_price) / self.vega()

            if abs(implied_vol - new_implied_vol) <= tolerance:
                break

            implied_vol = new_implied_vol

        return implied_vol

    def put_value(self, observed_call_price: float = None) -> float:
        """
        Use put call parity (incl. continuous dividend) to calculate the put option value

        :return: put option value
        """
        if observed_call_price is None:
            put_value = self.call_value() + exp(-self.r * self.T) * self.K - exp(-self.div_yield * self.T) * self.S0
        else:
            put_value = observed_call_price + exp(-self.r * self.T) * self.K - exp(-self.div_yield * self.T) * self.S0

        return put_value

    def lookback_BSM(self, option_type: str, max_share_price: float, min_share_price: float) -> float:
        """
        A European lookback call at maturity pays St - min(St).
        A European lookback put at maturity pays max(St) - St.
        min(St) is the minimum price over the life of the option
        max(St) is the maximum price over the life of the option
        Robert. L. MacDonald: Derivatives Markets (3rd. edition)
        Chapter 23: Exotic Option II
        Formula 23.47 (Exercise)
        :param option_type: call, put
        :param max_share_price: maximum share price
        :param min_share_price: minimum share price
        :return: value of lookback option
        """

        assert option_type == "call" or option_type == "put"

        if option_type == "call":
            self.w = 1
            self.s_bar = float(min_share_price)

        elif option_type == "put":
            self.w = -1
            self.s_bar = float(max_share_price)

        self.d5 = (log(self.K / self.s_bar) + (self.r - self.div_yield + 0.5 * (self.sigma ** 2)) * self.T) / (
                self.sigma * sqrt(self.T))
        self.d6 = self.d5 - self.sigma * sqrt(self.T)
        self.d7 = (log(self.s_bar / self.K) + (self.r - self.div_yield + 0.5 * (self.sigma ** 2)) * self.T) / (
                self.sigma * sqrt(self.T))
        self.d8 = self.d7 - self.sigma * sqrt(self.T)

        # Lookback option pricing
        self.lb_first_part = self.w * self.K * exp(-self.div_yield * self.T) * (
                stats.norm.cdf(self.w * self.d5) - (self.sigma ** 2) * stats.norm.cdf(-self.w * self.d5) / (
                2 * (self.r - self.div_yield)))
        self.lb_second_part = self.w * self.s_bar * exp(-self.r * self.T) * (stats.norm.cdf(self.w * self.d6) - (
                (self.sigma ** 2) / (2 * (self.r - self.div_yield)) * (self.K / self.s_bar) ** (
                1 - 2 * (self.r - self.div_yield) / (self.sigma ** 2))) * stats.norm.cdf(self.w * self.d8))

        return self.lb_first_part - self.lb_second_part

    def merton_jump_diffusion(self, option_type: str, avg_num_jumps: int, jump_size_mean: float, jump_size_std: float):
        """
        assuming jump size follows a log-normal distribution: ln(jump_size) ~ N(jump_size_mean, jump_size_std)
        Parameters
        ----------
        option_type: (str) call or put
        avg_num_jumps: (int) how many jumps in T
        jump_size_mean: (float) ln(jump_size) ~ N(jump_size_mean, jump_size_std)
        jump_size_std: (float) ln(jump_size) ~ N(jump_size_mean, jump_size_std)

        Returns
        -------

        """
        assert option_type == "call" or option_type == "put", "option type must be either call or put"

        LAMBDA = avg_num_jumps
        I = np.random.poisson(lam=LAMBDA)  # simulate the number of jumps based on poisson distribution

        alpha_j = jump_size_mean
        sigma_j, variance_j = jump_size_std, jump_size_std ** 2

        LAMBDA_hat = LAMBDA * exp(alpha_j + 0.5 * sigma_j ** 2)
        option_value = 0

        sigma, variance = self.sigma, self.sigma ** 2  # this is the raw volatility of the underlying asset
        r = self.r  # this is the raw interest rate
        d1, d2 = self.d1, self.d2  # this is the raw d1, d1 for Black-Scholes option pricing

        for i in range(I):
            # to calculate adjusted Black-Scholes option value
            # note this is ad hoc
            self.sigma = sqrt(variance + i * variance_j / self.T)
            self.r = r - LAMBDA * (exp(alpha_j + 0.5 * sigma_j ** 2) - 1) + i * (alpha_j + 0.5 * sigma_j ** 2) / self.T

            # re-calculate d1 d2 for Black-Scholes option pricing component
            # note this is ad hoc
            self._calculate_d1_d2()

            jump_diffusion_scale = exp(-LAMBDA_hat * self.T) * (LAMBDA_hat * self.T) ** i / np.math.factorial(i)
            print(jump_diffusion_scale)
            print(self.call_value())

            if option_type == "call":
                option_value += jump_diffusion_scale * self.call_value()
            else:
                option_value += jump_diffusion_scale * self.put_value()

        # re-assign the original variables in case other methods need to use the two values.
        self.sigma, self.r, self.d1, self.d2 = sigma, r, d1, d2

        return option_value


class GarmanKohlhagenForex(BSMOptionValuation):
    """
    Valuation of European call options in Black-Scholes-Merton Model (for forex)
    Garman, M. B. and Kohlhagen, S. W. "Foreign Currency Option Values." Journal of
    International Money and Finance 2, 231-237, 1983.
    Price, J. F. "Optional Mathematics is Not Optional." Not. Amer. Math. Soc. 43, 964-971, 1996.

    Attributes
    ==========
    S0: float
        current spot rate: units of domestic currency per unit of foreign currency
    K: float
        strike price: units of domestic currency per unit of foreign currency
    T: float
        maturity (in year fractions)
    rd: float
        domestic risk free interest rate
        assume flat term structure
    rf: float
        foreign risk free interest rate
    sigma: float
        volatility factor in diffusion term

    Methods:
    ==========
    call_value: float
        return value of a call option on forex (in domestic currecy)
    """

    def __init__(self, S0, K, T, rd, rf, sigma):
        BSMOptionValuation.__init__(S0, S0, K, T, rd, rf, sigma)
        assert sigma >= 0, 'volatility cannot be less than zero'
        assert S0 >= 0, 'initial stock price cannot be less than zero'
        assert T >= 0, 'time to maturity cannot be less than zero'

        self.S0 = float(S0)
        self.K = float(K)
        self.T = float(T)
        self.rf = float(rf)
        self.rd = float(rd)
        self.sigma = float(sigma)

        self.d1 = ((log(self.S0 / self.K) + (self.rd - self.rf + 0.5 * self.sigma ** 2) * self.T) / (
                self.sigma * sqrt(self.T)))

        self.d2 = self.d1 - self.sigma * sqrt(self.T)

    def call_value(self, empirical_put_price=None):
        """
        :return: call option value
        """
        if empirical_put_price is None:
            call_value = (self.S0 * exp(- self.rf * self.T) * stats.norm.cdf(self.d1, 0.0, 1.0) - self.K * exp(
                - self.rd * self.T) * stats.norm.cdf(self.d2, 0.0, 1.0))
        else:
            call_value = empirical_put_price + exp(-self.div_yield * self.T) * self.S0 - exp(-self.r * self.T) * self.K

        return call_value

    def put_value(self, empirical_call_price=None):
        """
        Use put call parity (incl. continuous dividend) to calculate the put option value
        :return: put option value
        """
        if empirical_call_price is None:
            put_value = self.K * exp(- self.rd * self.T) * stats.norm.cdf(- self.d2, 0.0, 1.0) - self.S0 * exp(
                - self.rf * self.T) * stats.norm.cdf(- self.d1, 0.0, 1.0)
        else:
            put_value = empirical_call_price + exp(-self.r * self.T) * self.K - exp(-self.div_yield * self.T) * self.S0

        return put_value
