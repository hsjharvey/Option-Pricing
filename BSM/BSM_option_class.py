#######################################################################
# Copyright (C) 2016 Shijie Huang (harveyh@student.unimelb.edu.au)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from math import log, sqrt, exp
from scipy import stats


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


    Methods:
    ==========
    value: float
        return present value of call option
    vega: float
        return vega of call option
    imp_vol: float
        return implied volatility given option quote
    """

    def __init__(self, S0, K, T, r, sigma, div_yield=0.0):
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

        self.d1 = ((log(self.S0 / self.K) + (self.r - self.div_yield + 0.5 * self.sigma ** 2) * self.T) / (
                self.sigma * sqrt(self.T)))
        self.d2 = self.d1 - self.sigma * sqrt(self.T)

    def call_value(self, observed_put_price=None):
        """
        :return: call option value
        """
        if observed_put_price is None:
            call_value = (self.S0 * exp(-self.div_yield * self.T) * stats.norm.cdf(self.d1, 0.0, 1.0) - self.K * exp(
                -self.r * self.T) * stats.norm.cdf(self.d2, 0.0, 1.0))
        else:
            call_value = observed_put_price + exp(-self.div_yield * self.T) * self.S0 - exp(-self.r * self.T) * self.K

        return call_value

    def delta(self):
        """
        Delta measures the change in the option price for a $1 change in the stock price
        :return: delta of the option
        """
        delta_call = exp(- self.div_yield * self.T) * stats.norm.cdf(self.d1, 0.0, 1.0)
        delta_put = -exp(- self.div_yield * self.T) * stats.norm.cdf(-self.d1, 0.0, 1.0)

        return delta_call, delta_put

    def gamma(self):
        """
        Gamma measures the change in delta when the stock price changes
        :return: gamma of the option
        """
        gamma = exp(-self.div_yield * self.T) * stats.norm.pdf(self.d1) / (self.S0 * self.sigma * sqrt(self.T))

        return gamma

    def theta(self):
        """
        Theta measures the change in the option price with respect to calendar time (t ),
        holding fixed time to expiration (T).

        If time to expiration is measured in years, theta will be the annualized change in the option value.
        To obtain a per-day theta, divide by 365.
        :return: theta of the option
        """
        part1 = self.div_yield * self.S0 * exp(-self.div_yield * self.T) * stats.norm.cdf(self.d1)
        part2 = self.r * self.K * stats.norm.cdf(self.d2)
        part3 = (self.K * exp(-self.r * self.T) * stats.norm.pdf(self.d2) * self.sigma) / (2 * sqrt(self.T))

        theta_call = part1 - part2 - part3
        theta_put = theta_call + self.r * self.K * exp(-self.r * self.T) - self.div_yield * self.S0 * exp(
            -self.div_yield * self.T)

        return theta_call, theta_put

    def vega(self):
        """
        Vega measures the change in the option price when volatility changes. Some writers also
        use the terms lambda or kappa to refer to this measure:
        It is common to report vega as the change in the option price per percentage point change
        in the volatility. This requires dividing the vega formula above by 100.
        :return: vega of option
        """
        vega = self.S0 * exp(-self.div_yield * self.T) * stats.norm.pdf(self.d1, 0.0, 1.0) * sqrt(self.T)

        return vega

    def rho(self):
        """
        Returns: call_rho, put_rho
        -------
        Rho is the partial derivative of the option price with respect to the interest rate.
        These expressions for rho assume a change in r of 1.0. We are typically interested in
        evaluating the effect of a change of 0.01 (100 basis points) or 0.0001 (1 basis point). To
        report rho as a change per percentage point in the interest rate, divide this measure by 100.
        To interpret it as a change per basis point, divide by 10,000.
        """
        call_rho = self.T * self.K * exp(-self.r * self.T) * stats.norm.cdf(self.d2)
        put_rho = -self.T * self.K * exp(-self.r * self.T) * stats.norm.cdf(-self.d2)

        return call_rho, put_rho

    def psi(self):
        """
        Returns: call_psi, put psi
        -------
        Psi is the partial derivative of the option price with respect to the continuous dividend yield:
        To interpret psi as a price change per percentage point change in the dividend yield, divide
        by 100.
        """
        call_psi = - self.T * self.S0 * exp(-self.div_yield * self.T) * stats.norm.cdf(self.d1)
        put_psi = self.T * self.S0 * exp(-self.div_yield * self.T) * stats.norm.cdf(-self.d1)

        return call_psi, put_psi

    def implied_vol(self, observed_call_price, iteration=1000):
        """
        Newton-Raphson iterative approach, assuming BSM model
        :param C0: observed call option value
        :param observed_call_price: call price from the market
        :param sigma_est: estimated volatility, starting value
        :param iteration: no. of iteration
        :return: implied volatility given option price
        """

        for _ in range(iteration):
            self.sigma -= (self.call_value() - observed_call_price) / self.vega()

        return self.sigma

    def put_value(self, observed_call_price=None):
        """
        Use put call parity (incl. continuous dividend) to calculate the put option value
        :return: put option value
        """
        if observed_call_price is None:
            put_value = self.call_value() + exp(-self.r * self.T) * self.K - exp(-self.div_yield * self.T) * self.S0
        else:
            put_value = observed_call_price + exp(-self.r * self.T) * self.K - exp(-self.div_yield * self.T) * self.S0

        return put_value

    def lookback_BSM(self, option_type, max_share_price, min_share_price):
        """
        A European lookback call at maturity pays St - min(St).
        A European lookback put at maturity pays max(St) - St.
        min(St) is the minimum price over the life of the option
        max(St) is the maximum price over the life of the option
        Robert. L. MacDonald: Derivatives Markets (3rd. edition)
        Chapter 23: Exotic Option II
        Formula 23.47 (Exercise)
        :param option_type: call, put
        share_price_max share_price_min
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
