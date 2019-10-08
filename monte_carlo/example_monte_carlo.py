# -*- coding:utf-8 -*-
#######################################################################
# Copyright (C) 2016 Shijie Huang(harveyh@student.unimelb.edu.au)     #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
from monte_carlo.monte_carlo_class import *

# initialize parameters
S0 = 100.0  # e.g. spot price = 35
K = 120.0  # e.g. exercise price = 40
T = 1.0  # e.g. one year
mue = 0.05  # e.g. expected return of the asset, under risk neutral assumption, mue = r
r = 0.05  # e.g. risk free rate = 1%
sigma = 0.25  # e.g. volatility = 5%
div_yield = 0.0  # e.g. dividend yield = 1%
no_of_slice = 252  # e.g. quarterly adjusted

barrier_price = 80.0  # barrier level for barrier options
barrier_condition = 21  # no.of consecutive trading days required for parisian options

# optional parameter
simulation_rounds = 100000  # For monte carlo simulation, a large number of simulations required

# initialize
MT = MonteCarloOptionPricing(r, S0, K, T, mue, sigma, div_yield, simulation_rounds=simulation_rounds,
                             no_of_slices=no_of_slice, fix_random_seed=True)
MT.stock_price_simulation()
# MT.stock_price_simulation_with_poisson_jump(jump_alpha=0.1, jump_std=0.25, poisson_lambda=0)
MT.european_call()
MT.asian_avg_price(avg_method='arithmetic', option_type='call')
MT.american_option_monte_carlo(poly_degree=2, option_type='put')
MT.down_and_in_parisian_monte_carlo(barrier_price=barrier_price, option_type='call',
                                    barrier_condition=barrier_condition)

MT.LookBackEuropean()
