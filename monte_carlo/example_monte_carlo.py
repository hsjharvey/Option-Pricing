from monte_carlo_class import *

# To compare answers, fix random seed
# To compare with results in matlab, please type rand('twister', 150000); in matlab code
random_seed = 150000

# initialize parameters
S0 = 100.0  # e.g. spot price = 35
K = 100.0  # e.g. exercise price = 40
T = 1  # e.g. one year
r = 0.05  # e.g. risk free rate = 1%
sigma = 0.25  # e.g. volatility = 5%
div_yield = 0.0  # e.g. dividend yield = 1%
no_of_slice = 252  # e.g. quarterly adjusted

barrier_price = 80.0  # barrier level for barrier options
barrier_condition = 21  # no.of days required for parisian options

# optional parameter
simulation_rounds = 500

# initialize
x = monte_carlo_option_pricing(S0, K, T, r, sigma, div_yield, simulation_rounds=simulation_rounds,
                               no_of_slice=no_of_slice, random_seed=random_seed)

# Stock price
# results = x.stock_price_simulation()

# European call
# results = European_call_price = x.european_call()

# European put
# results = European_put_price = x.european_put()

# Asian call (average price)
# results = Asian_call_price = x.asian_avg_price_call()

# Asian put (average price)
# results = Asian_put_price = x.asian_avg_price_put()

# American call
# results = American_call_price = x.american_call()

# American put
# results = American_put_price = x.american_put()

# Down-and-in
# results = x.down_and_in_option(barrier_price, "call") # option_type: call, put

# Down-and-out
# results = x.down_and_out_option(barrier_price, "call")

# Up-and-in
# results = x.up_and_in_option(barrier_price, "call")

# Up-and-out
# results = x.up_and_out_option(barrier_price, "call")

# Parisian option: down_and_in
# results = x.down_and_in_parisian_monte_carlo(barrier_price, barrier_condition, "call") # barrier condition must be satisfied for X days

# Parisian option: down_and_out
# results = x.down_and_out_parisian_monte_carlo(barrier_price, barrier_condition, "call")

# Parisian option: up_and_in
# results = x.up_and_in_parisian_monte_carlo(barrier_price, barrier_condition, "call")

# Parisian option: up_and_out
# results = x.up_and_out_parisian_monte_carlo(barrier_price, barrier_condition, "call")

# Lookback call
# results = x.lookback_monte_carlo("call")


# Value for special interest rate models

r0 = 0.05
alpha_r = 0.2
b_r = 0.05
interest_vol = 0.1

# Vasciek
# results = x.vasicek_model(r0=r0, alpha=alpha_r, b=b_r, interest_vol=interest_vol)

# Cox Ingersoll Ross
results = x.Cox_Ingersoll_Ross_model(r0=r0, alpha=alpha_r, b=b_r, interest_vol=interest_vol)

# CIR Heston
v0 = 0.25 ** 2
alpha_v = 0.1
b_v = 0.05
asset_vol = 0.1
# results = x.CIR_Heston(r0=r0, alpha_r=alpha_r, b_r=b_r,interest_vol=interest_vol,v0=v0,alpha_v=alpha_v,b_v=b_v,asset_vol=asset_vol)

print(results)
