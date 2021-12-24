# -*- coding:utf-8 -*-
#######################################################################
# Copyright (C) 2016 Shijie Huang (harveyh@student.unimelb.edu.au)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from BSM_option_class import BSMOptionValuation

S0 = 100  # e.g. spot price = 35
K = 100  # e.g. exercise price = 40
T = 1.0  # e.g. six months
r = 0.05  # e.g. risk free rate = 1%
sigma = 0.25  # e.g. volatility = 5%
div_yield = 0.0  # e.g. dividend yield = 1%

# initialize
x = BSMOptionValuation(S0, K, T, r, sigma, div_yield)

# Get observed call and put price
observed_call_price = 146.75
observed_put_price = 120

# Calculate call price
call_price_cal = x.call_value()  # calculated call price
call_price_obs = x.call_value(observed_put_price)  # observed call price

# Calculate greeks
delta = x.delta()
gamma = x.gamma()
theta = x.theta()
vega = x.vega()
rho = x.rho()
psi = x.psi()

# Calculate implied volatility
implied_volatility = x.implied_vol(observed_call_price=observed_call_price, num_iterations=1000, tolerance=1e-4)

# Calculate put price
put_price_cal = x.put_value()  # using calculated call price
put_price_obs = x.put_value(observed_call_price=observed_call_price)  # using observed call price

# Calculate lookback option price
# Step 1: simulate stock price over the option life (use Monte Carlo)
# Step 2: input Max and Min stock price and calculate the lookback option price
# (150, 50) is the (max, min) price over option life
lookback_call = x.lookback_BSM(option_type="call", max_share_price=150, min_share_price=50)

# Results
print("=" * 64)
print("Call price using calculations: %.3f" % call_price_cal)
print("Call price using observed put: %.3f" % call_price_obs)
print("Delta of the call: %.3f | put: %.3f" % delta)
print("Gamma is: %.3f" % gamma)
print("Theta of the call: %.3f | put: %.3f" % theta)
print("Vega is: %.3f" % vega)
print("Rho of the call: %.3f | put: %.3f" % rho)
print("Psi of the call: %.3f | put: %.3f" % psi)
print("Implied volatility: %.3f" % implied_volatility)
print("Put price using the calculated call: %.3f" % put_price_cal)
print("Put price using the observed call: %.3f" % put_price_obs)
print("Lookback call price is: " + str(lookback_call))
print("=" * 64)
