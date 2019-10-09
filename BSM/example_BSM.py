#######################################################################
# Copyright (C) 2016 Shijie Huang (harveyh@student.unimelb.edu.au)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from BSM.BSM_option_class import *

S0 = 2168.27  # e.g. spot price = 35
K = 2150  # e.g. exercise price = 40
T = 1.2  # e.g. six months
r = 0.006  # e.g. risk free rate = 1%
sigma = 0.1684  # e.g. volatility = 5%
div_yield = 0.0205  # e.g. dividend yield = 1%

# initialize
x = BSMOptionValuation(S0, K, T, r, sigma, div_yield)

# Get observed call and put price
observed_call_price = 146.75
observed_put_price = 120

# Calculate call price
call_price = x.call_value()
call_price_obs = x.call_value(observed_put_price)

# Calculate greeks
delta = x.delta()
gamma = x.gamma()
theta = x.theta()
vega = x.vega()

# Calculate implied volatility
implied_volatility = x.imp_vol(C0=observed_call_price)

# Calculate put price
put_price_cal = x.put_value()  # using calculated call price
put_price_obs = x.put_value(observed_call_price=observed_call_price)  # using observed call price

# Calculate lookback option price
# Step 1: simulate stock price over the option life (use Monte Carlo)
# Step 2: input Max and Min stock price and calculate the lookback option price
# (150, 50) is the (max, min) price over option life
lookback_call = x.lookback_BSM(option_type="call", max_share_price=150, min_share_price=50)

# Results
print("Call price using calculations: " + str(call_price))
print("Call price using observed put: " + str(call_price_obs))
print("Delta of the call: " + str(delta[0]) + "    " + "Delta of put: " + str(delta[1]))
print("Gamma is: " + str(gamma))
print("Theta of the call: " + str(theta[0]) + "    " + "Theta of put: " + str(theta[1]))
print("Vega is: " + str(vega))
print("Implied volatility: " + str(implied_volatility))
print("Put price using the calculated call: " + str(put_price_cal))
print("Put price using the observed call: " + str(put_price_obs))
print("=" * 64)
print("Lookback call price is: " + str(lookback_call))
