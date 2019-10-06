import numpy as np
from scipy.special import erfcinv

# Using complementary error correction function (erfc and erfc inverse)

p = 0.05 # probability

S0 = float(100)
T = float(1)
mue = float(0.1)
sigma = float(0.2)

St = np.exp(np.log(S0) + (mue - sigma**2 / 2) - sigma * np.sqrt(2 * T) * erfcinv(2 * p))


Value_at_Risk = S0 - St

print("St = " + str(St))
print("VaR = " + str(Value_at_Risk))