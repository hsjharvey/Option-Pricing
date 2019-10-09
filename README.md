# Option Pricing
The repo is a refactorization of the code for an assignment in FNCE40009 Advanced Financial Derivatives.
> If you have any questions or want to report a bug, please open an issue.
```
@misc{optionPricingHuang,
  author = {Shijie Huang},
  title = {option pricing},
  year = {2016},
  publisher = {GitHub},
  journal = {GitHub Repository},
}
```
Citation: The majority of the code is an implementation of the algorithms described in the following textbook.
```
@book{mcdonald2006derivatives,
  title={Derivatives markets},
  author={McDonald, Robert Lynch and Cassano, Mark and Fahlenbrach, R{\"u}diger},
  year={2006},
  publisher={Addison-Wesley Boston}
}
```
## Black-Scholes-Merton (BSM)



## Monte Carlo Simulation
T: period, in years. e.g. 1.0 = 1year

Number of slices: e.g. 252 trading days.


* `stock_price_simulation: `
$$ S_{t+h} = S_t * e^{} $$
Final estimated price is the average of all the simulated price at time T (column T).

| Stock price | h | 2h  | 3h | ... |T-h|T|
| :---------: |:---------:| -----:| -----:| -----:| -----:|----:|
| simulation 1 |   |   |  |  |  | $S_{1,T}$ |
| simulation 2 |   |   |  |  |  | $S_{2,T}$ |
| simulation 3 |   |   |  |  |  | $S_{3,T}$|
| ... |   |   |  |  |  | ... |

* `vasicek: ` and `Cox_Ingersoll_Ross_model` and `CIR_Heston`: interest rate and volatility simulation.
Note: the default case is two tables with constant values. You have to call these two functions before stock_price_simulation

| Interest rate r | h | 2h  | 3h | ... |T-h|T|
| :---------: |:---------:| -----:| -----:| -----:| -----:|----:|
| simulation 1 |   |   |  |  |  | $r_{1,T}$ |
| simulation 2 |   |   |  |  |  | $r_{2,T}$ |
| simulation 3 |   |   |  |  |  | $r_{3,T}$|
| ... |   |   |  |  |  | ... |

| Sigma or volatility| h | 2h  | 3h | ... |T-h|T|
| :---------: |:---------:| -----:| -----:| -----:| -----:|----:|
| simulation 1 |   |   |  |  |  | $\sigma_{1,T}$ |
| simulation 2 |   |   |  |  |  | $\sigma_{2,T}$ |
| simulation 3 |   |   |  |  |  | $\sigma_{3,T}$|
| ... |   |   |  |  |  | ... |