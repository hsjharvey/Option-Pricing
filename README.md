# Option Pricing
The repository is a re-factorization (and some extensions) of the code for an FNCE40009 Advanced Derivative Securities assignment.
> This repository is mainly designed for those who want to study option pricing.
> It is built purely on standard NumPy and SciPy packages.
>
> I've tried my best to utilize the speed offered by NumPy array operation rather than multiple for-loops, 
> without sacrificing the ability to refer the code back to the equations in the textbook.
>
> Debug/Improvements/AddOns will be implemented from time to time.
>
> If you have any questions or want to report a bug, please open an issue. I am open to any corrections/suggestions.
> If you like it, or if it benefits you in some way, kindly give it a star 😊

<br/>

Reference: most of this repository implements algorithms described in the following textbook.
```
@book{mcdonald2013derivatives,
  title={Derivatives markets (3rd edition)},
  author={McDonald, Robert Lynch},
  year={2013},
  publisher={Pearson Education}
}
```

## Binomial Excel Illustration

<br/>

<p align="center">
    <img width="700px" src="./binomial_excel/binomial_illustration.png">
</p>

<br/>

## Black-Scholes-Merton (BSM)
How to use: `example_BSM.py`
 * Chapter 12.1: European call `call_value()`, European put `put_value()`
 * Chapter 12.2: (class) `GarmanKohlhagenForex`
 * Chapter 21.5: Merton jump diffusion `merton_jump_diffusion()`
 * Chapter 12.5: implied volatility assume BSM model `implied_vol()`
 * Appendix 12.B: `delta()`, `gamma()`, `theta()`, `vega()`, `rho()`, `psi()`

<br/>


## Monte Carlo Simulation
How to use: `example_monte_carlo.py`
* Chapter 18 & 19.3: `stock_price_simulation()`
* Chapter 19.8: `stock_price_simulation_with_poisson_jump()`

<br/>

The final predicted price is the average of all the simulated prices at time T (column T).

| Stock price S | h | 2h  | 3h | ... |T-h|                T |
|:-------------:|:---------:| -----:| -----:| -----:| -----:|-----------------:|
| simulation 1  |   |   |  |  |  |  S<sub>1,T</sub> |
| simulation 2  |   |   |  |  |  |  S<sub>2,T</sub> |
| simulation 3  |   |   |  |  |  |  S<sub>3,T</sub> |
|      ...      |   |   |  |  |  |              ... |

* `vasicek()` and `cox_ingersoll_ross_model()` and `CIR_Heston()`: interest rate and volatility simulation.

Note: the default case is two tables with constant values. Call these two functions 
before running `stock_price_simulation()`.

| Interest rate r | h | 2h  | 3h | ... |T-h|               T |
| :---------: |:---------:| -----:| -----:| -----:| -----:|----------------:|
| simulation 1 |   |   |  |  |  | r<sub>1,T</sub> |
| simulation 2 |   |   |  |  |  | r<sub>2,T</sub> |
| simulation 3 |   |   |  |  |  | r<sub>3,T</sub> |
| ... |   |   |  |  |  |             ... |

| Volatility &sigma; | h | 2h  | 3h | ... |T-h|                     T |
|:------------------:|:---------:| -----:| -----:| -----:| -----:|----------------------:|
|    simulation 1    |   |   |  |  |  | &sigma;<sub>1,T</sub> |
|    simulation 2    |   |   |  |  |  | &sigma;<sub>2,T</sub> |
|    simulation 3    |   |   |  |  |  | &sigma;<sub>3,T</sub> |
|        ...         |   |   |  |  |  |                   ... |

* Chapter 19.4: `european_call()` & `european_put()`
* Chapter 14.2 & 19.4: `asian_avg_price()`
* Chapter 19.6: `american_option_longstaff_schwartz()`, Longstaff & Schwartz (2001).
* Chapter 14 & 23: exotic options, e.g. `barrier_option()`
* Chapter 23, exercise 23.12: `look_back_european()`

<br/>

## More References:
* Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. Journal of financial economics, 3(1-2), 125-144.
* Garman, M. B. and Kohlhagen, S. W. (1983). "Foreign Currency Option Values." Journal of International Money and Finance 2, 231-237.
* Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options by simulation: a simple least-squares approach. The review of financial studies, 14(1), 113-147.

<br/>

## For Fun:
![Alt](https://repobeats.axiom.co/api/embed/b002e14d33548265f332c023aa55c9df8ee9e6c1.svg "Repobeats analytics image")
