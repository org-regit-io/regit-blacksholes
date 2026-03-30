# MATH.md — regit-blackscholes

> Full formula derivations for every algorithm implemented in this crate.
> Each section maps to a source module and cites the primary paper reference.
> All formulas are shown in LaTeX-style notation using code blocks.

---

## Table of contents

1. [Normal PDF](#normal-pdf--srcmathrs)
2. [Normal CDF via erfc](#normal-cdf-via-erfc--srcmathrs)
3. [d1 and d2](#d1-and-d2--srcmathrs)
4. [Black-Scholes-Merton](#black-scholes-merton--srcmodelsblack_scholesrs)
5. [Black-76](#black-76--srcmodelsblack76rs)
6. [Bachelier (Normal Model)](#bachelier-normal-model--srcmodelsbachelierrs)
7. [Displaced Log-Normal](#displaced-log-normal--srcmodelsdisplacedrs)
8. [Greeks — all 17, analytic](#greeks--all-17-analytic--srcgreeksrs)
9. [Implied Volatility Solver Chain](#implied-volatility-solver-chain--srcivrs)

---

## Normal PDF — `src/math.rs`

**Source:** Direct from the definition of the standard normal density.

The standard normal probability density function:

```
phi(x) = exp(-x^2 / 2) / sqrt(2 * pi)
```

The implementation precomputes the constant `1 / sqrt(2 * pi) = 0.3989422804014327`.

### Overflow guard

When `x^2 / 2 > 700`, the exponential `exp(-x^2/2)` underflows below the f64 subnormal range (since `exp(-709.78) ~ 5e-309` and `exp(-710) = 0` in IEEE 754 double precision). The function returns `0.0` directly, preventing any NaN or Inf from propagating into downstream pricing formulas.

### Symmetry

`phi(-x) = phi(x)` holds exactly because the formula depends only on `x^2`.

---

## Normal CDF via erfc — `src/math.rs`

**Source:** Cody, W.J., "Rational Chebyshev approximations for the error function", *Mathematics of Computation*, 22(107):631-637 (1969). Implementation follows Sun fdlibm `s_erf.c`, which uses Cody's coefficients.

**Secondary reference:** Abramowitz & Stegun, *Handbook of Mathematical Functions*, section 26.2.17 (1964).

The standard normal CDF is defined as:

```
N(x) = integral from -inf to x of phi(t) dt
```

Rather than evaluating this integral directly, the implementation uses the relationship between the normal CDF and the complementary error function:

```
N(x) = (1/2) * erfc(-x / sqrt(2))
```

This formulation is numerically superior to the naive `N(x) = (1 + erf(x/sqrt(2))) / 2` because it avoids subtractive cancellation in both tails. The complementary form `Q(x) = 1 - N(x)` is provided separately as:

```
Q(x) = (1/2) * erfc(x / sqrt(2))
```

Computing `Q` via `1.0 - N(x)` would lose approximately 15 significant digits in the right tail (e.g., `Q(6) ~ 1e-9`). The direct erfc-based form preserves full f64 precision.

### erfc implementation — four regions

The complementary error function `erfc(a) = 1 - erf(a)` is evaluated over four regions, each using a different rational polynomial approximation optimised for that domain. All polynomial evaluations use Horner's method with explicit `mul_add` at every step.

**Negative arguments** are handled via the identity:

```
erfc(-a) = 2 - erfc(a)
```

So the core implementation only needs to handle `a >= 0`.

#### Region 1: Tiny — |x| < 2^(-56) ~ 1.39e-17

For arguments this small, `erf(x) ~ 2x/sqrt(pi)` to machine precision. The quadratic term `x^2` vanishes in f64 arithmetic. The implementation returns `erfc(x) = 1.0` directly (the correction from `erf(x)` is below the f64 unit roundoff).

#### Region 2: Small — |x| < 0.84375

```
erf(x) = x + x * P(x^2) / Q(x^2)
erfc(x) = 1 - erf(x)
```

where `P` is a degree-4 polynomial and `Q` is a degree-5 polynomial (monic, leading coefficient 1). The numerator coefficients are `PP0..PP4` and the denominator coefficients are `QQ1..QQ5`, all from fdlibm (Cody 1969, Table II).

Horner evaluation of the numerator:

```
P(z) = ((((PP4 * z + PP3) * z + PP2) * z + PP1) * z + PP0)
```

where `z = x^2`. Each multiplication-addition step is a single `mul_add` call.

#### Region 3: Medium — 0.84375 <= |x| < 1.25

```
erfc(x) = erfc(1) + P(|x| - 1) / Q(|x| - 1)
```

This is an expansion around `x = 1` where `erfc(1) = 0.15729920705028513` is precomputed to full f64 precision. `P` is degree 6, `Q` is degree 6 (monic). Coefficients are `PA0..PA6` and `QA1..QA6` from fdlibm (Cody 1969, Table III adapted).

Expanding around `x = 1` rather than using the small-x formula avoids loss of relative accuracy in the transition region where `erf(x)` is close to 1.

#### Region 4: Large — 1.25 <= |x| < 28

```
erfc(x) = exp(-z^2 - 0.5625) * exp((z - x)(z + x) + R(1/x^2) / S(1/x^2)) / x
```

where `z` is `x` truncated to approximately 28 significant bits (by masking the lower 32 bits of the f64 representation). This **exp-splitting** technique is critical for precision: `exp(-x^2)` loses significant digits when `x` is large because `x^2` grows much faster than the exponent can represent. By writing:

```
exp(-x^2) = exp(-z^2 - 0.5625) * exp(0.5625 + (z - x)(z + x))
```

the first factor `exp(-z^2 - 0.5625)` is computed with a simpler argument (because `z^2` is exactly representable), and the second factor `exp(0.5625 + (z-x)(z+x))` captures the residual with full precision.

Two sub-regions use different rational polynomial coefficients:

- **Sub-region a: 1.25 <= |x| < 1/0.35 ~ 2.857.** R is degree 7, S is degree 8 (monic). Coefficients: `RA0..RA7`, `SA1..SA8`.
- **Sub-region b: 2.857 <= |x| < 28.** R is degree 6, S is degree 7 (monic). Coefficients: `RB0..RB6`, `SB1..SB7`.

All coefficients are from fdlibm `s_erf.c` (Cody 1969, Tables III-IV).

#### Region 5: Extreme — |x| >= 28

`erfc(x)` underflows to `0.0` in f64 representation. The function returns `0.0` directly (for positive `x`) or `2.0` (for negative `x`).

### Accuracy target

Maximum absolute error < 1e-15 over the full f64 domain. Verified by the symmetry identity `N(x) + N(-x) = 1.0` holding to 1e-15 for `x` in `[-8, 8]` at step 0.1, and by comparison with known reference values from Abramowitz & Stegun.

---

## d1 and d2 — `src/math.rs`

**Source:** Black & Scholes, *JPE* (1973); Merton, *Bell Journal of Economics* (1973).

The Black-Scholes `d1` and `d2` parameters appear in every formula throughout the crate:

```
d1 = [ln(S / K) + (r - q + sigma^2 / 2) * T] / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)
```

where:
- `S` = spot price
- `K` = strike price
- `r` = risk-free rate (continuous, annualised)
- `q` = continuous dividend yield (annualised)
- `sigma` = implied volatility (annualised)
- `T` = time to expiry in years

### Interpretation

`d1` and `d2` are the standardised log-moneyness of the option, adjusted for the drift of the underlying. Under the risk-neutral measure:

- `N(d2)` = risk-neutral probability that the call expires in-the-money
- `N(d1)` = `N(d2)` adjusted for the delta-hedging numeraire change (share measure vs money-market measure)

### Implementation notes

`d2` is computed from a precomputed `d1` value to avoid redundant computation of `sigma * sqrt(T)` and `ln(S/K)`. The `d2` function takes `d1_val` as its first argument rather than recomputing from scratch.

---

## Black-Scholes-Merton — `src/models/black_scholes.rs`

**Source:** Black, F. & Scholes, M., "The Pricing of Options and Corporate Liabilities", *Journal of Political Economy*, 81(3):637-654 (1973). Merton, R.C., "Theory of Rational Option Pricing", *Bell Journal of Economics and Management Science*, 4(1):141-183 (1973).

### Derivation

Under the risk-neutral measure, the spot price follows geometric Brownian motion with continuous dividend yield `q`:

```
dS / S = (r - q) dt + sigma dW
```

The Merton (1973) continuous-dividend form prices a European option as the discounted expected payoff:

```
Call = exp(-r * T) * E[max(S_T - K, 0)]
Put  = exp(-r * T) * E[max(K - S_T, 0)]
```

Since `ln(S_T)` is normally distributed with mean `ln(S) + (r - q - sigma^2/2) * T` and variance `sigma^2 * T`, evaluating the expectation yields:

```
Call = S * exp(-q * T) * N(d1) - K * exp(-r * T) * N(d2)
Put  = K * exp(-r * T) * N(-d2) - S * exp(-q * T) * N(-d1)
```

where `d1` and `d2` are as defined above.

### Put-call parity

The formulas satisfy put-call parity exactly:

```
Call - Put = S * exp(-q * T) - K * exp(-r * T)
```

This identity is verified in tests to machine epsilon (1e-10).

### Edge cases

**T = 0 (at expiry).** The option value is the undiscounted intrinsic payoff:

```
Call = max(S - K, 0)
Put  = max(K - S, 0)
```

Returned via `PricingError::IntrinsicOnly` so the caller can distinguish this degenerate case.

**sigma = 0 (deterministic forward).** When volatility is zero, the forward price is known with certainty. `N(d1)` and `N(d2)` become step functions (0 or 1 depending on whether the forward is above or below the strike). The price reduces to the discounted intrinsic of the forward:

```
Call = max(S * exp(-q * T) - K * exp(-r * T), 0)
Put  = max(K * exp(-r * T) - S * exp(-q * T), 0)
```

The implementation handles this as a special case before computing `d1`/`d2` to avoid division by zero in the `d1` formula.

**Negative rates.** No special casing is needed. The formula handles `r < 0` directly — `exp(-r * T)` simply becomes greater than 1, meaning the discount factor amplifies rather than reduces the strike's present value.

**q > r.** No special casing. The forward `S * exp((r-q) * T)` naturally falls below `S` when `q > r`, reflecting the cost of dividends.

---

## Black-76 — `src/models/black76.rs`

**Source:** Black, F., "The pricing of commodity contracts", *Journal of Financial Economics*, 3(1-2):167-179 (1976).

### Derivation

Black-76 prices options on forwards and futures. The key difference from BSM is that the forward price `F` replaces the spot-dividend combination. Under the risk-neutral measure, the forward follows:

```
dF / F = sigma * dW
```

Note there is no drift term — the forward is already a martingale under the risk-neutral measure (since it prices to zero cost of entry). This eliminates the dividend yield parameter entirely.

The `d1` and `d2` for Black-76 are:

```
d1 = [ln(F / K) + sigma^2 * T / 2] / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)
```

The pricing formulas are:

```
Call = exp(-r * T) * [F * N(d1) - K * N(d2)]
Put  = exp(-r * T) * [K * N(-d2) - F * N(-d1)]
```

The `exp(-r * T)` discount factor appears because the payoff occurs at time `T` but the forward price `F` already reflects the cost of carry. When `F = S * exp((r - q) * T)` (the forward-spot relationship), these formulas reduce to the BSM formulas.

### Put-call parity (Black-76)

```
Call - Put = exp(-r * T) * (F - K)
```

At-the-money forward (`F = K`), call and put prices are identical. This is a direct consequence of the forward being driftless.

### Edge cases

**T = 0.** Returns `PricingError::IntrinsicOnly` with `max(F - K, 0)` for calls.

**sigma = 0.** Returns the discounted intrinsic `exp(-r * T) * max(F - K, 0)`. The implementation checks `sigma * sqrt(T) == 0` to catch this before the `d1` division.

---

## Bachelier (Normal Model) — `src/models/bachelier.rs`

**Source:** Bachelier, L., *Theorie de la Speculation*, Annales Scientifiques de l'Ecole Normale Superieure, 17:21-86 (1900). Schachermayer, W. & Teichmann, J., "How Close Are the Option Pricing Formulas of Bachelier and Black-Merton-Scholes?" (2008).

### Derivation

The Bachelier model assumes the forward price follows arithmetic (not geometric) Brownian motion:

```
dF = sigma_N * dW
```

where `sigma_N` is the **normal volatility** — an absolute quantity in price units, not a relative (percentage) quantity like Black-Scholes volatility.

Since `F_T = F + sigma_N * W_T` is normally distributed with mean `F` and variance `sigma_N^2 * T`, the expected payoff of a call is:

```
E[max(F_T - K, 0)] = (F - K) * N(d) + sigma_N * sqrt(T) * phi(d)
```

where:

```
d = (F - K) / (sigma_N * sqrt(T))
```

and `phi(d)` is the standard normal PDF. The discounted price is:

```
Call = exp(-r * T) * [(F - K) * N(d) + sigma_N * sqrt(T) * phi(d)]
Put  = exp(-r * T) * [(K - F) * N(-d) + sigma_N * sqrt(T) * phi(d)]
```

Note that the `phi(d)` term (the "time value" or "extrinsic value" component) is identical for calls and puts. Only the intrinsic component `(F - K) * N(d)` vs `(K - F) * N(-d)` differs.

### ATM proportionality

At the money (`F = K`), `d = 0`, so `N(0) = 0.5` and `phi(0) = 1/sqrt(2*pi)`. The ATM price simplifies to:

```
Call_ATM = Put_ATM = exp(-r * T) * sigma_N * sqrt(T) / sqrt(2 * pi)
```

This means the ATM Bachelier price is exactly proportional to `sigma_N` — doubling the normal vol doubles the ATM price. This linearity is tested explicitly.

### Put-call parity (Bachelier)

```
Call - Put = exp(-r * T) * (F - K)
```

Same form as Black-76. At-the-money forward, call equals put.

### Why Bachelier for rates

The log-normal models (BSM, Black-76) assume the underlying cannot go negative. For interest rates that can be near or below zero, the log of the rate is undefined or produces extreme volatility distortions. The Bachelier model handles negative underlyings naturally since arithmetic Brownian motion has no such constraint.

### Edge cases

**T = 0.** Returns intrinsic value via `PricingError::IntrinsicOnly`.

**sigma_N = 0.** Returns `exp(-r * T) * max(F - K, 0)` for calls. The implementation checks `sigma_N * sqrt(T) == 0` to avoid dividing by zero in the `d` formula.

---

## Displaced Log-Normal — `src/models/displaced.rs`

**Source:** Rubinstein, M., "Displaced Diffusion Option Pricing", *Journal of Finance*, 38(1):213-217 (1983).

### Derivation

The displaced diffusion model assumes that the shifted forward `F + beta` follows geometric Brownian motion:

```
d(F + beta) / (F + beta) = sigma * dW
```

where `beta` is the displacement parameter. This means `F + beta` is log-normally distributed, and we can apply the Black-76 formula with shifted inputs:

```
F' = F + beta    (shifted forward)
K' = K + beta    (shifted strike)
```

The pricing formulas are Black-76 applied to the shifted inputs:

```
d1 = [ln(F' / K') + sigma^2 * T / 2] / (sigma * sqrt(T))
d2 = d1 - sigma * sqrt(T)

Call = exp(-r * T) * [F' * N(d1) - K' * N(d2)]
Put  = exp(-r * T) * [K' * N(-d2) - F' * N(-d1)]
```

### Bridging property

The displaced model bridges two limiting cases:

- **beta = 0:** Reduces exactly to Black-76 (verified in tests to machine epsilon).
- **beta -> infinity:** As `beta` grows, the relative volatility `sigma * (F + beta)` approximates a constant absolute volatility `sigma * beta`, and the model converges to the Bachelier normal model with `sigma_N ~ sigma * beta`.

This bridging property makes the displaced model essential for SABR calibration, where the `beta` parameter (different from the displacement `beta` here, but conceptually related) controls the transition between normal and log-normal dynamics.

### Put-call parity (Displaced)

```
Call - Put = exp(-r * T) * (F - K)
```

The displacement cancels out in the parity relation because `F' - K' = (F + beta) - (K + beta) = F - K`.

### Edge cases

**F + beta <= 0 or K + beta <= 0.** Returns `PricingError::NegativeSpot` or `PricingError::NegativeStrike`. The shifted values must be positive for the log-normal assumption to hold.

**T = 0 or sigma = 0.** Returns `PricingError::IntrinsicOnly` with the discounted intrinsic value `exp(-r * T) * max(F - K, 0)`.

---

## Greeks — all 17, analytic — `src/greeks.rs`

**Source:** Black & Scholes, *JPE* (1973); Merton, *Bell Journal of Economics* (1973). All Greeks are derived by analytic differentiation of the BSM pricing formula.

### Intermediate caching

The following values are computed **once** and reused across all 17 Greek formulas:

```
d1, d2               (from math::d1, math::d2)
N(d1), N(d2)         (via ncdf)
N(-d1), N(-d2)       (via ncdf on negated arguments)
phi(d1), phi(d2)     (via npdf)
exp(-q * T), exp(-r * T)  (discount factors)
sigma * sqrt(T)      (vol-time product)
```

This caching strategy is critical for performance (< 80 ns for all 17 Greeks) and numerical consistency (all Greeks see exactly the same intermediate values).

### Notation

Throughout this section:

```
S     = spot price
K     = strike price
r     = risk-free rate (continuous, annualised)
q     = continuous dividend yield
sigma = implied volatility
T     = time to expiry in years
phi   = call (+1) or put (-1) indicator
```

---

### 1st order Greeks

#### Delta

Rate of change of price with respect to spot.

```
Delta_call = exp(-q * T) * N(d1)
Delta_put  = -exp(-q * T) * N(-d1)
```

**Derivation:** Differentiating the BSM call price `C = S * exp(-qT) * N(d1) - K * exp(-rT) * N(d2)` with respect to `S`. The terms involving `dN(d1)/dS` and `dN(d2)/dS` cancel exactly (this cancellation is a well-known result from risk-neutral pricing theory), leaving only the `exp(-qT) * N(d1)` coefficient.

**Properties:**
- Call delta in `(0, 1)` for all valid inputs
- Put delta in `(-1, 0)` for all valid inputs
- Delta parity: `Delta_call - Delta_put = exp(-q * T)`

#### Theta

Rate of change of price with respect to time (expressed per calendar day).

```
Theta_annual_call = -(S * exp(-qT) * phi(d1) * sigma) / (2 * sqrt(T))
                    - r * K * exp(-rT) * N(d2)
                    + q * S * exp(-qT) * N(d1)

Theta_annual_put  = -(S * exp(-qT) * phi(d1) * sigma) / (2 * sqrt(T))
                    + r * K * exp(-rT) * N(-d2)
                    - q * S * exp(-qT) * N(-d1)

Theta = Theta_annual / 365
```

**Derivation:** Differentiating the BSM price with respect to `T` (with the convention that theta is `dV/dT`, not `-dV/dT`). The first term comes from differentiating the `N(d1)` and `N(d2)` terms (via the chain rule through `d1` and `d2`, which both depend on `T`). The second and third terms come from the time dependence of the discount factors `exp(-rT)` and `exp(-qT)`.

**Convention:** The implementation divides the annualised theta by 365 to give a per-calendar-day figure. This is the market convention for equity options.

**Sign:** Theta is typically negative for long option positions (time decay erodes value). For calls, theta is more negative than for puts at the same strike/expiry because the rho and dividend terms partially offset time decay for puts.

#### Vega

Rate of change of price with respect to volatility.

```
Vega = S * exp(-q * T) * phi(d1) * sqrt(T)
```

**Derivation:** Differentiating BSM with respect to `sigma`. As with delta, the cross-terms from `dN(d1)/dsigma` and `dN(d2)/dsigma` cancel, leaving only the direct dependence through the `phi(d1)` density evaluated at `d1`.

**Properties:** Vega is always positive (higher vol means higher option value) and is symmetric for calls and puts (same formula for both).

#### Rho

Rate of change of price with respect to the risk-free rate.

```
Rho_call = K * T * exp(-r * T) * N(d2)
Rho_put  = -K * T * exp(-r * T) * N(-d2)
```

**Derivation:** Differentiating BSM with respect to `r`. The dominant term is from `d(exp(-rT))/dr = -T * exp(-rT)` applied to the strike leg. The spot leg's dependence on `r` through `d1` and `d2` again cancels.

**Properties:** Rho is positive for calls (higher rates increase the present value advantage of deferring the strike payment) and negative for puts.

#### Epsilon

Rate of change of price with respect to the dividend yield.

```
Epsilon_call = -T * S * exp(-q * T) * N(d1)
Epsilon_put  =  T * S * exp(-q * T) * N(-d1)
```

**Derivation:** Differentiating BSM with respect to `q`. Symmetric to rho but with opposite sign convention — higher dividends reduce the forward and hence reduce call value.

#### Lambda (leverage / elasticity)

Percentage change in price per percentage change in spot.

```
Lambda = Delta * S / V
```

where `V` is the option price. Lambda represents the option's leverage — for a call, lambda is always greater than 1, meaning the option amplifies spot movements.

**Guard:** When the option price `V` is near zero (< 1e-30), lambda is set to 0.0 to avoid division by zero.

#### Dual Delta

Rate of change of price with respect to strike.

```
DualDelta_call = -exp(-r * T) * N(d2)
DualDelta_put  =  exp(-r * T) * N(-d2)
```

**Derivation:** Differentiating BSM with respect to `K`. For calls, increasing the strike always reduces the price (the right to buy at a higher price is less valuable), so dual delta is negative.

---

### 2nd order Greeks

#### Gamma

Rate of change of delta with respect to spot. The curvature of the price surface.

```
Gamma = exp(-q * T) * phi(d1) / (S * sigma * sqrt(T))
```

**Derivation:** Differentiating delta with respect to `S`:

```
d(Delta)/dS = exp(-qT) * d(N(d1))/dS = exp(-qT) * phi(d1) * d(d1)/dS
```

Since `d(d1)/dS = 1 / (S * sigma * sqrt(T))`, this gives the formula above.

**Properties:** Gamma is always positive (convexity — the option holder benefits from large moves in either direction). Gamma is symmetric for calls and puts — both have the same curvature.

**Edge case:** When `sigma * sqrt(T) = 0`, gamma is returned as 0.0 (degenerate case where the price is a piecewise linear function of spot with a kink at the strike).

#### Vanna

Cross-derivative of price with respect to spot and volatility. Equivalently: `d(Delta)/d(sigma)` or `d(Vega)/d(S)`.

```
Vanna = -exp(-q * T) * phi(d1) * d2 / sigma
```

**Derivation:** Differentiating delta with respect to `sigma`, or equivalently differentiating vega with respect to `S`. The equality of these two cross-derivatives follows from the smoothness of the BSM price function (Schwarz's theorem).

**Properties:** Symmetric for calls and puts.

#### Charm (delta decay)

Rate of change of delta with respect to time. Tells how delta drifts as time passes.

```
Charm_call = q * exp(-qT) * N(d1)
           - exp(-qT) * phi(d1) * [2(r-q)T - d2 * sigma * sqrt(T)]
             / (2 * T * sigma * sqrt(T))

Charm_put  = -q * exp(-qT) * N(-d1)
           + exp(-qT) * phi(d1) * [2(r-q)T - d2 * sigma * sqrt(T)]
             / (2 * T * sigma * sqrt(T))
```

**Derivation:** `d(Delta)/d(T)` via the chain rule. The first term comes from the time dependence of the discount factor `exp(-qT)`, and the second from the time dependence of `N(d1)` through `d1(T)`.

**Use case:** Charm is critical for delta-hedging — it tells you how much your delta hedge will drift overnight.

#### Veta (vega decay)

Rate of change of vega with respect to time.

```
Veta = -S * exp(-qT) * phi(d1) * sqrt(T)
       * [q + (r - q) * d1 / (sigma * sqrt(T))
          - (1 + d1 * d2) / (2 * T)]
```

**Derivation:** `d(Vega)/d(T)`. The outer factor `S * exp(-qT) * phi(d1) * sqrt(T)` is vega itself; the bracketed expression is the logarithmic derivative of vega with respect to time.

**Properties:** Symmetric for calls and puts.

#### Vomma (volga / vega convexity)

Rate of change of vega with respect to volatility.

```
Vomma = Vega * d1 * d2 / sigma
```

**Derivation:** `d(Vega)/d(sigma)`. Since `Vega = S * exp(-qT) * phi(d1) * sqrt(T)`, differentiating with respect to `sigma` involves `d(phi(d1))/d(sigma) = -phi(d1) * d1 * d(d1)/d(sigma)` and `d(d1)/d(sigma)` simplifies to yield the `d1 * d2 / sigma` factor.

**Properties:** Symmetric for calls and puts. Also known as volga in the FX options market.

#### Dual Gamma

Second derivative of price with respect to strike.

```
DualGamma = exp(-r * T) * phi(d2) / (K * sigma * sqrt(T))
```

**Derivation:** Differentiating dual delta with respect to `K`. Analogous to gamma but on the strike axis instead of the spot axis. Note it uses `phi(d2)` rather than `phi(d1)`.

**Properties:** Always positive. Symmetric for calls and puts.

---

### 3rd order Greeks

#### Speed

Rate of change of gamma with respect to spot.

```
Speed = -Gamma * (d1 / (sigma * sqrt(T)) + 1) / S
```

**Derivation:** `d(Gamma)/d(S)`. Differentiating gamma involves `d(phi(d1))/dS = -phi(d1) * d1 / (S * sigma * sqrt(T))` and the `1/S` factor from gamma itself.

**Properties:** Symmetric for calls and puts. Tells how gamma changes as the underlying moves — important for gamma scalping strategies.

#### Zomma

Rate of change of gamma with respect to volatility.

```
Zomma = Gamma * (d1 * d2 - 1) / sigma
```

**Derivation:** `d(Gamma)/d(sigma)`. Similar structure to vomma but applied to gamma instead of vega.

**Properties:** Symmetric for calls and puts.

#### Color (gamma decay)

Rate of change of gamma with respect to time.

```
Color = -exp(-qT) * phi(d1)
        / (2 * S * T * sigma * sqrt(T))
        * [2 * q * T + 1
           + d1 * (2(r-q)T - d2 * sigma * sqrt(T)) / (sigma * sqrt(T))]
```

**Derivation:** `d(Gamma)/d(T)`. The most complex Greek formula — it involves the time derivative of both the `phi(d1)` density (through `d1(T)`) and the `1/(S * sigma * sqrt(T))` scaling factor.

**Use case:** Color tells you how your gamma exposure changes overnight, which is critical for portfolio management near expiry when gamma can change rapidly.

#### Ultima

Third derivative of price with respect to volatility.

```
Ultima = -(Vega / sigma^2) * [d1 * d2 * (1 - d1 * d2) + d1^2 + d2^2]
```

**Derivation:** `d^3(V)/d(sigma)^3`. Obtained by differentiating vomma with respect to sigma. The expression inside the brackets involves the third-order interaction between `d1` and `d2` through their sigma dependence.

**Properties:** Symmetric for calls and puts. Rarely used in practice but included for completeness in the 17-Greek set.

---

### Edge cases for all Greeks

**sigma * sqrt(T) = 0.** All Greeks that involve division by `sigma * sqrt(T)` (gamma, vanna, charm, veta, vomma, speed, zomma, color, ultima, dual gamma) return 0.0 in this degenerate case. The price surface becomes a piecewise linear function at expiry, so all curvature-related Greeks vanish.

**sigma = 0 alone.** Greeks that divide by `sigma` (vanna, vomma, zomma, ultima) also return 0.0.

**V ~ 0 (near-zero price).** Lambda uses a floor check (`|V| > 1e-30`) to avoid division by zero.

---

## Implied Volatility Solver Chain — `src/iv.rs`

**Problem statement:** Given an observed market price `V_market` for a European option with known parameters `(S, K, r, q, T, type)`, find the volatility `sigma` such that `BS_price(sigma) = V_market`.

This is a root-finding problem: find `sigma` such that `f(sigma) = BS_price(sigma) - V_market = 0`.

The difficulty is that `f(sigma)` is highly nonlinear, can have near-zero derivatives (vega) for deep OTM/ITM options, and must be solved rapidly (< 150 ns target) for real-time pricing applications.

### Constants

```
IV_TOL    = 1e-10    convergence tolerance |sigma_new - sigma_old|
IV_LOWER  = 1e-8     search domain lower bound
IV_UPPER  = 100.0    search domain upper bound
VEGA_FLOOR = 1e-12   minimum vega for Newton/Halley to proceed
MAX_ITER_HALLEY_NEWTON = 100
MAX_ITER_BRENT = 50
```

### Intrinsic value validation

Before any solver runs, the implementation checks that the market price is above the discounted intrinsic value:

```
Intrinsic_call = max(S * exp(-qT) - K * exp(-rT), 0)
Intrinsic_put  = max(K * exp(-rT) - S * exp(-qT), 0)
```

If `V_market < Intrinsic - 1e-10`, no valid volatility can produce that price (the BS price is always at least the intrinsic value), and `IvError::BelowIntrinsic` is returned.

---

### Step 1: Corrado-Miller initial guess

**Source:** Corrado, C.J. & Miller, T.W., "A note on a simple, accurate formula to compute implied standard deviations", *Journal of Banking & Finance*, 20:595-603 (1996).

The Corrado-Miller formula provides a closed-form approximation to implied volatility, typically accurate to 1-2 significant figures. It avoids the need for an arbitrary starting guess.

For puts, the market price is first converted to an equivalent call price via put-call parity:

```
C = P + S * exp(-qT) - K * exp(-rT)
```

Define the discounted forward and strike:

```
S' = S * exp(-qT)
K' = K * exp(-rT)
```

The formula is:

```
C_adj = C - (S' - K') / 2
inner = C_adj^2 - (S' - K')^2 / pi
correction = sqrt(max(inner, 0))
sigma_0 = sqrt(2 * pi / T) * (C_adj + correction) / (S' + K')
```

**Rationale for the formula:** Corrado-Miller (1996) derives this by expanding the Black-Scholes formula around the at-the-money point and inverting the resulting expression. The `C_adj` term removes the intrinsic component, and the `correction` term accounts for the curvature of the Black-Scholes formula away from ATM.

**Guard:** If `C_adj <= 0` (very deep ITM/OTM), a simpler estimate `sqrt(2*pi) * C / (mid * sqrt(T))` is used. A floor of `0.001` prevents starting at zero vol.

---

### Step 2: Halley's method (3rd order) — primary solver

**Source:** Standard numerical analysis; the application to IV follows from the BS price being a smooth, monotone function of sigma.

Halley's method achieves cubic convergence by using the first and second derivatives of `f(sigma)`:

```
f(sigma) = BS_price(sigma) - V_market
f'(sigma) = Vega = S * exp(-qT) * phi(d1) * sqrt(T)
f''(sigma) = Vomma = Vega * d1 * d2 / sigma
```

The Newton step is:

```
newton_step = f / f'
```

The Halley correction modifies this to:

```
halley_denom = 1 - (newton_step * f'') / (2 * f')
             = 1 - (1/2) * newton_step * d1 * d2 / sigma

sigma_new = sigma - newton_step / halley_denom
```

**Convergence:** Halley's method has cubic convergence (`|e_{n+1}| ~ C * |e_n|^3`) compared to Newton's quadratic convergence. From a Corrado-Miller initial guess accurate to ~1%, Halley typically converges in 2-3 iterations.

**Stability guard:** When `|halley_denom| < 0.1`, the Halley correction is unstable (the denominator is too close to zero). In this case, the implementation falls back to the plain Newton step for that iteration. This can happen when `d1 * d2` is large (deep ITM/OTM).

**Convergence criterion:** `|sigma_new - sigma_old| < 1e-10`. After each step, the result is clamped to `[1e-8, 100.0]`.

**Failure:** Returns `IvError::NearZeroVega` if vega drops below `1e-12` (the price surface is flat, so no root-finding method based on derivatives can proceed). Returns `IvError::MaxIterationsReached` after 100 iterations.

---

### Step 3: Newton-Raphson (2nd order) — fallback

**Source:** Standard numerical analysis.

Simpler than Halley — uses only the first derivative:

```
sigma_new = sigma - f(sigma) / f'(sigma)
          = sigma - (BS_price(sigma) - V_market) / Vega
```

**Convergence:** Quadratic (`|e_{n+1}| ~ C * |e_n|^2`). More robust than Halley because it does not require the vomma correction, which can introduce instability when `d1 * d2 / sigma` is poorly conditioned.

**Why it is the fallback, not the primary:** For well-behaved inputs (ATM, moderate OTM), Halley converges in 2-3 iterations vs Newton's 4-6. The Halley savings justify trying it first, with Newton as a reliable second choice.

**Same convergence criteria and failure modes as Halley.**

---

### Step 4: Jackel-inspired rational approximation — deep OTM/ITM

**Source:** Jackel, P., "Let's Be Rational", *Wilmott Magazine* (2016).

The Jackel approach is designed for regimes where Newton and Halley fail — specifically, deep out-of-the-money or in-the-money options where vega is near zero. The key insight is to work with a normalized Black price rather than the absolute price.

**Phase 1: Improved initial guess.** The implementation uses the normalized moneyness:

```
x = ln(F / K)     where F = S * exp((r - q) * T)  is the forward
```

- **Near ATM (|x| < 0.5):** Uses the Brenner-Subrahmanyam (1988) approximation: `sigma ~ sqrt(2*pi) * C / (S * exp(-qT) * sqrt(T))`
- **Away from ATM:** Uses an intrinsic-adjusted form based on the normalised time value and a rational function of the log-moneyness

**Phase 2: Newton refinement with Halley acceleration.** From the improved initial guess, the implementation runs Newton-Halley iterations (same as Step 2) but with a much tighter residual check (`|residual| < 1e-14`). Per Jackel (2016), a good initial guess plus 2 Newton iterations achieves full double-precision accuracy.

**Fallback:** If vega drops below `1e-30` during refinement, the implementation gives up on derivative-based methods and falls through to Brent's method.

---

### Step 5: Brent's method — last resort

**Source:** Brent, R., *Algorithms for Minimization Without Derivatives*, Prentice-Hall (1973), Chapter 4.

Brent's method is a bracketed root-finding algorithm that combines three strategies:

1. **Bisection** — always converges, worst-case log2(n) iterations
2. **Secant method** — superlinear convergence when the function is smooth
3. **Inverse quadratic interpolation** — even faster when three distinct function values are available

The algorithm maintains a bracket `[a, b]` such that `f(a)` and `f(b)` have opposite signs (guaranteeing a root exists between them). At each step, it tries the fastest method first (inverse quadratic, then secant) and falls back to bisection if the faster methods would step outside the bracket or converge too slowly.

**Initial bracket:** `[IV_LOWER, IV_UPPER] = [1e-8, 100.0]`. If `f(IV_LOWER)` and `f(IV_UPPER)` have the same sign, no root exists in the bracket and `IvError::NoSolution` is returned.

**Convergence:** Brent's method converges at least as fast as bisection (linear, with the bracket halving each iteration) and in practice achieves superlinear convergence for smooth functions. It never diverges — the bracket always contains the root.

**Why it is the last resort:** Brent requires evaluating `BS_price(sigma)` at each iteration (no analytical shortcut), and the superlinear convergence is slower than Halley's cubic convergence. For a typical input, Brent needs 15-30 iterations vs Halley's 2-3. But it is the only method that is **guaranteed** to converge for any input where a root exists.

**Convergence criterion:** `|b - a| < 1e-10` or `|f(b)| < 1e-14`.

---

### Why the chain order matters

The solver chain `Halley -> Newton -> Jackel -> Brent` is ordered by:

1. **Speed** (fastest first): Halley (cubic, ~2-3 iters) > Newton (quadratic, ~4-6 iters) > Jackel (improved guess + Newton, ~3-5 iters) > Brent (bracket, ~15-30 iters).

2. **Robustness** (most robust last): Halley can diverge when vomma is poorly conditioned; Newton can diverge when vega is near zero; Jackel can fail for extreme parameters; Brent never diverges.

3. **Domain coverage**: Halley covers ~90% of real-world inputs (moderate moneyness, reasonable vol). Newton catches the cases where Halley's vomma correction was unstable. Jackel handles deep OTM/ITM where both fail due to near-zero vega. Brent catches everything else.

The Corrado-Miller initial guess runs unconditionally before the chain starts, ensuring all iterative methods begin from a reasonable starting point rather than an arbitrary guess.

### Post-validation

After any solver succeeds, the result is checked against the bounds `[IV_LOWER, IV_UPPER]`. A result outside these bounds produces `IvError::BoundsExceeded` — this prevents returning a "converged" but nonsensical volatility (e.g., sigma = -0.001 from a Newton overshoot that happened to satisfy the tolerance).

---

## Algorithm references

| Algorithm | Primary reference |
|---|---|
| Normal CDF (erfc-based) | Cody, *Math. Comp.* 22(107):631-637 (1969); Sun fdlibm `s_erf.c` |
| Normal CDF (tables) | Abramowitz & Stegun, section 26.2.17 (1964) |
| Black-Scholes-Merton | Black & Scholes, *JPE* 81(3):637-654 (1973); Merton, *BJEMS* 4(1):141-183 (1973) |
| Black-76 | Black, *JFE* 3(1-2):167-179 (1976) |
| Bachelier model | Bachelier, *Ann. Sci. ENS* 17:21-86 (1900); Schachermayer & Teichmann (2008) |
| Displaced log-normal | Rubinstein, *J. Finance* 38(1):213-217 (1983) |
| Corrado-Miller IV guess | Corrado & Miller, *JBF* 20:595-603 (1996) |
| Brenner-Subrahmanyam | Brenner & Subrahmanyam, *Financial Analysts Journal* 44(5):80-83 (1988) |
| Halley's method | Halley, *Phil. Trans.* 18:136-148 (1694) |
| Newton-Raphson | Newton, *De analysi* (1669); Raphson, *Analysis Aequationum* (1690) |
| Jackel "Let's Be Rational" | Jackel, *Wilmott Magazine* (2016) |
| Brent's method | Brent, *Algorithms for Minimization Without Derivatives*, Prentice-Hall (1973) |

---

*Part of [Regit OS](https://www.regit.io) — the operating system for investment products. From Luxembourg.*
