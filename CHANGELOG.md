# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - Unreleased

### Added

#### Core types (`src/types.rs`)
- `Float` trait — minimal float abstraction satisfied by `f32` and `f64`, exposing
  `zero`, `one`, `ln`, `exp`, `sqrt`, `abs`, `mul_add`, `from_f64`, `to_f64`,
  `pi`, `is_nan`, `is_infinite`; both primitives implement it with `#[inline(always)]`
- `OptionType` enum — `Call` / `Put`; `Copy`, `Clone`, `PartialEq`, `Eq`, `Hash`, `Debug`
- `OptionParams<F: Float>` struct — `option_type`, `spot`, `strike`, `rate`,
  `div_yield`, `vol`, `time`; generic over `F`, `Copy`
- `Greeks<F: Float>` struct — all 17 analytic Greeks: `delta`, `gamma`, `theta`,
  `vega`, `rho`, `epsilon`, `lambda`, `vanna`, `charm`, `veta`, `vomma`, `speed`,
  `zomma`, `color`, `ultima`, `dual_delta`, `dual_gamma`; generic over `F`, `Copy`

#### Error types (`src/errors.rs`)
- `PricingError` enum — `NegativeSpot`, `NegativeStrike`, `NegativeTime`,
  `NegativeVolatility`, `IntrinsicOnly { intrinsic: f64 }`; implements `Display`,
  `Debug`, `std::error::Error`, `Copy`, `Clone`, `PartialEq`
- `IvError` enum — `NoSolution`, `BelowIntrinsic { intrinsic: f64 }`,
  `MaxIterationsReached { last_vol: f64, residual: f64 }`, `NearZeroVega`,
  `BoundsExceeded { vol: f64 }`; implements `Display`, `Debug`,
  `std::error::Error`, `Copy`, `Clone`, `PartialEq`

#### Mathematical primitives (`src/math.rs`)
- `ncdf(x: f64) -> f64` — standard normal CDF via hand-rolled `erfc` using
  Sun fdlibm / Cody (1969) rational Chebyshev coefficients; four domain regions
  (tiny, small, medium, large) targeting max absolute error < 1e-15
- `ncdf_complement(x: f64) -> f64` — complementary CDF (`1 - N(x)`) with
  improved precision for large positive `x`
- `npdf(x: f64) -> f64` — standard normal PDF with overflow guard (returns 0.0
  when `x²/2 > 700`)
- `d1(spot, strike, rate, div_yield, vol, time) -> f64` — log-moneyness adjusted
  drift term; `#[inline(always)]`
- `d2(d1_val, vol, time) -> f64` — `d1 − σ√T`; `#[inline(always)]`
- All polynomial evaluation uses `f64::mul_add` exclusively (Horner's method)
- Published constants: `FRAC_1_SQRT_2PI`, `SQRT_2`

#### Pricing models

**Black-Scholes-Merton** (`src/models/black_scholes.rs`)
- `validate<F: Float>(params: &OptionParams<F>) -> Result<(), PricingError>` —
  checks non-negative spot, strike, time, vol; returns `IntrinsicOnly` at `T = 0`
- `price<F: Float>(params: &OptionParams<F>) -> Result<f64, PricingError>` —
  continuous-dividend Merton (1973) form; handles `σ = 0` (discounted intrinsic),
  `σ → ∞` (bounded), and negative rates

**Black-76** (`src/models/black76.rs`)
- `Black76Params` struct — `option_type`, `forward`, `strike`, `rate`, `vol`, `time`
- `price(params: &Black76Params) -> Result<f64, PricingError>` — futures/forwards
  pricing (Black 1976); discount factor `exp(-rT)` applied to `[F·N(d1) - K·N(d2)]`

**Bachelier** (`src/models/bachelier.rs`)
- `BachelierParams` struct — `option_type`, `forward`, `strike`, `rate`,
  `normal_vol`, `time`; `normal_vol` is absolute price-unit vol, not percentage
- `price(params: &BachelierParams) -> Result<f64, PricingError>` — normal
  (Bachelier 1900) model for rates near/below zero; formula uses `(F-K)·N(d) + σ_N·√T·φ(d)`

**Displaced Log-Normal** (`src/models/displaced.rs`)
- `DisplacedParams` struct — `option_type`, `forward`, `strike`, `rate`, `vol`,
  `time`, `displacement`; bridges Black-76 (`β=0`) and Bachelier (`β→∞`)
- `price(params: &DisplacedParams) -> Result<f64, PricingError>` — Rubinstein
  (1983) shifted log-normal; reduces to Black-76 on shifted inputs `F'=F+β`, `K'=K+β`

#### Analytic Greeks (`src/greeks.rs`)
- `compute_greeks(params: &OptionParams<f64>) -> Result<Greeks<f64>, PricingError>` —
  all 17 Greeks computed analytically from a single shared set of intermediates
  (`d1`, `d2`, `N(d1)`, `N(d2)`, `φ(d1)`); no finite differences, no recomputation;
  `#[inline(always)]`
- 1st order: delta, theta (per calendar day), vega, rho, epsilon, lambda, dual delta
- 2nd order: gamma, vanna, charm, veta, vomma, dual gamma
- 3rd order: speed, zomma, color, ultima
- Numerically stable at `T → 0`, `σ → 0`, deep ITM, deep OTM

#### Implied volatility solver (`src/iv.rs`)
- `IvSolver` enum — `Auto`, `Halley`, `Newton`, `Jackel`, `Brent`
- `implied_vol(params: &OptionParams<f64>, market_price: f64, solver: IvSolver) -> Result<f64, IvError>` —
  multi-strategy chain: Corrado-Miller (1996) initial guess → Halley (3rd order,
  primary) → Newton-Raphson (2nd order fallback) → Jackel "Let's Be Rational"
  inspired rational approximation (deep OTM/ITM) → Brent bracketed bisection
  (last resort); convergence tolerance `1e-10`, search bounds `[1e-8, 100.0]`,
  max 100 iterations (Halley/Newton), 50 iterations (Brent); pure Rust, no FFI

#### Integration tests (`tests/integration.rs`) — 109 tests
- `golden` module — regression anchors for BS, Black-76, Bachelier, and Displaced
  prices against computed reference values (ATM, OTM, ITM call and put; negative
  rate; high vol; short-dated; long-dated; zero dividend; high dividend)
- `parity` module — put-call parity for all four models; Black-76 / displaced
  equivalence at `β=0`; Bachelier / displaced convergence at large `β`;
  symmetry and monotonicity identities
- `greeks_tests` module — cross-Greek relationships (call+put delta sum,
  gamma equality, vega equality, rho/epsilon signs); finite-difference verification
  of delta and vega at `STANDARD` tolerance; golden value anchors for all 17 Greeks
- `boundaries` module — `T=0` intrinsic, `σ=0` discounted intrinsic, deep OTM
  near-zero price, deep ITM near-intrinsic, large spot/strike/vol/time stability,
  negative rates, zero dividend, IV round-trip recovery, IV error variants
- `properties` module — proptest invariants (1,000 cases each): call delta in
  `[0,1]`, put delta in `[-1,0]`, gamma non-negative, vega non-negative, price
  monotone increasing in vol

#### Criterion benchmarks (`benches/blackscholes.rs`)
- `math` group — `ncdf`, `npdf` (target: < 5 ns each)
- `pricing` group — `bs_call_price`, `bs_put_price`, `black76_price`,
  `bachelier_price` (target: < 15 ns)
- `greeks` group — `greeks_call`, `greeks_put` (target: < 80 ns)
- `iv` group — `iv_atm_auto`, `iv_deep_otm_auto`, `iv_atm_halley`,
  `iv_atm_newton` (targets: < 150 ns ATM, < 300 ns deep OTM)

#### Project infrastructure
- `#![forbid(unsafe_code)]` crate-wide; zero-dependency (no `statrs`, `libm`,
  `nalgebra`, `cc` FFI); `std`-only implementation
- `approx` and `proptest` as dev-dependencies for test tolerances and property tests
- Apache-2.0 license; copyright headers on every `.rs` file
